import os
import torch
import torch.nn as nn
from torch import autograd
from model.networks import Generator, LocalDis, GlobalDis
from utils.tools import get_model_list, local_patch, spatial_discounting_mask
from utils.logger import get_logger
from torchvision import utils as vutils

logger = get_logger()

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids, 
                              dropout_p=self.config.get('dropout_p', 0.3))
        self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.localD.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

        # 使用字典存储每个 sample_id 的历史 inpainting map，保存最近 5 张
        self.inpaint_history_dict = {}
        # 一致性损失权重
        self.consistency_loss_alpha = self.config.get('consistency_loss_alpha', 0.1)

    def forward(self, x, masks, ground_truth, sample_ids, steps, compute_g_loss=False, past_input=None):
        """
        x: 当前部分地图, shape [B, input_dim, H, W]
        masks: 遮罩, shape [B, 1, H, W]
        ground_truth: 完整地图, shape [B, 1, H, W]
        sample_ids: [B]，每个元素为对应的地图ID（整型）
        steps: [B]，每个元素为当前探索步数（整型）
        past_input: 过去提示, shape [B, past_channels, H, W]；若为 None，则内部补零
        compute_g_loss: 是否计算生成器损失
        """
        if self.training:
            l1_loss_fn = nn.L1Loss()
            losses = {}

            # 前向传播：将过去提示传入 Generator
            x1, x2, offset_flow = self.netG(x, masks, past=past_input, dropout=True)
            x1_inpaint = x1 * masks + x * (1. - masks)
            x2_inpaint = x2 * masks + x * (1. - masks)

            # 计算按样本逐一对齐的一致性损失（只对比历史中步数小于当前的结果）
            consistency_list = []
            B = x2_inpaint.size(0)
            for i in range(B):
                sid = sample_ids[i].item() if torch.is_tensor(sample_ids) else int(sample_ids[i])
                stp = steps[i].item() if torch.is_tensor(steps) else int(steps[i])
                curr_map = x2_inpaint[i:i+1]  # shape: [1, C, H, W]
                if sid in self.inpaint_history_dict:
                    local_sum = 0.0
                    count = 0
                    for (old_step, old_map) in self.inpaint_history_dict[sid]:
                        if old_step < stp:
                            local_sum += l1_loss_fn(curr_map, old_map)
                            count += 1
                    if count > 0:
                        local_sum = local_sum / count
                    consistency_list.append(local_sum)
                else:
                    consistency_list.append(torch.tensor(0.0, device=x2_inpaint.device))
            if len(consistency_list) > 0:
                consistency_loss = sum(consistency_list) / len(consistency_list)
            else:
                consistency_loss = 0.0
            losses['consistency'] = consistency_loss

            # 判别器部分：WGAN D loss
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                self.localD, ground_truth, x2_inpaint.detach())
            global_real_pred, global_fake_pred = self.dis_forward(
                self.globalD, ground_truth, x2_inpaint.detach())
            losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
                torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']

            # 梯度惩罚
            local_penalty = self.calc_gradient_penalty(self.localD, ground_truth, x2_inpaint.detach())
            global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
            losses['wgan_gp'] = local_penalty + global_penalty

            # 生成器部分
            if compute_g_loss:
                sd_mask = spatial_discounting_mask(self.config)
                losses['l1'] = l1_loss_fn(x1_inpaint * sd_mask, ground_truth * sd_mask) * \
                    self.config['coarse_l1_alpha'] + \
                    l1_loss_fn(x2_inpaint * sd_mask, ground_truth * sd_mask)
                losses['ae'] = l1_loss_fn(x1 * (1. - masks), ground_truth * (1. - masks)) * \
                    self.config['coarse_l1_alpha'] + \
                    l1_loss_fn(x2 * (1. - masks), ground_truth * (1. - masks))
                local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                    self.localD, ground_truth, x2_inpaint)
                global_real_pred, global_fake_pred = self.dis_forward(
                    self.globalD, ground_truth, x2_inpaint)
                losses['wgan_g'] = - torch.mean(local_patch_fake_pred) - \
                    torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

                losses['g'] = (losses['l1'] * self.config['l1_loss_alpha'] +
                               losses['ae'] * self.config['ae_loss_alpha'] +
                               losses['wgan_g'] * self.config['gan_loss_alpha'] +
                               self.consistency_loss_alpha * consistency_loss)

            # 更新历史缓存：对每个样本更新其历史 inpainting 输出（只保留最近5张）
            for i in range(B):
                sid = sample_ids[i].item() if torch.is_tensor(sample_ids) else int(sample_ids[i])
                stp = steps[i].item() if torch.is_tensor(steps) else int(steps[i])
                curr_map = x2_inpaint[i:i+1].detach()
                if sid not in self.inpaint_history_dict:
                    self.inpaint_history_dict[sid] = []
                self.inpaint_history_dict[sid].append((stp, curr_map))
                if len(self.inpaint_history_dict[sid]) > 5:
                    self.inpaint_history_dict[sid].pop(0)

            return losses, x2_inpaint, offset_flow

        else:
            # Evaluation mode：传入 past_input 同样生效
            with torch.no_grad():
                x1, x2, offset_flow = self.netG(x, masks, past=past_input, dropout=False)
                x2_inpaint = x2 * masks + x * (1. - masks)
            return {}, x2_inpaint, offset_flow

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)
        return real_pred, fake_pred

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()
        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size()).to(real_data.device)
        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def inference(self, x, masks, past_input=None):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks, past=past_input, dropout=False)
        x2_inpaint = x2 * masks + x * (1. - masks)
        return x2_inpaint, offset_flow

    def inference_with_uncertainty(self, x, masks, num_samples=10, past_input=None):
        self.eval()
        predictions = []
        for _ in range(num_samples):
            self.netG.train()
            with torch.no_grad():
                x1, x2, offset_flow = self.netG(x, masks, past=past_input, dropout=True)
                x2_inpaint = x2 * masks + x * (1. - masks)
                predictions.append(x2_inpaint.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        self.eval()
        print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
        return mean, std, offset_flow

    def save_model(self, checkpoint_dir, iteration):
        if isinstance(iteration, int):
            gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
            dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        else:
            gen_name = os.path.join(checkpoint_dir, f'gen_{iteration}.pt')
            dis_name = os.path.join(checkpoint_dir, f'dis_{iteration}.pt')
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save({'localD': self.localD.state_dict(),
                    'globalD': self.globalD.state_dict()}, dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(),
                    'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        self.netG.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])
        if not test:
            last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            state_dict = torch.load(last_model_name)
            self.localD.load_state_dict(state_dict['localD'])
            self.globalD.load_state_dict(state_dict['globalD'])
            state_dict = torch.load(os.path.join(checkpoint_path, 'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            self.optimizer_g.load_state_dict(state_dict['gen'])
        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        return iteration
