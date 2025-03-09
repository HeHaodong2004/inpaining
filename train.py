import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from trainer import Trainer
from data.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image
from utils.logger import get_logger

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    writer = SummaryWriter(logdir=checkpoint_path)
    logger = get_logger(checkpoint_path)

    logger.info("Arguments: {}".format(args))
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Configuration: {}".format(config))

    try:
        # 加载训练数据集（返回：ground_truth, x, mask, sample_ids, steps）
        logger.info("Training on dataset: {}".format(config['dataset_name']))
        train_dataset = Dataset(data_path=config['train_data_path'],
                                image_shape=config['image_shape'],
                                random_crop=config['random_crop'])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'])
        # 加载验证数据集
        logger.info("Validating on dataset: {}".format(config['dataset_name']))
        val_dataset = Dataset(data_path=config['val_data_path'],
                              image_shape=config['image_shape'],
                              random_crop=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=config['batch_size'],
                                                 shuffle=False,
                                                 num_workers=config['num_workers'])

        # 定义 Trainer
        trainer = Trainer(config)
        logger.info("\n{}".format(trainer.netG))
        logger.info("\n{}".format(trainer.localD))
        logger.info("\n{}".format(trainer.globalD))
        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1
        iterable_train_loader = iter(train_loader)
        time_count = time.time()
        best_val_loss = float('inf')
        past_channels = trainer_module.netG.past_channels  # 例如 5

        for iteration in range(start_iteration, config['niter'] + 1):
            trainer_module.train()
            try:
                ground_truth, x, mask, sample_ids, steps = next(iterable_train_loader)
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                ground_truth, x, mask, sample_ids, steps = next(iterable_train_loader)
            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()
            '''B, _, H, W = x.shape

            # 构造 past_input：对每个样本构造形状为 [1, past_channels, H, W] 的提示
            past_input_list = []
            for i in range(B):
                sid = sample_ids[i].item() if torch.is_tensor(sample_ids) else int(sample_ids[i])
                stp = steps[i].item() if torch.is_tensor(steps) else int(steps[i])
                # 从历史字典中取该样本的历史 inpainting 输出（只取步数小于当前的）
                if sid in trainer_module.inpaint_history_dict:
                    hist = []
                    for (old_step, old_map) in trainer_module.inpaint_history_dict[sid]:
                        if old_step < stp:
                            # 确保历史记录形状为 [1, 1, H, W]
                            if old_map.dim() != 4:
                                old_map = old_map.view(1, 1, H, W)
                            hist.append(old_map)
                    if len(hist) < past_channels:
                        pad = [torch.zeros(1, 1, H, W, device=x.device) for _ in range(past_channels - len(hist))]
                        hist = pad + hist
                    else:
                        hist = hist[-past_channels:]
                else:
                    hist = [torch.zeros(1, 1, H, W, device=x.device) for _ in range(past_channels)]
                # 拼接成 [1, past_channels, H, W]
                past_i = torch.cat(hist, dim=0)
                past_input_list.append(past_i.unsqueeze(0))
            if len(past_input_list) > 0:
                past_input = torch.cat(past_input_list, dim=0)  # shape: [B, past_channels, H, W]
            else:
                past_input = torch.zeros(B, past_channels, H, W, device=x.device)'''


            B, _, H, W = x.shape

            # 构造 past_input：对每个样本构造形状为 [1, past_channels, H, W] 的提示
            past_input_list = []
            for i in range(B):
                sid = sample_ids[i].item() if torch.is_tensor(sample_ids) else int(sample_ids[i])
                stp = steps[i].item() if torch.is_tensor(steps) else int(steps[i])
                # 从历史字典中取该样本的历史 inpainting 输出（只取步数小于当前的）
                if sid in trainer_module.inpaint_history_dict:
                    hist = []
                    for (old_step, old_map) in trainer_module.inpaint_history_dict[sid]:
                        if old_step < stp:
                            # 确保历史记录形状为 [1, 1, H, W]
                            if old_map.dim() != 4:
                                old_map = old_map.view(1, 1, H, W)
                            hist.append(old_map)
                    if len(hist) < past_channels:
                        pad = [torch.zeros(1, 1, H, W, device=x.device) for _ in range(past_channels - len(hist))]
                        hist = pad + hist
                    else:
                        hist = hist[-past_channels:]
                else:
                    hist = [torch.zeros(1, 1, H, W, device=x.device) for _ in range(past_channels)]
                # 拼接成 [1, past_channels, H, W]，注意此时 past_i 的形状为 [past_channels, 1, H, W]
                past_i = torch.cat(hist, dim=0)
                # 在最前面增加一个 batch 维度，变成 [1, past_channels, 1, H, W]
                past_input_list.append(past_i.unsqueeze(0))
            if len(past_input_list) > 0:
                # 拼接所有样本，得到形状 [B, past_channels, 1, H, W]
                past_input = torch.cat(past_input_list, dim=0)
            else:
                past_input = torch.zeros(B, past_channels, 1, H, W, device=x.device)
            # 将 shape 从 [B, past_channels, 1, H, W] 调整为 [B, past_channels, H, W]
            past_input = past_input.squeeze(2)
                        

            ###### Forward pass ######
            compute_g_loss = iteration % config['n_critic'] == 0
            losses, inpainted_result, offset_flow = trainer(x, mask, ground_truth,
                                                             sample_ids, steps,
                                                             compute_g_loss=compute_g_loss,
                                                             past_input=past_input)
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()
            if compute_g_loss:
                trainer_module.optimizer_g.zero_grad()
                losses['g'] = (losses['l1'] * config['l1_loss_alpha'] +
                               losses['ae'] * config['ae_loss_alpha'] +
                               losses['wgan_g'] * config['gan_loss_alpha'] +
                               losses['consistency'] * config.get('consistency_loss_alpha', 0.1))
                losses['g'].backward()
                trainer_module.optimizer_g.step()
            trainer_module.optimizer_d.step()

            ###### Log and visualization ######
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd', 'consistency']
            if iteration % config['print_iter'] == 0:
                elapsed = time.time() - time_count
                speed = config['print_iter'] / elapsed
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()
                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = losses.get(k, 0.)
                    writer.add_scalar(k, v, iteration)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)

            if iteration % config['viz_iter'] == 0:
                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
                                              offset_flow[:viz_max_out]], dim=1)
                else:
                    viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
                viz_images = viz_images.view(-1, *list(x.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (checkpoint_path, iteration),
                                  nrow=3 * 4,
                                  normalize=True)

            if 'uncertainty_iter' in config and iteration % config['uncertainty_iter'] == 0:
                logger.info(f"Starting uncertainty estimation at iteration {iteration}")
                mean, std, offset_flow_uncertainty = trainer_module.inference_with_uncertainty(x, mask, num_samples=config['mc_dropout_samples'], past_input=past_input)
                print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
                vutils.save_image(mean, os.path.join(checkpoint_path, f'mean_{iteration:08d}.png'), normalize=True)
                vutils.save_image(std, os.path.join(checkpoint_path, f'std_{iteration:08d}.png'), normalize=True)
                if mean.dim() == 3:
                    writer.add_image('Mean', mean, iteration)
                elif mean.dim() == 4 and mean.size(0) == 1:
                    writer.add_image('Mean', mean.squeeze(0), iteration)
                else:
                    logger.warning(f"Mean tensor has unexpected shape: {mean.shape}")
                if std.dim() == 3:
                    writer.add_image('Std', std, iteration)
                elif std.dim() == 4 and std.size(0) == 1:
                    writer.add_image('Std', std.squeeze(0), iteration)
                else:
                    logger.warning(f"Std tensor has unexpected shape: {std.shape}")
                logger.info(f"Saved uncertainty maps at iteration {iteration}")

            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)

            if iteration % config.get('val_iter', 1000) == 0:
                logger.info(f"Starting validation at iteration {iteration}")
                val_losses = []
                trainer_module.eval()
                with torch.no_grad():
                    for val_ground_truth, val_x, val_mask, _, _ in val_loader:
                        if cuda:
                            val_x = val_x.cuda()
                            val_mask = val_mask.cuda()
                            val_ground_truth = val_ground_truth.cuda()
                        # 验证时不需要 past_input，可传 None
                        val_losses_batch, val_inpainted_result, _ = trainer(val_x, val_mask, val_ground_truth, [], [], compute_g_loss=False, past_input=None)
                        l1 = nn.L1Loss()(val_inpainted_result, val_ground_truth)
                        val_losses.append(l1.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                writer.add_scalar('val/l1_loss', avg_val_loss, iteration)
                logger.info(f"Validation Iteration {iteration}: Average L1 Loss: {avg_val_loss:.6f}")
                if config.get('save_best_model', True):
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_name = config.get('best_model_name', 'best')
                        trainer_module.save_model(checkpoint_path, best_model_name)
                        logger.info(f"Saved best model ({best_model_name}) with validation loss: {best_val_loss:.6f}")
                trainer_module.train()

    except Exception as e:
        logger.error("{}".format(e))
        raise e

if __name__ == '__main__':
    main()
