import os
import random
import imageio
import glob
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml',
                    help="testing configuration")
parser.add_argument('--seed', type=int, help='manual seed')
# parser.add_argument('--image', type=str, default='dataset/full/1.png')
# parser.add_argument('--mask', type=str, default=None)
parser.add_argument('--image', type=str, default='dataset/part/1_0.png')
parser.add_argument('--mask', type=str, default='dataset/mask/1_0.png')
parser.add_argument('--output', type=str, default='dataset/1_0_p.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='part_map_inpainting')
parser.add_argument('--iter', type=int, default=0)

def main(netG, args, config):
    try:  # for unexpected error logging
        with torch.no_grad():   # enter no grad context
            if is_image_file(args.image):
                if args.mask and is_image_file(args.mask):
                    # Test a single masked image with a given mask
                    x = default_loader(args.image)
                    mask = default_loader(args.mask)
                    x = transforms.Resize(config['image_shape'][:-1])(x)
                    x = transforms.CenterCrop(config['image_shape'][:-1])(x)
                    mask = transforms.Resize(config['image_shape'][:-1])(mask)
                    mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
                    x = transforms.ToTensor()(x)
                    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
                    x = normalize(x)
                    x = x * (1. - mask)
                    x = x.unsqueeze(dim=0)
                    mask = mask.unsqueeze(dim=0)
                elif args.mask:
                    raise TypeError("{} is not an image file.".format(args.mask))
                else:
                    # Test a single ground-truth image with a random mask
                    ground_truth = default_loader(args.image)
                    ground_truth = transforms.Resize(config['image_shape'][:-1])(ground_truth)
                    ground_truth = transforms.CenterCrop(config['image_shape'][:-1])(ground_truth)
                    ground_truth = transforms.ToTensor()(ground_truth)
                    ground_truth = normalize(ground_truth)
                    ground_truth = ground_truth.unsqueeze(dim=0)
                    bboxes = random_bbox(config, batch_size=ground_truth.size(0))
                    x, mask = mask_image(ground_truth, bboxes, config)

                if config['cuda']:
                    netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                    x = x.cuda()
                    mask = mask.cuda()

                # Inference
                x1, x2, offset_flow = netG(x, mask)
                inpainted_result = x2 * mask + x * (1. - mask)

                if "ground_truth" in locals():
                    x = x + ground_truth.cuda() if config['cuda'] else x + ground_truth
                    x /= 2

                viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
                viz_images = viz_images.view(-1, *list(x.size())[1:])

                vutils.save_image(viz_images, args.output, padding=0, normalize=True)
                print("Saved the inpainted result to {}".format(args.output))
                if args.flow:
                    vutils.save_image(offset_flow, args.flow, padding=0, normalize=True)
                    print("Saved offset flow to {}".format(args.flow))
            else:
                raise TypeError("{} is not an image file.".format)
        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.config)
    print("Arguments: {}".format(args))

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True
    print("Configuration: {}".format(config))

    # Define the trainer
    netG = Generator(config['netG'], cuda, device_ids)
    # Resume weight
    last_model_name = get_model_list(args.checkpoint_path, "gen", iteration=args.iter)
    netG.load_state_dict(torch.load(last_model_name))
    model_iteration = int(last_model_name[-11:-3])
    print("Resume from {} at iteration {}".format(args.checkpoint_path, model_iteration))

    img_id = 0
    frame_files = []
    n_img = len(glob.glob(f'dataset/part/{img_id}_*.png'))
    print(f'number of images: {n_img}')

    for i in range(n_img):
        args.image = f'dataset/part/{img_id}_{i}.png'
        args.mask = f'dataset/mask/{img_id}_{i}.png'
        args.output = f'dataset/{img_id}_{i}_predict.png'
        frame_files.append(args.output)
        main(netG, args, config)

    with imageio.get_writer(f'dataset/{img_id}_predict.gif', mode='I', duration=2) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print(f'gif completed: dataset/{img_id}_predict.gif')
    for filename in frame_files:
        os.remove(filename)
