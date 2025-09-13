#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   evaluate.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Evaluation Scripts (modified to support both CPU and GPU)
@License :   This source code is licensed under the license found in the 
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import network
from datasets import SCHPDataset, transform_logits


dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}

def get_arguments():
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--restore-weight", type=str, default='', help="restore pretrained model parameters.")
    parser.add_argument("--input", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")
    parser.add_argument("--argmax_logits", action='store_true', default=False, help="Save logits in compressed argmax form")
    parser.add_argument("--postfix_filename", default="", help="add a postfix to the filenames, before the file extension")

    return parser.parse_args()


def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def main():
    args = get_arguments()

    # Select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']

    model = network(num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.restore_weight, map_location=device)

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("context_encoding.stages.0.2.", "context_encoding.stages.0.2.bn.")
        name = name.replace("context_encoding.stages.1.2.", "context_encoding.stages.1.2.bn.")
        name = name.replace("context_encoding.stages.2.2.", "context_encoding.stages.2.2.bn.")
        name = name.replace("context_encoding.stages.3.2.", "context_encoding.stages.3.2.bn.")
        name = name.replace("context_encoding.bottleneck.1.", "context_encoding.bottleneck.1.bn.")
        name = name.replace("edge.conv1.1.", "edge.conv1.1.bn.")
        name = name.replace("edge.conv2.1.", "edge.conv2.1.bn.")
        name = name.replace("edge.conv3.1.", "edge.conv3.1.bn.")
        name = name.replace("decoder.conv1.1.", "decoder.conv1.1.bn.")
        name = name.replace("decoder.conv2.1.", "decoder.conv2.1.bn.")
        name = name.replace("decoder.conv3.1.", "decoder.conv3.1.bn.")
        name = name.replace("decoder.conv3.3.", "decoder.conv3.3.bn.")
        name = name.replace("fushion.1.", "fushion.1.bn.")
        name = name.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # Send model to device
    model = nn.DataParallel(model).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SCHPDataset(root=args.input, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    palette = get_palette(num_classes)

    print(f"Found {len(dataloader)} files")
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, meta = batch
            image = image.to(device)

            img_name = meta['name'][0]
            img_path = meta['path'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image)

            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output)
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)

            parsing_result = np.argmax(logits_result, axis=2)

            img_subpath = img_path.replace(args.input, "").lstrip("/")
            parsing_result_path = os.path.join(args.output, img_subpath[:-4] + args.postfix_filename + '.png')
            os.makedirs(os.path.dirname(parsing_result_path), exist_ok=True)

            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)

            if args.logits:
                logits_result_path = os.path.join(args.output, img_subpath[:-4] + args.postfix_filename + '.npy')
                if args.argmax_logits:
                    logits_result_path += "c"
                    result = parsing_result
                else:
                    result = logits_result
                np.save(logits_result_path, result)


if __name__ == '__main__':
    main()
