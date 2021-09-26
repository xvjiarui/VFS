import argparse
from collections import OrderedDict

import torch


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # convert to pytorch style
    state_dict = OrderedDict()
    src_dict = torch.load(src)
    src_state_dict = src_dict.get('state_dict', src_dict)
    for k, v in src_state_dict.items():
        if not k.startswith('backbone'):
            continue
        b_k = k.replace('backbone.', '')
        b_k_splits = b_k.split('.')
        tail = b_k_splits[-1]
        if b_k.startswith('conv1'):
            if b_k_splits[1] == 'conv':
                name = f'conv1.{tail}'
            elif b_k_splits[1] == 'bn':
                name = f'bn1.{tail}'
            elif b_k_splits[1] == 'gn':
                name = f'gn1.{tail}'
            else:
                raise RuntimeError(b_k)
        elif b_k.startswith('layer'):
            layer_idx = int(b_k_splits[0][-1])
            block_idx = int(b_k_splits[1])
            if b_k_splits[2] == 'downsample':
                # downsample
                if b_k_splits[3] == 'conv':
                    name = f'layer{layer_idx}.{block_idx}.downsample.0.{tail}'
                elif b_k_splits[3] == 'bn':
                    name = f'layer{layer_idx}.{block_idx}.downsample.1.{tail}'
                elif b_k_splits[3] == 'gn':
                    name = f'layer{layer_idx}.{block_idx}.downsample.1.{tail}'
                else:
                    raise RuntimeError(b_k)
            elif b_k_splits[3] == 'conv':
                conv_module_idx = int(b_k_splits[2][-1])
                name = f'layer{layer_idx}.{block_idx}.' \
                       f'conv{conv_module_idx}.{tail}'
            elif b_k_splits[3] == 'bn':
                conv_module_idx = int(b_k_splits[2][-1])
                name = f'layer{layer_idx}.{block_idx}.' \
                       f'bn{conv_module_idx}.{tail}'
            elif b_k_splits[3] == 'gn':
                conv_module_idx = int(b_k_splits[2][-1])
                name = f'layer{layer_idx}.{block_idx}.' \
                       f'gn{conv_module_idx}.{tail}'
            else:
                raise RuntimeError(b_k)
        else:
            raise RuntimeError(f'{b_k}')
        state_dict[name] = v
        print(f'{k} --> {name}')

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    checkpoint['meta'] = dict()
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
