import cv2
import argparse
import torch
from mmdet.apis import init_detector
import torchvision.transforms as T
from PIL import Image
import struct


def gen_wts(model, filename):
    f = open(filename + '.wts', 'w')
    f.write('{}\n'.format(len(model.state_dict().keys()) + 72))
    for k, v in model.state_dict().items():
        if 'in_proj' in k:
            dim = int(v.size(0) / 3)
            q_weight = v[:dim].reshape(-1).cpu().numpy()
            k_weight = v[dim:2*dim].reshape(-1).cpu().numpy()
            v_weight = v[2*dim:].reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k + '_q', len(q_weight)))
            for vv in q_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

            f.write('{} {} '.format(k + '_k', len(k_weight)))
            for vv in k_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

            f.write('{} {} '.format(k + '_v', len(v_weight)))
            for vv in v_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

        elif 'level_embeds' in k:
            # v 4,
            num_level = v.size(0)
            for i in range(num_level):
                vr = v[i].reshape(-1).cpu().numpy()
                f.write('{} {} '.format(k+'.'+str(i), len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f',float(vv)).hex())
                f.write('\n')
        else:
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f',float(vv)).hex())
            f.write('\n')
    f.close()


def parse():
    opt = argparse.ArgumentParser()
    opt.add_argument("--config",type=str)
    opt.add_argument("--checkpoint",type=str)
    opt.add_argument("--output",type=str)
    return opt.parse_args()

def main():
    opt = parse()

    model = init_detector(opt.config,
                        opt.checkpoint,
                        device='cpu')

    gen_wts(model, opt.output)


if __name__ == '__main__':
    main()