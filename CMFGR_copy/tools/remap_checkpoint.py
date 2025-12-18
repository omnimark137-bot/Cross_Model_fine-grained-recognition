#!/usr/bin/env python3
"""Normalize checkpoint keys to match current model state_dict and save a remapped checkpoint.

Usage:
  python tools/remap_checkpoint.py --ckpt weights/HOSS_TransOSS.pth --cfg configs/aircraft_transoss.yml --out weights/HOSS_TransOSS.remapped.pth
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch
from config import cfg
from model.make_model import make_model


def norm_key(k):
    prefixes = ['module.', 'model.', 'state_dict.', 'base.']
    for p in prefixes:
        if k.startswith(p):
            k = k[len(p):]
    return k


def generate_candidates(k):
    # yield several normalized candidate names we might map to
    k0 = k
    yield k0
    yield norm_key(k0)
    # drop first namespace segment (e.g. encoder.layer -> layer)
    if '.' in k0:
        yield k0.split('.', 1)[-1]
        yield norm_key(k0.split('.', 1)[-1])
    # swap common head/classifier names
    swaps = [('head', 'classifier'), ('classifier', 'head'), ('fc', 'head'), ('head', 'fc')]
    for a, b in swaps:
        if a in k0:
            yield k0.replace(a, b)
            yield norm_key(k0.replace(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num_class', type=int, default=100)
    args = parser.parse_args()

    print('Loading checkpoint:', args.ckpt)
    ck = torch.load(args.ckpt, map_location='cpu')
    if isinstance(ck, dict):
        if 'state_dict' in ck:
            ck_dict = ck['state_dict']
        elif 'model' in ck:
            ck_dict = ck['model']
        else:
            ck_dict = ck
    else:
        ck_dict = {}

    print('Building model from cfg:', args.cfg)
    cfg.merge_from_file(args.cfg)
    cfg.defrost()
    try:
        cfg.MODEL.PRETRAIN_CHOICE = False
        cfg.MODEL.PRETRAIN_PATH = ''
    except Exception:
        pass
    cfg.freeze()

    model = make_model(cfg, num_class=args.num_class, camera_num=0)
    model_keys = set(model.state_dict().keys())

    remapped = {}
    matched = []
    unmatched = []
    for k, v in ck_dict.items():
        found = False
        for cand in generate_candidates(k):
            if cand in model_keys:
                remapped[cand] = v
                matched.append((k, cand))
                found = True
                break
        if not found:
            unmatched.append(k)

    print(f'matched {len(matched)} keys, unmatched {len(unmatched)} keys')
    if len(matched) > 0:
        print('sample matched:')
        for a, b in matched[:40]:
            print(f"  {a} -> {b}")
    if len(unmatched) > 0:
        print('sample unmatched:')
        for k in unmatched[:40]:
            print('  ', k)

    out = {'state_dict': remapped}
    print('Saving remapped checkpoint to', args.out)
    torch.save(out, args.out)


if __name__ == '__main__':
    main()
