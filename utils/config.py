#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: config.py
@time: 2020/5/12 22:12
'''
import argparse
import yaml
import warnings

warnings.filterwarnings('ignore')

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--gpu', type=str, default='0', help='GPU idx')
    parser.add_argument('--config', type=str, default='opt/JS3C_default_kitti.yaml', help='path to config file')
    parser.add_argument('--log_dir', type=str, default=None, help='path to log file')
    parser.add_argument('--debug', default=False, action='store_true')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f)
    config['gpu'] = args_cfg.gpu
    config['config'] = args_cfg.config
    config['log_dir'] = args_cfg.log_dir
    config['GENERAL']['debug'] = args_cfg.debug
    return config

cfg = get_parser()