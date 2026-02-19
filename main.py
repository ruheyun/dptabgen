# coding:UTF-8
# RuHe  2025/10/11 19:09
import json
import os
import argparse
import tomli_w
import torch
from scripts.data_wrapper import process_wrapper
# from scripts.train import train
from scripts.train_dp import train
from scripts.sample import sample
from scripts.eval_catboost import train_catboost
from scripts.eval_mlp import train_mlp
from scripts.eval_simple import train_simple
from lib import load_config, load_json
import warnings
warnings.filterwarnings('ignore')


def save_config(parent_dir, config):
    os.makedirs(parent_dir, exist_ok=True)
    filepath = os.path.join(parent_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    # 启用训练、采样、测试
    parser.add_argument('--config', metavar='FILE', default='config/buddy/config_cb_best.toml')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--change_val', action='store_true', default=False)
    # 评估模型设置
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=-1)
    parser.add_argument('--eval_model', type=str, choices=['mlp', 'catboost', 'simple', 'None'], default='None')

    args = parser.parse_args()
    raw_config = load_config(args.config)
    raw_info = load_json(os.path.join(raw_config['data_path'], 'info.json'))
    if args.batch_size != -1:
        raw_config['train']['main']['batch_size'] = args.batch_size
    if args.steps != -1:
        raw_config['train']['main']['steps'] = args.steps
    if args.eval_model != 'None':
        raw_config['eval']['type']['eval_model'] = args.eval_model

    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cpu')
    save_config(raw_config['exp_path'], raw_config)
    if args.train:
        folder_path = os.path.join(raw_config['exp_path'], raw_config['train']['T']['cat_encoding'])
        os.makedirs(folder_path, exist_ok=True)
        if not os.listdir(folder_path):
            process_wrapper(
                exp_path=raw_config['exp_path'],
                data_path=raw_config['data_path'],
                cat_encode=raw_config['train']['T']['cat_encoding'],
                task_type=raw_info['task_type']
            )

        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            exp_dir=raw_config['exp_path'],
            # data_path=raw_config['data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            dp_params=raw_config['dp'],
            device=device,
            change_val=args.change_val
        )

    if args.sample:
        sample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            exp_dir=raw_config['exp_path'],
            data_path=raw_config['data_path'],
            model_path=os.path.join(raw_config['exp_path'], 'model_ema.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=args.change_val
        )

    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                exp_dir=raw_config['exp_path'],
                data_path=raw_config['data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        elif raw_config['eval']['type']['eval_model'] == 'mlp':
            train_mlp(
                exp_dir=raw_config['exp_path'],
                data_path=raw_config['data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['train']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                device=device
            )
        elif raw_config['eval']['type']['eval_model'] == 'simple':
            train_simple(
                exp_dir=raw_config['exp_path'],
                data_path=raw_config['data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['train']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        else:
            print('No eval model!')


if __name__ == '__main__':
    main()
