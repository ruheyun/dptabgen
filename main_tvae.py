# coding:UTF-8
# RuHe  2025/12/25 15:11
import os
import argparse
from scripts.train_sample_tvae import train_tvae, sample_tvae
from scripts.eval_catboost import train_catboost
from scripts.eval_mlp import train_mlp
import lib
import json


def save_config(parent_dir, config):
    os.makedirs(parent_dir, exist_ok=True)
    filepath = os.path.join(parent_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', default='config/adult/config_tvae.toml')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=True)
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--change_val', action='store_true', default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    save_config(raw_config['exp_path'], raw_config)
    tvae = None
    if args.train:
        tvae = train_tvae(
            parent_dir=raw_config['exp_path'],
            real_data_path=raw_config['data_path'],
            train_params=raw_config['train_params'],
            change_val=args.change_val,
            device=raw_config['device']
        )
    if args.sample:
        sample_tvae(
            synthesizer=tvae,
            parent_dir=raw_config['exp_path'],
            real_data_path=raw_config['data_path'],
            num_samples=raw_config['sample']['num_samples'],
            train_params=raw_config['train_params'],
            change_val=args.change_val,
            seed=raw_config['sample']['seed'],
            device=raw_config['device']
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
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )


if __name__ == '__main__':
    main()
