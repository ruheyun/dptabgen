import subprocess
import lib
import os
import sys
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path


def _suggest_mlp_layers(trial):
    """根据 trial 建议 MLP 层结构"""
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t

    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)

    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2 else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last

    return d_layers


def objective(trial, args, base_config_path, exps_path, train_size, eval_type,
              pipeline, python_exec):
    """Optuna 目标函数"""

    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    d_layers = _suggest_mlp_layers(trial)
    batch_size = trial.suggest_categorical('batch_size', [512])
    steps = trial.suggest_categorical('steps', [5000])
    num_timesteps = trial.suggest_categorical('num_timesteps', [1000])
    num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))

    # 加载并修改配置
    base_config = lib.load_config(base_config_path)
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers

    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    base_config['dp']['0']['T'] = num_timesteps
    base_config['dp']['0']['steps'] = steps

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size

    base_config['sample']['num_samples'] = num_samples

    base_config['eval']['type']['eval_type'] = eval_type

    exp_dir = exps_path / f"{trial.number}"
    base_config['exp_path'] = str(exp_dir)
    base_config['eval']['type']['eval_model'] = args.eval_model

    trial.set_user_attr("config", base_config)

    # 保存配置
    lib.dump_config(base_config, exps_path / 'config.toml')

    # 训练阶段
    subprocess.run([
        python_exec, f'{pipeline}',
        '--config', f'{exps_path / "config.toml"}',
        '--train'#, '--change_val'
    ], check=True)

    # 评估阶段
    n_datasets = 5
    score = 0.0
    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')

        subprocess.run([
            python_exec, f'{pipeline}',
            '--config', f'{exps_path / "config.toml"}',
            '--sample', '--eval'#, '--change_val'
        ], check=True)

        report_path = str(Path(base_config['exp_path']) / f'results_{args.eval_model}.json')
        report = lib.load_json(report_path)

        if 'r2' in report['metrics']['val']:
            score += report['metrics']['val']['r2']
        else:
            score += report['metrics']['val']['macro avg']['f1-score']

    # 清理临时实验文件
    shutil.rmtree(exp_dir, ignore_errors=True)

    return score / n_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, default='adult')
    parser.add_argument('--train_size', type=int, default=26048)
    parser.add_argument('--eval_type', type=str, default='synthetic')
    parser.add_argument('--eval_model', type=str, default='mlp')
    parser.add_argument('--prefix', type=str, default='dm')
    parser.add_argument('--eval_seeds', action='store_true', default=False)
    parser.add_argument('--num_trials', type=int, default=25)

    args = parser.parse_args()

    # 自动检测 Python 解释器（Windows/Linux 通用）
    python_exec = sys.executable
    print(f"[INFO] Using Python executable: {python_exec}")

    train_size = args.train_size
    ds_name = args.ds_name
    eval_type = args.eval_type
    eval_model = args.eval_model
    num_trials = args.num_trials
    assert eval_type in ('merged', 'synthetic')
    prefix = str(args.prefix + '_' + eval_model)

    pipeline = f'main.py'
    base_config_path = f'config/{ds_name}/config.toml'
    parent_path = Path(f'expdir/{ds_name}/')
    exps_path = parent_path / 'many-exps'
    eval_seeds = f'eval_seeds.py'

    os.makedirs(exps_path, exist_ok=True)

    # 建立 Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0)
    )

    func = lambda trial: objective(
        trial, args, base_config_path, exps_path, train_size, eval_type,
        pipeline, python_exec
    )

    study.optimize(func, n_trials=num_trials, show_progress_bar=True) # 50

    # 保存最优配置
    best_config_path = parent_path / f'{prefix}_best/config.toml'
    best_config = study.best_trial.user_attrs['config']
    best_config["exp_path"] = str(parent_path / f'{prefix}_best/')

    os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
    lib.dump_config(best_config, best_config_path)
    lib.dump_json(
        optuna.importance.get_param_importances(study),
        parent_path / f'{prefix}_best/importance.json'
    )

    # 使用最优配置重新训练和采样
    subprocess.run([
        python_exec, f'{pipeline}',
        '--config', f'{best_config_path}',
        '--train', '--sample'
    ], check=True)

    # 如果启用 eval_seeds，则进一步评估
    if args.eval_seeds:
        best_exp = str(best_config_path)
        subprocess.run([
            python_exec, f'{eval_seeds}',
            '--config', f'{best_exp}',
            '--n_seeds', '10',
            '--eval_type', eval_type,
            '--model_type', args.eval_model,
            '--n_datasets', '5'
        ], check=True)


if __name__ == '__main__':
    main()
