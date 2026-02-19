import argparse
import subprocess
import tempfile
import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import lib

import shutil
from pathlib import Path
from copy import deepcopy
from scripts.eval_catboost import train_catboost
from scripts.eval_mlp import train_mlp
from scripts.eval_simple import train_simple


def eval_seeds(
        raw_config,
        n_seeds,
        eval_type,
        model_type="catboost",
        n_datasets=1,
        dump=True,
        change_val=False
):
    if model_type == 'simple':
        models = ["tree", "lr", "rf", "mlp"]
        metrics_seeds_report = {
            k: lib.SeedsMetricsReport() for k in models
        }
    else:
        metrics_seeds_report = lib.SeedsMetricsReport()

    exp_dir = Path(raw_config["exp_path"])

    if eval_type == 'real':
        n_datasets = 1

    temp_config = deepcopy(raw_config)
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        temp_config["exp_path"] = str(dir_)
        shutil.copy2(exp_dir / "model.pt", dir_)
        shutil.copy2(exp_dir / "model_ema.pt", dir_)
        shutil.copytree(exp_dir / 'analogbit', dir_ / 'analogbit')

        for sample_seed in range(n_datasets):
            temp_config['sample']['seed'] = sample_seed
            lib.dump_config(temp_config, dir_ / "config.toml")
            if eval_type != 'real' and n_datasets > 1:
                subprocess.run([sys.executable, 'main.py', '--config', f'{str(dir_ / "config.toml")}',
                                '--sample'], check=True)

            for seed in range(n_seeds):
                print(f'**Eval Iter: {sample_seed * n_seeds + (seed + 1)}/{n_seeds * n_datasets}**')
                if model_type == "catboost":
                    T_dict = deepcopy(raw_config['eval']['T'])
                    metric_report = train_catboost(
                        exp_dir=temp_config['exp_path'],
                        data_path=temp_config['data_path'],
                        eval_type=eval_type,
                        T_dict=T_dict,
                        seed=seed,
                        change_val=change_val
                    )
                    metrics_seeds_report.add_report(metric_report)
                elif model_type == "mlp":
                    T_dict = deepcopy(raw_config['train']['T'])
                    metric_report = train_mlp(
                        exp_dir=temp_config['exp_path'],
                        data_path=temp_config['data_path'],
                        eval_type=eval_type,
                        T_dict=T_dict,
                        seed=seed,
                        change_val=change_val
                    )
                    metrics_seeds_report.add_report(metric_report)
                elif model_type == 'simple':
                    T_dict = deepcopy(raw_config['train']['T'])
                    metric_reports = train_simple(
                        exp_dir=temp_config['exp_path'],
                        data_path=temp_config['data_path'],
                        eval_type=eval_type,
                        T_dict=T_dict,
                        seed=seed,
                        change_val=change_val
                    )
                    for model in models:
                        metrics_seeds_report[model].add_report(metric_reports[model])
    if model_type == 'simple':
        for model in models:
            metrics_seeds_report[model].get_mean_std()
        res = {k: metrics_seeds_report[k].print_result() for k in models}
        m1, m2 = ("r2-mean", "rmse-mean") if "r2-mean" in res["tree"]["val"] else ("f1-mean", "acc-mean")
        res["avg"] = {
            "val": {
                m1: np.around(np.mean([res[k]["val"][m1] for k in models]), 4),
                m2: np.around(np.mean([res[k]["val"][m2] for k in models]), 4)
            },
            "test": {
                m1: np.around(np.mean([res[k]["test"][m1] for k in models]), 4),
                m2: np.around(np.mean([res[k]["test"][m2] for k in models]), 4)
            },
        }
    else:
        metrics_seeds_report.get_mean_std()
        res = metrics_seeds_report.print_result()

    if os.path.exists(exp_dir / f"eval_{model_type}.json"):
        eval_dict = lib.load_json(exp_dir / f"eval_{model_type}.json")
        eval_dict = eval_dict | {eval_type: res}
    else:
        eval_dict = {eval_type: res}

    if dump:
        lib.dump_json(eval_dict, exp_dir / f"eval_{model_type}.json")

    raw_config['sample']['seed'] = 0
    lib.dump_config(raw_config, exp_dir / 'config.toml')
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', default='config/adult/config.toml')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--eval_type', type=str, default='synthetic')
    parser.add_argument('--model_type', type=str, default='catboost')
    parser.add_argument('--n_datasets', type=int, default=5)
    parser.add_argument('--no_dump', action='store_false', default=True)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    eval_seeds(
        raw_config,
        n_seeds=args.n_seeds,
        eval_type=args.eval_type,
        model_type=args.model_type,
        n_datasets=args.n_datasets,
        dump=args.no_dump
    )


if __name__ == '__main__':
    main()
