# coding:UTF-8
# RuHe  2025/11/26 15:36
import numpy as np
import pandas as pd
import math
import json
from typing import Optional, Tuple, Dict
import os


def gaussian_sigma_for_mean(C_l: float, C_h: float, n: int, eps: float, delta: float) -> float:
    """
    :param C_l: 裁剪下界
    :param C_h: 裁剪上界
    :param n: 样本总数
    :param eps: 隐私预算
    :param delta: 错误概率
    :return: 高斯噪声标准差
    """
    sens = 2 * C / n
    return math.sqrt(2 * math.log(1.25 / delta)) * sens / eps


def gaussian_sigma_for_moment(C: float, n: int, eps: float, delta: float) -> float:
    """
    :param C: 裁剪因子
    :param n: 样本总数
    :param eps: 隐私预算
    :param delta: 错误概率
    :return: 高斯噪声标准差
    """
    sens = (C * C) / n
    return math.sqrt(2 * math.log(1.25 / delta)) * sens / eps


def gaussian_sigma_for_p(n: int, eps: float, delta: float) -> float:
    """
    :param n: 样本总数
    :param eps: 隐私预算
    :param delta: 错误概率
    :return: 高斯噪声标准差
    """
    sens = 1.0 / n
    return math.sqrt(2 * math.log(1.25 / delta)) * sens / eps


def compute_noisy_marginals(
        X_num: np.ndarray,
        X_cat: np.ndarray,
        C_dict: Dict[str, float],
        epsilon: float = 1.0,
        delta: float = 1e-5,
        col_num_names: Optional[list] = None,
        col_cat_names: Optional[list] = None,
        eps_allocation: Optional[Dict] = None
) -> Dict:
    N = X_num.shape[0]
    d_num = X_num.shape[1] if X_num is not None else 0
    d_cat = X_cat.shape[1] if X_cat is not None else 0

    # names
    if col_num_names is None:
        col_num_names = [f"num_{i}" for i in range(d_num)]
    if col_cat_names is None:
        col_cat_names = [f"cat_{i}" for i in range(d_cat)]

    total_stats = 0
    for i in range(d_num):
        total_stats += 2
    cat_categories = {}
    for j in range(d_cat):
        vals, counts = np.unique(X_cat[:, j], return_counts=True)
        cat_categories[j] = list(vals)
        total_stats += len(vals)

    # if user did not provide an allocation, split epsilon evenly per-stat (simple)
    if eps_allocation is None:
        eps_per_stat = epsilon / max(1, total_stats)
    else:
        raise NotImplementedError("Custom eps_allocation not implemented in this simple helper")

    noisy = {'numeric': {}, 'categorical': {},
             'meta': {'N': int(N), 'total_stats': int(total_stats), 'eps_per_stat': float(eps_per_stat)}}

    # numeric columns
    for i in range(d_num):
        colname = col_num_names[i]
        C = C_dict.get(colname, None)
        if C is None:
            C = C_dict.get(i, None)
        if C is None:
            raise ValueError(f"Clipping bound B must be provided for numeric column {colname} (or index).")
        arr = X_num[:, i].astype(float)
        clipped = np.clip(arr, -C, C)
        mean = float(clipped.mean())
        second_moment = float((clipped ** 2).mean())
        sigma_mean = gaussian_sigma_for_mean(C, N, eps_per_stat, delta)
        sigma_m2 = gaussian_sigma_for_moment(C, N, eps_per_stat, delta)
        noisy_mean = mean + np.random.normal(0.0, sigma_mean)
        noisy_m2 = second_moment + np.random.normal(0.0, sigma_m2)
        noisy_var = max(1e-12, noisy_m2 - noisy_mean ** 2)
        noisy['numeric'][colname] = {'mean': float(noisy_mean), 'var': float(noisy_var),
                                     'sigma_mean': float(sigma_mean), 'sigma_m2': float(sigma_m2),
                                     'C': float(C)}
    # categorical columns
    for j in range(d_cat):
        colname = col_cat_names[j]
        vals = cat_categories[j]
        counts = np.array([(X_cat[:, j] == v).sum() for v in vals], dtype=float)
        sigma_count = gaussian_sigma_for_p(N, eps_per_stat, delta)
        noisy_counts = counts + np.random.normal(0.0, sigma_count, size=counts.shape)
        noisy_counts = np.clip(noisy_counts, 0.0, None)
        if noisy_counts.sum() <= 0:
            props = np.ones_like(noisy_counts) / float(len(noisy_counts))
        else:
            props = noisy_counts / noisy_counts.sum()
        noisy['categorical'][colname] = {'categories': list(map(str, vals)), 'props': props.tolist(),
                                         'sigma_count': float(sigma_count)}
    return noisy


def synthesize_table_from_stats(noisy_stats: Dict, n_synth: int = 10000,
                                numeric_method: str = 'gaussian') -> pd.DataFrame:
    X_num_random = []
    for k, v in noisy_stats['numeric'].items():
        mu = float(v['mean'])
        var = float(v['var'])
        if var <= 0:
            var = 1e-6
        if numeric_method == 'gaussian':
            X_num_random.append(np.random.normal(mu, math.sqrt(var), size=n_synth))
        else:
            raise NotImplementedError("Only 'gaussian' numeric_method implemented in this helper.")
    X_num_random = np.stack(X_num_random, axis=1)
    X_cat_random = []
    for k, v in noisy_stats['categorical'].items():
        cats = v['categories']
        props = np.array(v['props'], dtype=float)
        if props.sum() <= 0:
            props = np.ones_like(props) / len(props)
        else:
            props = props / props.sum()
        X_cat_random.append(np.random.choice(cats, size=n_synth, p=props))
    X_cat_random = np.stack(X_cat_random, axis=1)
    return X_num_random, X_cat_random


def main(
        num_path: str,
        cat_path: str,
        C_dict: Dict,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        n_synth: int = 10000,
        col_num_names: Optional[list] = None,
        col_cat_names: Optional[list] = None,
        out_path: str = ''
) -> Tuple[pd.DataFrame, Dict]:
    X_num = np.load(num_path)
    X_cat = np.load(cat_path).astype(object)
    noisy = compute_noisy_marginals(X_num, X_cat, C_dict, epsilon, delta, col_num_names, col_cat_names)

    X_num_r, X_cat_r = synthesize_table_from_stats(noisy, n_synth=n_synth)
    if X_num is not None:
        np.save(out_path, X_num_r)
    if X_cat is not None:
        np.save(out_path, X_cat_r)
    return X_num_r, X_cat_r, noisy


if __name__ == '__main__':
    pass
