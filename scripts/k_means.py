# coding:UTF-8
# RuHe  2025/12/4 14:55
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import math
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def main(
        num_timesteps,
        schedule_name,
        k
):
    # 生成调度
    X = np.arange(num_timesteps)
    print(f'======================== {schedule_name} ========================')
    beta = get_named_beta_schedule(schedule_name, num_timesteps)
    alpha = 1.0 - beta
    alpha = alpha.reshape(-1, 1)
    alpha_cumprod = np.cumprod(alpha).reshape(-1, 1)

    sqrt_alpha = np.sqrt(alpha)
    sqrt_one_minus_alpha = np.sqrt(1.0 - alpha)  #
    sqrt_alpha_cumprod = np.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - alpha_cumprod).reshape(-1, 1)  #
    noise_signal_ratio = sqrt_alpha / sqrt_one_minus_alpha  #
    noise_signal_ratio_cumprod = sqrt_alpha_cumprod / sqrt_one_minus_alpha_cumprod #

    # K-Means 聚类（基于噪声强度）
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(sqrt_one_minus_alpha)

    # 计算分界点（基于噪声强度）
    centers = kmeans.cluster_centers_.flatten()
    sorted_centers = np.sort(centers)  # 从小到大：噪声少 → 噪声多
    sorted_indices = np.argsort(centers)
    print("排序后的簇质心:", sorted_centers)

    # 构建映射：old_label → new_label (0=最小噪声, 2=最大噪声)
    label_map = np.empty(k, dtype=int)
    for new_label, old_label in enumerate(sorted_indices):
        label_map[old_label] = new_label
    # 应用映射
    cluster_labels_correct = label_map[cluster_labels]

    # 相邻质心中点作为分界点
    boundaries_noise = (sorted_centers[:-1] + sorted_centers[1:]) / 2
    print("噪声强度分界点:", boundaries_noise)

    # 找对应的时间步分界（近似）
    t_boundaries = []
    for b in boundaries_noise:
        idx = np.argmin(np.abs(sqrt_one_minus_alpha.flatten() - b))
        t_boundaries.append(X[idx])
    t_boundaries = np.array(t_boundaries)
    print("对应的时间步分界点（近似）:", t_boundaries)

    # 输出每类的时间步与参数范围
    for group_id in range(k):
        mask = (cluster_labels_correct == group_id)
        if np.any(mask):
            t_min, t_max = X[mask].min(), X[mask].max()
            noise_min = sqrt_one_minus_alpha[mask].min()
            noise_max = sqrt_one_minus_alpha[mask].max()
            beta_min = beta[mask].min()
            beta_max = beta[mask].max()
            print(f"类别 {group_id}: 时间步 [{t_min}, {t_max}], "
                  f"噪声强度 ∈ [{noise_min:.4f}, {noise_max:.4f}], "
                  f"beta ∈ [{beta_min:.5f}, {beta_max:.5f}]")

    return {
        'X': X,
        'sqrt_alpha': sqrt_alpha,
        'sqrt_one_minus_alpha': sqrt_one_minus_alpha,
        'sqrt_alpha_cumprod': sqrt_alpha_cumprod,
        'sqrt_one_minus_alpha_cumprod': sqrt_one_minus_alpha_cumprod,
        'noise_signal_ratio': noise_signal_ratio,
        'noise_signal_ratio_cumprod': noise_signal_ratio_cumprod,
        't_boundaries': t_boundaries,
        'cluster_labels': cluster_labels
    }


def plot(
        params_linear,
        params_cosine,
        y_name,
        title
):
    plt.figure(figsize=(8, 4))
    # 左 y 轴：linear 调度
    ax1 = plt.gca()
    # plt.plot(params_linear['X'], params_linear['sqrt_one_minus_alpha'], color='black', linewidth=1.5, label='linear')
    my_cmap = ListedColormap(['red', 'yellow', 'blue'])
    sc1 = ax1.scatter(
        params_linear['X'], params_linear[y_name],
        c=params_linear['cluster_labels'], cmap=my_cmap,
        s=20, alpha=0.8, label="linear"
    )
    ax1.set_ylabel(f"{y_name} (linear)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    # 右 y 轴：cosine 调度
    ax2 = ax1.twinx()
    # plt.plot(params_cosine['X'], params_cosine['sqrt_one_minus_alpha'], color='blue', linewidth=1.5, label='cosine')
    sc2 = ax2.scatter(
        params_cosine['X'], params_cosine[y_name],
        c=params_cosine['cluster_labels'], cmap='viridis',
        s=20, alpha=0.8, label="cosine"
    )
    ax2.set_ylabel(f"{y_name} (cosine)", color="green")
    ax2.tick_params(axis='y', labelcolor="green")

    plt.title(title)
    ax1.set_xlabel("timestep")
    plt.legend(handles=[sc1, sc2], labels=["linear", "cosine"])
    # plt.grid(True)
    plt.show()


if __name__ == '__main__':
    num_timesteps = 100
    k = 3

    schedule_name = 'linear'
    params_linear = main(num_timesteps, schedule_name, k)

    schedule_name = 'cosine'
    params_cosine = main(num_timesteps, schedule_name, k)

    # 保存
    data = np.concatenate([
        params_linear['sqrt_one_minus_alpha'],
        params_linear['sqrt_one_minus_alpha_cumprod'],
        params_linear['noise_signal_ratio'],
        params_linear['noise_signal_ratio_cumprod'],
        params_cosine['sqrt_one_minus_alpha'],
        params_cosine['sqrt_one_minus_alpha_cumprod'],
        params_cosine['noise_signal_ratio'],
        params_cosine['noise_signal_ratio_cumprod']
    ], axis=1)

    df = pd.DataFrame(
        data,
        columns=[
            "l_sqrt_one_minus_alpha",
            "l_sqrt_one_minus_alpha_cumprod",
            "l_noise_signal_ratio",
            "l_noise_signal_ratio_cumprod",
            "c_sqrt_one_minus_alpha",
            "c_sqrt_one_minus_alpha_cumprod",
            "c_noise_signal_ratio",
            "c_noise_signal_ratio_cumprod"
        ]
    )

    df.to_csv("diffusion_params.csv", index=True)

    # 可视化 1：sqrt beta 曲线 + 聚类着色
    plot(
        params_linear,
        params_cosine,
        'sqrt_one_minus_alpha',
        "Clustering by Noise Strength (Linear vs Cosine)"
    )

    # 可视化 2：噪声强度曲线 + 分界线
    plot(
        params_linear,
        params_cosine,
        'sqrt_one_minus_alpha_cumprod',
        'Noise Strength Over Timesteps with Cluster Boundaries'
    )

    # 噪声-信号比
    plot(
        params_linear,
        params_cosine,
        'noise_signal_ratio',
        'noise to signal ratio'
    )

    # 噪声-信号比（累积）
    plot(
        params_linear,
        params_cosine,
        'noise_signal_ratio_cumprod',
        'noise to signal ratio (cumprod)'
    )
