import json
import pickle
import pandas as pd
import torch
import numpy as np
import delu
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from ddpm.gaussian_diffusion import GaussianDiffusion
from utils.utils_train import get_model, make_dataset
import lib
from lib import round_columns
from sklearn.preprocessing import LabelEncoder


def sample(
        exp_dir='expdir',
        data_path='data/adult',
        batch_size=2000,
        num_samples=0,
        model_type='mlp',
        model_params=None,
        model_path=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        disbalance=None,
        device=torch.device('cuda:0'),
        seed=0,
        change_val=False
):
    data_path = os.path.normpath(data_path)
    exp_dir = os.path.normpath(exp_dir)
    trans_data_path = os.path.join(exp_dir, T_dict['cat_encoding'])

    delu.random.seed(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        trans_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    model_params['d_in'] = int(num_numerical_features_)

    model = get_model(
        model_type,
        model_params
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()
    K = np.array([0])
    # diffusion = GaussianMultinomialDiffusion(
    #     K,
    #     num_numerical_features=num_numerical_features_,
    #     denoise_fn=model,
    #     num_timesteps=num_timesteps,
    #     gaussian_loss_type=gaussian_loss_type,
    #     scheduler=scheduler,
    #     device=device
    # )
    diffusion = GaussianDiffusion(
        input_dim=num_numerical_features_,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )

    diffusion.to(device)
    diffusion.eval()

    print('Starting sampling...')
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    if disbalance == 'fix':
        # 交换类别比例（反转不平衡），只适用于二分类
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen_, y_gen_ = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        # 补足少数类至与多数类持平（过采样），适用于二分类和多分类
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen_list, y_gen_list = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
            x_gen_list.append(x_temp)
            y_gen_list.append(y_temp)

        x_gen_ = torch.cat(x_gen_list, dim=0)
        y_gen_ = torch.cat(y_gen_list, dim=0)

    else:
        # 按原始分布生成，适用于二分类和多分类
        x_gen_, y_gen_ = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    X_gen_, y_gen_ = x_gen_.cpu().numpy(), y_gen_.cpu().numpy()

    # if y_gen_.ndim == 1:
    #     y_gen_ = y_gen_[:, None]

    # Save as standard TabDDPM format (only X_num and y)
    np.save(os.path.join(exp_dir, 'X_gen_unreverse.npy'), X_gen_)
    np.save(os.path.join(exp_dir, 'y_gen_unreverse.npy'), y_gen_)

    print(f"Raw samples saved to {exp_dir}")
    print(f"X shape: {X_gen_.shape}, y shape: {y_gen_.shape}")

    # Load the DataWrapper used for preprocessing
    with open(f"{trans_data_path}/data_wrapper.pkl", "rb") as f:
        data_wrapper = pickle.load(f)

    df_synthetic = data_wrapper.Reverse(X_gen_)

    with open(os.path.join(trans_data_path, "info.json"), 'r') as f:
        info = json.load(f)

    if info['task_type'] == 'regression':
        y_gen = df_synthetic.iloc[:, 0]
        X_gen = df_synthetic.iloc[:, 1:]
    else:
        encoder = LabelEncoder()
        encoder.classes_ = np.array(info['classes'])
        y_gen = encoder.inverse_transform(y_gen_)
        X_gen = df_synthetic

    y_gen = pd.DataFrame(y_gen)
    y_gen.columns = ['label']
    X_gen = pd.DataFrame(X_gen)

    # Get column names from DataWrapper
    num_cols = [f"num_{i}" for i in range(data_wrapper.num_dim - int(info['task_type'] == 'regression'))]
    cat_cols = [f"cat_{i}" for i in range(len(data_wrapper.all_distinct_values))]

    # Extract X_num and X_cat
    X_num = X_gen[num_cols].values.astype(np.float32)  # 数值列
    X_cat = X_gen[cat_cols].values.astype(object)      # 类别列（字符串）

    if X_num is not None:
        X_num_real = np.load(os.path.join(data_path, "X_num_train.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    # Save as standard TabDDPM format
    np.save(os.path.join(exp_dir, 'X_num_gen.npy'), X_num)
    np.save(os.path.join(exp_dir, 'X_cat_gen.npy'), X_cat)
    np.save(os.path.join(exp_dir, 'y_gen.npy'), y_gen)

    sample_data = pd.concat([X_gen, y_gen], axis=1)

    sample_data.to_csv(os.path.join(exp_dir, 'gen.csv'), index=False, header=True)
    print('Sample done!')
