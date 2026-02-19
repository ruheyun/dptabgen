# coding:UTF-8
# RuHe  2025/10/16 11:35
# ========== 将 npy 的原始数据通过模拟位编码，得到 npy 文件数据 ==========
import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
import re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils.data_utils import DataWrapper, num_missing_values, cat_missing_values, y_build_target
from lib import load_config, load_json
from scripts.npy2csv import read_files_in_folder


def process_wrapper(
        exp_path='expdir/ault/',
        data_path='data/ault/',
        cat_encode='',
        task_type='binclass'
):
    print('Start preprocessing the data...')
    output_dir = os.path.join(exp_path, cat_encode)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load .npy files
    npy_files, _, _ = read_files_in_folder(data_path)
    splits = ['train', 'val', 'test']
    types = ['num', 'cat', 'y']
    data = {s: {t: pd.DataFrame() for t in types} for s in splits}
    for f in npy_files:
        splits_name = re.split(r'[._]', f)[-2]
        types_name = re.split(r'[._]', f)[-3]
        if splits_name in data and types_name in data[splits_name]:
            data[splits_name][types_name] = pd.DataFrame(np.load(os.path.join(data_path, f), allow_pickle=True))

    # Step 2: Handling null values in numeric and categorical columns and y column
    if not data['train']['num'].empty:
        data = num_missing_values(data, policy=None)
    if not data['train']['cat'].empty:
        data = cat_missing_values(data, policy=None)

    # 打印每个类别特征的类别数
    cat_sizes = []
    for i in range(data['train']['cat'].shape[1]):
        unique_vals = np.unique(data['train']['cat'].iloc[:, i])
        n_classes = len(unique_vals)
        cat_sizes.append(n_classes)
    print(f"Total categorical features: {cat_sizes}")
    col_trans_analog_nums = sum(int(np.ceil(np.log2(max(k, 1)))) for k in cat_sizes)
    print(f"Total binary bits after encoding: {col_trans_analog_nums}")

    # Step 3: Reconstruct DataFrame for fitting
    num_cols = [f"num_{i}" for i in range(data['train']['num'].shape[1])] if data['train']['num'].size > 0 else []
    cat_cols = [f"cat_{i}" for i in range(data['train']['cat'].shape[1])] if data['train']['cat'].size > 0 else []
    y_cols = ['label'] if data['train']['y'].size > 0 else []
    if task_type == 'regression':
        print(f'Total column: {y_cols + num_cols + cat_cols}')
        df_train = pd.concat([data['train']['y'], data['train']['num'], data['train']['cat']], axis=1)
        df_train.columns = y_cols + num_cols + cat_cols
    else:
        print(f'Total column: {num_cols + cat_cols}')
        df_train = pd.concat([data['train']['num'], data['train']['cat']], axis=1)
        df_train.columns = num_cols + cat_cols

    # Step 4: Fit DataWrapper
    train_wrapper = DataWrapper()
    train_wrapper.fit(df_train, all_category=False)  # all_category=False: let it auto-detect

    # Step 5: Transform all splits
    def to_dataframe(X_num, X_cat, y, num_cols, cat_cols, y_cols, task_type):
        if X_num.size == 0:
            X_combined = X_cat
            cols = cat_cols
        elif X_cat.size == 0:
            X_combined = X_num
            cols = num_cols
        else:
            X_combined = pd.concat([X_num, X_cat], axis=1)
            cols = num_cols + cat_cols
        if task_type == 'regression':
            X_combined = pd.concat([y, X_combined], axis=1)
            cols = y_cols + cols
        X_combined.columns = cols
        return X_combined

    df_val = to_dataframe(data['val']['num'], data['val']['cat'], data['val']['y'], num_cols, cat_cols, y_cols,
                          task_type)
    df_test = to_dataframe(data['test']['num'], data['test']['cat'], data['test']['y'], num_cols, cat_cols, y_cols,
                           task_type)

    # Transform
    X_train_processed = train_wrapper.transform(df_train)
    X_val_processed = train_wrapper.transform(df_val)
    X_test_processed = train_wrapper.transform(df_test)

    # Step 6: Processing label
    data, y_info = y_build_target(data, task_type)
    for s in splits:
        data[s]['y'].columns = ['label']

    # Step 7: Save as new X_num_*.npy (no X_cat!)
    np.save(os.path.join(output_dir, "X_num_train.npy"), X_train_processed)
    np.save(os.path.join(output_dir, "X_num_val.npy"), X_val_processed)
    np.save(os.path.join(output_dir, "X_num_test.npy"), X_test_processed)

    np.save(os.path.join(output_dir, "y_train.npy"), data['train']['y'])
    np.save(os.path.join(output_dir, "y_val.npy"), data['val']['y'])
    np.save(os.path.join(output_dir, "y_test.npy"), data['test']['y'])

    # Step 8: Save info.json
    n_features = X_train_processed.shape[1]
    # Determine task type from original info.json if available
    info_path = os.path.join(data_path, "info.json")

    assert n_features == col_trans_analog_nums + len(num_cols) + int(
        task_type == 'regression'), 'Category dimension conversion error!'

    with open(info_path, 'r') as f:
        info = json.load(f)
        info["n_cat_features"] = col_trans_analog_nums
        info['n_features'] = n_features
        info['X_num_train_path'] = os.path.join(output_dir, 'X_num_train.npy')
        info['X_num_val_path'] = os.path.join(output_dir, 'X_num_val.npy')
        info['X_num_test_path'] = os.path.join(output_dir, 'X_num_test.npy')
        info['y_train_path'] = os.path.join(output_dir, 'y_train.npy')
        info['y_val_path'] = os.path.join(output_dir, 'y_val.npy')
        info['y_test_path'] = os.path.join(output_dir, 'y_test.npy')

        for k in y_info:
            if k not in info:
                info[k] = y_info[k]

    with open(os.path.join(output_dir, "info.json"), 'w') as f:
        json.dump(info, f)

    # Step 7: Save DataWrapper for later reverse
    with open(os.path.join(output_dir, "data_wrapper.pkl"), 'wb') as f:
        pickle.dump(train_wrapper, f)

    print(f"Preprocessing done! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', default='config/buddy/config.toml')

    args = parser.parse_args()
    raw_config = load_config(args.config)
    raw_info = load_json(os.path.join(raw_config['data_path'], 'info.json'))

    folder_path = os.path.join(raw_config['exp_path'], raw_config['train']['T']['cat_encoding'])
    os.makedirs(folder_path, exist_ok=True)
    if not os.listdir(folder_path):
        print('Start converting the data now!')
        process_wrapper(
            exp_path=raw_config['exp_path'],
            data_path=raw_config['data_path'],
            cat_encode=raw_config['train']['T']['cat_encoding'],
            task_type=raw_info['task_type']
        )


if __name__ == '__main__':
    main()
