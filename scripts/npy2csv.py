import numpy as np
import pandas as pd
import json
import re
import os


def npy2csv(path, files):
    """
    npy 文件转化为 csv 文件
    """
    assert files[0].split('.')[-1] == 'npy', f'输入不是 npy 为后缀的二进制文件'

    splits = ['train', 'val', 'test']
    types = ['num', 'cat', 'y']

    data = {s: {t: pd.DataFrame() for t in types} for s in splits}

    for f in files:
        splits_name = re.split(r'[._]', f)[-2]
        types_name = re.split(r'[._]', f)[-3]
        if splits_name in data and types_name in data[splits_name]:
            data[splits_name][types_name] = pd.DataFrame(np.load(os.path.join(path, f), allow_pickle=True))

    # 定义列名
    num_cols = [f'C{i + 1}' for i in range(data['train']['num'].shape[1])]
    cat_cols = [f'C{i + len(num_cols) + 1}' for i in range(data['train']['cat'].shape[1])]
    all_cols = num_cols + cat_cols + ['label']

    for sn in splits:
        df = pd.concat([data[sn]['num'], data[sn]['cat'], data[sn]['y']], axis=1)
        df.columns = all_cols
        df.to_csv(os.path.join(path, f'{sn}.csv'), index=False, header=True)

    print('已将 npy 数据集转化为 csv 格式。')

    # 生成数据集信息 info.json
    name = path.split('/')[-1]
    num_classes = int(data['train']['y'].iloc[:, 0].nunique())  # 假设 y 是单列 DataFrame
    task_type = 'binclass' if num_classes == 2 else 'multiclass' if num_classes < 50 else 'regression'
    dataset_info = {
        "name": name,  # 数据集名字
        "id": f"{name}--default",  # 同上，数据集命名一致
        "task_type": task_type,  # 二分类任务，可改为 multiclass / regression 等
        "num_classes": num_classes if num_classes < 50 else None,
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "train_size": len(data['train']['y']),
        "val_size": len(data['val']['y']),
        "test_size": len(data['test']['y']),
    }

    json_path = os.path.join(path, 'info.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    print('已生成 json 数据集信息文件。')


def read_files(path, files):
    if len(files):
        file_type = files[0].split('.')[-1]
    else:
        return

    for f in files:
        try:
            # 分别处理三类文件
            if file_type == 'npy':
                data = np.load(os.path.join(path, f), allow_pickle=True)
                print(f'文件：{f} \n 第一个元素: {data[0]}，形状: {data.shape} \n')
            elif file_type == 'csv':
                df = pd.read_csv(os.path.join(path, f))
                print(f'文件: {f} \n 第一行: {df.head(1)}，形状：{df.shape} \n')
            elif file_type == 'json':
                with open(os.path.join(path, f), 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                print(f'文件：{f} \n 内容：{data} \n')
        except Exception as e:
            print(f"读取 {f} 出错: {e}")


def read_files_in_folder(path):
    """
    读取指定文件夹中的所有文件，并尝试读取每个文件内容。
    """
    assert os.path.isdir(path), f'错误：{path} 不是一个有效的文件夹。'

    # 获取文件夹中所有文件，排除子目录
    npy_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.npy')]
    csv_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]
    json_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.json')]

    return npy_files, csv_files, json_files


if __name__ == '__main__':
    # 对于不同数据集，只需改下面一个参数即可
    path = "../data/king"

    # 读取文件夹下所有文件
    npy_files, csv_files, json_files = read_files_in_folder(path)

    # 读取文件内容并打印
    read_files(path, npy_files)
    # read_files(path, csv_files)
    # read_files(path, json_files)

    # npy文件转为csv文件
    npy2csv(path, npy_files)

