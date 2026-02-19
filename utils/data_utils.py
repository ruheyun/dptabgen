# coding:UTF-8
# RuHe  2025/10/11 19:09
import tomllib
import json
import pickle
import os
import pandas as pd
import numpy as np
from typing import Union, Any, Callable, Dict, cast
from pathlib import Path
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from copy import deepcopy

RawConfig = Dict[str, Any]
_CONFIG_NONE = '__none__'
CAT_MISSING_VALUE = '__nan__'


# def _replace(obj: Any, cond: Callable[[Any], bool], value: Any) -> Any:
#     """
#     递归遍历 dict / list，将满足 cond 的值替换为 value。
#     """
#     if isinstance(obj, dict):
#         return {k: _replace(v, cond, value) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [_replace(v, cond, value) for v in obj]
#     elif cond(obj):
#         return value
#     else:
#         return obj


# def unpack_config(config: RawConfig) -> RawConfig:
#     """
#     替换 TOML 配置中所有 _CONFIG_NONE 为 Python None
#     """
#     config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
#     return config


# def load_config(path: Union[Path, str]) -> Any:
#     """
#     读取 TOML 配置文件。
#     """
#     path = Path(path)
#     with path.open("rb") as f:
#         config = tomllib.load(f)
#     return unpack_config(config)


# def load_json(path: Union[Path, str]) -> Any:
#     """
#     读取 json 配置文件
#     """
#     path = Path(path)
#     with open(path, "r") as f:
#         data = json.load(f)
#     return data


def data_preprocessing(raw_data, label, save_dir=None):
    data_wrapper = DataWrapper()
    label_wrapper = DataWrapper()
    data_wrapper.fit(raw_data)
    label_wrapper.fit(raw_data[[label]])

    if save_dir is not None:
        save_pickle(data=data_wrapper, path=os.path.join(save_dir, 'data_wrapper.pkl'))
        save_pickle(data=label_wrapper, path=os.path.join(save_dir, 'label_wrapper.pkl'))
    return data_wrapper, label_wrapper


def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


class DataWrapper:
    def __init__(self, num_encoder="quantile", seed=0):
        self.num_encoder = num_encoder
        self.seed = seed
        self.raw_dim = None
        self.raw_columns = None
        self.all_distinct_values = {}
        self.num_normalizer = {}
        self.num_dim = 0
        self.columns = []
        self.col_dim = []
        self.col_dtype = {}

    def fit(self, dataframe, all_category=False):
        self.raw_dim = dataframe.shape[1]
        self.raw_columns = dataframe.columns

        for i, col in enumerate(self.raw_columns):
            if all_category:
                break
            if is_numeric_dtype(dataframe[col]):
                col_data = dataframe.loc[pd.notna(dataframe[col])][col]
                self.col_dtype[col] = col_data.dtype
                if self.num_encoder == "quantile":
                    self.num_normalizer[col] = QuantileTransformer(
                        output_distribution='normal',  # normal  uniform
                        n_quantiles=max(min(len(col_data) // 30, 1000), 10),
                        subsample=1000000000,
                        random_state=self.seed, )
                elif self.num_encoder == "standard":
                    self.num_normalizer[col] = StandardScaler()
                elif self.num_encoder == "minmax":
                    self.num_normalizer[col] = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown num encoder: {self.num_encoder}")
                self.num_normalizer[col].fit(col_data.values.reshape(-1, 1))
                self.columns.append(col)
                self.num_dim += 1
                self.col_dim.append(1)
        for i, col in enumerate(self.raw_columns):
            if col not in self.num_normalizer.keys():
                col_data = dataframe.loc[pd.notna(dataframe[col])][col]
                self.col_dtype[col] = col_data.dtype
                distinct_values = col_data.unique()
                distinct_values.sort()
                self.all_distinct_values[col] = distinct_values
                self.columns.append(col)
                self.col_dim.append(max(1, int(np.ceil(np.log2(len(distinct_values))))))

    def transform(self, data):
        reorder_data = data[self.columns].values
        norm_data = []
        for i, col in enumerate(self.columns):
            col_data = reorder_data[:, i]
            if col in self.all_distinct_values.keys():
                col_data = self.CatValsToNum(col, col_data).reshape(-1, 1)
                col_data = self.ValsToBit(col_data, self.col_dim[i])
                norm_data.append(col_data)
            elif col in self.num_normalizer.keys():
                norm_data.append(self.num_normalizer[col].transform(col_data.reshape(-1, 1)).reshape(-1, 1))
        norm_data = np.concatenate(norm_data, axis=1)
        norm_data = norm_data.astype(np.float32)
        return norm_data

    def ReOrderColumns(self, data: pd.DataFrame):
        ndf = pd.DataFrame([])
        for col in self.raw_columns:
            ndf[col] = data[col]
        return ndf

    def GetColData(self, data, col_id):
        col_index = np.cumsum(self.col_dim)
        col_data = data.copy()
        if col_id == 0:
            return col_data[:, :col_index[0]]
        else:
            return col_data[:, col_index[col_id - 1]:col_index[col_id]]

    def ValsToBit(self, values, bits):
        bit_values = np.zeros((values.shape[0], bits))
        for i in range(values.shape[0]):
            bit_val = np.mod(np.right_shift(int(values[i]), list(reversed(np.arange(bits)))), 2)
            bit_values[i, :] = bit_val
        return bit_values

    def BitsToVals(self, bit_values):
        bits = bit_values.shape[1]
        values = bit_values.astype(int)
        values = values * (2 ** np.array(list((reversed(np.arange(bits))))))
        values = np.sum(values, axis=1)
        return values

    def CatValsToNum(self, col, values):
        num_values = pd.Categorical(values, categories=self.all_distinct_values[col]).codes
        # num_values = np.zeros_like(values)
        # for i, val in enumerate(values):
        # 	ind = np.where(self.all_distinct_values[col] == val)
        # 	num_values[i] = ind[0][0]
        return num_values

    def NumValsToCat(self, col, values):
        cat_values = np.zeros_like(values).astype(object)
        # print(col_name, values)
        values = np.clip(values, 0, len(self.all_distinct_values[col]) - 1)
        for i, val in enumerate(values):
            # val = np.clip(val, self.Mins[col_id], self.Maxs[col_id])
            cat_values[i] = self.all_distinct_values[col][int(val)]
        return cat_values

    def ReverseToOrdi(self, data):
        reverse_data = []

        for i, col in enumerate(self.columns):
            col_data = self.GetColData(data, i)
            if col in self.all_distinct_values.keys():
                col_data = np.round(col_data)
                col_data = self.BitsToVals(col_data)
                col_data = col_data.astype(np.int32)
            else:
                col_data = self.num_normalizer[col].inverse_transform(col_data.reshape(-1, 1))
                if self.col_dtype[col] == np.int32 or self.col_dtype[col] == np.int64:
                    col_data = np.round(col_data).astype(int)
            # col_data = self.NumValsToCat(col, col_data)
            # col_data = col_data.astype(self.raw_data[col].dtype)
            reverse_data.append(col_data.reshape(-1, 1))
        reverse_data = np.concatenate(reverse_data, axis=1)
        return reverse_data

    def ReverseToCat(self, data):
        reverse_data = []
        for i, col in enumerate(self.columns):
            col_data = data[:, i]
            if col in self.all_distinct_values.keys():
                col_data = self.NumValsToCat(col, col_data)
            reverse_data.append(col_data.reshape(-1, 1))
        reverse_data = np.concatenate(reverse_data, axis=1)
        return reverse_data

    def Reverse(self, data):
        data = self.ReverseToOrdi(data)
        data = self.ReverseToCat(data)
        data = pd.DataFrame(data, columns=self.columns)
        return self.ReOrderColumns(data)

    def RejectSample(self, sample):
        all_index = set(range(sample.shape[0]))
        allow_index = set(range(sample.shape[0]))
        for i, col in enumerate(self.columns):
            if col in self.all_distinct_values.keys():
                allow_index = allow_index & set(np.where(sample[:, i] < len(self.all_distinct_values[col]))[0])
                allow_index = allow_index & set(np.where(sample[:, i] >= 0)[0])
        reject_index = all_index - allow_index
        allow_index = np.array(list(allow_index))
        reject_index = np.array(list(reject_index))
        # allow_sample = sample[allow_index, :]
        return allow_index, reject_index


def num_missing_values(data, policy='mean'):
    """
    对数值特征判空和处理
    :param data: 需要处理的字典数据集
    :param policy: 处理策略
    :return: 处理后的字典数据集
    """
    if policy is None:
        return data

    data = deepcopy(data)
    splits = ['train', 'val', 'test']
    has_nan = any(
        ('num' in data[split]) and (not data[split]['num'].empty) and data[split]['num'].isna().any().any()
        for split in splits
    )
    if not has_nan:
        return data

    if policy == 'drop-rows':
        if ('num' in data['test']) and (not data['test']['num'].empty) and data['test']['num'].isna().any().any():
            raise ValueError("Test set contains missing values. Cannot apply 'drop-rows' policy.")

        for split in ['train', 'val']:
            df = data[split]['num']
            if df.empty:
                continue
            valid_rows = df.notna().all(axis=1)
            for key in ['num', 'cat', 'y']:
                if key in data[split] and not data[split][key].empty:
                    data[split][key] = data[split][key][valid_rows]

    elif policy == 'mean':
        train_df = data['train']['num']
        if train_df.empty or train_df.select_dtypes(include=[np.number]).empty:
            raise ValueError("Training numerical data is empty or non-numeric; cannot compute mean.")
        fill_values = train_df.mean()
        for split in splits:
            if 'num' in data[split] and not data[split]['num'].empty:
                data[split]['num'] = data[split]['num'].fillna(fill_values)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return data


def cat_missing_values(data, policy='most_frequent'):
    """
    对 data[split]['cat'] 中的类别型 DataFrame 进行缺失值处理。
    支持 np.nan 和自定义缺失符号。
    仅在 train 上拟合 imputer，避免数据泄露。

    Args:
        data: dict, 形如 data[split]['cat']，每个都是 pandas.DataFrame
        policy: None 或 'most_frequent'
        CAT_MISSING_VALUE: 用于标记缺失的特殊值（如 "__MISSING__"）

    Returns:
        处理后的 data（深拷贝）
    """
    if policy is None:
        return data

    data = deepcopy(data)
    splits = ['train', 'val', 'test']
    for split in splits:
        data[split]['cat'] = normalize_missing(data[split]['cat'])

    # 检查是否有缺失值（包括 np.nan 和自定义缺失标记）
    has_missing = any(
        (
                data[split]['cat'].isna().any().any() or
                (data[split]['cat'] == CAT_MISSING_VALUE).any().any()
        )
        for split in splits
        if 'cat' in data[split] and not data[split]['cat'].empty
    )

    if not has_missing or policy is None:
        return data  # 无缺失或不处理

    if policy == 'most_frequent':
        # 取训练集类别数据
        df_train_cat = data['train']['cat']
        if df_train_cat.empty:
            raise ValueError("Training categorical data is empty; cannot fit imputer.")

        # 检查是否全列缺失
        all_missing = (
                df_train_cat.isna().all() |
                (df_train_cat == CAT_MISSING_VALUE).all()
        )
        if all_missing.any():
            bad_cols = list(df_train_cat.columns[all_missing])
            raise ValueError(f"Columns entirely missing in train: {bad_cols}")

        # 初始化填充器：默认识别 np.nan；我们先将自定义缺失标记替换为 np.nan
        df_train_cat = df_train_cat.replace(CAT_MISSING_VALUE, np.nan)

        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(df_train_cat)

        # 对所有 split 的 cat 进行填充
        for split in splits:
            if 'cat' not in data[split] or data[split]['cat'].empty:
                continue

            df_cat = data[split]['cat'].replace(CAT_MISSING_VALUE, np.nan)
            filled = imputer.transform(df_cat)
            filled_df = pd.DataFrame(filled, columns=df_cat.columns, index=df_cat.index)

            # 恢复原列类型（例如 category）
            data[split]['cat'] = filled_df.astype(df_cat.dtypes.to_dict())

    else:
        raise ValueError(f"Unknown categorical NaN policy: {policy}")

    return data


def normalize_missing(df):
    return df.replace(["nan", "NaN", "NAN", None, ""], CAT_MISSING_VALUE)


def y_build_target(data, task_type):
    """
    对 data[split]['y'] 应用 build_target 逻辑：
      - 回归：标准化（仅用 train 统计量）
      - 分类：LabelEncoder（仅用 train 拟合）
    保持 DataFrame 结构不变。
    """
    splits = ['train', 'val', 'test']
    info: Dict[str, Any] = {'policy': None, 'task_type': task_type}

    # 提取 train y（必须非空）
    y_train_df = data['train']['y']
    if y_train_df.empty:
        raise ValueError("Training labels (y) are empty.")

    # -------------------- 回归任务 --------------------
    if task_type == 'regression':
        y_train = y_train_df.values.astype(float).squeeze()
        if y_train.ndim > 1:
            raise ValueError("Multi-output regression not supported.")

        mean = float(np.mean(y_train))
        std = float(np.std(y_train)) or 1.0

        info.update({
            'policy': 'standardize',
            'mean': mean,
            'std': std
            # 'inverse_fn': lambda x: x * std + mean
        })

        for split in splits:
            y_df = data[split]['y']
            if not y_df.empty:
                y_vals = y_df.values.astype(float)
                y_norm = (y_vals - mean) / std
                data[split]['y'] = pd.DataFrame(
                    y_norm, columns=y_df.columns, index=y_df.index
                )

    # -------------------- 分类任务 --------------------
    elif task_type == 'binclass' or task_type == 'multiclass':
        y_train = y_train_df.values.squeeze()

        # 使用 LabelEncoder（仅用 train 拟合）
        encoder = LabelEncoder()
        encoder.fit(y_train)

        info.update({
            'policy': 'label_encode',
            'classes': encoder.classes_.tolist()
        })

        data = fast_encode_label(data, encoder)

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return data, info


# 反编码y
# encoder = info['encoder']
# y_pred_labels = encoder.inverse_transform(y_pred_encoded)

def fast_encode_label(data, encoder, splits=('train', 'val', 'test')):
    # 构建类别 → 编码的映射字典
    mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}

    for split in splits:
        y_df = data[split]['y']
        if y_df.empty:
            continue

        # 提取 Series（保证形状对齐）
        y_series = y_df.squeeze()

        # 使用 Series.map 快速替换
        y_encoded = y_series.map(mapping)

        # 将未知类别（map结果为 NaN）填成 -1
        y_encoded = y_encoded.fillna(-1).astype(int)

        # 恢复成 DataFrame
        data[split]['y'] = pd.DataFrame(
            y_encoded.values.reshape(-1, 1),
            columns=y_df.columns,
            index=y_df.index
        )
    return data
