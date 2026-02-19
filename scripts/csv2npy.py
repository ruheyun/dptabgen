import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import os

# ==================== 配置区域 ====================
CSV_TRAIN_PATH = 'data/default/default_train.csv'
CSV_TEST_PATH = 'data/default/default_test.csv'
OUTPUT_DIR = 'data/default'

# 缺失值填充策略
FILL_NUM_STRATEGY = 'median'  # 'mean' 或 'median' 或 0
FILL_CAT_STRATEGY = 'Unknown'  # 'Unknown' 或 'mode'
# ==================================================

def main():
    print("=" * 60)
    print("CSV 转 NPY 转换器")
    print("=" * 60)
    
    # 1. 读取CSV文件
    print("\n[1/8] 读取CSV文件...")
    train_df = pd.read_csv(CSV_TRAIN_PATH)
    test_df = pd.read_csv(CSV_TEST_PATH)
    
    print(f"  训练集: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
    print(f"  测试集: {test_df.shape[0]} 行 × {test_df.shape[1]} 列")
    
    # 2. 检查缺失值
    print("\n[2/8] 检查缺失值...")
    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()
    
    if train_missing.sum() > 0:
        print("  训练集缺失值:")
        for col, count in train_missing.items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(train_df)*100:.2f}%)")
    else:
        print("   训练集无缺失值")
    
    if test_missing.sum() > 0:
        print("  测试集缺失值:")
        for col, count in test_missing.items():
            if count > 0:
                print(f"    {col}: {count} ({count/len(test_df)*100:.2f}%)")
    else:
        print("   测试集无缺失值")
    
    # 3. 识别特征类型
    print("\n[3/8] 识别特征类型...")
    all_columns = train_df.columns.tolist()
    target_col = all_columns[-1]  # 假设最后一列是目标变量
    feature_cols = all_columns[:-1]
    
    num_features = []
    cat_features = []
    
    for col in feature_cols:
        if train_df[col].dtype in ['int64', 'float64']:
            num_features.append(col)
        else:
            cat_features.append(col)
    
    print(f"  数值型特征 ({len(num_features)}): {num_features}")
    print(f"  分类型特征 ({len(cat_features)}): {cat_features}")
    print(f"  目标变量: {target_col}")
    
    # 4. 分离特征和标签
    print("\n[4/8] 分离特征和标签...")
    X_train_full = train_df[feature_cols].copy()
    y_train_full = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()
    
    # 5. 处理缺失值
    print("\n[5/8] 处理缺失值...")
    
    # 数值型特征填充
    for col in num_features:
        if X_train_full[col].isnull().sum() > 0 or X_test[col].isnull().sum() > 0:
            if FILL_NUM_STRATEGY == 'median':
                fill_value = X_train_full[col].median()
            elif FILL_NUM_STRATEGY == 'mean':
                fill_value = X_train_full[col].mean()
            elif FILL_NUM_STRATEGY == '0':
                fill_value = 0
            else:
                fill_value = X_train_full[col].median()
            
            X_train_full[col].fillna(fill_value, inplace=True)
            X_test[col].fillna(fill_value, inplace=True)
            print(f"   {col} (数值型): 用{FILL_NUM_STRATEGY}={fill_value:.4f} 填充")
    
    # 分类型特征填充
    for col in cat_features:
        if X_train_full[col].isnull().sum() > 0 or X_test[col].isnull().sum() > 0:
            if FILL_CAT_STRATEGY == 'mode':
                fill_value = X_train_full[col].mode()[0]
            else:
                fill_value = FILL_CAT_STRATEGY
            
            X_train_full[col].fillna(fill_value, inplace=True)
            X_test[col].fillna(fill_value, inplace=True)
            print(f"   {col} (分类型): 用'{fill_value}' 填充")
    
    # 目标变量处理（如果有缺失，删除对应行）
    if y_train_full.isnull().sum() > 0:
        mask = y_train_full.notna()
        X_train_full = X_train_full[mask]
        y_train_full = y_train_full[mask]
        print(f"   目标变量: 删除 {mask.sum() - len(y_train_full)} 行缺失值")
    
    if y_test.isnull().sum() > 0:
        mask = y_test.notna()
        X_test = X_test[mask]
        y_test = y_test[mask]
        print(f"   测试集目标变量: 删除 {mask.sum() - len(y_test)} 行缺失值")
    
    # 6. 分割训练集为train和validation
    print("\n[6/8] 分割训练集 (train/val)...")
    
    # 检查是否可以分层抽样
    try:
        if y_train_full.dtype in ['int64', 'float64'] or len(y_train_full.unique()) <= 20:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_train_full
            )
            print("   使用分层抽样 (stratified split)")
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, 
                test_size=0.2, 
                random_state=42
            )
            print("   使用随机抽样 (random split)")
    except:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.2, 
            random_state=42
        )
        print("   使用随机抽样 (random split)")
    
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 7. 分离数值型和分类型特征并保存
    print("\n[7/8] 保存为NPY文件...")
    
    # 数值型特征
    X_train_num = X_train[num_features].values.astype(np.float32)
    X_val_num = X_val[num_features].values.astype(np.float32)
    X_test_num = X_test[num_features].values.astype(np.float32)
    
    # 分类型特征（保持字符串，使用object类型）
    X_train_cat = X_train[cat_features].values.astype(object)
    X_val_cat = X_val[cat_features].values.astype(object)
    X_test_cat = X_test[cat_features].values.astype(object)
    
    # 目标变量
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # 保存文件
    np.save(os.path.join(OUTPUT_DIR, 'X_num_train.npy'), X_train_num)
    np.save(os.path.join(OUTPUT_DIR, 'X_cat_train.npy'), X_train_cat)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_num_val.npy'), X_val_num)
    np.save(os.path.join(OUTPUT_DIR, 'X_cat_val.npy'), X_val_cat)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_num_test.npy'), X_test_num)
    np.save(os.path.join(OUTPUT_DIR, 'X_cat_test.npy'), X_test_cat)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    
    print("   已保存:")
    print("    - X_num_train.npy, X_cat_train.npy, y_train.npy")
    print("    - X_num_val.npy, X_cat_val.npy, y_val.npy")
    print("    - X_num_test.npy, X_cat_test.npy, y_test.npy")
    
    # 8. 生成info.json
    print("\n[8/8] 生成info.json...")
    
    # 确定任务类型
    if y_train.dtype in ['int64', 'float64']:
        num_classes = len(np.unique(y_train))
        if num_classes == 2:
            task_type = "binclass"
        elif num_classes > 2:
            task_type = "multiclass"
        else:
            task_type = "regression"
    else:
        num_classes = len(np.unique(y_train))
        task_type = "binclass" if num_classes == 2 else "multiclass"
    
    info = {
        "name": "default",
        "id": "default--default",
        "task_type": task_type,
        "num_classes": int(num_classes),
        "n_num_features": len(num_features),
        "n_cat_features": len(cat_features),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test))
    }
    
    with open(os.path.join(OUTPUT_DIR, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    
    print("   info.json 内容:")
    print(json.dumps(info, indent=4))
    
    print("\n" + "=" * 60)
    print(" 转换完成！所有文件已保存。")
    print("=" * 60)
    print("\n 注意事项:")
    print("  1. 分类特征以字符串形式保存 (dtype=object)")
    print("  2. 后续使用前需要手动编码分类特征")
    print("  3. 所有缺失值已处理完毕")
    print("\n 生成的文件位于:", OUTPUT_DIR)

if __name__ == "__main__":
    main()