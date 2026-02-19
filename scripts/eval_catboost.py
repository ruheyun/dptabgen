from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import os
import delu
import lib
from pprint import pprint
from lib import concat_features, read_pure_data, get_catboost_config, read_changed_val


def train_catboost(
        exp_dir,
        data_path,
        eval_type,
        T_dict,
        seed=0,
        params=None,
        change_val=True
):
    delu.random.seed(seed)

    synthetic_data_path = os.path.normpath(exp_dir)
    info = lib.load_json(os.path.join(exp_dir, 'analogbit/info.json'))
    T = lib.Transformations(**T_dict)

    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(data_path, val_size=0.2)

    X = None
    print('-' * 100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(data_path)
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path)

        y = np.concatenate([y_real, y_fake], axis=0)

        X_num = None
        if X_num_real is not None:
            X_num = np.concatenate([X_num_real, X_num_fake], axis=0)

        X_cat = None
        if X_cat_real is not None:
            X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    elif eval_type == 'synthetic':
        print(f'loading synthetic data: {synthetic_data_path}')
        X_num, X_cat, y = read_pure_data(synthetic_data_path, split='gen')

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num, X_cat, y = read_pure_data(data_path)
    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(data_path, 'val')
    X_num_test, X_cat_test, y_test = read_pure_data(data_path, 'test')

    y = y.reshape(-1)

    D = lib.Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        lib.TaskType(info['task_type']),
        info.get('num_classes')
    )

    # D = lib.transform_dataset(D, T, None)
    X = concat_features(D)
    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')

    if params is None:
        catboost_config = get_catboost_config(data_path)
    else:
        catboost_config = params

    if 'cat_features' not in catboost_config:
        catboost_config['cat_features'] = list(range(D.n_num_features, D.n_features))

    for col in range(D.n_features):
        for split in X.keys():
            if col in catboost_config['cat_features']:
                X[split][col] = X[split][col].astype(str)
            else:
                X[split][col] = X[split][col].astype(float)
    print(T_dict)
    pprint(catboost_config, width=100)
    print('-' * 100)

    if D.is_regression:
        model = CatBoostRegressor(
            **catboost_config,
            eval_metric='RMSE',
            random_seed=seed,
            allow_writing_files=False
        )
        predict = model.predict
    else:
        # for split in ['train', 'val', 'test']:
        #     D.y[split] = D.y[split].astype('int64')
        model = CatBoostClassifier(
            loss_function="MultiClass" if D.is_multiclass else "Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=seed,
            class_names=[str(i) for i in range(D.n_classes)] if D.is_multiclass else ["0", "1"],
            allow_writing_files=False
        )
        predict = (
            model.predict_proba
            if D.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]
        )
    if D.y['train'].ndim > 1:
        D.y['train'] = D.y['train'].squeeze()

    model.fit(X['train'], D.y['train'], eval_set=(X['val'], D.y['val']), verbose=100)
    predictions = {k: predict(v) for k, v in X.items()}
    print(predictions['train'].shape)

    report = {'eval_type': eval_type, 'dataset': data_path,
              'metrics': D.calculate_metrics(predictions, None if D.is_regression else 'probs')}

    metrics_report = lib.MetricsReport(report['metrics'], D.task_type)
    metrics_report.print_metrics()

    if exp_dir is not None:
        lib.dump_json(report, os.path.join(exp_dir, "results_catboost.json"))

    return metrics_report
