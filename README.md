安装环境
```
conda create -n dptabdm python==3.12.8

conda activate dptabdm

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install -r requirements.txt
```

训练模型（采样、评估）
```
python main.py --config config/adult/config.toml --train --sample --eval --eval_type mlp
```

不同种子采样、不同种子评估(先运行训练模型的命令)
```
python eval_seeds.py --config config/adult/config.toml --eval_type synthetic --model_type catboost
```

评估模型超参数自动调优
```
python tune_evaluation_model.py --ds_name adult --model mlp --tune_type val
```

数据集编码
```
python scripts/data_wrapper.py --config config/adult/config.toml
```