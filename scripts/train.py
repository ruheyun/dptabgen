# coding:UTF-8
# RuHe  2025/10/19 16:59
from copy import deepcopy
import torch
import os
import numpy as np
import delu
import sys
from tqdm import trange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from ddpm import GaussianDiffusion
from utils.utils_train import get_model, make_dataset, update_ema
import pandas as pd
import lib


class Trainer:
    def __init__(self, diffusion, ema_model, train_iter, lr, optimizer, steps, dp_params, privacy_engine,
                 stage, loss_history, device=torch.device('cuda:0')):
        self.diffusion = diffusion
        self.ema_model = ema_model
        self.train_iter = train_iter
        self.steps = steps
        self.dp_params = dp_params
        self.privacy_engine = privacy_engine
        self.init_lr = lr
        self.optimizer = optimizer
        self.device = device
        self.stage = stage
        self.loss_history = loss_history
        self.log_every = 100
        self.print_every = 20
        self.ema_every = 100

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict, step):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.diffusion.compute_loss(x, out_dict, stage=self.stage)
        loss.backward()

        # 打印所有参数的梯度
        if step % 1000 == 0:
            total = torch.tensor(0.0, device=self.device)
            for name, param in self.diffusion._denoise_fn.named_parameters():
                grad = param.grad
                print(f"{name}.grad = {grad}, grad.size = {grad.size()}")
                influence = grad.pow(2).sum()
                total += influence
            print(total, total.sqrt())

        self.optimizer.step()

        return loss

    def _run_step_ts(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)

        def compute_grad_norm(model, loss):
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            total = 0
            for g in grads:
                total += (g ** 2).sum()
            return total.sqrt()

        grad_norms = []
        t_list = torch.arange(0, 1000, 10, device=self.device)
        for t in range(100):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.diffusion.compute_loss(x, out_dict, stage=self.stage)
            # loss.backward()
            grad_norm = compute_grad_norm(self.diffusion._denoise_fn, loss)
            grad_norms.append(grad_norm.item())
        print(grad_norms)
        df = pd.DataFrame({'grad_norm': grad_norms})
        df.to_csv('grad_norms.csv')
        return grad_norm

    def run_loop(self):
        step = 0
        curr_loss_gauss = 0.0
        curr_count = 0

        print(f'Start training stage: {self.stage}...')
        with trange(self.steps, desc="Training", unit="step", dynamic_ncols=True) as pbar:
            while step < self.steps:
                for x, out_dict in self.train_iter:
                    out_dict = {'y': out_dict}
                    batch_loss_gauss = self._run_step(x, out_dict, step)

                    curr_count += len(x)
                    curr_loss_gauss += batch_loss_gauss.item() * len(x)

                    self._anneal_lr(step)

                    if (step + 1) % self.print_every == 0:
                        gloss = np.around(curr_loss_gauss / curr_count, 3)
                        epsilon = (self.privacy_engine.accountant.get_epsilon(self.dp_params['common']['delta'])
                                   if self.privacy_engine else 0)
                        pbar.set_postfix({
                            'Loss': round(gloss, 3),
                            'Epsilon': round(epsilon, 3)
                        })
                        if (step + 1) % self.log_every == 0:
                            self.loss_history.loc[len(self.loss_history)] = [step + 1, gloss, epsilon]
                            curr_count = 0
                            curr_loss_gauss = 0.0

                    update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                    step += 1
                    pbar.update(1)
                    if step >= self.steps:
                        break
        print(f'Training stage: {self.stage} done!')


def train(
        exp_dir='expdir',
        # data_path='data/adult',
        steps=1000,
        lr=0.002,
        weight_decay=1e-4,
        batch_size=1024,
        model_type='mlp',
        model_params=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        dp_params=None,
        device=torch.device('cuda:0'),
        seed=0,
        change_val=False
):
    delu.random.seed(seed)

    exp_dir = os.path.normpath(exp_dir)
    trans_data_path = os.path.join(exp_dir, T_dict['cat_encoding'])

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        trans_data_path,  # trans_data_path
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    model_params['d_in'] = num_numerical_features
    print(num_numerical_features)

    print(model_params)

    loss_history = pd.DataFrame(columns=['step', 'loss', 'epsilon'])

    assert dp_params['common']['enabled_stages'] == ['0'], 'dp_params set false'

    model = get_model(
        model_type,
        model_params
    )

    # model.load_state_dict(
    #     torch.load('expdir/buddy/dm_mlp_best/model.pt', map_location="cpu")
    # )
    model.to(device)

    diffusion = GaussianDiffusion(
        input_dim=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
        dp_params=dp_params
    )
    diffusion.to(device)
    diffusion.train()

    ema_model = deepcopy(diffusion._denoise_fn)

    for param in ema_model.parameters():
        param.detach_()

    train_loader = lib.prepare_opacus_dataloader(dataset, split='train', batch_size=batch_size)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)

    privacy_engine = None
    stage = '0'
    trainer = Trainer(
        diffusion,
        ema_model,
        train_loader,
        lr,
        optimizer,
        steps,
        dp_params,
        privacy_engine,
        stage,
        loss_history,
        device
    )
    trainer.run_loop()

    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(exp_dir, 'model.pt'))
    torch.save(ema_model.state_dict(), os.path.join(exp_dir, 'model_ema.pt'))

    loss_history.to_csv(os.path.join(exp_dir, 'loss.csv'), index=False)

