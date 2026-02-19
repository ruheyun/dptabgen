# coding:UTF-8
# RuHe  2025/10/19 16:59
from copy import deepcopy
import torch
import os
import numpy as np
import delu
import sys
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import trange

from scripts.dp_clip import DPGradClipAdvisor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from ddpm import GaussianDiffusion
from utils.utils_train import get_model, make_dataset, update_ema
import pandas as pd
import lib


class Trainer:
    def __init__(self, diffusion, ema_model, train_iter, lr, optimizer, dp_params, privacy_engine,
                steps, loss_history, device=torch.device('cuda:0')):
        self.diffusion = diffusion
        self.ema_model = ema_model
        self.train_iter = train_iter
        self.dp_params = dp_params
        self.privacy_engine = privacy_engine
        self.init_lr = lr
        self.optimizer = optimizer
        self.device = device
        self.loss_history = loss_history
        self.log_every = 10
        self.print_every = 10
        self.ema_every = 100
        self.total_steps = 0
        self.steps = steps

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict, stage):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.diffusion.compute_loss(x, out_dict, stage=stage)
        loss.backward()
        self.optimizer.step()
        return loss

    def run(self):
        for stage in self.dp_params['common']['enabled_stages']:
            if stage == '0':
                self.run_loop(stage)
            elif stage == '1':
                self.privacy_engine = PrivacyEngine()
                self.diffusion, self.optimizer, self.train_iter = self.privacy_engine.make_private(
                    module=self.diffusion,
                    optimizer=self.optimizer,
                    data_loader=self.train_iter,
                    max_grad_norm=self.dp_params[stage]['max_grad_norm'],
                    noise_multiplier=self.dp_params[stage]['sigma']
                )
                self.diffusion.compute_loss = self.diffusion._module.compute_loss
                self.run_loop_dp(stage)
            else:
                self.optimizer.max_grad_norm = self.dp_params[stage]['max_grad_norm']
                self.optimizer.noise_multiplier = self.dp_params[stage]['sigma']
                self.run_loop_dp(stage)

    def run_loop(self, stage):
        step = 0
        curr_loss_gauss = 0.0
        curr_count = 0
        steps = self.dp_params[stage]['steps']
        print(f'Start training stage: {stage}...')
        with trange(steps, unit="step", dynamic_ncols=True) as pbar:
            while step < steps:
                for x, out_dict in self.train_iter:
                    out_dict = {'y': out_dict}
                    batch_loss_gauss = self._run_step(x, out_dict, stage)

                    curr_count += len(x)
                    curr_loss_gauss += batch_loss_gauss.item() * len(x)

                    self._anneal_lr(step + self.total_steps)

                    if (step + 1) % self.print_every == 0:
                        gloss = np.around(curr_loss_gauss / curr_count, 3)
                        epsilon = 0
                        pbar.set_postfix({
                            'Loss': round(gloss, 3),
                            'Epsilon': round(epsilon, 3)
                        })
                        if (step + 1) % self.log_every == 0:
                            self.loss_history.loc[len(self.loss_history)] = [self.total_steps + step + 1, gloss, epsilon]
                            curr_count = 0
                            curr_loss_gauss = 0.0

                    update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                    step += 1
                    pbar.update(1)
                    if step >= steps:
                        break
        self.total_steps += steps
        print(f'Training stage: {stage} done!')

    def run_loop_dp(self, stage):
        step = 0
        curr_loss_gauss = 0.0
        curr_count = 0
        steps = self.dp_params[stage]['steps']
        print(f'Start training stage: {stage}...')
        with trange(steps, unit="step", dynamic_ncols=True) as pbar:
            while step < steps:
                with BatchMemoryManager(data_loader=self.train_iter, max_physical_batch_size=128, optimizer=self.optimizer) as memory_safe_data_loader:
                    for x, out_dict in memory_safe_data_loader:
                        out_dict = {'y': out_dict}
                        batch_loss_gauss = self._run_step(x, out_dict, stage)

                        curr_count += len(x)
                        curr_loss_gauss += batch_loss_gauss.item() * len(x)

                        if not getattr(self.optimizer, "_is_last_step_skipped", False):
                            self._anneal_lr(step + self.total_steps)

                            if (step + 1) % self.print_every == 0:
                                gloss = np.around(curr_loss_gauss / curr_count, 3)
                                epsilon = (self.privacy_engine.get_epsilon(self.dp_params['common']['delta'])
                                           if self.privacy_engine else 0)
                                pbar.set_postfix({
                                    'Loss': round(gloss, 3),
                                    'Epsilon': round(epsilon, 3)
                                })
                                if (step + 1) % self.log_every == 0:
                                    self.loss_history.loc[len(self.loss_history)] = [self.total_steps + step + 1, gloss, epsilon]
                                    curr_count = 0
                                    curr_loss_gauss = 0.0

                            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                            step += 1
                            pbar.update(1)
                            if step >= steps:
                                break
        self.total_steps += steps
        print(f'Training stage: {stage} done!')


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

    assert dp_params['common']['enabled_stages'] in [['0'], ['1'], ['0', '1'], ['0', '1', '2']], 'dp_params set false'

    model = get_model(
        model_type,
        model_params
    )
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
    trainer = Trainer(
        diffusion,
        ema_model,
        train_loader,
        lr,
        optimizer,
        dp_params,
        privacy_engine,
        steps,
        loss_history,
        device
    )
    trainer.run()

    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(exp_dir, 'model.pt'))
    torch.save(ema_model.state_dict(), os.path.join(exp_dir, 'model_ema.pt'))

    loss_history.to_csv(os.path.join(exp_dir, 'loss.csv'), index=False)

