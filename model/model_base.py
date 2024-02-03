import os
from abc import abstractmethod, ABC
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .coulomb.potential import get_potentials
from .dag.dag import DAG
from .model_utils import calc_gradient_penalty, set_require_grads, slice_volume_along_xyz, TrainClock
from .networks import get_network

import pdb


class SSGmodelBase(ABC):
    """Base class for single shape generative model"""

    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.config = config
        self.train_depth = config.train_depth
        self.device = torch.device(f"cuda:{config.gpu_ids}" if config.gpu_ids >= 0 else "cpu")

        # self.dag = DAG(self.D_loss_func, self.G_loss_func, policy=[self.config.aug_type], policy_weight=[1.0])
        policy = self.config.aug_type.split(',')
        self.dag = DAG(self.D_loss_func, self.G_loss_func, policy=policy, policy_weight=[1.0] * len(policy))
        self.config.n_augs = self.dag.get_num_of_augments_from_policy()

        self.scale = 0
        self.netD = get_network(config, 'D').to(self.device)
        self.netG = get_network(config, 'G').to(self.device)
        self.noiseOpt_init = None  # fixed noise at first scale
        self.noiseAmp_list = []  # gaussian noise std for each scale
        self.real_sizes = []  # real data spatial dimensions

    @abstractmethod
    def _netG_trainable_params(self, lr_g: float, lr_sigma: float, train_depth: int):
        """trainable parameters of generator at current scale.

        Args:
            lr_g (float): generator learning rate
            lr_sigma (float): lr scaling for lower scale when train_depth > 1
            train_depth (int): number of concurrent training scales

        Returns:
            a list of trainable parameters with learning rate
        """
        raise NotImplementedError

    @abstractmethod
    def _draw_fake_in_training(self, mode: str):
        """draw a fake sample (generated by netG) at training"""
        raise NotImplementedError

    @abstractmethod
    def draw_noises_list(self, mode: str, scale=None, resize_factor=(1.0, 1.0, 1.0)):
        """draw a list of noise for netG

        Args:
            mode (str): "rec" or "rand"
            scale (int, optional): at which scale, i.e., list length. Defaults to None.
            resize_factor (tuple, optional): resize factors, compared to training data. Defaults to (1.0, 1.0, 1.0).

        Returns:
            a list of noise tensors
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, mode: str, scale=None, resize_factor=(1.0, 1.0, 1.0), upsample=1, return_each=False):
        """use netG to generate a sample

        Args:
            mode (str): "rec" or "rand"
            scale (int, optional): at which scale, i.e., list length. Defaults to None.
            resize_factor (tuple, optional): resize factors, compared to training data. Defaults to (1.0, 1.0, 1.0).
            upsample (int, optional): upsample factor (to increase resolution). Defaults to 1.
            return_each (bool, optional): return output at each scale. Defaults to False.

        Returns:
            generated sample(s)
        """
        raise NotImplementedError

    def _set_optimizer(self):
        """set optimizer for netG and netD"""

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.lr_d, betas=(self.config.beta1, 0.999))

        parameter_list = self._netG_trainable_params(self.config.lr_g, self.config.lr_sigma, self.config.train_depth)
        self.optimizerG = optim.Adam(parameter_list, lr=self.config.lr_g, betas=(self.config.beta1, 0.999))

    def _set_tbwriter(self):
        """set tensorboard writer"""
        path = os.path.join(self.log_dir, f'train_s{self.scale}.events')
        self.train_tb = SummaryWriter(path)

    def save_ckpt(self, name=None):
        """save checkpoint for future restore"""

        if name is None:
            save_name = f"ckpt_scale{self.scale}_step{self.clock.step}.pth"
        else:
            save_name = f"scale{self.scale}_{name}.pth"
        save_path = os.path.join(self.model_dir, save_name)
        print(f"Save checkpoint at {save_path}.")

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'netD_state_dict': self.netD.cpu().state_dict(),
            'netG_state_dict': self.netG.cpu().state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'noiseOpt_init': self.noiseOpt_init.detach().cpu(),
            'noiseAmp_list': self.noiseAmp_list,
            'realSizes_list': self.real_sizes,
        }, save_path)

        self.netD.to(self.device)
        self.netG.to(self.device)

    def load_ckpt(self, n_scale: int, name="latest"):
        """load saved checkpoint"""

        load_path = os.path.join(self.model_dir, f"scale{n_scale}_{name}.pth")
        if not os.path.exists(load_path):
            raise ValueError(f"Checkpoint {load_path} not exists.")
        print(f"Load checkpoint from {load_path}.")
        checkpoint = torch.load(load_path)

        self.noiseOpt_init = checkpoint['noiseOpt_init'].to(self.device)
        self.noiseAmp_list = checkpoint['noiseAmp_list']
        self.real_sizes = checkpoint['realSizes_list']

        for s in range(n_scale + 1):
            if s == 0:
                level = 0
            else:
                level = int(np.ceil(np.log2(min(*self.real_sizes[s]) / self.config.min_res)))

            # self.netG.init_next_scale()
            self.netG.init_next_scale(level, config=self.config, device=self.device)

        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.netG.to(self.device)
        self.netD.to(self.device)

        self._set_optimizer()
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

        self.scale = n_scale

    def D_loss_func(self, real_data, generated_data, netD, dag=False, dag_idx=0):
        if dag == False:
            # input real and generated data to netD
            disc_y = self.netD(real_data, dag_id=0)
            disc_x = self.netD(generated_data.detach(), dag_idx=0)
        else:
            disc_y = netD(real_data, dag_idx=dag_idx)
            disc_x = netD(generated_data.detach(), dag_idx=dag_idx)

        kernel_dim = 3
        epsilon = 1
        pot_x, pot_y = get_potentials(generated_data, real_data, kernel_dim, epsilon, margin=self.config.margin)

        # # adversarial loss
        # loss_r = -d_real.mean()
        # loss_f = d_generated.mean()
        # loss = loss_f + loss_r

        loss_d_x = ((disc_x.mean(dim=(1, 2, 3, 4)) - pot_x) ** 2).mean()
        loss_d_y = ((disc_y.mean(dim=(1, 2, 3, 4)) - pot_y) ** 2).mean()
        loss = loss_d_x + loss_d_y

        # get gradient penalty
        if self.config.lambda_grad:
            gradient_penalty = self.config.lambda_grad * \
                               calc_gradient_penalty(self.netD, real_data, generated_data, self.device, dag, dag_idx)
            loss += gradient_penalty
        else:
            gradient_penalty = torch.tensor([0])

        # return loss_r, loss_f, gradient_penalty, loss
        return loss_d_y, loss_d_x, gradient_penalty, loss

    def _critic_wgan_iteration(self, real_data: torch.Tensor):
        """critic (discriminator) training iterations

        Args:
            real_data (torch.Tensor): a real 3D shape of shape (1, 1, H, W, D)
        """
        # require grads
        set_require_grads(self.netD, True)

        # get generated data
        generated_data = self._draw_fake_in_training('rand')

        # zero grads
        self.optimizerD.zero_grad()

        loss_r, loss_f, gradient_penalty, loss = (
            self.dag.compute_discriminator_loss(real_data, generated_data.detach(), self.netD))

        # # input real and generated data to netD
        # d_real = self.netD(real_data)
        # d_generated = self.netD(generated_data.detach())
        #
        # # adversarial loss
        # loss_r = -d_real.mean()
        # loss_f = d_generated.mean()
        # loss = loss_f + loss_r
        #
        # # get gradient penalty
        # if self.config.lambda_grad:
        #     gradient_penalty = self.config.lambda_grad * \
        #                        calc_gradient_penalty(self.netD, real_data, generated_data, self.device)
        #     loss += gradient_penalty

        # backward loss
        loss.backward()
        self.optimizerD.step()

        # record loss value
        loss_values = {'D': loss.data.item(), 'D_r': loss_r.data.item(), 'D_f': loss_f.data.item()}
        if self.config.lambda_grad:
            loss_values.update({'D_gp': gradient_penalty.data.item()})
        self._update_loss_dict(loss_values)

    def G_loss_func(self, fake_data, netD, dag=False, dag_idx=0):
        if dag == False:
            d_generated = netD(fake_data, dag_idx=0)
        else:
            d_generated = netD(fake_data, dag_idx=dag_idx)

        loss = 0.
        # adversarial loss
        loss_adv = -d_generated.mean()
        loss += loss_adv

        return loss_adv, loss

    def _generator_iteration(self, real_data: torch.Tensor):
        """generator training iterations

        Args:
            real_data (torch.Tensor): a real 3D shape of shape (1, 1, H, W, D)
        """
        # netD does not receive grads
        set_require_grads(self.netD, False)

        # zero grads
        self.optimizerG.zero_grad()

        # get generated data and input to netD
        fake_data = self._draw_fake_in_training('rand')

        loss_adv, loss = self.dag.compute_generator_loss(fake_data, self.netD)

        # loss = 0.
        #
        # d_generated = self.netD(fake_data)
        #
        # # adversarial loss
        # loss_adv = -d_generated.mean()
        # loss += loss_adv

        # reconstruction loss
        if self.config.alpha:
            generated_data_rec = self._draw_fake_in_training('rec')  # use fixed initialized noise

            loss_recon = F.mse_loss(generated_data_rec, real_data) * self.config.alpha
            loss += loss_recon

        # backward loss
        loss.backward()
        self.optimizerG.step()

        # record loss value
        loss_values = {'G': loss.data.item(), 'G_adv': loss_adv.data.item()}
        if self.config.alpha:
            loss_values.update({'G_rec': loss_recon.data.item()})
        self._update_loss_dict(loss_values)

    def _update_loss_dict(self, loss_dict: dict = None):
        """update loss recording dict during training"""

        if loss_dict is None:
            self.losses = {}
        else:
            for k, v in loss_dict.items():
                if k in self.losses:
                    self.losses[k].append(v)
                else:
                    self.losses[k] = [v]

    def _record_losses(self):
        """record loss values on tensorboard"""

        avg_loss = {k: np.mean(v) for k, v in self.losses.items()}
        self.train_tb.add_scalars("loss", avg_loss, global_step=self.clock.step)
        Wasserstein_D = np.mean(
            [-self.losses['D_r'][i] - self.losses['D_f'][i] for i in range(len(self.losses['D_r']))])
        self.train_tb.add_scalar("wasserstein distance", Wasserstein_D, global_step=self.clock.step)

        return avg_loss

    def _updateStep(self, real_data: torch.Tensor):
        """a training iteration (step), including both G and D updates"""

        self._update_loss_dict(None)
        self.netD.train()
        self.netG.train()

        # (1) Update D network: maximize D(x) + D(G(z))
        for j in range(self.config.Dsteps):
            self._critic_wgan_iteration(real_data)

        # (2) Update G network: maximize D(G(z)) + rec(G(noise_opt), real_data)
        for j in range(self.config.Gsteps):
            self._generator_iteration(real_data)

        avg_loss = self._record_losses()
        if self.config.alpha > 0:
            return {'D': avg_loss['D'], 'G_adv': avg_loss['G_adv'], 'G_rec': avg_loss['G_rec']}

        return {'D': avg_loss['D'], 'G_adv': avg_loss['G_adv']}

    def _train_single_scale(self, real_data: torch.Tensor):
        """train current scale for n iterations"""

        print(f"scale: {self.scale}, real shape dimensions: {real_data.shape}, noise amp: {self.noiseAmp_list[-1]}")
        pbar = tqdm(range(self.config.n_iters), desc=f"Train scale {self.scale}")
        self.prev_opt_feats = None  # buffer of prev scale features for reconstruction

        for _ in pbar:
            losses = self._updateStep(real_data)
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            if self.config.vis_frequency is not None and self.clock.step % self.config.vis_frequency == 0:
                self._visualize_in_training(real_data)

            self.clock.tick()

            if self.clock.step % self.config.save_frequency == 0:
                self.save_ckpt()

        self.prev_opt_feats = None  # FIXME: move this variable to _draw_fake_in_training
        self.save_ckpt('latest')

    def train(self, real_data_list: list):
        """train on a list of multi-scale 3D shapes (coarse-to-fine).

        Args:
            real_data_list (list): a list of torch.Tensor of shape (H_i, W_i, D_i)
        """
        self._set_real_data(real_data_list)  # convert to tensor, each element: (1, 1, H, W, D)
        self.n_scales = len(self.real_list)

        for s in range(self.scale, self.n_scales):
            # init networks and optimizers for each scale
            # self.netD is reused directly

            if s == 0:
                level = 0
            else:
                level = int(np.ceil(np.log2(min(*self.real_sizes[s]) / self.config.min_res)))

            self.netG.init_next_scale(level, config=self.config, device=self.device)
            self.netG.to(self.device)
            assert self.netG.n_scales == s + 1

            self._set_optimizer()
            self._set_tbwriter()
            self.clock.reset()

            # draw fixed noise for reconstruction
            if self.noiseOpt_init is None:
                torch.manual_seed(1234)
                self.noiseOpt_init = torch.randn_like(self.real_list[0])

            # compute std of added gaussian noise
            noise_amp = self._compute_noise_sigma(s)  # Eq. (8) in SSG
            self.noiseAmp_list.append(noise_amp)

            # train for current scale
            self._train_single_scale(self.real_list[s])
            self.scale += 1

    def _set_real_data(self, real_data_list: list):
        """set a list of multi-scale 3D shapes for training"""

        print("real data dimensions: ", [x.shape for x in real_data_list])
        self.real_list = [torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0) for x in
                          real_data_list]
        self.real_sizes = [x.shape[-3:] for x in self.real_list]

    def _compute_noise_sigma(self, scale: int):
        """compute std of added gaussian noise at scale i"""
        s = scale
        if self.config.alpha > 0:
            if s > 0:
                prev_rec = self.generate('rec', s - 1)
                prev_rec = F.interpolate(prev_rec, size=self.real_list[s].shape[2:], mode='trilinear',
                                         align_corners=False)
                noise_amp = self.config.base_noise_amp * torch.sqrt(
                    F.mse_loss(self.real_list[s], prev_rec)).item()  # Eq.(8) in SSG
            else:
                noise_amp = 1.0
        else:
            noise_amp = self.config.base_noise_amp if s > 0 else 1.0

        return noise_amp

    def draw_init_noise(self, mode: str, resize_factor=(1.0, 1.0, 1.0)):
        """draw input noise (for scale 0)"""
        if mode == 'rec':
            return self.noiseOpt_init
        else:
            if resize_factor[0] != 1.0 or resize_factor[1] != 1.0 or resize_factor[2] != 1.0:
                init_size = self.real_sizes[0][-3:]
                # init_size = [1, 1] + [round(init_size[i] * resize_factor[i]) for i in range(3)]
                init_size = [self.config.batch_size, 1] + [round(init_size[i] * resize_factor[i]) for i in range(3)]

                return torch.randn(*init_size, device=self.device)

            # return torch.randn_like(self.noiseOpt_init)
            return torch.randn((self.config.batch_size, *self.noiseOpt_init.shape[1:]), device=self.device)

    def _visualize_in_training(self, real_data: torch.Tensor):
        """write 3D volume slices on tensorboard for quick visualization"""
        if self.clock.step == 0:
            real_data_ = real_data.detach().cpu().numpy()[0, 0]
            self.train_tb.add_image('real', slice_volume_along_xyz(real_data_), self.clock.step, dataformats='HW')

        with torch.no_grad():
            fake1_ = self.generate('rand')
            rec_ = self.generate('rec')

        fake1_ = fake1_.detach().cpu().numpy()[0, 0]
        self.train_tb.add_image('fake1', slice_volume_along_xyz(fake1_), self.clock.step, dataformats='HW')
        rec_ = rec_.detach().cpu().numpy()[0, 0]
        self.train_tb.add_image('rec', slice_volume_along_xyz(rec_), self.clock.step, dataformats='HW')
