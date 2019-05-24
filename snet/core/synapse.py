# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: synapse
# @Date: 2019-04-17-14-58
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


import torch
import torchvision
import os
import matplotlib.pyplot as plt
from math import sqrt, ceil
from torch.distributions.normal import Normal
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


class AbstractSynapse(object):
    """
    Abstract class for synapse.
    """

    def __init__(self, pre_layer, post_layer, net):
        """
        :param pre_layer:   <Layer>
        :param post_layer:  <Layer>
        :param net:         <Network>
        """

        self.pre_layer = pre_layer
        self.post_layer = post_layer

        self.network = net

        # weights related
        self.w_min = self.options.get('w_min')
        self.w_max = self.options.get('w_max')

        def _init_weights(w_init):
            weights = torch.ones(self.pre_layer.size, self.post_layer.size)

            if w_init == 'min':
                return self.w_min * weights
            elif w_init == 'max':
                return self.w_max * weights
            elif w_init == 'random':
                return self.w_min + (self.w_max - self.w_min) * torch.rand_like(weights)
            elif w_init == 'fixed':
                cwd = os.path.dirname(__file__)
                weights_file = os.path.join(cwd, 'default_weights.pt')

                if os.path.exists(weights_file):
                    return torch.load(weights_file)
                else:
                    weights = self.w_min + (self.w_max - self.w_min) * torch.rand_like(weights)
                    torch.save(weights, weights_file)
                    return weights
            else:
                raise ValueError("Wrong configuration for w_init.")

        self.weights = _init_weights(self.options.get('w_init'))

        self.failure_rate = self.options.get('failure_rate', 0.)
        failure_count = int(self.weights.numel() * self.failure_rate)
        indices = torch.randperm(self.weights.numel())[:failure_count]
        self.failure_mask = torch.zeros_like(self.weights).byte().view(-1)
        if len(indices) > 0:
            self.failure_mask[indices] = 1
        self.failure_mask = self.failure_mask.view(*self.weights.shape)

        self.update_counts = torch.zeros_like(self.weights)

        # recording
        self._last_pre_spike_time = -torch.ones(self.pre_layer.size)    # -1 means never fired before
        self._last_post_spike_time = -torch.ones(self.post_layer.size)

        # static mode
        self.static = False

    @property
    def options(self):
        return self.network.options

    def forward(self):
        """
        Fetches output of `pre_layer` and computes results as input of `post_layer`.
        """
        pre = self.pre_layer.o

        self.post_layer.i = torch.matmul(pre, self.weights)

    def _clamp(self):
        self.weights.clamp_(min=self.w_min, max=self.w_max)

    @property
    def time(self):
        return self.network.time

    def update_on_pre_spikes(self):
        """
        Updates weights when new pre-spikes come.
        """
        raise NotImplementedError("update_on_pre_spikes() is not implemented.")

    def update_on_post_spikes(self):
        """
        Updates weights when new post-spikes come.
        :return:
        """
        raise NotImplementedError("update_on_post_spikes() is not implemented.")

    def plot_weight_map(self, out_file=None):
        """
        Plots weight map.
        """
        image_size = self.network.options.get('image_size', (28, 28))
        output_number = self.network.OUTPUT.size
        cols = ceil(sqrt(output_number))

        w = torchvision.utils.make_grid(self.weights.cpu().view(1, *image_size, -1).permute(3, 0, 1, 2), nrow=cols,
                                        normalize=True, range=(self.options.get('w_min'), self.options.get('w_max')))

        plt.figure(1)
        plt.clf()
        plt.imshow(w.permute(1, 2, 0)[:, :, 0], cmap='viridis')
        plt.axis('off')

        if out_file:
            plt.savefig(out_file)

        plt.pause(0.5)

    def plot_update_map(self, out_file=None):
        image_size = self.network.options.get('image_size', (28, 28))
        output_number = self.network.OUTPUT.size
        cols = ceil(sqrt(output_number))

        m = torchvision.utils.make_grid(self.update_counts.cpu().view(1, *image_size, -1).permute(3, 0, 1, 2), nrow=cols,
                                        normalize=True)

        m = m.permute(1, 2, 0)[:, :, 0]
        m = (self.update_counts.max() - self.update_counts.min()) * m + self.update_counts.min()

        plt.figure(2)
        plt.clf()
        plt.matshow(m, fignum=2, cmap='Reds')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=20)
        plt.axis('off')

        if out_file:
            plt.savefig(out_file)

        plt.pause(0.5)

    def ger_device(self, tensor1, tensor2, use_gpu=True):

        if use_gpu and torch.cuda.is_available():
            return torch.ger(tensor1.cuda().float(), tensor2.cuda().float()).byte()
        else:
            return torch.ger(tensor1, tensor2)


class ExponentialSTDPSynapse(AbstractSynapse):
    """
    Learning rule: exponential STDP
    """

    def __init__(self, *args, **kwargs):
        super(ExponentialSTDPSynapse, self).__init__(*args, **kwargs)

        # learning rate options
        self.learning_rate_p = self.options.get('learning_rate_p')
        self.learning_rate_m = self.options.get('learning_rate_m')
        self.tau_p = self.options.get('tau_p')
        self.tau_m = self.options.get('tau_m')

        self.decay = self.options.get('decay', 0.)

    def update_on_pre_spikes(self):
        if self.static:
            return

        # decay first
        dw = self.decay * (self.w_max - self.w_min)
        self.weights[~self.failure_mask] -= dw
        self._clamp()

        # record new pre-spikes
        self._last_pre_spike_time[self.pre_layer.firing_mask] = self.time

        # mask
        post_active = (self._last_post_spike_time >= 0).to(device=dev)
        active = self.ger_device(self.pre_layer.firing_mask, post_active)   # new pre-spikes and fired post-spikes

        # calculate timing difference (where new pre-spikes timing is now)
        dt = (self._last_pre_spike_time.repeat(self.post_layer.size, 1).t() -
              self._last_post_spike_time.repeat(self.pre_layer.size, 1))

        window_mask = (dt <= 2 * self.tau_m).to(device=dev)
        active &= window_mask
        active &= ~self.failure_mask

        self.update_counts[active] += 1

        # weights decrease, because pre-spikes come after post-spikes
        dw = self.learning_rate_m * (self.weights - self.w_min) * torch.exp(-dt / self.tau_m)
        dw.masked_fill_(~active, 0)
        self.weights -= dw
        self._clamp()

    def update_on_post_spikes(self):
        if self.static:
            return

        # record new post-spikes
        self._last_post_spike_time[self.post_layer.firing_mask] = self.time

        # mask
        pre_active = (self._last_pre_spike_time >= 0).to(device=dev)
        active = self.ger_device(pre_active, self.post_layer.firing_mask)

        # calculate timing difference (where new post-spikes timing is now)
        dt = (self._last_post_spike_time.repeat(self.pre_layer.size, 1) -
              self._last_pre_spike_time.repeat(self.post_layer.size, 1).t())

        window_mask = (dt <= 2 * self.tau_p).to(device=dev)
        active &= window_mask
        active &= ~self.failure_mask

        self.update_counts[active] += 1

        # weights decrease, because pre-spikes come after post-spikes
        dw = self.learning_rate_p * (self.w_max - self.weights) * torch.exp(-dt / self.tau_p)
        dw.masked_fill_(~active, 0)
        self.weights += dw
        self._clamp()


class RRAMSynapse(ExponentialSTDPSynapse):
    def __init__(self, *args, **kwargs):
        super(RRAMSynapse, self).__init__(*args, **kwargs)

        # variation option
        learning_rate_d2d_variation = self.options.get('learning_rate_d2d_variation', 0.)
        if learning_rate_d2d_variation > 0:
            dist = Normal(1.0, learning_rate_d2d_variation)
            self.learning_rate_m = self.learning_rate_m * dist.sample(self.weights.shape)
            self.learning_rate_p = self.learning_rate_p * dist.sample(self.weights.shape)

        learning_rate_c2c_variation = self.options.get('learning_rate_c2c_variation', 0.)
        self.distribution = Normal(1.0, learning_rate_c2c_variation)

        window_d2d_variation = self.options.get('window_d2d_variation', 0.)
        if window_d2d_variation > 0:
            dist = Normal(1.0, window_d2d_variation)
            self.w_max = self.w_max * dist.sample(self.weights.shape)
            self.w_min = self.w_min * dist.sample(self.weights.shape)

        window_c2c_variation = self.options.get('window_c2c_variation', 0.)
        self.window_dist = Normal(1.0, window_c2c_variation)

    def _clamp(self):
        self.weights.clamp_(min=0.)

    def update_on_pre_spikes(self):
        if self.static:
            return

        # record new pre-spikes
        self._last_pre_spike_time[self.pre_layer.firing_mask] = self.time

        # mask
        post_active = (self._last_post_spike_time >= 0).to(device=dev)
        active = self.ger_device(self.pre_layer.firing_mask, post_active)   # new pre-spikes and fired post-spikes

        # calculate timing difference (where new pre-spikes timing is now)
        dt = (self._last_pre_spike_time.repeat(self.post_layer.size, 1).t() -
              self._last_post_spike_time.repeat(self.pre_layer.size, 1))

        window_mask = (dt <= 2 * self.tau_m).to(device=dev)
        active &= window_mask
        active &= ~self.failure_mask

        self.update_counts[active] += 1

        # weights decrease, because pre-spikes come after post-spikes
        dw = (self.learning_rate_m * self.distribution.sample(self.weights.shape) *
              (self.weights - self.w_min * self.window_dist.sample(self.weights.shape)) * torch.exp(-dt / self.tau_m))
        dw.masked_fill_(~active, 0)
        self.weights -= dw
        self._clamp()

    def update_on_post_spikes(self):
        if self.static:
            return

        # record new post-spikes
        self._last_post_spike_time[self.post_layer.firing_mask] = self.time

        # mask
        pre_active = (self._last_pre_spike_time >= 0).to(device=dev)
        active = self.ger_device(pre_active, self.post_layer.firing_mask)

        # calculate timing difference (where new post-spikes timing is now)
        dt = (self._last_post_spike_time.repeat(self.pre_layer.size, 1) -
              self._last_pre_spike_time.repeat(self.post_layer.size, 1).t())

        window_mask = (dt <= 2 * self.tau_p).to(device=dev)
        active &= window_mask
        active &= ~self.failure_mask

        self.update_counts[active] += 1

        # weights decrease, because pre-spikes come after post-spikes
        dw = (self.learning_rate_p * self.distribution.sample(self.weights.shape) *
              (self.w_max * self.window_dist.sample(self.weights.shape) - self.weights) * torch.exp(-dt / self.tau_p))
        dw.masked_fill_(~active, 0)
        self.weights += dw
        self._clamp()
