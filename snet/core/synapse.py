# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: synapse
# @Date: 2019-04-17-14-58
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


import torch
import math
import matplotlib.pyplot as plt


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
            else:
                raise ValueError("Wrong configuration for w_init.")

        self.weights = _init_weights(self.options.get('w_init'))

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

    def plot_weight_map(self):
        """
        Plots weight map.
        """
        output_num = self.network.OUTPUT.size
        col_num = math.ceil(math.sqrt(output_num))
        row_num = math.ceil(output_num / col_num)
        image_size = self.network.options.get('image_size', (28, 28))

        plt.figure(1)
        for i in range(output_num):
            plt.subplot(row_num, col_num, i + 1)
            plt.matshow(self.weights[:, i].view(*image_size), fignum=False, vmin=self.w_min, vmax=self.w_max)

        plt.pause(0.5)


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

    def update_on_pre_spikes(self):
        if self.static:
            return

        # record new pre-spikes
        self._last_pre_spike_time[self.pre_layer.firing_mask] = self.time

        # mask
        post_active = self._last_post_spike_time >= 0
        active = torch.ger(self.pre_layer.firing_mask, post_active)  # new pre-spikes and fired post-spikes

        # calculate timing difference (where new pre-spikes timing is now)
        dt = (self._last_pre_spike_time.repeat(self.post_layer.size, 1).t() -
              self._last_post_spike_time.repeat(self.pre_layer.size, 1))

        window_mask = (dt <= 2 * self.tau_m)
        active &= window_mask

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
        pre_active = self._last_pre_spike_time >= 0
        active = torch.ger(pre_active, self.post_layer.firing_mask)     # fired pre-spikes and new post-spikes

        # calculate timing difference (where new post-spikes timing is now)
        dt = (self._last_post_spike_time.repeat(self.pre_layer.size, 1) -
              self._last_pre_spike_time.repeat(self.post_layer.size, 1).t())

        window_mask = (dt <= 2 * self.tau_p)
        active &= window_mask

        # weights decrease, because pre-spikes come after post-spikes
        dw = self.learning_rate_p * (self.w_max - self.weights) * torch.exp(-dt / self.tau_p)
        dw.masked_fill_(~active, 0)
        self.weights += dw
        self._clamp()
