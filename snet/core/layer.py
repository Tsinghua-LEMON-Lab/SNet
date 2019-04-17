# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: layer
# @Date: 2019-04-17-14-07
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


import torch


class Layer(object):
    """
    Abstract class <Layer>.
    """

    def __init__(self, size, net):
        """
        :param size:    <int>
        :param net:     <Network>   reference to the network
        """

        self.size = size
        self.network = net

        # voltage spec.
        self.o_rest = 0.
        self.o_peak = 1.

        self.i = torch.zeros(self.size, dtype=torch.float)                  # input port
        self.o = torch.ones(self.size, dtype=torch.float) * self.o_rest     # output port

        # spike related
        self.firing_mask = torch.zeros(self.size, dtype=torch.uint8)
        self.spike_counts = torch.zeros(self.size, dtype=torch.float)

        # adaptive thresholds
        self.adaptive = False

        # lateral inhibition
        self.inhibition = False

    def process(self):
        """
        Processes in one timestep.
        """
        raise NotImplementedError("Layer.process() is not implemented.")

    def _clear(self):
        """
        Clears output according to `firing_mask`.
        """
        self.o.masked_fill_(self.firing_mask, self.o_rest)
        self.firing_mask = torch.zeros_like(self.firing_mask)

    def _fire(self):
        """
        Fires spikes.
        """
        self.o.masked_fill_(self.firing_mask, self.o_peak)
        self.spike_counts[self.firing_mask] += 1

    def _reset(self):
        """
        Resets internal state.
        """
        raise NotImplementedError("Layer._reset() is not implemented.")

    def _fire_and_reset(self):
        self._fire()
        self._reset()

    def adapt(self):
        pass

    def clear_spike_counts(self):
        self.spike_counts = torch.zeros_like(self.spike_counts)


class PoissonLayer(Layer):
    """
    Layer of Poisson neurons.
    """

    def __init__(self, size, net):
        super(PoissonLayer, self).__init__(size, net)

        self.image = None
        self.image_norm = None

    def process(self):
        self._clear()

        ref = self.image / self.image_norm

        x = torch.rand_like(self.image, dtype=torch.float)

        self.firing_mask = (x <= ref)

        self._fire_and_reset()

    def _reset(self):
        pass


class LIFLayer(Layer):
    """
    Layer of LIF neurons.
    """

    def __init__(self, size, net):
        super(LIFLayer, self).__init__(size, net)

        # membrane potential related
        self.v_rest = 0.
        self.v_th_rest = 0.

        self.v = torch.ones(self.size, dtype=torch.float) * self.v_rest
        self.v_th = torch.ones_like(self.v) * self.v_th_rest

        # LIF params
        self.refractory = 0
        self.tau = None
        self.res = None

        # refractory related
        # firing events are allowed only when `rest_time` exceeds `refractory` period
        self._resting_time = torch.ones(self.size, dtype=torch.int) * self.refractory

        # adaptive thresholds
        self.adaptive = True

        # lateral inhibition
        self.inhibition = True

    @property
    def input_size(self):
        return self.network.INPUT.size

    def process(self):
        """
        Leaks, integrates, and fires.
        """
        self._clear()

        # leak
        self.v -= (self.v - self.v_rest) / self.tau

        # during refractory?
        self._resting_time += 1
        allowed = (self._resting_time >= self.refractory)

        # integrate
        self.v[allowed] += self.res * self.i[allowed] / self.input_size

        self.firing_mask = torch.zeros_like(self.firing_mask)

        # lateral inhibition (winner-take-all)
        if self.inhibition:
            overshoot = self.v - self.v_th
            overshoot_mask = overshoot > 0

            _, indices = torch.sort(overshoot, descending=True)

            overshoot_mask = overshoot_mask.index_select(0, indices)

            indices = indices.masked_select(overshoot_mask)

            if len(indices) > 0:
                index = indices[0]

                self.firing_mask[index] = 1

        self._fire_and_reset()

    def _reset(self):
        self.v[self.firing_mask] = self.v_rest
        self._resting_time[self.firing_mask] = 0

    def adapt(self):
        """
        Adapts thresholds.
        """
        if self.adaptive:
            pass