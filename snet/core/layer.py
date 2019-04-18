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

    @property
    def options(self):
        return self.network.options

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

        # get Pattern&Background Phase options
        self.pb_phases = self.options.get('pb_phases', False)
        self.pattern_firing_rate = self.options.get('pattern_firing_rate', 1.)
        self.background_firing_rate = self.options.get('background_firing_rate', None)
        self.t_background_phase = self.options.get('t_background_phase', 0)

        # input image fields
        self.image = None
        self.image_norm = None

        # phase flag
        self.is_pattern = True

        # duration options
        self.local_t = 0
        self.t_training_image = self.options.get('t_training_image')
        self.t_testing_image = self.options.get('t_testing_image')

    @property
    def finished(self):
        if self.network.inference:
            if self.local_t >= self.t_testing_image:
                return True
        else:
            if self.pb_phases:
                if self.local_t >= self.t_training_image + self.t_background_phase:
                    return True
            else:
                if self.local_t >= self.t_training_image:
                    return True

        return False

    def process(self):
        self._clear()

        ref = self.image / self.image_norm * self.firing_rate

        x = torch.rand_like(self.image, dtype=torch.float)

        self.firing_mask = (x <= ref)

        self._fire_and_reset()

    def _reset(self):
        self.local_t += 1

        if not self.network.inference and self.pb_phases and self.local_t >= self.t_training_image:
            self.background_phase()

    def feed_image(self, image):
        self.image = image.view(-1)
        self.image_norm = self.image.sum()
        self.pattern_phase()

    def pattern_phase(self):
        self.is_pattern = True

        self.local_t = 0

    def background_phase(self):
        self.is_pattern = False

        self.image = 1 - self.image
        self.image_norm = self.image.sum()

    @property
    def firing_rate(self):
        if self.is_pattern:
            return self.pattern_firing_rate

        return self.background_firing_rate


class LIFLayer(Layer):
    """
    Layer of LIF neurons.
    """

    def __init__(self, size, net):
        super(LIFLayer, self).__init__(size, net)

        # membrane potential related
        self.v_rest = 0.
        self.v_th_rest = self.options.get('v_th_rest')

        self.v = torch.ones(self.size, dtype=torch.float) * self.v_rest
        self.v_th = torch.ones_like(self.v) * self.v_th_rest

        # LIF params
        self.refractory = self.options.get('refractory')
        self.tau = self.options.get('tau')
        self.res = self.options.get('res')

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

    def clear_v(self):
        self.v = torch.ones_like(self.v) * self.v_rest
