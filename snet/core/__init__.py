# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: __init__
# @Date: 2019-04-17-13-53
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snet.core.layer import PoissonLayer, LIFLayer
from snet.core.synapse import ExponentialSTDPSynapse, RRAMSynapse
import os
import torch
import pickle


class Network(object):
    """
    Spiking Neural Network model of fixed structure:
    One input layer, one output layer, fully-connected synapses
    """

    def __init__(self, options):
        """
        Initialize <Network> instance, with options.
        :param options:     <dict>
        """
        # save options
        self.options = options

        # instantiate layers
        self.INPUT = PoissonLayer(options['input_number'], self)
        self.OUTPUT = LIFLayer(options['output_number'], self)

        # instantiate synapse
        # update_variation = self.options.get('update_variation', 0.)
        # if update_variation > 0:
        self.W = RRAMSynapse(self.INPUT, self.OUTPUT, self)
        # else:
        # self.W = ExponentialSTDPSynapse(self.INPUT, self.OUTPUT, self)

        # clock (count in unit of `dt`)
        self.dt = options.get('dt')
        self.time = 0

        # training / inference
        self.inference = False

        # greedy training option
        self.greedy = options.get('greedy', False)

    def training_mode(self):
        # self.time = 0

        self.inference = False

        self.OUTPUT.adaptive = True
        self.OUTPUT.inhibition = True
        self.OUTPUT.tracker_size = self.options.get('tracker_size')

        self.W.static = False

    def inference_mode(self):
        self.time = 0

        self.inference = True

        self.OUTPUT.adaptive = False
        # self.OUTPUT.inhibition = False
        self.OUTPUT.tracker_size = 1

        self.W.static = True

    def feed_image(self, image, clear_v=True):
        self.INPUT.feed_image(image)
        if clear_v:
            self.OUTPUT.clear_v()

    def learn_current_image(self, force_greedy=False):
        """
        Runs the network in timesteps, until one full iteration of current image (reaches the stimulation time of one
        training/testing image.
        """
        self.learn_pattern(force_greedy)

        self.post_learn()

        self.learn_background()

    def learn_in_dt(self, forward=True):
        # processes Poisson layer
        self.INPUT.process()

        # modulates weights on pre-spikes
        self.W.update_on_pre_spikes()

        # feeds forward
        if forward:
            self.W.forward()

            # processes LIF layer
            self.OUTPUT.process()

            # modulates weights on post-spikes
            self.W.update_on_post_spikes()

        self.time += 1

        self.INPUT.next()
        self.OUTPUT.next()

    def learn_pattern(self, force_greedy=False):
        self.INPUT.pattern_phase()

        if not force_greedy:
            while not self.INPUT.finished:
                self.learn_in_dt()

                if not self.inference and self.greedy:
                    if self.OUTPUT.spike_counts.sum() >= 1:
                        return
        else:
            while True:
                self.learn_in_dt()

                if self.greedy:
                    if self.OUTPUT.spike_counts.sum() >= 1:
                        return

    def learn_background(self):
        self.INPUT.feed_image(1 - self.INPUT.image)
        self.INPUT.background_phase()

        while not self.INPUT.finished:
            self.learn_in_dt(forward=False)

    def post_learn(self):
        # track
        self.OUTPUT.track()

        # adapt thresholds
        self.OUTPUT.adapt()

    def save_model(self, path, prefix=''):
        # save weights
        weights_file = os.path.join(path, prefix + 'weights.pt')
        torch.save(self.W.weights, weights_file)

        # save thresholds
        v_th_file = os.path.join(path, prefix + 'v_th.pt')
        torch.save(self.OUTPUT.v_th, v_th_file)

        # pickling <Network> object
        with open(os.path.join(path, prefix + 'network.pickle'), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
