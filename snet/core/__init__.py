# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: __init__
# @Date: 2019-04-17-13-53
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snet.core.layer import PoissonLayer, LIFLayer
from snet.core.synapse import ExponentialSTDPSynapse


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
        self.W = ExponentialSTDPSynapse(self.INPUT, self.OUTPUT, self)

        # clock (count in unit of `dt`)
        self.dt = options.get('dt')
        self.time = 0

        # training / inference
        self.inference = False

        # greedy training option
        self.greedy = options.get('greedy', False)

    def training_mode(self):
        self.inference = False

        self.OUTPUT.adaptive = True
        self.OUTPUT.inhibition = True

        self.W.static = False

    def inference_mode(self):
        self.inference = True

        self.OUTPUT.adaptive = False
        # self.OUTPUT.inhibition = False

        self.W.static = True

    def feed_image(self, image):
        self.INPUT.feed_image(image)
        self.OUTPUT.clear_v()

    def learn_current_image(self):
        """
        Runs the network in timesteps, until one full iteration of current image (reaches the stimulation time of one
        training/testing image.
        :return:
        """

        while not self.INPUT.finished:
            # processes Poisson layer
            self.INPUT.process()

            # modulates weights on pre-spikes
            self.W.update_on_pre_spikes()

            # TODO: feedforward

            # processes LIF layer
            self.OUTPUT.process()

            # modulates weights on post-spikes
            self.W.update_on_post_spikes()

            self.time += 1
