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

        # define network structure
        self.INPUT = None
        self.OUTPUT = None

        self.W = None

        # clock
        self.time = 0.

        # training / inference
        self.inference = False


class NetworkFactory(object):
    """
    Builds <Network> instance from options.
    """

    def build(self, options):
        """
        Builds <Network> instance from options.
        :param options:     <dict>
        :return:            <Network>
        """
        net = Network(options)

        # instantiate layers
        net.INPUT = PoissonLayer(options['input_number'], net)
        net.OUTPUT = LIFLayer(options['output_number'], net)

        # instantiate synapse
        net.W = ExponentialSTDPSynapse(net.INPUT, net.OUTPUT, net)

        return net

    def get_default_options(self):
        """
        :return:    <dict>      default options
        """
        # defined in snetapp
        pass
