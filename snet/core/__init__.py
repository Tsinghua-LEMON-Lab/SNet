# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: __init__
# @Date: 2019-04-17-13-53
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


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
    Build <Network> instance from options.
    """

    def build(self, options):
        """
        Build <Network> instance from options.
        :param options:     <dict>
        :return:            <Network>
        """