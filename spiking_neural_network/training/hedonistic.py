import numpy as np
from spiking_neural_network.training.stdp import STDP

class HedonisticSTDP(STDP):
    """
    Implements the hedonistic variant of the STDP method.

    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self.tau_c = 1
        self.tau_d = 0.2
        self.d = 0
        self.reward = 0.1

    def initialize(self, neuron):
        super(self.__class__, self).initialize(neuron)
        neuron.c = np.zeros(len(neuron.inputs))

    def calculate_c(self, neuron, stdp_weights):
        delta = 1
        neuron.c = np.add(np.divide(neuron.c, self.tau_c), np.multiply(stdp_weights, delta))

    def calculate_s(self, neuron):
        return np.app(np.multiply(neuron.c, self.d), neuron.weights)

    def calculate_d(self):
        self.d = (self.d / self.tau_d) + self.reward

    def update(self, layers):
        for layer in layers:
            for neuron in layer:
                if not self.is_init:
                    self.init_neuron(neuron)
                correlated_adjustment = self.adjustment if neuron.refactory_time is 0 else -self.adjustment
                stdp_weight_updates = self.update_weights(neuron, correlated_adjustment)
                self.calculate_c(neuron, stdp_weight_updates)
                self.calculate_d()
                neuron.weights = self.calculate_s(neuron)
        if not self.is_init:
            self.is_init = True
        self.time += 1
