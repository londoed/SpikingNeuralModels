import numpy as np
from spiking_neural_network.neurons.neuron import Neuron

class LeakyIntegrateAndFire(Neuron):
    """
    Implements the Leaky Integrate and Fire neuron model.

    """
    def __init__(self, weights=0):
        super(self.__class__, self).__init__(weights)
        self.degredation = 0.9
        self.refractory_time = 0

    def calculate_potential(self, inputs):
        self.potential = (self.potential + np.sum(np.multiply(inputs, self.weights))) * self.degredation
        return self.potential

    def pulse(self, inputs):
        self.inputs = inputs
        if self.refractory_time > 0:
            self.refractory_time = self.refractory_time - 1
        else:
            self.value = self.calculate_potential(inputs)
        self.fire()
        if self.fired:
            self.refractory_time = 2
        return self.fired
