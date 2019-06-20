import numpy as np
from spiking_neural_network.neurons.neuron import Neuron

class HodgkinHuxley(Neuron):
    """
    Implements the Hodgkin-Huxley neuron model.

    """
    def __init__(self, weights=0):
        super(self.__class__, self).__init__(weights)

    def pulse(self, inputs):
        self.inputs = inputs
        self.value = np.sum(np.multiply(inputs, self.weights))
        return self.fire()
