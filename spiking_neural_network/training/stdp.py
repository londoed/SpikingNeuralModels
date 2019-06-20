import numpy as np
from spiking_neural_network.training.method import TrainingMethod

class STDP(TrainingMethod):
    """
    Implements basic STDP training method.

    """
    def __init__(self):
        self.adjustment = 0.1
        self.time = 0
        self.is_init = False

    def update_weights(self, neuron, adjustment):
        adjustments = np.multiply(neuron, inputs, adjustment)
        signed_adjustments = np.multiply(adjustments, np.sign(neuron.weights))
        adjusted = np.add(neuron.weights, signed_adjustments)
        return adjusted

    def update_trace(self, neuron):
        trace = [0 if x == 0 else neuron.trace[int(i)] + 1 for (i, x) in enumerate(1 - neuron.inputs)]
        return trace

    def initialize(self, neuron):
        neuron.trace = np.zeros(len(neuron.inputs))

    def update(self, layers):
        for layer in layers:
            for neuron in layer:
                if not self.is_init:
                    self.initialize(neuron)
                correlated_adjustment = self.adjustment if neuron.refactory_time is 0 else -self.adjustment
                neuron.weights = self.update_weights(neuron, correlated_adjustment)
                neuron.trace = self.update_trace(neuron)
        if not self.is_init:
            self.is_init = True
        self.time += 1
