import numpy as np
from spiking_neural_network.models.snn import SpikingNeuralNetwork

class SpikingRewardNetwork(SpikingNeuralNetwork):
    """
    Implementation of a reward based Spiking Neural Network.

    """
    def __init__(self, num_input, hidden_layers, num_output, neuron, training, reward):
        super(SpikingRewardNetwork, self.__class__, self).__init__(num_input, hidden_layers, num_output, neuron, training, reward)
        self.reward = reward

    def fit(self, input, individual):
        prev_layer = np.array(input)
        for (i, layer) in enumerate(self.layers):
            new_prev_layer = np.array([])
            for neuron in layer:
                neuron.reward = self.reward(individual)
                new_prev_layer = np.append(new_prev_layer, neuron.pulse(prev_layer))
            prev_layer = new_prev_layer
        self.adjust_weights()
        return prev_layer
