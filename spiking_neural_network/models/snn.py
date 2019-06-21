import numpy as np

class SpikingNeuralNetwork():
    """
    Class implementing a Spiking Neural Network model.

    """
    def __init__(self, num_input, hidden_layers, num_output, neuron, training):
        self.layers = []
        self.neuron = neuron
        self.training = training
        self.initialize(num_input, hidden_layers, num_output)

    def initialize(self, num_input, hidden_layers, num_output):
        self.init_layer(num_input)
        self.init_hidden(hidden_layers)
        self.init_layer(num_output)

    def init_layer(self, num_neurons):
        layer_neurons = np.array([])
        for _ in range(num_neurons):
            input_weights = len(self.layers[-1]) if len(self.layers) > 0 else num_neurons
            layer_neurons = np.append(layer_neurons, self.neuron_class(input_weights))
        self.layers.append(layer_neurons)

    def init_hidden(self, hidden_layers):
        if type(hidden_layers) is int:
            self.init_layer(hidden_layers)
        else:
            for layer in hidden_layers:
                self.init_layer(layer)

    def adjust_weights(self):
        self.training.update(self.layers)

    def fit(self, input):
        prev_layer = np.array(input)
        for (i, layer) in enumerate(self.layers):
            new_prev_layer = np.array([])
            for neuron in layer:
                new_prev_layer = np.append(new_prev_layer, neuron.pulse(prev_layer))
            prev_layer = new_prev_layer
        self.adjust_weights()
        return prev_layer
