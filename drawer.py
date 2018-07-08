# https://github.com/miloharper/visualise-neural-network

from matplotlib import pyplot
from math import cos, sin, atan


class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius, color='black'):
        circle = pyplot.Circle(
            (self.x, self.y), radius=neuron_radius, fill=False, color=color)
        pyplot.gca().add_patch(circle)


class Layer:
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, number_of_layers):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.number_of_layers = number_of_layers
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(
            number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        # if self.previous_layer:
        #     return self.previous_layer.y + self.vertical_distance_between_layers
        # else:
        #     return 0

        if self.previous_layer:
            return self.previous_layer.y - self.vertical_distance_between_layers
        else:
            return self.number_of_layers * self.vertical_distance_between_layers

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x + x_adjustment, neuron2.x - x_adjustment),
                             (neuron1.y + y_adjustment, neuron2.y - y_adjustment))

        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for i, neuron in enumerate(self.neurons):
            neuron.draw(self.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(
                        neuron, previous_layer_neuron)
        x_text = self.number_of_neurons_in_widest_layer * \
            self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize=12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize=12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer ' +
                        str(layerType), fontsize=12)


class NetworkDrawer:
    def __init__(self, number_of_neurons_in_widest_layer, number_of_layers):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.number_of_layers = number_of_layers
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons,
                      self.number_of_neurons_in_widest_layer,
                      self.number_of_layers)
        self.layers.append(layer)

    def update_link(layer1, neuron1, layer2, neuron2):
        pass

    def draw(self):
        pyplot.figure()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw(i)
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Spike Neural Network', fontsize=15)
        pyplot.show()


class DrawNN:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NetworkDrawer(widest_layer, len(self.neural_network))
        for l in self.neural_network:
            network.add_layer(l)
        network.draw()


if __name__ == '__main__':
    network = DrawNN([2, 8, 8, 1])
    network.draw()
