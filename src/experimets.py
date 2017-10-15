import os
import sys

from data_manipulation import DataManipulation
from neural_network import NeuralNetwork

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.utilities.graphs_generator import GraphsGenerator


class Experiments:
    '''
    Divide data in batches to training ML
    '''

    def __init__(self):
        self.div_number = 1

        self.data_manipulation = DataManipulation()
        self.input, self.output_expected = self.data_manipulation.read_file()

        self.run_experiment()

    def run_experiment(self):
        result_vector_perceptron = []
        result_vector_multi_perceptron = []

        for i in range(self.div_number):
            input_training  = []
            output_training = []
            input_test      = []
            output_test     = []

            for j in range(len(self.input)):
                if j % 4 == i:
                    input_test.append(self.input[j])
                    output_test.append(self.output_expected[j])
                else:
                    input_training.append(self.input[j])
                    output_training.append(self.output_expected[j])

            rn = NeuralNetwork(input_training, output_training, input_test, output_test)
            result_vector_multi_perceptron.append(rn.neural_network_run('multi_percptron', 4) * 100)
            result_vector_multi_perceptron.append(rn.neural_network_run('multi_percptron', 8) * 100)
            result_vector_multi_perceptron.append(rn.neural_network_run('multi_percptron', 16) * 100)
            result_vector_multi_perceptron.append(rn.neural_network_run('multi_percptron', 32) * 100)

            result_vector_perceptron.append(rn.neural_network_run('perceptron', 4) * 100)
            result_vector_perceptron.append(rn.neural_network_run('perceptron', 8) * 100)
            result_vector_perceptron.append(rn.neural_network_run('perceptron', 16) * 100)
            result_vector_perceptron.append(rn.neural_network_run('perceptron', 32) * 100)


        GraphsGenerator.plot_perceptron_compare('perceptron', result_vector_perceptron,
                                                result_vector_multi_perceptron, [5, 10, 20, 40])


if __name__ == '__main__':
    ex = Experiments()