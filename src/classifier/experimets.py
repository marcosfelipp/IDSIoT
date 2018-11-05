import os
import sys

from data_manipulation import DataManipulation
from neural_network import NeuralNetwork

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.utilities.graphs_generator import GraphsGenerator
from src.utilities.config_parser import Configuration


class Experiments:
    '''
    Divide data in batches to training ML
    '''

    def __init__(self):
        self.div_number = 1

        self.data_manipulation = DataManipulation()
        self.input, self.output_expected = self.data_manipulation.read_file('features_train')
        self.input_test, self.output_test_expected = self.data_manipulation.read_file('features_test')

        self.run_experiment()

    def run_experiment(self):

        input_training  = self.input
        output_training = self.output_expected

        input_test      = self.input_test
        output_test     = self.output_test_expected

        rn = NeuralNetwork(input_training, output_training, input_test, output_test)

        # run first 4 experiments varing epochs
        for experiment_number in range(1, 5):
            result_vector = []

            experiment_name = 'EXPERIMENT' + str(experiment_number)

            title_graph = Configuration.get(experiment_name, 'name_graph')
            description = Configuration.get(experiment_name, 'description')
            n_layers = Configuration.get_int(experiment_name, 'n_layers')
            n_epochs = Configuration.get_list(experiment_name, 'n_epochs')
            n_neurons = Configuration.get_int(experiment_name, 'n_neurons')



            for epoch in n_epochs:
                average_calc = 0

                for average in range(3):
                    average_calc += rn.neural_network_run(n_layers, epoch, n_neurons)

                average_calc = average_calc / 3
                result_vector.append(average_calc * 100)

            GraphsGenerator.plot_epoch_compare(title_graph, result_vector, n_epochs, experiment_number)

        # run experiments 5,6 and 7 varing neurons number
        for experiment_number in range(5, 8):
            result_vector = []
            experiment_name = 'EXPERIMENT' + str(experiment_number)

            title_graph = Configuration.get(experiment_name, 'name_graph')
            description = Configuration.get(experiment_name, 'description')
            n_layers = Configuration.get_int(experiment_name, 'n_layers')
            n_epochs = Configuration.get_int(experiment_name, 'n_epochs')
            n_neurons = Configuration.get_list(experiment_name, 'n_neurons')


            for neuron in n_neurons:
                average_calc = 0

                for average in range(3):
                    average_calc += rn.neural_network_run(n_layers, n_epochs, neuron)

                average_calc = average_calc / 3
                result_vector.append(average_calc * 100)

            GraphsGenerator.plot_epoch_compare(title_graph, result_vector, n_neurons, experiment_number)


if __name__ == '__main__':
    ex = Experiments()
