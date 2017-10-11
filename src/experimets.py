import os
import sys

from data_manipulation import DataManipulation
from neural_network import NeuralNetwork

sys.path.append( os.path.join( os.path.dirname(__file__), "../" ) )

from src.utilities import Log

class Experiments:
    '''
    Divide data in batches to training ML
    '''

    def __init__(self):
        self.div_number = 4

        self.data_manipulation = DataManipulation()
        self.input, self.output_expected = self.data_manipulation.read_file()

        self.run_experiment()

    def run_experiment(self):
        for i in range(self.div_number):
            Log.info("Starting test: %d" %i)

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


if __name__ == '__main__':
    ex = Experiments()