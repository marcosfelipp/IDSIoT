from neural_network import NeuralNetwork
from data_manipulation import DataManipulation


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

            print(len(input_training), len(output_training), len(input_test), len(output_test))
            rn = NeuralNetwork(input_training, output_training, input_test, output_test)


if __name__ == '__main__':
    ex = Experiments()