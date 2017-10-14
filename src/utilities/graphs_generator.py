import matplotlib.pyplot as mpl


class GraphsGenerator:
    def __init__(self):
        pass

    @staticmethod
    def plot_perceptron_compare(perceptron_hits, multi_perceptron_hits, epochs_number):
        '''
        Plot graph comparing Perceptron e Multilayer Perceptron in relation of epochs number
        :param perceptron_hits: list containing hits percentage of perceptron
        :param multi_perceptron_hits: list containing hits percentage of multilayer perceptron
        :param epochs_number:
        :return: None
        '''

        mpl.plot(epochs_number, perceptron_hits, epochs_number, multi_perceptron_hits)

        mpl.axis([0, 40, 70, 100])

        mpl.ylabel('% Acertos')
        mpl.xlabel('Epochs')
        mpl.savefig('acertosXepochs.png')

    @staticmethod
    def plot_neurons_amount_compare(hits, neurons_amount):
        '''
        Plot graph comparing neurons number with hits percentage
        :param hits: a
        :param neurons_amount:
        :return: None
        '''

        mpl.plot(hits, neurons_amount)
        mpl.axis([0, 10, 70, 100])

        mpl.ylabel('% Acertos')
        mpl.xlabel('Qtd. Neuronios camdada 2')
        mpl.savefig('acertosXepochs.png')


if __name__ == "__main__":
    gg = GraphsGenerator()