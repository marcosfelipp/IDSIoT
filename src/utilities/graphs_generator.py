import matplotlib.pyplot as mpl


class GraphsGenerator:
    def __init__(self):
        pass

    @staticmethod
    def plot_perceptron_compare(description, perceptron_hits, multi_perceptron_hits, epochs_number):
        '''
        Plot graph comparing Perceptron e Multilayer Perceptron in relation of epochs number
        :param description :
        :param perceptron_hits: list containing hits percentage of perceptron
        :param multi_perceptron_hits: list containing hits percentage of multilayer perceptron
        :param epochs_number:
        :return: None
        '''

        mpl.plot(epochs_number, perceptron_hits, epochs_number, multi_perceptron_hits)

        mpl.axis([4, 32, 40, 100])
        mpl.title(description)
        mpl.ylabel('% Acertos')
        mpl.xlabel('Epochs')
        mpl.savefig('acertosXepochs.png')

    @staticmethod
    def plot_neurons_amount_compare(description, neurons_amount, hits):
        '''
        Plot graph comparing neurons number with hits percentage
        :param description :
        :param hits: a
        :param neurons_amount:
        :return: None
        '''

        mpl.plot(hits, neurons_amount)
        mpl.axis([1, 4, 70, 100])
        mpl.title(description)
        mpl.ylabel('% Acertos')
        mpl.xlabel('Qtd. Neuronios camdada 2')
        mpl.savefig('neuronsXhits.png')
