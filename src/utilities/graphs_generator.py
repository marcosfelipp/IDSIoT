import matplotlib.pyplot as mpl

class GraphsGenerator:
    def __init__(self):

        acertos_perceptron = [90,85,92,94]
        acertos_multi_perceptron = [99,85,88,99]

        numero_epochs = [5,10,20,40]
        self.perceptron_compare(acertos_perceptron, acertos_multi_perceptron, numero_epochs)

    def perceptron_compare(self, perceptron_hits, multi_perceptron_hits, epochs_number):
        '''
        Compare Perceptron e Multilayer Perceptron in relation of epochs number
        :param perceptron_hits: list containing hits percentage of perceptron
        :param multi_perceptron_hits: list containing hits percentage of multilayer perceptron
        :param epochs_number:
        :return: None
        '''

        mpl.plot(epochs_number, perceptron_hits, epochs_number, multi_perceptron_hits)

        mpl.axis([0, 40, 70, 100])

        mpl.ylabel('% Acertos')
        mpl.xlabel('Epochs')
        mpl.savefig('foo.png')

    def multilayer_perceptron_compare(self, hits, layers_amount):
        mpl.plot(hits, layers_amount)
        mpl.axis([0, 10, 70, 100])

if __name__ =="__main__":
    gg = GraphsGenerator()