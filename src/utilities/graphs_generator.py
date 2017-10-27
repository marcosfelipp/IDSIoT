import matplotlib.pyplot as mpl
import numpy as np

class GraphsGenerator:
    def __init__(self):
        pass

    @staticmethod
    def plot_epoch_compare(title, acurace, x_plot, experiment_number):
        '''
        Plot graph and save
        :param title: Title to plot in graph
        :param acurace: 
        :param x_plot: 
        :param experiment_number: 
        :return: None
        '''
        y_pos = np.arange(len(x_plot))

        mpl.bar(y_pos, acurace, 0.3, color='#1874CD',edgecolor = '#104E8B', align='center', alpha=0.5)
        mpl.xticks(y_pos, x_plot)

        for x, y in zip(y_pos, acurace):
            mpl.text(x , y , y, ha='center', va='bottom')

        mpl.ylim(0, 100)
        mpl.ylabel('% Acertos')
        mpl.title(title)
        mpl.savefig(('experiment' + str(experiment_number) + '.png'))
        mpl.close()
