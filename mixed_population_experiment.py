import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from core_functions_mixed import HelperFuncs

# Initialize a mixture of hw and comp with tags at index 0
# Let only the comp orgs reorganize their genes
# Calculate fitness of the mixed population
# Selection based on these fitness values, but note that competency needs to be phenotypic only.
# Get new population. Note down the number of hw and comp organisms. 
# Plot a graph of hw_f, comp_f vs #gen

class MixedFit():
    def __init__(self, conf, cfs):
        self.config = conf
        self.cfs = cfs

        self.initialize()

    def initialize(self):
        self.hw_organisms = np.random.randint(low=self.config['LOW'], high=self.config['HIGH'], size=(self.config['N_hw'], self.config['SIZE'])) # hw = -2

        self.comp_organisms = np.random.randint(low=self.config['LOW'], high=self.config['HIGH'], size=(self.config['N_comp'], self.config['SIZE'])) # comp = -1

        self.hw_organisms = np.array([np.insert(i, 0, -2, axis=0) for i in self.hw_organisms]) 
        self.comp_organisms = np.array([np.insert(i, 0, -1, axis=0) for i in self.comp_organisms])
        self.total_orgs = np.append(self.hw_organisms, self.comp_organisms, axis = 0)

    def count_orgs(self):
        t = [i for i in self.total_orgs if i[0] == -1]
        self.comp_count = len(t) / len(self.total_orgs)
        self.hw_count = 1 - self.comp_count 

    def run_ga(self):
        generation = 0
        self.counts = []
        self.count_orgs()
        print('Generation: {} | %HW: {} | %COMP: {}'.format(generation, self.hw_count, self.comp_count))
        self.counts.append((self.hw_count, self.comp_count))

        while self.comp_count < 1.0 and self.hw_count < 1.0:  
        
            fittest_organisms, _ = self.cfs.combined_selection(self.total_orgs)
            new_population = self.cfs.crossover_mutation(fittest_organisms)
            mutated_population = self.cfs.mutation_flip(new_population)

            self.total_orgs = mutated_population.copy()               

            generation +=1
            self.count_orgs()
            self.counts.append((self.hw_count, self.comp_count)) 
            print('Generation: {} | %HW: {} | %COMP: {}'.format(generation, self.hw_count, self.comp_count))


    def plot_counts(self, x_lab='Generations', y_lab='%'): 
        hw_count = [hw for hw,_ in self.counts]
        comp_count = [comp for _, comp in self.counts]
        plt.rcParams["figure.figsize"] = (20,10)
        plt.plot(list(range(len(self.counts))), hw_count, label = 'HW count')
        plt.plot(list(range(len(self.counts))),comp_count, label = 'Comp count')
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.title('Mixed population counts per Generation')
        plt.legend()
        plt.savefig('counts')


if __name__ == "__main__":

    conf = {
    'LOW' :1,
    'HIGH':51,
    'N_hw' : 4500,
    'N_comp': 10,
    'N_organisms': 4510,
    'SIZE' : 50,
    'Mutation_Probability' : 0.6,
    'LIMIT': 50,
    'MAX_ITER' : 1000,
    'MAX_FIELD' : 50,
    'MODE': 'normal',
    'REORG_TYPE': 2,
    'ExpFitness': True,
    }

    cfs = HelperFuncs(conf)
    run1 = MixedFit(conf, cfs)

    run1.run_ga()
    run1.plot_counts()
    #run1.percent_fit_plot()
    #run1.tsne_plot(6000)
    #run1.autoencoder_encoding(-1, batch_size = 32, total_epochs = 100)