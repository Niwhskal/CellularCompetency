import math
import numpy as np
import matplotlib.pyplot as plt
from core_functions import HelperFuncs
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class TwoFit():
  def __init__(self, conf, cfs):

    self.config = conf
    self.cfs = cfs
    self.init_organisms = np.random.randint(low=self.config['LOW'], high=self.config['HIGH'], size=(self.config['N_organisms'], self.config['SIZE']))

  def run_ga(self):

    generation = 0

    self.raw_fitness = []
    self.app_fitness = []

    self.tsne_raw_fitness=[]
    self.tsne_ap_fitness = []

#    self.competent_mod, _ = self.cfs.reorganize(self.init_organisms)
    self.competent_mod = self.cfs.advanced_reorganize(self.init_organisms, flag=self.config['MODE'])


    r_fit = [self.cfs.fitness(i) for i in self.init_organisms]
    ap_fit = [self.cfs.fitness(j) for j in self.competent_mod]

    max_raw = max(r_fit)

    self.raw_fitness.append(max_raw)
    self.app_fitness.append(max(ap_fit))

    self.to_tsne_organisms = self.init_organisms.copy()

    self.tsne_raw_fitness += r_fit 
    self.tsne_ap_fitness += ap_fit 

    while round(max_raw, 2) != 1.00 and generation < self.config['MAX_ITER']: 
      
      fittest_organisms, most_fit_organism = self.cfs.selection(self.init_organisms, ap_fit)
      new_population = self.cfs.crossover_mutation(fittest_organisms)
      mutated_population = self.cfs.mutation_flip(new_population)

      self.init_organisms = mutated_population.copy()

      self.to_tsne_organisms = np.append(self.to_tsne_organisms, self.init_organisms, axis = 0)


      #self.competent_mod, _ = self.cfs.reorganize(self.init_organisms)
      self.competent_mod = self.cfs.advanced_reorganize(self.init_organisms, flag=self.config['MODE'])

      r_fit = [self.cfs.fitness(i) for i in self.init_organisms]
      ap_fit = [self.cfs.fitness(j) for j in self.competent_mod]

      self.tsne_raw_fitness += r_fit 
      self.tsne_ap_fitness += ap_fit

      max_raw = max(r_fit)
      max_ap = max(ap_fit)

      self.raw_fitness.append(max_raw)
      self.app_fitness.append(max_ap)

      generation +=1

      print("GEN: {} | RAW_FITNESS: {:.2f} | APP_FITNESS: {:.2f} | Tsne_size: {}".format(generation, max_raw, max_ap, (self.to_tsne_organisms.shape, len(self.tsne_raw_fitness)))) 


  def plot_fitness(self, tit, x_lab='Generations', y_lab='Max Fitness'): 
    plt.rcParams["figure.figsize"] = (20,10)
    plt.plot(list(range(len(self.raw_fitness))), self.raw_fitness, label = 'Raw_fitness')
    plt.plot(list(range(len(self.app_fitness))), self.app_fitness, label = 'App_fitness')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(tit)
    plt.legend()
    plt.savefig('Fitness')

  def tsne_plot(self, nos):
    plt.rcParams["figure.figsize"] = (20,10)
    print('TSNE in progress for data matrix of size: {}'.format(self.to_tsne_organisms[:nos, :].shape))
    X_embedded = TSNE(n_components=2).fit_transform(self.to_tsne_organisms[:nos, :])

    print('TSNE completed\n')
    print('Embedding Dimension: {}'.format(X_embedded.shape))
    
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(X_embedded[:,0], X_embedded[:,1], self.tsne_raw_fitness[:nos], label='raw curve')
    ax.scatter(X_embedded[:,0], X_embedded[:,1], self.tsne_ap_fitness[:nos], label='app curve')
    ax.legend()
    plt.savefig('tsne')


if __name__ == "__main__":

  conf = {
  'LOW' :1,
  'HIGH':51,
  'N_organisms' : 100,
  'SIZE' : 50,
  'Mutation_Probability' : 0.6,
  'LIMIT': 50,
  'MAX_ITER' : 1000,
  'MAX_FIELD' : 50,
  'MODE': 'normal'
  }

  cfs = HelperFuncs(conf)
  run1 = TwoFit(conf, cfs)

  run1.run_ga()
  run1.plot_fitness(tit='Only N, field: restricted')
  #run1.tsne_plot(6000)