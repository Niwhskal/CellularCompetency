import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from core_functions import HelperFuncs


class TwoFit():
  def __init__(self, conf, cfs):

    self.config = conf
    self.cfs = cfs
    self.init_organisms = np.random.randint(low=self.config['LOW'], high=self.config['HIGH'], size=(self.config['N_organisms'], self.config['SIZE']))
    # fov_genes = np.random.randint(low = self.config['LOW'], high = self.config['HIGH']//2, size= (self.config['N_organisms'], 1))

    # self.init_population = np.append(self.init_population, fov_genes, axis = 1)


  def run_ga(self):

    generation = 1

    self.raw_fitness = []
    self.app_fitness = []

    self.competent_mod = self.cfs.bubble_sort(self.init_organisms)

    r_fit = [self.cfs.fitness(i)[0] for i in self.init_organisms]
    ap_fit = [self.cfs.fitness(j)[0] for j in self.competent_mod]

    self.raw_fitness.append(max(r_fit))
    self.app_fitness.append(max(ap_fit))

    while generation <N_generens: 
      
      fittest_organisms, most_fit_organism = self.cfs.selection(self.init_organisms, ap_fit)
      new_population = self.cfs.crossover_mutation(fittest_organisms)
      mutated_population = self.cfs.mutation_flip(new_population)

      self.init_organisms = mutated_population.copy()

      self.competent_mod = self.cfs.bubble_sort(self.init_organisms)
 
      r_fit = [self.cfs.fitness(i)[0] for i in self.init_organisms]
      ap_fit = [self.cfs.fitness(j)[0] for j in self.competent_mod]

      max_raw = max(r_fit)
      max_ap = max(ap_fit)

      self.raw_fitness.append(max_raw)
      self.app_fitness.append(max_ap)

      generation +=1
      
      print("GEN: {} | RAW_FITNESS: {:.3f} | APP_FITNESS: {:.3f}".format(generation, max_raw, max_ap)) 

    # self.final_iter_rawFitness = np.array(r_fit).copy()
    # self.final_iter_appFitness = np.array(ap_fit).copy()

    # self.final_iter_rawFitness = self.final_iter_rawFitness[self.final_iter_rawFitness >=0.95] 
    # self.final_iter_appFitness = self.final_iter_appFitness[self.final_iter_appFitness >=0.95] 

    # self.final_iter_rawFitness = np.round(self.final_iter_rawFitness, 2)
    # self.final_iter_appFitness = np.round(self.final_iter_appFitness, 2)

    return self.raw_fitness, self.app_fitness
  
  # def percent_fit_plot(self): 
  #   plt.rcParams["figure.figsize"] = (20,10)
  #   unique_raw, counts_raw = np.unique(self.final_iter_rawFitness, return_counts=True)
  #   unique_comp, counts_comp = np.unique(self.final_iter_appFitness, return_counts=True)

  #   counts_raw = (counts_raw/self.config['N_organisms']) * 100
  #   counts_comp = (counts_comp/self.config['N_organisms']) * 100

  #   plt.bar(x = unique_raw, height = counts_raw, width = 9e-03, alpha = 0.3, label='Hardwired')
  #   plt.bar(x = unique_comp, height = counts_comp, width = 9e-03, alpha = 0.3, label = 'Competent')

  #   #plt.hist(self.final_iter_rawFitness[self.final_iter_rawFitness >= 0.9], density=True, bins= np.arange(0.96, 1.20, 0.01) , label='Hardwired', alpha=0.3)
  #   #plt.hist(self.final_iter_appFitness[self.final_iter_appFitness >= 0.9], density=True, bins= np.arange(0.96, 1.20, 0.01), label='Competent', alpha=0.3)
  #   plt.xlabel('Fitness')
  #   plt.ylabel('%')
  #   plt.legend()
  #   plt.savefig('percentHistogram')

  def plot_fitness(self, tit, x_lab='Generations', y_lab='Max Fitness'): 
    plt.rcParams["figure.figsize"] = (20,10)
    plt.plot(list(range(len(self.raw_fitness))), self.raw_fitness, label = 'Raw_fitness')
    plt.plot(list(range(len(self.app_fitness))), self.app_fitness, label = 'App_fitness')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(tit)
    plt.legend()
    plt.savefig('Baldwin Effect')
 
#The reason why our fitness graph is not monotonic: Code ensures that there is always a fitness improvement within a generation. However, due to mutations and cross-mutations, Every generation has scrambled genes, which when reorganized still lead to better fitness within that generation, but the fitness can be lower when compared to the previous generation. 


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
  # 'MODE': 'normal',
  # 'REORG_TYPE': 3,
  'ExpFitness': True,
  'BubbleLimit': 20,
  # 'viewfield': 1,
  # 'Loops': 10,
  }

  N_runs = 10
  N_generens = 150

  cfs = HelperFuncs(conf)
  run1 = TwoFit(conf, cfs)

  results_hw = np.zeros((N_runs, N_generens))
  results_comp = np.zeros((N_runs, N_generens))

  for r in tqdm(range(N_runs)):
    print('In run {} / {}'.format(r+1, N_runs))
    hwft, cmpft = run1.run_ga()
    results_hw[r] = hwft
    results_comp[r] = cmpft 

  np.save('./weights/hw_baldwin_bubblesort', results_hw)
  np.save('./weights/comp_baldwin_bubblesort', results_comp)
  mr_hw = np.mean(results_hw, axis=0)
  vr_hw = np.std(results_hw, axis =0)/np.sqrt(N_runs)

  mr_cmp = np.mean(results_comp, axis = 0)
  vr_cmp = np.std(results_comp, axis = 0)/np.sqrt(N_runs)

  plt.plot(np.arange(1, N_generens+1), mr_hw, label='Hardwired Fitness')
  plt.fill_between(np.arange(1,N_generens+1), mr_hw-2*vr_hw, mr_hw+2*vr_hw, alpha =0.2) 

  plt.plot(np.arange(1, N_generens+1), mr_cmp, label='Apparent Fitness ({} swaps)'.format(conf['BubbleLimit']))
  plt.fill_between(np.arange(1,N_generens+1), mr_cmp-2*vr_cmp, mr_cmp+2*vr_cmp, alpha =0.2)
  plt.xlabel('Generation')
  plt.ylabel('Fitness')
  plt.legend(loc='lower right')
  plt.savefig('Baldwin_effectBubbleSort')
  