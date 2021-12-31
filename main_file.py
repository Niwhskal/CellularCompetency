import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from core_functions import HelperFuncs
from core_functions import AE
from core_functions import linearDataset
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor 



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

    func_tag = self.config['REORG_TYPE']
    if func_tag == 0:
      self.competent_mod, _ = self.cfs.reorganize(self.init_organisms)
    elif func_tag == 1:
      self.competent_mod = self.cfs.advanced_reorganize(self.init_organisms, flag=self.config['MODE'])
    elif func_tag == 2:
      self.competent_mod = self.cfs.super_advanced_reorganize(self.init_organisms, flag=self.config['MODE'])


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

      if func_tag == 0:
        self.competent_mod, _ = self.cfs.reorganize(self.init_organisms)
      elif func_tag == 1:
        self.competent_mod = self.cfs.advanced_reorganize(self.init_organisms, flag=self.config['MODE'])
      elif func_tag == 2:
        self.competent_mod = self.cfs.super_advanced_reorganize(self.init_organisms, flag=self.config['MODE'])
        
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

    self.final_iter_rawFitness = np.array(r_fit).copy()
    self.final_iter_appFitness = np.array(ap_fit).copy()

    self.final_iter_rawFitness = self.final_iter_rawFitness[self.final_iter_rawFitness >=0.95] 
    self.final_iter_appFitness = self.final_iter_appFitness[self.final_iter_appFitness >=0.95] 

    self.final_iter_rawFitness = np.round(self.final_iter_rawFitness, 2)
    self.final_iter_appFitness = np.round(self.final_iter_appFitness, 2)
  
  def percent_fit_plot(self): 
    plt.rcParams["figure.figsize"] = (20,10)
    unique_raw, counts_raw = np.unique(self.final_iter_rawFitness, return_counts=True)
    unique_comp, counts_comp = np.unique(self.final_iter_appFitness, return_counts=True)

    counts_raw = (counts_raw/self.config['N_organisms']) * 100
    counts_comp = (counts_comp/self.config['N_organisms']) * 100

    plt.bar(x = unique_raw, height = counts_raw, width = 9e-03, alpha = 0.3, label='Hardwired')
    plt.bar(x = unique_comp, height = counts_comp, width = 9e-03, alpha = 0.3, label = 'Competent')

    #plt.hist(self.final_iter_rawFitness[self.final_iter_rawFitness >= 0.9], density=True, bins= np.arange(0.96, 1.20, 0.01) , label='Hardwired', alpha=0.3)
    #plt.hist(self.final_iter_appFitness[self.final_iter_appFitness >= 0.9], density=True, bins= np.arange(0.96, 1.20, 0.01), label='Competent', alpha=0.3)
    plt.xlabel('Fitness')
    plt.ylabel('%')
    plt.legend()
    plt.savefig('percentHistogram')

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

  def autoencoder_encoding(self, nos, batch_size, total_epochs):
    plt.rcParams["figure.figsize"] = (20,10)
    print('Building Autoencoder \n')
    inp_shape = self.to_tsne_organisms[:nos, :].shape
    print('AE in progress for data matrix of size: {} \n'.format(inp_shape))

    ae_net = AE(inp_shape, self.config)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ae_net.parameters(), lr = 0.001, momentum =0.9)

    lin_data = linearDataset(self.to_tsne_organisms[:nos, :], self.config)
    trainloader = DataLoader(lin_data, batch_size=batch_size, shuffle=True)

    print('Training \n')

    for epoch in range(total_epochs):
      epoch_loss = 0.0
      for i, data in enumerate(trainloader, 0):
        x_in = data
        optimizer.zero_grad()
        embedding, x_reconstruct = ae_net(x_in)
        loss = loss_fn(x_reconstruct, x_in)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if i % 10 == 0:
          print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, epoch_loss / 2000))
          epoch_loss = 0.0

    print('Done Training\n')
    print('Plotting..\n')
    test_data = torch.from_numpy(self.to_tsne_organisms)
    test_data = test_data.type(torch.float)
    X_embedding, _ = ae_net(test_data)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(X_embedding[:,0].detach().numpy(), X_embedding[:,1].detach().numpy(), self.tsne_raw_fitness, label='raw curve')
    ax.scatter(X_embedding[:,0].detach().numpy(), X_embedding[:,1].detach().numpy(), self.tsne_ap_fitness, label='app curve')
    ax.legend()
    plt.savefig('AE_plot')





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
  'MODE': 'normal',
  'REORG_TYPE': 2,
  'ExpFitness': True,
  }

  cfs = HelperFuncs(conf)
  run1 = TwoFit(conf, cfs)

  run1.run_ga()
  run1.plot_fitness(tit='N and M, with exponential fitness function')
  #run1.percent_fit_plot()
  #run1.tsne_plot(6000)
  run1.autoencoder_encoding(-1, batch_size = 32, total_epochs = 100)