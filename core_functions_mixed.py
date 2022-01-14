import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class HelperFuncs():
  def __init__(self, conf):
    self.config = conf
    self.ncr = math.factorial(self.config['SIZE']) / (math.factorial(self.config['SIZE']-2)* 2)

  def fitness(self, o):
    inv_count = 0
    for i in range(len(o)):
      for j in range(i + 1, len(o)):
        if (o[i] < o[j]):
          inv_count += 1

    x_val = inv_count/self.ncr
    if self.config['ExpFitness'] == True:
      return (9**x_val) / 9 
    else:
      return x_val


  def selection(self, ory, fi):
    organisms = ory.copy()
    fitne = fi.copy()
    fit_index = {k: i for k, i in enumerate(fitne)}
    fitness_organisms = {k: v for k, v in sorted(fit_index.items(), key=lambda item: item[1], reverse=True)} 
    orgs_keys = [k for k,v in fitness_organisms.items()] 
    orgs_vals = list(fitness_organisms.values())

    new_orgs_keys = orgs_keys[: round(0.1*self.config['N_organisms'])]
    new_orgs_vals = orgs_vals[: round(0.1*self.config['N_organisms'])]

    new_orgs = [organisms[j] for j in new_orgs_keys]
    max_fitness = new_orgs_vals[0] 
    return new_orgs, max_fitness 

  def combined_selection(self, ory):
    organisms = ory.copy()
    pseudo_organisms = np.array([i[1:] if i[0] == -2 else self.super_advanced_reorganize(i[1:], single_sample=True) for i in organisms])
    fitnessess = [self.fitness(j) for j in pseudo_organisms]

    return(self.selection(organisms, fitnessess))


  def shuffle_mutation(self, organism):
    organisms = organism.copy()
    L = len(organisms)
    while (len(organisms)<= self.config['N_organisms']):
      new_org = organisms[np.random.randint(L)].copy() 
      np.random.shuffle(new_org)
      organisms = np.append(organisms, [new_org], axis =0) 

    return organisms


  def mutation_flip(self, organi):
    organisms = organi.copy()

    for i in range(len(organisms)):
      if np.random.rand(1) > self.config['Mutation_Probability']:
        organisms[i][np.random.randint(low = 1, high=self.config['SIZE'])] = np.random.randint(low = self.config['LOW'], high=self.config['HIGH'])

    return organisms     


  def crossover_mutation(self, organis):
    organisms = organis.copy()
    L = len(organisms)
    while (len(organisms) < self.config['N_organisms']):
      i = np.random.randint(L)
      j = np.random.randint(L)
      if i!=j:
        random_pair = (i, j)
      else:
        continue

      cross_over_point = np.random.randint(low = self.config['LOW']+10, high=self.config['HIGH'])
      new_org_1 = organisms[random_pair[0]]
      new_org_2 = organisms[random_pair[1]]

  #    print(random_pair, cross_over_point, new_org_1, new_org_2)

      temp = new_org_1[:cross_over_point].copy()
      new_org_1[:cross_over_point] = new_org_2[:cross_over_point]
      new_org_2[:cross_over_point] = temp 

  #    print(temp, new_org_1, new_org_2)
  #    print('\n')
      organisms = np.append(organisms, [new_org_1, new_org_2], axis =0) 
  #    print(organisms)
  #    print('\n')
  #    print('\n ====')
    return organisms


  def reorganize(self, orgas):
    orgs = orgas.copy()
    count = 0
    for nos, org in enumerate(orgs):
      n = 0
      while n < len(org)-1:
        if org[n] > org[n+1]:
          temp = org[n]
          org[n] = org[n+1] 
          org[n+1] = temp
          count +=1
        if count >self.config['LIMIT']:
          count =0
          break
        n+=1
    return orgs, count/len(orgs)

  def advanced_reorganize(self, all_orgs, flag='normal'):
    a_orgs = all_orgs.copy()
    all_fits = [self.fitness(i) for i in a_orgs]
    for nos, org in enumerate(a_orgs):
      if flag == 'normal':
        fov = round((3**(-2*all_fits[nos])) * self.config['MAX_FIELD'])
        N = np.random.randint(self.config['LOW'], fov)
      elif flag == 'random':
        N = np.random.randint(self.config['LOW'], self.config['HIGH'])
      for pos, current_cell in enumerate(org):
        if pos-N >=0:
          if org[pos-N] > current_cell:
            temp = org[pos]
            org[pos] = org[pos-N]
            org[pos-N] = temp
        if pos+N < self.config['SIZE']:
          if org[pos+N] < current_cell:
            temp = org[pos]
            org[pos] = org[pos+N]
            org[pos+N] = temp
    
    return a_orgs

  def super_advanced_reorganize(self, all_orgs, flag='normal', single_sample =False):
    a_orgs = all_orgs.copy()
    if single_sample:
      a_orgs = a_orgs.reshape((1,-1))
    all_fits = [self.fitness(i) for i in a_orgs]
    for nos, org in enumerate(a_orgs):
      M = np.random.randint(self.config['LOW'], self.config['HIGH'])
      if flag == 'normal':
        fov = round((3**(-2*all_fits[nos])) * self.config['MAX_FIELD'])
        N = np.random.randint(self.config['LOW'], fov)
      elif flag == 'random':
        N = np.random.randint(self.config['LOW'], self.config['HIGH'])
      for pos in range(self.config['SIZE']): 
        if pos-N >=0: # get left slice
          l_slice = org[pos-N: pos]
          l_indexes = np.where(l_slice > (org[pos]+M))[0]
          if len(l_indexes):
            l_pos = l_indexes[-1] + (pos-N)
            temp = org[pos]
            org[pos] = org[l_pos]
            org[l_pos] = temp

        if pos+N < self.config['SIZE']:
          r_slice = org[pos: pos+N] 

          r_indexes = np.where(r_slice < abs(org[pos]-M))[0]
          if len(r_indexes):
            r_pos = r_indexes[0] + pos
            temp = org[pos]
            org[pos] = org[r_pos]
            org[r_pos] = temp    

    if single_sample:
      a_orgs = a_orgs.reshape((-1))
      return a_orgs 

    return a_orgs

  def percentage_fit(self, r_fits, c_fits):
    r_f = r_fits.copy()
    c_f = c_fits.copy()

    r_f = np.array(r_f)
    c_f = np.array(c_f)

    percent_fit_raw = len(np.where(r_f == 1.0)[0])/len(r_f)
    perfect_fit_comp = len(np.where(c_f == 1.0)[0])/len(c_f)

    return perfect_fit_raw, perfect_fit_comp

class AE(nn.Module):
  def __init__(self, inp_shape, config):
    super(AE, self).__init__()   
    
    self.config = config
    self.l1 = nn.Linear(self.config['SIZE'], 25)
    self.emb = nn.Linear(25, 2)
    self.rev_l1 = nn.Linear(2, 25)
    self.out = nn.Linear(25, self.config['SIZE'])

  def forward(self, x):
    x = F.relu(self.l1(x)) 
    embed = F.relu(self.emb(x))
    rev_x = F.relu(self.rev_l1(embed))
    o = self.out(rev_x)

    return embed, o

class linearDataset(Dataset):
  def __init__(self, x, con):
    self.config = con 
    self.X = x

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    sample = self.X[idx, :] / self.config['SIZE']
    ten = torch.from_numpy(sample)
    return ten.type(torch.float)



if __name__=='__main__':
  conf = {
    'LOW' :1,
    'HIGH':11,
    'N_organisms' : 5,
    'SIZE' : 10,
    'Mutation_Probability' : 0.6,
    'MAX_FIELD' : 15,
    }
  funcs = HelperFuncs(conf)
  test_org = np.random.randint(conf['LOW'], conf['HIGH'], (conf['N_organisms'], conf['SIZE']))
  funcs.super_advanced_reorganize(test_org, flag ='normal')
