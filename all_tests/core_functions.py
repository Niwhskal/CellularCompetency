#!/usr/env/bin python3

import math
import random
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
      return (9**x_val) / 9, inv_count
    else:
      return x_val, inv_count


  def check_bubble_fit(self, og,  si, cfit):
    origi = og.copy()
    temp = og[si]
    og[si] = og[si + 1]
    og[si+1] = temp

    if ((self.fitness(og) > self.fitness(origi)) and (self.fitness(og) > cfit)):
      return 1, self.fitness(og)
    
    else:
      return 0, cfit

  def check_bubble_fit_mod(self, og,  si, midx, cfit):
    origi = og.copy()
    temp = og[si]
    og[si] = og[midx]
    og[midx] = temp

    if ((self.fitness(og)[0] > self.fitness(origi)[0]) and (self.fitness(og)[0] > cfit)):
      return 1, og, self.fitness(og)[0]
    
    else:
      return 0, [0], cfit

  def check_feasibility(self, person, i):
    per = person.copy()
    temp = per[i]
    per[i] = per[i+1]
    per[i+1] = temp

    if self.fitness(per)[0] > self.fitness(person)[0]:
      return per, True
    else:
      return [0], False

  def bubble_sort(self, orgs):

    individuals = orgs.copy() 
    for n, singleIndv in enumerate(individuals):
      counter = 0
      _, total_swaps = self.fitness(individuals[n])

      defecit_swaps = total_swaps - self.config['BubbleLimit']

      if defecit_swaps >0:
        swapsToUse = self.config['BubbleLimit']
      else:
        swapsToUse = total_swaps

      notswapped = 0
      while counter < swapsToUse and notswapped <=self.config['SIZE']-2: 
        for pos in range(self.config['SIZE']-1):
          if (individuals[n][pos] > individuals[n][pos+1]):
            mod_ind, feasible = self.check_feasibility(individuals[n], pos)
            if feasible:
              individuals[n] = mod_ind
              counter +=1
            else:
              notswapped +=1

          else:
            notswapped +=1

          if (counter >=swapsToUse) or (notswapped >self.config['SIZE']-2):
            break

    return individuals

  
  def bubble_sortevolve(self, orgs):
    individuals = orgs.copy()
    genes = [i[-1] for i in individuals]
    individuals = [i[:-1] for i in individuals]

    for n, singleIndv in enumerate(individuals):
      counter = 0
      _, total_swaps = self.fitness(individuals[n])

      defecit_swaps = total_swaps - genes[n]

      if defecit_swaps >0:
        swapsToUse = genes[n]
      else:
        swapsToUse = total_swaps

      notswapped = 0
      while counter < swapsToUse and notswapped <=self.config['SIZE']-2: 
        for pos in range(self.config['SIZE']-1):
          if (individuals[n][pos] > individuals[n][pos+1]):
            mod_ind, feasible = self.check_feasibility(individuals[n], pos)
            if feasible:
              individuals[n] = mod_ind
              counter +=1
            else:
              notswapped +=1

          else:
            notswapped +=1

          if (counter >=swapsToUse) or (notswapped >self.config['SIZE']-2):
            break

    individuals = np.append(individuals, np.array(genes).reshape(-1,1), axis = 1)

    return individuals

  # def bubble_sortevolve(self, orgs):
  #   individuals = orgs.copy()
  #   genes = [i[-1] for i in individuals] 
  #   for oneIndv in individuals:
      


  # def bubble_reorg(self, orgsi):
  #   orgs = orgsi.copy()

  #   test_yolo = max([self.fitness(i) for i in orgs])
  #   print(test_yolo)

  #   for org in orgs:
  #     count = 0
  #     quit_count = 0
  #     curr_fit = self.fitness(org)
  #     while count < self.config['BubbleLimit'] and quit_count < (self.config['HIGH']*4):
  #       swap_index = np.random.randint(low = self.config['LOW']-1, high = (self.config['HIGH']-2))
  #       if org[swap_index] > org[swap_index + 1]: 
  #         ch_flag, curr_fit = self.check_bubble_fit(org, swap_index, curr_fit)
  #         if ch_flag ==0:
  #           quit_count +=1
  #           continue
  #         else:
  #           count +=1

  #       else:
  #         quit_count += 1

  #   test_yolo = max([self.fitness(i) for i in orgs])
  #   print(test_yolo)
  #   print('-'*10)


  def stress_bubble_reorg(self, indivi):
    individuals = indivi.copy()
    comparison = np.arange(self.config['LOW'], self.config['HIGH'])

    for org in individuals:
      count = 0
      curr_fit = self.fitness(org)
      order_array = np.abs(comparison - org) 
      while count < self.config['BubbleLimit'] and np.max(order_array): 
        indx = np.argmax(order_array)
        LC, RC = False, False
        if indx != self.config['SIZE']-1 and org[indx] > org[indx +1]: 
          ch_flag, curr_fit = self.check_bubble_fit(org, indx, curr_fit)
          if ch_flag:
            RC = True
            order_array[indx] = 0
            count +=1
          else:
            continue

        elif indx != 0 and org[indx] < org[indx -1]:
          ch_flag, curr_fit = self.check_bubble_fit(org, indx-1, curr_fit)
          if ch_flag:
            LC = True
            order_array[indx] = 0
            count += 1
          else:
            continue

        if not LC and not RC:
          order_array[indx] = 0

    return individuals


  def stress_bubble_reorg_field(self, indivi):
    individuals = indivi.copy()
    comparison = np.arange(self.config['LOW'], self.config['HIGH'])

    for org in individuals:
      count = 0
      curr_fit = self.fitness(org)
      order_array = np.abs(comparison - org) 
      mask = np.ones(len(order_array))
      while count < self.config['BubbleLimit'] and np.max(order_array): 
        order_array = np.abs(comparison - org) 
        order_array = order_array * mask
        indx = np.argmax(order_array)
        LC, RC = False, False
        r_indx, l_indx= indx.copy(), indx.copy()

        tick_counter =0
        while tick_counter <= self.config['viewfield']: 
          r_indx += 1
          l_indx -= 1

          #check_right
          if r_indx <= self.config['SIZE']-1:
            if org[indx] > org[r_indx]:
              ch_flag, curr_fit = self.check_bubble_fit_mod(org, indx, r_indx, curr_fit)
              if ch_flag:
                RC= True
                #order_array[indx] =0
                count +=1
                break
          
          if l_indx >0:
            if org[indx] < org[l_indx]:
              ch_flag, curr_fit = self.check_bubble_fit_mod(org, l_indx, indx, curr_fit)
              if ch_flag:
                LC = True
                #order_array[indx] =0
                count +=1
                break

          else:
            tick_counter += 1

        if not LC and not RC:
          mask[indx] = 0

    return individuals 


  def getStress(self, org, stress_distance):
    cellStress = np.zeros(len(org))
    for n, i in enumerate(org):
      l_slice = []
      r_slice = []

      if n-stress_distance >=0:
        l_slice = org[n-stress_distance: n]
      
      else:
        l_slice = org[:n]

      if n+1+stress_distance <=self.config["SIZE"]:
        r_slice = org[n+1 : n+1+stress_distance]

      else:
        r_slice = org[n+1:]

      # while cells_included < stress_distance:
      #   #check left first
      #   if n-l_distance >=0:
      #     l_slice.append(org[n-l_distance])
      #     l_distance +=1
      #     cells_included += 1

      #   elif n+r_distance < self.config['SIZE']:
      #     r_slice.append(org[n+r_distance])
      #     r_distance +=1
      #     cells_included += 1

      subslice = np.append(l_slice, np.append([i], r_slice))
      cellStress[n] = np.sum(np.abs(subslice - np.sort(subslice)))

    return cellStress

  def getBestSwap(self, mainidx, llist, rlist, og, curr_fit):
    org = og.copy()
    # left slice first
    d_left = {}
    for idx in llist:
      org = og.copy()
      if org[idx] > org[mainidx]:
        temp = org[idx]
        org[idx] = org[mainidx]
        org[mainidx] = temp
        d_left[idx] = self.fitness(org)[0]

    d_right = {}

    for idx in rlist:
      org = og.copy()
      if org[idx] < org[mainidx]:
        temp = org[idx]
        org[idx] = org[mainidx]
        org[mainidx] = temp
        d_right[idx] = self.fitness(org)[0]  

    final_dict = d_left | d_right
    final_sort = {k:v for k, v in sorted(final_dict.items(), key = lambda item:item[1], reverse=True)}
    if len(final_sort)==0:
      return og, curr_fit, False

    swap_tuple = list(final_sort.items())[0]
    if swap_tuple[1] > curr_fit:
      sidx = swap_tuple[0]
      temp = og[sidx]
      og[sidx] = og[mainidx]
      og[mainidx] = temp

      return og, swap_tuple[1], True

    else:
      return og, curr_fit, False

    
  def stress_bubble_reorg_fieldgene(self, individ):
    individuals = individ.copy()
    fov_genes = [i[-2] for i in individuals]
    stress_genes = [i[-1] for i in individuals]
    individuals = np.array([ind[:-2] for ind in individuals])

    for n, org in enumerate(individuals):
      count = 0
      curr_fit = self.fitness(org)[0]
      order_array = self.getStress(org, stress_genes[n]) 
      mask = np.ones(len(order_array))
      while count <= self.config['BubbleLimit'] and np.max(order_array): 
        order_array = self.getStress(org, stress_genes[n]) 
        order_array = order_array * mask
        indx = np.argmax(order_array)

        if indx - fov_genes[n] >=0:
          left_check_slice = range(indx-fov_genes[n], indx) #org[indx-fov_genes[n]:indx]
        
        else:
          left_check_slice = range(indx) #org[:indx]

        if indx+1+fov_genes[n] <= self.config['SIZE']:
          right_check_slice = range(indx+1, indx+1+fov_genes[n]) #org[indx+1: indx+1+fov_genes[n]]
        
        else:
          right_check_slice = range(indx+1, len(org)-1) #org[indx+1:]

        individuals[n], curr_fit, chkflag = self.getBestSwap(indx, left_check_slice, right_check_slice, org, curr_fit)

        if chkflag:
          count += 1

        else:
          mask[indx] = 0
    
    individuals = np.append(individuals, np.array(fov_genes).reshape((-1,1)), axis = 1)
    individuals = np.append(individuals, np.array(stress_genes).reshape((-1,1)), axis = 1)

    return individuals



  def selection(self, ory, fi, noevolution=False):
    organisms = ory.copy()
    fitne = fi.copy()
    fit_index = {k: i for k, i in enumerate(fitne)}
    fitness_organisms = {k: v for k, v in sorted(fit_index.items(), key=lambda item: item[1], reverse=True)} 
    orgs_keys = [k for k,v in fitness_organisms.items()] 
    orgs_vals = list(fitness_organisms.values())

    new_orgs_keys = orgs_keys[: round(0.1*self.config['N_organisms'])]
    new_orgs_vals = orgs_vals[: round(0.1*self.config['N_organisms'])]

    if noevolution:
      new_orgs_keys = orgs_keys[:]
      new_orgs_vals = orgs_vals[:]

    new_orgs = [organisms[j] for j in new_orgs_keys]
    max_fitness = new_orgs_vals[0] 
    return new_orgs, max_fitness 

  def combined_selection(self, ory):
    organisms = ory.copy()
    pseudo_organisms = np.array([i[1:] if i[0] == -2 else self.bubble_sort(i[1:].reshape(1,-1)).reshape(-1) for i in organisms])
    fitnessess = [self.fitness(j)[0] for j in pseudo_organisms]

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
        organisms[i][np.random.randint(self.config['SIZE'])] = np.random.randint(low = self.config['LOW'], high=self.config['HIGH'])

    return organisms     

  def mutation_flip_withoutlastIndex(self, organi):
    organisms = organi.copy()

    for i in range(len(organisms)):
      if np.random.rand(1) > self.config['Mutation_Probability']:
        organisms[i][np.random.randint(low = 0, high= self.config['SIZE'])] = np.random.randint(low = self.config['LOW'], high=self.config['HIGH'])

    return organisms     
  
  def mutation_flip_stressgene(self, organi, gene_pos):
    organisms = organi.copy()

    for i in range(len(organisms)):
      if np.random.rand() > self.config['Mutation_Probability_stress']:
        organisms[i][gene_pos] = np.random.randint(low = self.config['LOW'], high=400)
    return organisms     

  def mutation_flip_stressgene_restricted_val(self, organi, gene_pos, restricted):
    organisms = organi.copy()

    for i in range(len(organisms)):
      if np.random.rand(1) > self.config['Mutation_Probability_stress']:
        organisms[i][gene_pos] = restricted #np.random.randint(low = 1, high=1)
        
    return organisms     

  def mutate_singleInd(self, orgi):
    organisms = orgi.copy()

    N_mutations = 1 #np.random.randint(1, 5)

    for _ in range(N_mutations):
      mutation_point = np.random.randint(low = self.config['LOW'], high=self.config['SIZE'])
      new_org_1 = organisms[0].copy()

      new_org_1[mutation_point] = np.random.randint(low = self.config['LOW'], high = self.config['HIGH']) 

    return new_org_1.reshape(1,-1)



  def mutate_all(self, orgs):
    organisms = orgs.copy()
    L = len(organisms)
    while(len(organisms) < self.config['N_organisms']):
      i = np.random.randint(L)

      N_mutations =1 #np.random.randint(1, 35)

      for _ in range(N_mutations):
        mutation_point =np.random.randint(self.config['LOW'], high=self.config['SIZE'])
        new_org_1 = organisms[i].copy()

        new_org_1[mutation_point] = np.random.randint(low = self.config['LOW'], high = self.config['HIGH'])

      organisms = np.append(organisms, [new_org_1], axis =0) 

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

      cross_over_point = np.random.randint(low = self.config['LOW']+10, high=self.config['SIZE'])
      new_org_1 = organisms[random_pair[0]].copy()
      new_org_2 = organisms[random_pair[1]].copy()

  #    print(random_pair, cross_over_point, new_org_1, new_org_2)

      temp = new_org_1[:cross_over_point].copy()
      new_org_1[:cross_over_point] = new_org_2[:cross_over_point]
      new_org_2[:cross_over_point] = temp 

  #    print(temp, new_org_1, new_org_2)
  #    print('\n')
      organisms = np.append(organisms, [new_org_1, new_org_2], axis =0) 
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

  
  def get_slice_fitness(self, indx, torg, curr_pos, field, lflag):
    prev_fitness = self.fitness(torg)
    if len(indx) == 0:
      return 0, [0]
    else:
      if lflag:
        l_pos = indx[-1] + (curr_pos-field)
        temp = torg[curr_pos]
        torg[curr_pos] = torg[l_pos]
        torg[l_pos] = temp
        l_modOrg = torg.copy()
        after_fitness = self.fitness(l_modOrg)
        if after_fitness > prev_fitness:
          return 1, l_modOrg
        else:
          return 0, [0]

      else:
        r_pos = indx[0] + curr_pos
        temp = torg[curr_pos]
        torg[curr_pos] = torg[r_pos]
        torg[r_pos] = temp    
        r_modOrg = torg.copy()
        after_fitness = self.fitness(r_modOrg)
        if after_fitness > prev_fitness:
          return 1, r_modOrg
        else:
          return 0,[0]


  def super_advanced_reorganize_copy(self, all_orgs, flag='normal', single_sample =False):
    a_orgs = all_orgs.copy()
    if single_sample:
      a_orgs = a_orgs.reshape((1,-1))
    all_fits = [self.fitness(i) for i in a_orgs]
    for nos, org in enumerate(a_orgs):
      M = 0 #np.random.randint(self.config['LOW'], self.config['HIGH'])
      if flag == 'normal':
        N = round((3**(-2*all_fits[nos])) * self.config['MAX_FIELD'])

      for pos in range(self.config['SIZE']): 
        if pos-N >=0: # get left slice
          l_slice = org[pos-N: pos]
          l_indexes = np.where(l_slice > (org[pos]+M))[0]
          
          check_one, redone_org = self.get_slice_fitness(l_indexes, org, pos, N, lflag=True)
          if check_one:
            org = redone_org.copy()

        if pos+N < self.config['SIZE']:
          r_slice = org[pos: pos+N] 

          r_indexes = np.where(r_slice < abs(org[pos]-M))[0]
          check_one, redone_org = self.get_slice_fitness(r_indexes, org, pos, N, lflag=False)
          if check_one:
            org = redone_org.copy()

          
    if single_sample:
      a_orgs = a_orgs.reshape((-1))
      return a_orgs 

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
    'MAX_FIELD' : 10,
    'ExpFitness': True,
    'BubbleLimit': 10,
    'viewfield': 3,
    }
  funcs = HelperFuncs(conf)

  test_org =np.array([10,8,7,6,9,2,1,5,3,4]) #np.arange(1,11) #np.random.randint(conf['LOW'], conf['HIGH'], size = (5, 10))
  # #test_org = np.array([[10, 8, 6, 7, 9, 2, 1, 5, 3, 4],
  #                     [6, 7, 1, 10, 3, 9, 4, 2, 8, 5],
  #                     [1,4,3,10,8,7,5,6,9,5],
  #                     [6,1,8, 9,3,5,4,2,8,7]])
  # print([funcs.fitness(i) for i in test_org])
  print(funcs.fitness(test_org))

  # org_C = funcs.bubble_sortevolve(test_org)
# 
  # print([funcs.fitness(i) for i in org_C])

  # def generate(l, h, s):
  #   t = []
  #   for i in range(s):
  #     q = np.arange(l,h)
  #     random.shuffle(q)
  #     t.append(q)
  #   return np.array(t)

  # #print(generate(1, 11, 5))
  # test_org = generate(1, 12 , 1)
  # print(test_org.shape)
  # for i in range(10):
  #   test_reorg = funcs.stress_bubble_reorg_fieldgene(test_org) 
  #   test_org = test_reorg.copy()


  # print([funcs.fitness(i[:-1]) for i in test_org])
  # print('-'*10)
  # print([funcs.fitness(i[:-1]) for i in test_reorg])

  #for i in range(200):
    #q = funcs.super_advanced_reorganize_copy(test_org, flag ='normal')
  #fit = [funcs.fitness(i) for i in test_org]
  #ft = funcs.selection(q, fit)
  #print(test_reorg)
  # print('---')
  # print(q)
  # print('---')
  # print(ft)

