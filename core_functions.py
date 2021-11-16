import math
import numpy as np


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
    return inv_count/self.ncr


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


  if __name__=='__main__':
    conf = {
      'LOW' :1,
      'HIGH':101,
      'N_organisms' : 100,
      'SIZE' : 100,
      'Mutation_Probability' : 0.6,
      }
    funcs = HelperFuncs(conf)