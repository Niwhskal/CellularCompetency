from core_functions import HelperFuncs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_ga (config, HW):
    np.random.seed(0)

    init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE']))

    fcs = HelperFuncs(config)

    HTracker = []
    CTracker = []

    for g in tqdm(range(1, config['RUNS'])):
        HWFitness = [fcs.fitness(org) for org in init_population]

        HTracker.append(np.max(HWFitness))

        C_population = fcs.stress_bubble_reorg_field(init_population)

        CFitness = [fcs.fitness(org) for org in C_population]

        CTracker.append(np.max(CFitness))

        if HW:
            print('HW Selection RUN')
            SPopulation, _ = fcs.selection(init_population, HWFitness)

        else:
            print('Competent Selection RUN')
            SPopulation, _ = fcs.selection(init_population, CFitness)

        ROPopulation = fcs.crossover_mutation(SPopulation)

        RFPopulation = fcs.mutation_flip(ROPopulation)

        init_population = RFPopulation.copy()

    if HW:
        print('Plotting HW')
        plt.plot(HTracker, label='HW')

    else:
        print('Plotting Comp')
        plt.plot(CTracker, label = 'Comp; {} bubble cycles'.format(config['BubbleLimit']))


if __name__ == '__main__':
    config = {'LOW': 1,
                    'HIGH' : 51, 
                    'SIZE' : 50,
                    'N_organisms' : 100,
                    'RUNS' : 70,
                    'Mutation_Probability' : 0.6,
                    'LIMIT': 50,
                    'MAX_FIELD' : 50,
                    'MODE': 'normal',
                    'REORG_TYPE': 3,
                    'ExpFitness': True,
                    'BubbleLimit': -1,
                    'viewfield': 20,
        }

    run_ga(config, HW=True)
    bubbles = np.array([0, 20, 100, 400])#np.arange(0, 250, 50)

    for r in tqdm(bubbles): 
        config['BubbleLimit'] = r
        run_ga(config, HW=False)

    plt.xlabel('Generation')
    plt.ylabel('Max Fitness')
    plt.legend()
    plt.savefig('bubble_cycle')