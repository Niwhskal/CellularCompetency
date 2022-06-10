from core_functions import HelperFuncs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (7, 9.5)

def run_ga (config, HW):
    #np.random.seed(0)

    init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE']))

    fcs = HelperFuncs(config)

    HTracker = []
    CTracker = []

    for g in tqdm(range(1, config['RUNS']+1)):
        HWFitness = [fcs.fitness(org)[0] for org in init_population]

        HTracker.append(np.max(HWFitness))
 
        if HW:
            SPopulation, _ = fcs.selection(init_population, HWFitness)

        else:
            C_population = fcs.bubble_sort(init_population)

            CFitness = [fcs.fitness(org)[0] for org in C_population]

            CTracker.append(np.max(CFitness))

            SPopulation, _ = fcs.selection(init_population, CFitness)

        ROPopulation = fcs.crossover_mutation(SPopulation)

        RFPopulation = fcs.mutation_flip(ROPopulation)

        init_population = RFPopulation.copy()

    if HW:
        print('Plotting HW')
        return HTracker

    else:
        print('Plotting Comp')
        return HTracker, CTracker

def plot_all(bubbles):

    hw_runs = np.load('./exp1/Hw_bubblesort.npy')
    comp_genome_runs = np.load('./exp1/Comp_genome_bubblesort.npy')
    comp_phenotype_runs = np.load('./exp1/Comp_phenotype_bubblesort.npy')

    m_hw = np.mean(hw_runs, axis = 0)
    var_hw = np.std(hw_runs, axis = 0)/np.sqrt(config['Loops'])
    plt.plot(range(1, config['RUNS']+1), m_hw, label='Hardwired Genotypic-Fitness (no swaps)', linestyle='--', color='black', linewidth=2)
    plt.fill_between(range(1, config['RUNS']+1), m_hw-2*var_hw, m_hw+2*var_hw, alpha = 0.2)

    for b in range(len(bubbles)):
        m_comp = np.mean(comp_genome_runs[b, : , :], axis = 0)
        var_comp = np.std(comp_genome_runs[b, :, :], axis = 0)/np.sqrt(config['Loops'])
        plt.plot(range(1, config['RUNS']+1), m_comp, label = 'Competent Genotypic-Fitness ({} swaps)'.format(bubbles[b]), linestyle='--')
        plt.fill_between(range(1, config['RUNS']+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2)

        m_comp = np.mean(comp_phenotype_runs[b, : , :], axis = 0)
        var_comp = np.std(comp_phenotype_runs[b, :, :], axis = 0)/np.sqrt(config['Loops'])
        plt.plot(range(1, config['RUNS']+1), m_comp, label = 'Competent Phenotypic-Fitness ({} swaps)'.format(bubbles[b]))
        plt.fill_between(range(1, config['RUNS']+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2)

 
    plt.xlabel('Generation')
    plt.ylabel('Fitness of best Individual')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('exp1_compVsHw')


if __name__ == '__main__':
    config = {'LOW': 1,
            'HIGH' : 51, 
            'SIZE' : 50,
            'N_organisms' : 100,
            'RUNS' : 100,
            'Mutation_Probability' : 0.6,
            'LIMIT': 50,
            'MAX_FIELD' : 50,
            'MODE': 'normal',
            #'REORG_TYPE': 3,
            'ExpFitness': True,
            'BubbleLimit': 0,
            'viewfield': 1,
            'Loops': 10,
        }

    bubbles = np.array([0, 20, 100, 400])
    hw_runs = np.zeros((config['Loops'], config['RUNS']))
    comp_genome_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))
    comp_phenotype_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    for yu in tqdm(range(config['Loops'])):
        print('In Run : {}\n'.format(yu))
        print('**'*10)
        fits = run_ga(config, HW=True)
        hw_runs[yu, :] = fits

        for k, r in enumerate(bubbles): 
            config['BubbleLimit'] = r
            cp_gen_fits, cp_cmp_fits = run_ga(config, HW=False)
            comp_genome_runs[k, yu, :], comp_phenotype_runs[k, yu, :] = cp_gen_fits, cp_cmp_fits 

    np.save('./exp1/Hw_bubblesort', hw_runs)
    np.save('./exp1/Comp_genome_bubblesort', comp_genome_runs)
    np.save('./exp1/Comp_phenotype_bubblesort', comp_phenotype_runs) 

    plot_all(bubbles)