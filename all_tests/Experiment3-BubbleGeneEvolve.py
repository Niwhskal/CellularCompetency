import numpy as np
from core_functions import HelperFuncs
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_ga (config, HW): 
    init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE']))

    #fov_genes = np.random.randint(low = config['LOW'], high = config['HIGH']//2, size= (config['N_organisms'], 1))
    #stress_genes = np.random.randint(low = config['LOW'], high = config['HIGH']//2, size= (config['N_organisms'], 1))

    nswapGenes = np.random.randint(low = config['LOW'], high = 15, size= (config['N_organisms'], 1))

    # init_population = np.append(init_population, fov_genes, axis = 1)
    # init_population = np.append(init_population, stress_genes, axis = 1)

    init_population = np.append(init_population, nswapGenes, axis = 1)

    fcs = HelperFuncs(config)

    HTracker = []
    CTracker = []
    # fov_genetracker = []
    # stress_genetracker =[]
    genetracker = []

    for g in tqdm(range(1, config['RUNS']+1)):
        # HWFitness = [fcs.fitness(org[:-2])[0] for org in init_population]

        HWFitness = [fcs.fitness(org[:-1])[0] for org in init_population]

        HTracker.append(np.max(HWFitness))

        # fov_genetracker.append(init_population[np.argmax(HWFitness), :][-2])
        # stress_genetracker.append(init_population[np.argmax(HWFitness), :][-1])

        genetracker.append(init_population[np.argmax(HWFitness), :][-1])

        # C_population = fcs.stress_bubble_reorg_fieldgene(init_population)
 

        if HW:
            print('HW Selection RUN, shape: {}'.format(init_population.shape))
            SPopulation, _ = fcs.selection(init_population, HWFitness)

        else:
            print('Competent Selection RUN')
            C_population = fcs.bubble_sortevolve(init_population)

            CFitness = [fcs.fitness(org[:-1])[0] for org in C_population]

            CTracker.append(np.max(CFitness))

            SPopulation, _ = fcs.selection(init_population, CFitness)

        ROPopulation = fcs.crossover_mutation(SPopulation)

        RFPopulation = fcs.mutation_flip_withoutlastIndex(ROPopulation)


        #stPopulation_fov = fcs.mutation_flip_stressgene(RFPopulation, -2)
        stPopulation_bubble = fcs.mutation_flip_stressgene(RFPopulation, -1)

        init_population = stPopulation_bubble.copy() #stPopulation_stress.copy()
        
        print('Run shape: {}'.format(init_population.shape))

    if HW:
        print('Plotting HW')
        #ax1.plot(HTracker, label='Hardwired')
        return HTracker

    else:
        print('Plotting Comp\n')
        #ax2.scatter(range(1, len(genetracker)+1), genetracker, label='Gene; for {} bubble cycles'.format(config['BubbleLimit']))
        #ax1.plot(CTracker, label = 'Competent ({} swaps)'.format(config['BubbleLimit']))
        # return HTracker, (fov_genetracker, stress_genetracker)
        return HTracker, genetracker, CTracker


def plot_all():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize =(25, 9))
    hw_runs = np.load('./weights/Hw_2genes.npy')
    comp_runs = np.load('./weights/Comp_2genes.npy')
    # fov_gene_tracker = np.load('./weights/fov_genetracker_2genes.npy')
    # stress_gene_tracker = np.load('./weights/stress_genetracker_2genes.npy')
    gene_tracker = np.load('./weights/genetracker_bubble.npy')
    corrs = np.load('./weights/10segCorr.npy')

    m_hw = np.mean(hw_runs, axis = 0)
    var_hw = np.std(hw_runs, axis = 0)/np.sqrt(config['Loops'])
    ax1.plot(range(1, config['RUNS']+1), m_hw, label='Genotypic Fitness (evolving swaps)')
    ax1.fill_between(range(1, config['RUNS']+1), m_hw-2*var_hw, m_hw+2*var_hw, alpha = 0.2)

    m_comp = np.mean(comp_runs, axis = 0)
    var_comp = np.std(comp_runs, axis = 0)/np.sqrt(config['Loops'])
    ax1.plot(range(1, config['RUNS']+1), m_comp, label = 'Phenotypic Fitness (evolving swaps)')
    ax1.fill_between(range(1, config['RUNS']+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2)

    m_comp = np.mean(gene_tracker, axis = 0)
    var_comp = np.std(gene_tracker, axis = 0)/np.sqrt(config['Loops'])
    ax2.plot(range(1, config['RUNS']+1), m_comp, label = 'Genomic (Evolving Swaps)', marker = 'x')

    ax3.plot(corrs, label='Correlation of Genotypic and Phenotypic Fitness', color='green')


        # m_comp = np.mean(stress_gene_tracker[j, :, :], axis = 0)
        # var_comp = np.std(stress_gene_tracker[j, :, :], axis = 0)/np.sqrt(config['Loops'])
        # ax3.scatter(range(1, config['RUNS']+1), m_comp, label = 'Competent ({} swaps)'.format(bubbles[j]), marker='x')

    ax1.set(xlabel = 'Generation', ylabel='Fitness of best Individual')
    ax2.set(xlabel = 'Generation', ylabel='Number of swaps of Best Individual')
    ax3.set(xlabel = 'Generation', ylabel='Correlation of Best Individual (over segments of 10 generations)')
    #ax1.set_title('Fitness plot')
    #ax2.set_title('Preferred gene (Max fitness individual)')
    #ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('Exp3:BubbleSwapEvol')



if __name__ == '__main__':
    config = {'LOW': 1,
                    'HIGH' : 51, 
                    'SIZE' : 50,
                    'N_organisms' : 100,
                    'RUNS' : 100,
                    'Mutation_Probability' : 0.6,
                    'Mutation_Probability_stress' : 0.95,
                    'LIMIT': 50,
                    'MAX_FIELD' : 50,
                    'MODE': 'normal',
                    'REORG_TYPE': 3,
                    'ExpFitness': True,
                    'BubbleLimit': 0,
                    #'viewfield': 1,
                    'Loops': 10,
        }

    # bubbles = np.array([1, 20, 100, 400])#np.arange(0, 250, 50)
    # plot_all(bubbles)

    hw_runs = np.zeros((config['Loops'], config['RUNS']))
    comp_runs = np.zeros((config['Loops'], config['RUNS']))

    #comp_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    # fov_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))
    # stress_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    bubble_genetracker = np.zeros((config['Loops'], config['RUNS']))
    

    for yu in tqdm(range(config['Loops'])):
        print('In Run : {}\n'.format(yu))
        print('**'*10)
        # fits = run_ga(config, HW=True)
        fits, gtrck, compfits = run_ga(config, HW=False)
        hw_runs[yu, :] = fits

        # cp_fits, gtrck = run_ga(config, HW=False)
        comp_runs[yu, :] = compfits
        # fov_gene_tracker[k, yu, :] = gtrck[0]
        # stress_gene_tracker[k, yu, :] = gtrck[1]
        bubble_genetracker[yu, :] = gtrck

        hw_mean = np.mean(hw_runs, axis =0)
        comp_mean = np.mean(comp_runs, axis = 0)

        hw_mean_splits = np.split(hw_mean, config['RUNS']//10)
        comp_mean_splits = np.split(comp_mean, config['RUNS']//10)

        print(len(hw_runs))
        corrs = np.array([np.corrcoef(i, j)[0,1] for i, j in zip(hw_mean_splits, comp_mean_splits)])

    
    np.save('./weights/Hw_2genes', hw_runs)
    np.save('./weights/Comp_2genes', comp_runs)
    # np.save('./weights/fov_genetracker_2genes', fov_gene_tracker)
    # np.save('./weights/stress_genetracker_2genes', stress_gene_tracker)
    np.save('./weights/genetracker_bubble', bubble_genetracker)
    np.save('./weights/10segCorr', corrs)

    plot_all()
