from core_functions import HelperFuncs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_ga (config, HW, restricted): 
    #np.random.seed(0)

    init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE']))

    fov_genes = restricted*np.ones((config['N_organisms'], 1), dtype=np.int8) #np.random.randint(low = config['LOW'], high = 2, size= (config['N_organisms'], 1))
    stress_genes = restricted*np.ones((config['N_organisms'], 1), dtype = np.int8) #np.random.randint(low = config['LOW'], high = 2, size= (config['N_organisms'], 1))

    # else:
    #     fov_genes = np.random.randint(low = config['LOW'], high = config['HIGH']//2, size= (config['N_organisms'], 1))
    #     stress_genes = np.random.randint(low = config['LOW'], high = config['HIGH']//2, size= (config['N_organisms'], 1))


    init_population = np.append(init_population, fov_genes, axis = 1)
    init_population = np.append(init_population, stress_genes, axis = 1)

    fcs = HelperFuncs(config)

    HTracker = []
    CTracker = []
    fov_genetracker = []
    stress_genetracker =[]

    for g in tqdm(range(1, config['RUNS']+1)):
        HWFitness = [fcs.fitness(org[:-2])[0] for org in init_population]

        HTracker.append(np.max(HWFitness))

        fov_genetracker.append(init_population[np.argmax(HWFitness), :][-2])
        stress_genetracker.append(init_population[np.argmax(HWFitness), :][-1])

        C_population = fcs.stress_bubble_reorg_fieldgene(init_population)

        CFitness = [fcs.fitness(org[:-2])[0] for org in C_population]

        CTracker.append(np.max(CFitness))


        if HW:
            print('HW Selection RUN, shape: {}'.format(init_population.shape))
            SPopulation, _ = fcs.selection(init_population, HWFitness)

        else:
            print('Competent Selection RUN')
            SPopulation, _ = fcs.selection(init_population, CFitness)

        ROPopulation = fcs.crossover_mutation(SPopulation)

        RFPopulation = fcs.mutation_flip_withoutlastIndex(ROPopulation)
        #RFPopulation = fcs.mutate_all(SPopulation)


        #stPopulation_fov = fcs.mutation_flip_stressgene_restricted_val(RFPopulation, -2, restricted=restricted)
        #stPopulation_stress = fcs.mutation_flip_stressgene_restricted_val(stPopulation_fov, -1, restricted = restricted)

        init_population = RFPopulation.copy() #stPopulation_stress.copy()
        
        print('Run shape: {}'.format(init_population.shape))

    if HW:
        print('Plotting HW')
        #ax1.plot(HTracker, label='Hardwired')
        return HTracker

    else:
        print('Plotting Comp\n')
        #ax2.scatter(range(1, len(genetracker)+1), genetracker, label='Gene; for {} bubble cycles'.format(config['BubbleLimit']))
        #ax1.plot(CTracker, label = 'Competent ({} swaps)'.format(config['BubbleLimit']))
        return HTracker, (fov_genetracker, stress_genetracker)


def plot_all(bubbles, fov_1, fov_10):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize =(25, 9))
    restricted_runs = np.load('./weights/restricted.npy')
    unrestricted_runs = np.load('./weights/unrestricted.npy')
    fov_gene_tracker = np.load('./weights/fov.npy')
    stress_gene_tracker = np.load('./weights/stress.npy')
    un_fov_gene_tracker = np.load('./weights/fov_un.npy')
    un_stress_gene_tracker = np.load('./weights/stress_un.npy')


    for j in range(len(bubbles)):
        m_hw = np.mean(restricted_runs[j, :, :], axis = 0)
        var_hw = np.std(restricted_runs[j, : , :], axis = 0)/np.sqrt(config['Loops'])
        ax1.plot(range(1, config['RUNS']+1), m_hw, label='{} swaps; gene value= {}'.format(bubbles[j], fov_10))
        ax1.fill_between(range(1, config['RUNS']+1), m_hw-2*var_hw, m_hw+2*var_hw, alpha = 0.2)

    for j in range(len(bubbles)):
        m_comp = np.mean(unrestricted_runs[j, :, :], axis = 0)
        var_comp = np.std(unrestricted_runs[j, :, :], axis = 0)/np.sqrt(config['Loops'])
        ax1.plot(range(1, config['RUNS']+1), m_comp, label ='{} swaps; gene value= {}'.format(bubbles[j], fov_1)) 
        ax1.fill_between(range(1, config['RUNS']+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2)

    for j in range(len(bubbles)):
        m_comp = np.mean(fov_gene_tracker[j, :, :], axis = 0)
        var_comp = np.std(fov_gene_tracker[j, :, :], axis = 0)/np.sqrt(config['Loops'])
        ax2.scatter(range(1, config['RUNS']+1), m_comp)# label = 'Fov gene (restricted)', marker='x')

        m_comp = np.mean(stress_gene_tracker[j, :, :], axis = 0)
        var_comp = np.std(stress_gene_tracker[j, :, :], axis = 0)/np.sqrt(config['Loops'])
        ax3.scatter(range(1, config['RUNS']+1), m_comp)# label = 'Stress gene (restricted)', marker='x')

        m_comp = np.mean(un_fov_gene_tracker[j, :, :], axis = 0)
        var_comp = np.std(un_fov_gene_tracker[j, :, :], axis = 0)/np.sqrt(config['Loops'])
        ax2.scatter(range(1, config['RUNS']+1), m_comp)# label = 'Fov Gene (unrestricted)', marker='x')

        m_comp = np.mean(un_stress_gene_tracker[j, :, :], axis = 0)
        var_comp = np.std(un_stress_gene_tracker[j, :, :], axis = 0)/np.sqrt(config['Loops'])
        ax3.scatter(range(1, config['RUNS']+1), m_comp)# label = 'Stress gene (unrestricted)', marker='x')

    ax1.set(xlabel = 'Generation', ylabel='Fitness')
    ax2.set(xlabel = 'Generation', ylabel='Fov Gene Value of Best Individual')
    ax3.set(xlabel = 'Generation', ylabel='Stress Gene value of Best Individual')
    #ax1.set_title('Fitness plot')
    #ax2.set_title('Preferred gene (Max fitness individual)')
    #ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('./plots/dot_gene')




if __name__ == '__main__':
    config = {'LOW': 1,
                    'HIGH' : 51, 
                    'SIZE' : 50,
                    'N_organisms' : 100,
                    'RUNS' : 100,
                    'Mutation_Probability' : 0.6,
                    'Mutation_Probability_stress' : 0.92,
                    'LIMIT': 50,
                    'MAX_FIELD' : 50,
                    'MODE': 'normal',
                    'REORG_TYPE': 3,
                    'ExpFitness': True,
                    'BubbleLimit': 0,
                    #'viewfield': 1,
                    'Loops': 1,
        }

    fov_1 = 1
    fov_10 = 5

    bubbles = np.array([10])#np.arange(0, 250, 50)
    #plot_all(bubbles)
    #hw_runs = np.zeros((config['Loops'], config['RUNS']))
    restricted_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))
    unrestricted_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    un_fov_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))
    un_stress_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    fov_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))
    stress_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))


    for yu in range(config['Loops']):
        print('In Run : {}\n'.format(yu))
        print('**'*10)

        for k, r in tqdm(enumerate(bubbles)): 
            config['BubbleLimit'] = r
            cp_fits, gtrck = run_ga(config, HW=False, restricted=fov_1)
            unrestricted_runs[k, yu, :] = cp_fits

            un_fov_gene_tracker[k, yu, :] = gtrck[0]
            un_stress_gene_tracker[k, yu, :] = gtrck[1]

            cp_fits, gtrck = run_ga(config, HW=False, restricted=fov_10)
            restricted_runs[k, yu, :] = cp_fits

            fov_gene_tracker[k, yu, :] = gtrck[0]
            stress_gene_tracker[k, yu, :] = gtrck[1]

    np.save('./weights/restricted', restricted_runs)
    np.save('./weights/unrestricted', unrestricted_runs)
    np.save('./weights/fov_un', un_fov_gene_tracker)
    np.save('./weights/stress_un', un_stress_gene_tracker)
    np.save('./weights/fov', fov_gene_tracker)
    np.save('./weights/stress', stress_gene_tracker)


    plot_all(bubbles, fov_1, fov_10)
