#Filename:	ga.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 18 Feb 2021 02:18:24  WIB

import numpy as np

def cal_pop_fitness(equation_inputs, pop):

    fitness = np.sum(pop * equation_inputs, axis = 1)
    return fitness

def select_matting_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999


    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]

        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring

def mutation(offspring_crossover):

    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.uniform(-1.0, 1., 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value

    return offspring_crossover

