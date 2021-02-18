#Filename:	test.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 18 Feb 2021 02:32:50  WIB

import ga
import numpy as np

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""
if __name__ == "__main__":
    equation_inputs = [4,-2,3.5,5,-11,-4.7]
    num_weights = 6

    sol_per_pop = 8
    num_parents_mating = 4

    pop_size = (sol_per_pop, num_weights)
    new_population = np.random.uniform(low = -4.0, high = 4.0, size = pop_size)
    print(new_population)

    num_generation = 5
    for generation in range(num_generation):

        print("Generation:\t", generation)
        fitness = ga.cal_pop_fitness(equation_inputs, new_population)
        parents = ga.select_matting_pool(new_population, fitness, num_parents_mating)
        offspring_crossover = ga.crossover(parents, offspring_size = (pop_size[0] - parents.shape[0], num_weights))
        offspring_mutation = ga.mutation(offspring_crossover)

        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        print("Best result:\t", np.max(np.sum(new_population * equation_inputs, axis = 1)))


    fitness = ga.cal_pop_fitness(equation_inputs, new_population)
    best_match_idx = np.where(fitness == np.max(fitness))

    print("Best solution:\t", new_population[best_match_idx, :])
    print("Best solution fitness:\t", fitness[best_match_idx])

