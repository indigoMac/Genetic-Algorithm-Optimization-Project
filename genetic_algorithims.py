# required packages and functions
from random import randint, random, sample
from operator import add
from functools import reduce
from statistics import mean
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

class GA_optimiser:
    """
    Class for the Genetic Algorithm optimizer.
    
    Attributes:
        target (int): The target value the GA aims to reach.
        fitness_func (str): The fitness function type ('bits' or 'value').
        selection_type (str): Type of selection process ('roulette', 'tournament', etc.).
        retain (float): Proportion of top-fitted individuals to retain for the next generation.
        random_select (float): Probability of selecting a lower-fitted individual.
        mutate (float): Mutation rate in the population.
        bits (int): Number of bits to represent the individual.
    """

    def __init__(self, target, fitness_func='bits', selection_type='roulette',
                 retain=0.5, random_select=0.2, mutate=0.01, bits=12): # defualt values

        # shift everything by half the max range, then subtract at the end
        # better than use negative numbers (more bits)
        self.shift = 2**bits / 2
        #self.target = target + self.shift
        self.target = target + self.shift if target is not None else None
        self.bits = bits

        self.retain = retain
        self.random_select = random_select
        self.mutate = mutate

        self.fitness_func = fitness_func
        self.selection_type = selection_type

    # Utility function to convert bit string list to integer
    def bit_2_int(self, val):
        return int(''.join(map(str, val)), 2)

    # Utility function to convert integer to a bit string list
    def int_2_bit(self, val):
        return list(map(int, format(val, f'0{self.bits}b')))

    # Create an induvidual with a initial random value
    def individual(self):
        return self.int_2_bit(randint(0, 2**self.bits - 1))

    # Create the population
    def create_population(self, count):
        self.population = [self.individual() for x in range(count)]
        self.fitness_history = [self.grade()]
        self.pop_history = [self.population_mean()]

    # find mean value of the population
    def population_mean(self):
        # convert population back to int, find mean, and remove shift
        return mean([self.bit_2_int(i) for i in self.population]) - self.shift #because of shift

    # find average fitness of the population
    def grade(self):
        fit_summed = sum(self.fitness(x) for x in self.population)
        return fit_summed / len(self.population)

    # Check fitness by comparing to target value
    def fitness(self, individual):
        if self.fitness_func == 'value':
            if self.target is None:
                return float('inf')  # Return a high value if target is not set
            # Convert back to decimal value and return difference
            return abs(self.target - self.bit_2_int(individual))
        elif self.fitness_func == 'bits':
            if self.target is None:
                return len(individual)  # Return the length of the individual
            target_bit_list = self.int_2_bit(int(self.target))
            diff = 0
            for bit1, bit2 in zip(individual, target_bit_list):
                if bit1 != bit2:
                    diff += 1
            return diff

    def random_selection(self):
        """Selects parents randomly from the population."""
        return [self.population[randint(0, len(self.population)-1)] for _ in range(2)]

    def tournament_selection(self):
        """Selects parents using tournament selection."""
        parents = []
        for _ in range(2):
            sampled = sample(list(zip(self.population, [self.fitness(x) for x in self.population])), 3)
            winner = min(sampled, key=lambda x: x[1])
            parents.append(winner[0])
        return parents

    def rank_selection(self):
        """Selects parents based on rank."""
        parents = []
        graded = [(self.fitness(x), x) for x in self.population]
        ranked = [x[1] for x in sorted(graded)]

        for _ in range(2):
            sel = randint(0, sum(range(0, len(ranked))))
            p = 0
            for x, fit in zip(ranked, range(len(ranked), 0, -1)):
                p += fit
                if p > sel:
                    parents.append(x)
                    break
        return parents

    def roulette_selection(self):
        """Selects parents using roulette wheel selection."""
        parents = []
        fitness = [1.0 / self.fitness(x) if self.fitness(x) != 0 else 1 for x in self.population]
        fitness_sum = sum(fitness)
        fitness = [x / fitness_sum for x in fitness]

        for _ in range(2):
            sel = random()
            fit_count = 0
            for x, fit in zip(self.population, fitness):
                fit_count += fit
                if fit_count > sel:
                    parents.append(x)
                    break
        return parents

    def selection(self):
        """Selects parents based on the specified selection type."""
        if self.selection_type == 'random':
            return self.random_selection()
        elif self.selection_type == 'tournament':
            return self.tournament_selection()
        elif self.selection_type == 'rank':
            return self.rank_selection()
        elif self.selection_type == 'roulette':
            return self.roulette_selection()


    def perform_crossover(self, parents):
        """Performs crossover on the parent population to generate children."""
        children = []
        wanted_offspring = len(self.population) - len(parents)

        while len(children) < wanted_offspring:
            male, female = self.selection()
            half = int(len(male) / 2)
            child = male[:half] + female[half:]
            children.append(child)

        return children

    def perform_mutation(self, population):
        """Performs mutation on the given population."""
        for individual in population:
            if self.mutate > random():
                bit_index = randint(0, len(individual)-1)
                individual[bit_index] ^= 1

    def update_population(self, parents, children):
        """Updates the population with parents and children for the next generation."""
        parents.extend(children)
        self.population = parents

    def evolve(self):
        """Evolve the population over a single generation."""
        graded = [(self.fitness(i), i) for i in self.population]
        graded = [i[1] for i in sorted(graded)]
        retain_len = int(len(graded) * self.retain)
        parents = graded[:retain_len]

        # Random selection for diversity
        for individual in graded[retain_len:]:
            if self.random_select > random():
                parents.append(individual)

        children = self.perform_crossover(parents)
        self.perform_mutation(parents + children)
        self.update_population(parents, children)

    def optimization_loop(self, generations):
        for i in range(generations):
            self.evolve()
            fitness = self.grade()
            self.fitness_history.append(fitness)
            self.pop_history.append(self.population_mean())

            # break if mean population value reaches threshold
            if fitness <= 0.5:
                break

    def plot_fitness_history(self, fitness_history):
        plt.plot(fitness_history)
        plt.title('Number Optimization')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.show()

    def optimize_coefficients(self, coefficients, generations, population_size):
        pHistory_lists = []
        for coeff in coefficients:
            self.target = coeff + self.shift  # Update target for each coefficient
            self.create_population(population_size)
            self.optimization_loop(generations)
            pHistory_lists.append(self.pop_history)

        self.plot_coefficient_optimization(pHistory_lists, coefficients)

    def plot_coefficient_optimization(self, pHistory_lists, coefficients):
        # find longest pop_history list
        longest_list = max([len(i) for i in pHistory_lists])

        # make array with a shape that matches the number of coefficients and the longest history list
        pHis_array = np.empty((len(coefficients), longest_list))

        for i, j in enumerate(pHistory_lists):
            # make all pop_history lists the same size with padding
            pHis_array[i] = np.pad(j, (0, longest_list-len(j)), 'constant', constant_values=(0, j[-1]))

        # Transpose so each row is made up of coefficients from each generation
        pHis_array = np.transpose(pHis_array)

        # Plot the target polynomial to compare against
        t_poly = np.polynomial.polynomial.Polynomial(np.flip(coefficients))
        x_target, y_target = t_poly.linspace(100, [-100, 100])
        plt.plot(x_target, y_target, label='Target Solution')

        # plot polynomial for each generation
        for current_gen, row in enumerate(pHis_array):
            coeffs = np.flip(row) # Flip so coefficients are lowest -> highest order
            poly = np.polynomial.polynomial.Polynomial(coeffs)  # plot generation solution

            x, y = poly.linspace(100, [-100, 100])

            plt.plot(x, y, label=f'Gen {current_gen + 1}')

        plt.title('Polynomial Plots')
        plt.grid(True, linestyle='-')
        plt.legend()
        plt.show()

### Initiation ###
if __name__ == '__main__':
    print("Genetic Algorithm Optimizer")
    print("1. Optimize a single numerical target")
    print("2. Optimize polynomial coefficients")
    choice = input("Enter your choice: ")

    # Common settings for GA
    retain = 0.1
    random_select = 0.05
    mutate = 0.01

    darwin = GA_optimiser(target=None, selection_type='rank', retain=retain, random_select=random_select, mutate=mutate)

    if choice == '1':
      ## example values:
        # target = 550
        # population_size = 40
        # generations = 40
        target = int(input("Enter the target value: "))
        population_size = int(input("Enter the population_size: "))
        generations = int(input("Enter the number of generations: "))
        darwin.target = target
        darwin.create_population(population_size)
        darwin.optimization_loop(generations)
        darwin.plot_fitness_history(darwin.fitness_history)

    elif choice == '2':
      ## example values:
        # target = [25, 18, 31, -14, 7, -19]
        # population_size = 100
        # generations = 5
        coefficients = list(map(int, input("Enter coefficients (comma-separated): ").split(',')))
        population_size = int(input("Enter the population_size: "))
        generations = int(input("Enter the number of generations: "))
        darwin.optimize_coefficients(coefficients, generations, population_size)
    