import random

import numpy as np
from typing import List, Tuple, Any
from itertools import combinations

from config import *
from ocr.algorithm import OCRAlgorithm
from ocr.fitness import PixelFitnessCalculator
from ocr.plotter import Plotter


class OCRIndividual:
    def __init__(self, y_indexes=None, x_indexes=None, fitness=NULL_FITNESS):
        self.y_indexes = y_indexes
        self.x_indexes = x_indexes
        self.fitness = fitness

    def __eq__(self, other):
        return self.y_indexes == other.y_indexes and self.x_indexes == other.x_indexes

    def get_combination(self) -> Tuple[List[int], List[int]]:
        return self.y_indexes, self.x_indexes


class OCRGenetic(OCRAlgorithm):
    def __init__(self, fitness_calculator: PixelFitnessCalculator, plotter: Plotter):
        super().__init__(fitness_calculator, plotter)
        self.population: List[OCRIndividual] = []
        self.mutation_probability = 0

    def create_first_generation(self, pixel_count: int, y_index_array: List[int], x_index_array: List[int]):

        super().shuffle_index_arrays(y_index_array, x_index_array, RANDOM_SEED)

        y_comb = combinations(y_index_array, pixel_count)
        x_comb = combinations(x_index_array, pixel_count)
        comb_count = 0

        population = []
        for y, x in zip(y_comb, x_comb):
            calculated_fitness = self.fitness_calculator.calculate_fitness(y, x)
            population += [OCRIndividual(y, x, calculated_fitness)]

            comb_count += 1
            if comb_count >= POPULATION_SIZE:
                break

        if comb_count < POPULATION_SIZE:
            raise Exception("Population size cannot be large than number of pixels")

        self.population = population

    def recalculate_fitness(self):
        for individual in self.population:
            individual.fitness = self.fitness_calculator.calculate_fitness(individual.y_indexes, individual.x_indexes)

        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def select_new_generation(self):
        new_generation = []

        print(f"previous population size {len(self.population)}")

        for _ in range(POPULATION_SIZE):
            # for i in np.random.randint(low=0, high=POPULATION_SIZE, size=TOURNAMENT_SIZE):
            a = random.randint(0, POPULATION_SIZE-1)
            b = random.randint(0, POPULATION_SIZE-1)
            new_generation += [max(self.population[a], self.population[b], key=lambda x: x.fitness)]

        self.population = new_generation
        print(f"new population size {len(new_generation)}")

    def crossover(self):
        child_generation = []

        print(f"parents count {len(self.population)}")

        for i in range(POPULATION_SIZE-1):
            j = i+1

            new_a = OCRIndividual()

            # for i in np.random.randint(low=0, high=POPULATION_SIZE, size=TOURNAMENT_SIZE):
            a = random.randint(0, POPULATION_SIZE-1)
            b = random.randint(0, POPULATION_SIZE-1)
            child_generation += [max(self.population[a], self.population[b], key=lambda x: x.fitness)]

        self.population = child_generation
        print(f"children count {len(child_generation)}")


    def calculate_for_k_pixels(self, pixel_count: int, y_index_array: List[int], x_index_array: List[int])\
            -> Tuple[bool, Tuple[List[int], List[int]]]:
        """Tries all possible solutions for a given number of chosen pixels."""
        self.population = []
        self.mutation_probability = 0

        self.create_first_generation(pixel_count, y_index_array, x_index_array)
        self.recalculate_fitness()

        best_fitness = self.population[0].fitness
        best_combination = self.population[0].get_combination()
        last_fitness = self.population[0].fitness

        for gen in range(MAX_GENERATIONS):
            # selection
            self.select_new_generation()

            # crossover
            self.crossover()

            # recalculate
            self.recalculate_fitness()

            # mutate
            if self.mutation_probability > 0:
                # todo mutate
                pass

            local_best_fitness = self.population[0].fitness
            self.plotter.add_record(local_best_fitness)

            # print
            print(f"Generation number {gen}:")
            print(f"Local best fitness: {local_best_fitness}")
            print(f"Local best combination: {self.population[0].get_combination()}")
            # print(f"y_indexes: {self.population[0].y_indexes}")
            # print(f"x_indexes: {self.population[0].x_indexes}")

            if last_fitness > local_best_fitness:
                self.mutation_probability += MUTATION_INCREASE_STEP

            # check
            if best_fitness == NULL_FITNESS or local_best_fitness > best_fitness:
                best_fitness = local_best_fitness
                best_combination = self.population[0].get_combination()

                # stop when desired final solution has been found
                if best_fitness == MAX_FITNESS:
                    break

        return best_fitness == MAX_FITNESS, best_combination
