import random

import numpy as np
from typing import List, Tuple, Any
from itertools import combinations

from config import *
from ocr.algorithm import OCRAlgorithm
from ocr.fitness import PixelFitnessCalculator
from ocr.plotter import Plotter


class OCRIndividual:
    def __init__(self, indexes: np.ndarray, fitness=NULL_FITNESS):
        self.indexes = indexes
        self.fitness = fitness

    def __eq__(self, other):
        return np.array_equal(self.indexes, other.indexes)


class OCRGenetic(OCRAlgorithm):
    def __init__(self, fitness_calculator: PixelFitnessCalculator, plotter: Plotter):
        super().__init__(fitness_calculator, plotter)
        self.population: List[OCRIndividual] = []
        self.mutation_probability = 0
        self.pixel_count = 0

    def create_first_generation(self, indexes_array: np.ndarray):

        super().shuffle_index_array(indexes_array, RANDOM_SEED)

        index_combinations = combinations(indexes_array, self.pixel_count)
        comb_count = 0

        population = []
        for index_combination in (np.array(comb) for comb in index_combinations):
            calculated_fitness = self.fitness_calculator.calculate_fitness(index_combination)
            population += [OCRIndividual(index_combination, calculated_fitness)]

            comb_count += 1
            if comb_count >= POPULATION_SIZE:
                break

        if comb_count < POPULATION_SIZE:
            raise Exception("Population size cannot be large than number of pixels")

        self.population = population

    def recalculate_fitness(self):
        for individual in self.population:
            individual.fitness = self.fitness_calculator.calculate_fitness(individual.indexes)

        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def select_new_generation(self):
        new_generation = []
        for _ in range(POPULATION_SIZE):
            a = random.randint(0, POPULATION_SIZE-1)
            b = random.randint(0, POPULATION_SIZE-1)
            new_generation += [max(self.population[a], self.population[b], key=lambda x: x.fitness)]

        self.population = new_generation

    def crossover(self):
        child_generation = []

        # print(f"parents count {len(self.population)}")

        for i in range(POPULATION_SIZE):
            j = i-1

            # todo write more efficiently

            # print("self.population[i]")
            # print(self.population[i].indexes)
            # print("self.population[j]")
            # print(self.population[j].indexes)

            concatenated_array = np.concatenate((self.population[i].indexes, self.population[j].indexes))
            available_indexes_array = np.unique(concatenated_array, axis=0)
            new_array_indexes = np.random.choice(len(available_indexes_array), size=self.pixel_count, replace=False)
            new_array = available_indexes_array[new_array_indexes]

            child_generation += [OCRIndividual(new_array)]

        self.population = child_generation
        # print(f"children count {len(child_generation)}")

    def calculate_for_k_pixels(self, pixel_count: int, indexes_array: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Tries all possible solutions for a given number of chosen pixels."""
        self.population = []
        self.mutation_probability = 0
        self.pixel_count = pixel_count

        self.create_first_generation(indexes_array)
        self.recalculate_fitness()

        best_fitness = NULL_FITNESS
        last_fitness = self.population[0].fitness
        best_combination = self.population[0].indexes

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

            fitness = self.population[0].fitness
            self.plotter.add_record(fitness)

            # print
            # print(f"Generation number {gen}:")
            # print(f"Local good fitness: {fitness}")
            # print(f"Local good combination: {self.population[0].indexes}")
            # print(f"y_indexes: {self.population[0].y_indexes}")
            # print(f"x_indexes: {self.population[0].x_indexes}")

            if last_fitness > fitness:
                self.mutation_probability += MUTATION_INCREASE_STEP

            # check
            if best_fitness == NULL_FITNESS or fitness > best_fitness:
                best_fitness = fitness
                best_combination = self.population[0].indexes.copy()

                # stop when desired final solution has been found
                if best_fitness == MAX_FITNESS:
                    print("Found best fitness")
                    break

        return best_fitness == MAX_FITNESS, best_combination
