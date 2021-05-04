from typing import List
from itertools import combinations

from config import *
from ocr.algorithms.algorithm import OCRAlgorithm
from ocr.gui.painter import Painter
from ocr.utils.fitness import PixelFitnessCalculator
from ocr.utils.plotter import Plotter


class OCRIndividual:
    def __init__(self, indexes: np.ndarray, fitness=NULL_FITNESS):
        self.indexes = indexes
        self.fitness = fitness

    def __eq__(self, other):
        return np.array_equal(self.indexes, other.indexes)


class OCRGenetic(OCRAlgorithm):
    """Class for calculation pixel combination using genetic algorithms"""
    def __init__(self, pixel_count: int, indexes_array: np.ndarray,
                 fitness_calculator: PixelFitnessCalculator,
                 plotter: Plotter, painter: Painter):
        super().__init__(pixel_count, indexes_array, fitness_calculator, plotter, painter)
        self.population: List[OCRIndividual] = []
        self.mutation_probability = 0.0
        self.mutate_swap_count = 0

    def create_first_generation(self):
        """Creates an initial generation of randomly generated combinations of pixels"""

        # randomly shuffles array of indexes
        self.shuffle_index_array(self.indexes_array, RANDOM_SEED)

        # creates iterator which can generate all possible combinations of a given length
        index_combinations = combinations(self.indexes_array, self.pixel_count)
        comb_count = 0

        population = []
        # generates new index combinations
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
        """Calculates fitness for each individual in a new generations"""
        for individual in self.population:
            individual.fitness = self.fitness_calculator.calculate_fitness(individual.indexes)

        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def select(self):
        """Selects individual combinations of indexes from the old generation using the tournament selection."""
        tournament_size = 10

        new_generation = []
        for _ in range(POPULATION_SIZE):
            chosen_individuals = tuple(np.random.choice(POPULATION_SIZE, int(POPULATION_SIZE*TOURNAMENT_SIZE_PERCENTAGE)))
            # chosen_individuals = (10, 20,30)
            # a = random.randint(0, POPULATION_SIZE-1)
            # b = random.randint(0, POPULATION_SIZE-1)
            # c = random.randint(0, POPULATION_SIZE-1)
            # new_generation += [max(self.population[a], self.population[b], self.population[c], key=lambda x: x.fitness)]
            new_generation += [max([self.population[x] for x in chosen_individuals], key=lambda x: x.fitness)]

        self.population = new_generation

    def crossover(self):
        """Crosses over each two individual pixel combinations"""

        child_generation = []

        for i in range(POPULATION_SIZE):
            j = i-1

            # todo write more efficiently

            concatenated_array = np.concatenate((self.population[i].indexes, self.population[j].indexes))
            available_indexes_array = np.unique(concatenated_array, axis=0)
            new_array_indexes = np.random.choice(len(available_indexes_array), size=self.pixel_count, replace=False)
            new_array = available_indexes_array[new_array_indexes]

            child_generation += [OCRIndividual(new_array)]

        self.population = child_generation

    def mutate(self):
        """Randomly mutates individuals by replacing pixels"""

        # goes thought entire populations
        for individual in self.population:
            if self.mutation_probability > np.random.uniform():
                for _ in range(np.random.choice(int(self.pixel_count * MUTATE_PIXELS_MAX_PERCENTAGE), 1)[0]):
                    for index_new in (np.random.choice(len(self.indexes_array), 1) for x in range(len(self.indexes_array))):
                        # replaces pixels if not already present in pixel combination
                        if self.indexes_array[index_new] not in individual.indexes:
                            index_old = np.random.choice(self.pixel_count, 1)
                            individual.indexes[index_old] = self.indexes_array[index_new]
                            self.mutate_swap_count += 1
                            break

    def calculate_for_k_pixels(self) -> Tuple[bool, np.ndarray]:
        """Uses genetic algorithm to find a combination of pixels."""
        self.population = []
        self.mutation_probability = 0.0

        self.create_first_generation()
        self.recalculate_fitness()

        best_fitness = self.population[0].fitness
        last_fitness = self.population[0].fitness
        best_combination = self.population[0].indexes

        for gen in range(MAX_GENERATIONS):
            # selection
            self.select()

            # crossover
            self.crossover()

            # mutate
            if self.mutation_probability > 0.0:
                self.mutate()
                # pass

            # recalculate
            self.recalculate_fitness()

            # get best combinations of current generation
            fitness = self.population[0].fitness
            self.plotter.add_record(fitness)
            self.painter.change_chosen_pixels(self.population[0].indexes)

            if last_fitness >= fitness:
                self.mutation_probability += MUTATION_INCREASE_STEP
            last_fitness = fitness

            # check if fitness has improved
            if best_fitness == NULL_FITNESS or fitness > best_fitness:
                best_fitness = fitness
                best_combination = self.population[0].indexes.copy()

                # stop when desired final solution has been found
                if best_fitness == MAX_FITNESS:
                    print("Found best fitness")
                    break

        print(f"For {self.pixel_count} pixels, mutation swap count: {self.mutate_swap_count}")
        print(f"Best fitness: {best_fitness}")
        print(f"Mutation probability: {self.mutation_probability}")
        return best_fitness == MAX_FITNESS, best_combination
