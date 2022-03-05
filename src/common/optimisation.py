import abc
import random
from typing import Tuple

import numpy as np
from pydantic import confloat, PositiveInt, validate_arguments


class OptimisationError(Exception):
    pass


class AllowArbitraryTypes:
    """Pydantic class to allow arbitrary types"""

    arbitrary_types_allowed = True


class BaseFitnessFunction(abc.ABC):
    """Base class for function evaluating fitness"""

    @abc.abstractmethod
    def __call__(self, state: np.ndarray) -> float:
        """Evaluate the fitness of a state

        Args:
            state: State of problem to be evaluated

        Returns:
            fitness: Value of fitness function
        """
        pass

    @property
    @abc.abstractmethod
    def length(self) -> int:
        """Length of valid state vectors

        Returns:
            int: Valid state vector length
        """
        pass


class BaseOptimisationProblem(abc.ABC):
    """Base class for optimisation problems"""

    def __init__(self, fitness_fn: BaseFitnessFunction, maximise: bool):
        self.fitness_fn = fitness_fn
        self.length = fitness_fn.length
        self.maximise = maximise

        self.state = None
        self._population = None
        self._population_fitness = None
        self.reset()

    @property
    def fitness(self) -> float:
        """Returns fitness of current state"""
        return self.calc_fitness(self.state)

    @property
    def population(self) -> np.ndarray:
        """Returns state of entire population"""
        return self._population

    @population.setter
    def population(self, population: np.ndarray):
        """Sets state of entire population. Also calculates fitness"""
        self._population = population
        self._population_fitness = np.array(
            [self.calc_fitness(state) for state in population]
        )

    @property
    def best_child(self) -> np.ndarray:
        """Returns state of best child in population"""
        return self._population[np.argmax(self._population_fitness)]

    def calc_fitness(self, state: np.ndarray) -> float:
        """Calculate fitness for given state

        Args:
            state: State to calculate fitness for

        Returns:
            fitness value
        """
        fitness = self.fitness_fn(state)
        return fitness if self.maximise else -fitness

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the state of the problem"""
        pass

    @abc.abstractmethod
    def generate_population(self, n: int) -> int:
        """Generate a random population of n nodes

        Args:
            n: Number of nodes in population

        Returns:
            Number of nodes actually generated
        """
        pass

    @property
    @abc.abstractmethod
    def mate_probabilities(self) -> np.ndarray:
        """Returns desired mate probability for all nodes in population"""
        pass

    @abc.abstractmethod
    def reproduce(
        self, parent_1: np.ndarray, parent_2: np.ndarray, mutation_prob: float
    ) -> np.ndarray:
        """Produce a child node from two parent nodes

        Args:
            parent_1: Parent node 1
            parent_2: Parent node 2
            mutation_prob: Probability of mutation of child node

        Returns:
            child node
        """
        pass


@validate_arguments(config=AllowArbitraryTypes)
def genetic_algorithm(
    problem: BaseOptimisationProblem,
    n_population: PositiveInt = 100,
    max_attempts: PositiveInt = 10,
    max_iterations: PositiveInt = np.inf,
    mutation_probability: confloat(ge=0, le=1) = 0.2,
    return_history: bool = False,
    random_seed: PositiveInt = None,
) -> Tuple[np.array, float, np.array]:
    """Implementation of a standard genetic algorithm for optimisation

    Args:
        problem: Optimisation problem
        n_population: Population size
        max_attempts: Maximum number of attempts to find a better state at each step
        max_iterations: Maximum number of iterations
        mutation_probability: Probability of a mutation during reproduction
        return_history: Option to return the history of fitness values
        random_seed: Random seed used for random number generation

    Returns:
        best_state: State that optimizes the fitness function
        best_fitness: Value of fitness function at best state.
        fitness_history: Fitness of the entire population at every iteration
    """
    fitness_history = []

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    problem.reset()
    n_population = problem.generate_population(n_population)
    attempts = 0
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        # Create next generation of population
        next_gen = []

        for _ in range(n_population):
            idx_1, idx_2 = np.random.choice(
                n_population, 2, p=problem.mate_probabilities
            )
            parent_1 = problem.population[idx_1]
            parent_2 = problem.population[idx_2]

            child = problem.reproduce(parent_1, parent_2, mutation_probability)
            next_gen.append(child)

        problem.population = np.array(next_gen)
        best_child_fitness = problem.calc_fitness(problem.best_child)

        # Move to best child if that is better than current state,
        # otherwise, reproduce with current population again
        if best_child_fitness > problem.fitness:
            problem.state = problem.best_child
            attempts = 0
        else:
            attempts += 1
            if attempts > max_attempts:
                break

        if return_history:
            fitness_history.append(problem.fitness)

    best_fitness = problem.fitness if problem.maximise else -problem.fitness
    best_state = problem.state

    return best_state, best_fitness, fitness_history
