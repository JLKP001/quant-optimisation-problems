from typing import Tuple
from collections import deque

import numpy as np

from ..common.optimisation import (
    BaseFitnessFunction,
    BaseOptimisationProblem,
    OptimisationError,
)


class TSPFitnessFunction(BaseFitnessFunction):
    """Evaluates TSP fitness of state for a set of coordinates"""

    def __init__(self, coordinates: Tuple[Tuple[float, float], ...]):
        self.coordinates = coordinates

    def _validate_state(self, state: np.array) -> None:
        if len(state) != self.length:
            raise OptimisationError(
                f"Invalid state length. Expected {self.length}, "
                f"got {len(state)} instead"
            )

        if not len(state) == len(set(state)):
            raise OptimisationError(
                "Index for each coordinate must appear exactly once"
            )

        if min(state) < 0 or max(state) >= self.length:
            raise OptimisationError("Invalid coordinate index in state")

    def __call__(self, state: np.array) -> float:
        """Evaluate the fitness of a state

        Args:
            state: State of TSP problem
                Index for each coordinate must appear exactly once

        Returns:
            fitness: Value of fitness function
        """
        self._validate_state(state)

        coords = np.array(self.coordinates)
        rotated = deque(state)
        rotated.rotate(-1)
        rotated = list(rotated)

        # Build array of source and destination nodes
        source_nodes = coords[state]
        dest_nodes = coords[rotated]

        # Sum distances between all source and dest nodes
        return sum(np.linalg.norm(source_nodes - dest_nodes, axis=1))

    @property
    def length(self) -> int:
        """Length of valid state vectors

        Returns:
            int: Valid state vector length
        """
        return len(self.coordinates)


class TSPOptimisation(BaseOptimisationProblem):
    """Travelling Salesman optimisation problem definition"""

    def reset(self) -> None:
        """Resets the state of the TSP"""
        self.state = self._random()[0]

    def generate_population(self, n: int) -> None:
        """Generate a random population of n nodes

        Args:
            n: Number of nodes in population
        """
        self.population = np.array(self._random(n))

    @property
    def mate_probabilities(self) -> np.ndarray:
        """Returns desired mate probability for all nodes in population"""
        total_fitness = np.sum(self._population_fitness)
        n_population = len(self._population_fitness)

        if total_fitness == 0:
            return np.ones(n_population) / n_population
        return self._population_fitness / total_fitness

    def _random(self, n: int = 1) -> list:
        """Generates n random permutations of possible states

        Args:
            n: Number of desired permuatations. Defaults to 1.

        Returns:
            List of state permutations
        """
        return [np.random.permutation(self.length) for _ in range(n)]

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
        # Randomly select a breakpoint and combine two parents at that point
        break_idx = np.random.randint(self.length - 1)
        first_part = parent_1[: break_idx + 1]
        second_part = [node for node in parent_2 if node not in first_part]

        child = np.array([0] * self.length)
        child[: break_idx + 1] = first_part
        child[break_idx + 1 :] = second_part

        # Random mutation of child (permutation of nodes)
        rand = np.random.uniform(size=self.length)
        mutate = np.where(rand < mutation_prob)[0]
        if len(mutate) > 1:
            mutate_perm = np.random.permutation(mutate)
            child[mutate] = child[mutate_perm]
        return child
