import pytest
import numpy as np

from src.common.optimisation import genetic_algorithm
from src.travelling_salesman_problem import TSPFitnessFunction, TSPOptimisation


@pytest.mark.parametrize(
    "state, expected",
    (
        (np.array([0, 1, 2, 3, 4]), 11.81256),
        (np.array([0, 1, 4, 3, 2]), 13.86138),
        (np.array([3, 1, 4, 2, 0]), 18.04241),
        (np.array([4, 3, 1, 0, 2]), 14.37894),
        (np.array([1, 3, 2, 0, 4]), 16.73255),
    ),
)
def test_tsp_fitness(state, expected):
    coordinates = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]
    fitness_fn = TSPFitnessFunction(coordinates=coordinates)
    fitness = fitness_fn(state)
    assert pytest.approx(fitness) == expected


def test_travelling_salesman_problem():
    coordinates = [
        (np.random.randint(-25, 25), np.random.randint(-25, 25)) for _ in range(15)
    ]
    fitness_fn = TSPFitnessFunction(coordinates=coordinates)

    problem = TSPOptimisation(fitness_fn, False)

    result = genetic_algorithm(
        problem, 20000, 10, mutation_probability=0.35, return_history=True
    )
    print(result)

    import matplotlib.pyplot as plt

    for x, y in coordinates:
        plt.scatter(x, y)

    for idx, i in enumerate(result[0]):
        point1 = coordinates[i]
        try:
            point2 = coordinates[result[0][idx + 1]]
        except IndexError:
            point2 = coordinates[result[0][0]]

        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        print(point1, point2)
        plt.plot(x_values, y_values)
    plt.show()
