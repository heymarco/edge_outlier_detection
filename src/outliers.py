import math
import random
import numpy as np

from .data import add_gaussian_noise, normalize


def generate_subspace_outliers(blueprint, num_outliers):
    """
    Generate outliers in a random subspace of the data
    :param blueprint: The blueprint data set
    :param num_outliers: The number of outliers
    :return: The outliers
    """
    o_indices = np.random.choice(range(len(blueprint)), num_outliers)
    outliers = np.empty(len(blueprint))
    outliers.fill(-1)
    for i in o_indices:
        outliers[i] = 1.0 if blueprint[i] > 0.5 else 0.0
    return outliers


def generate_contextual_outliers(blueprint):
    """
    In contextual outliers, the data points are only outlying in a certain context
    :param blueprint: The data set blueprint
    :return: The outliers
    """
    outliers = np.random.uniform(low=-1, high=1, size=len(blueprint))
    outliers = blueprint + outliers
    return outliers


# TODO: Generate nontrivial outliers

