"""Utility functions for map generation and comparison

Scoring functions: ref. Leaderboard/leaderboard/statistics/statistics_manager.py

"""

import sys
import numpy as np

sys.path.append(os.path.abspath("../../Leaderboard/"))

from leaderboard.agents.geometric_map import ROCK_UNCOMPLETED_VALUE

GEOMETRIC_MAP_MAX_SCORE = 300.0
GEOMETRIC_MAP_MIN_SCORE = 0.0
GEOMETRIC_MAP_THRESHOLD = 0.05

ROCK_MIN_SCORE = 0.0
ROCK_MAX_SCORE = 300.0
ROCK_UNCOMPLETED_VALUE = -np.inf


def get_geometric_score(ground_map: np.ndarray, agent_map: np.ndarray) -> float:
    """Compare the calculated heights vs the real ones"""
    if agent_map is None or ground_map is None:
        return GEOMETRIC_MAP_MIN_SCORE

    true_heights = ground_map[:, :, 2]
    agent_heights = agent_map[:, :, 2]
    error_heights = np.sum(np.abs(true_heights - agent_heights) < GEOMETRIC_MAP_THRESHOLD)
    score_rate = error_heights / true_heights.size
    return GEOMETRIC_MAP_MAX_SCORE * score_rate


def get_rocks_score(ground_map: np.ndarray, agent_map: np.ndarray) -> float:
    """
    Compare the number of rocks found vs the real ones using an F1 score. Uncompleted values
    will be supposed False, increasing the amount of false negatives.
    """
    if agent_map is None or ground_map is None:
        return ROCK_MIN_SCORE

    true_rocks = ground_map[:, :, 3]
    if np.sum(true_rocks) == 0:
        # Special case, preset has no rocks, disable the
        return ROCK_MIN_SCORE

    agent_rocks = np.copy(agent_map[:, :, 3])
    agent_rocks[agent_rocks == ROCK_UNCOMPLETED_VALUE] = False  # Uncompleted will

    tp = np.sum(np.logical_and(agent_rocks == True, true_rocks == True))
    fp = np.sum(np.logical_and(agent_rocks == True, true_rocks == False))
    fn = np.sum(np.logical_and(agent_rocks == False, true_rocks == True))

    score_rate = (2 * tp) / (2 * tp + fp + fn)
    return ROCK_MAX_SCORE * score_rate
