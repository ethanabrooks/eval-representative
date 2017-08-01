from unittest import TestCase
import unittest
import sys

import numpy as np
import tensorflow as tf

import location_data_pb2
from main import build_history, get_prediction, prediction_to_index


class StubModel(tf.contrib.learn.Estimator):
    def __init__(self, constant_output):
        self.constant_output = constant_output

    def predict(self):
        return self.constant_output


SEED = 0
NUM_FIELDS = 4
TIMESTAMP_INDEX = 2
IS_OPTIMAL_INDEX = 3


def random_int():
    return np.random.randint(low=0, high=100)


def build_random_time_sorted_array(num_protos):
    shuffled_history = np.random.randint(low=0, high=1000,
                                         size=(num_protos, NUM_FIELDS - 1))
    time_order = shuffled_history[:, TIMESTAMP_INDEX].argsort()
    optimality = np.random.choice(2, size=(num_protos, 1))
    time_ordered = shuffled_history[time_order]
    return np.hstack([time_ordered, optimality])


def row_to_proto(i, row):
    assert row.size == NUM_FIELDS
    d = location_data_pb2.LocationData()
    d.latlng.latE7, d.latlng.lngE7, d.timestamp, d.is_optimal = tuple(row)
    d.index = i
    return d


def array_to_protos_list(array):
    return [row_to_proto(i, row) for i, row in enumerate(array)]


def build_protos(num_protos):
    expected = build_random_time_sorted_array(num_protos)
    return iter(array_to_protos_list(expected))


class TestBuild_history(TestCase):
    def setUp(self):
        np.random.seed(SEED)

    def test_build_history(self):
        # Params
        size_history = random_int()
        num_protos = size_history + random_int()
        array = build_random_time_sorted_array(num_protos)
        is_optimal = array[:, IS_OPTIMAL_INDEX]
        array = array[is_optimal is True]
        protos = array_to_protos_list(array)
        expected = array[:, :IS_OPTIMAL_INDEX].flatten()
        actual = build_history(protos)
        assert np.array_equal(actual, expected)


class TestPrediction_to_index(TestCase):
    def setUp(self):
        np.random.seed(SEED)

    def test_prediction_to_index(self):
        size_history = random_int()
        num_protos = size_history + random_int()

        relative_index = np.random.randint(low=0, high=num_protos - 1)
        expected = relative_index
        array = build_random_time_sorted_array(num_protos)
        protos = array_to_protos_list(array)

        expected_timestamp = protos[relative_index].timestamp
        subsequent_timestamp = protos[relative_index + 1].timestamp
        assert subsequent_timestamp >= expected_timestamp
        delta = subsequent_timestamp - expected_timestamp

        # random offset, but still closer to timestamp of expected_index
        offset = np.random.uniform(low=0, high=delta / 2)

        first_timestamp = protos[0].timestamp
        prediction = expected_timestamp + offset - first_timestamp
        actual, _ = prediction_to_index(prediction, iter(protos))
        assert actual == expected


if __name__ == '__main__':
    unittest.main()
