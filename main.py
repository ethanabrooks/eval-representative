from collections import deque
from queue import Queue
import itertools

from location_data_pb2 import LocationData
import numpy as np
from typing import Iterable, Tuple
import tensorflow as tf

from make_protos import get_location_data


def prediction_to_index(time_delta: float, protos: Iterable[LocationData]) -> \
        Tuple[int, Iterable[LocationData]]:
    """
    :param time_delta: predicted delta, in terms of milliseconds from the
    first item in `protos`
    :param protos: time-sorted LocationData iterator
    :return: tuple
     - the index of the item in protos that is closest to `time_delta` from first
    item in `protos`
     - the protos that were past by the iterator.
    """
    first = next(protos)
    absolute_time = first.timestamp + time_delta
    previous = first.timestamp
    past_protos = []
    for i, proto in enumerate(protos):
        past_protos.append(proto)
        if proto.timestamp < absolute_time:
            previous = proto.timestamp
        else:
            time_since_previous = np.abs(previous - absolute_time)
            time_until_next = np.abs(proto.timestamp - absolute_time)
            closer_to_previous = time_since_previous < time_until_next
            return i if closer_to_previous else i + 1, past_protos


def build_history(protos: Iterable[LocationData]):
    field_fns = [
        lambda proto: proto.latlng.latE7,
        lambda proto: proto.latlng.lngE7,
        lambda proto: proto.timestamp,
    ]
    return np.array([field_fn(proto)
                     for proto in protos
                     for field_fn in field_fns
                     if proto.is_optimal])


def get_prediction(history: np.array, model: tf.estimator.Estimator):
    return model.predict(history)


class MemoryIterator:
    def __init__(self, iterator, buffer=tuple()):
        self.iterator = iterator
        self.buffer = deque()
        for x in buffer:
            self.buffer.put(x)

    def __getitem__(self, item):
        if isinstance(item, int):
            try:
                return self.buffer[item]
            except IndexError:
                return self.iterate(item - len(self.buffer))
        if isinstance(item, slice):
            if item.step != 1 and item.step is not None:
                raise ValueError("Stepping by other than 1 is not supported.")
            _, iterator_copy = itertools.tee(self.iterator)
            if item.start < len(self.buffer):
                return MemoryIterator(iterator_copy, self.buffer[item:])
            else:
                for _ in range(item.start - len(self.buffer)):
                    next(iterator_copy)
                return MemoryIterator(iterator_copy)

        else:
            raise TypeError(
                'list indices must be integers or slices, '
                'not {}'.format(type(item)))

    def iterate(self, n=1):
        assert n > 0
        for _ in range(n):
            item = next(self.iterator)
            self.buffer.put(item)
        return item

    def __next__(self):
        if self.buffer.empty():
            return self.iterate()
        else:
            self.buffer.get()

    def __iter__(self):
        return self


if __name__ == '__main__':
    protos = []
    previous = None
    for i in range(10):
        proto = get_location_data(i, np.random.choice(2), previous)
        protos.append(proto)
        previous = proto.timestamp
    protos = MemoryIterator(protos)
    indexes = []
    model = tf.contrib.learn.LinearRegressor()
    with tf.Session() as sess:
        while True:
            if len(protos[:10]) < 10:
                break
            history = build_history(protos[:10])
            prediction = sess.run(model.predict(history))
            index, _ = prediction_to_index(prediction, protos[10:])
            indexes.append(index)
            protos = protos[1:]

