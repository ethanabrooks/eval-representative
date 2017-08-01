#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import location_data_pb2

num_records = 1
num_optimal = 0
example_path = 'example.proto'
location_data_path = 'location_data_points.proto'


def random_timestamp():
    return np.random.randint(low=0, high=100000, size=())


def random_timedelta():
    return np.random.randint(low=0, high=1000, size=())


def random_degree_e7():
    return np.random.randint(low=0, high=100000, size=())


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_location_data(i, optimal, previous_timestamp):
    record = location_data_pb2.LocationData()
    record.latlng.latE7 = random_degree_e7()
    record.latlng.lngE7 = random_degree_e7()
    if previous_timestamp:
        record.timestamp = previous_timestamp + random_timedelta()
    else:
        record.timestamp = random_timestamp()
    record.is_optimal = i in optimal
    record.index = i
    return record


if __name__ == '__main__':
    optimal = np.random.randint(low=0, high=num_records, size=num_optimal)

    location_data_protos = []

    with open(location_data_path, 'wb') as f:
        for i in range(num_records):
            record = get_location_data(i)
            f.write(record.SerializeToString())
            location_data_protos.append(record)
    print('done')

    # with tf.python_io.TFRecordWriter(example_path) as writer:
    #     for i in optimal.tolist():
    #         record = location_data_protos[i]
    #         example = tf.train.Example(
    #             features=tf.train.Features(
    #                 feature={
    #                     'timestamp': _int64_feature(record.timestamp),
    #                     'lat': _int64_feature(record.latlng.latE7),
    #                     'lng': _int64_feature(record.latlng.lngE7),
    #                 }))
    #         writer.write(example.SerializeToString())
