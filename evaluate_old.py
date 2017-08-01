import tensorflow as tf
from make_protos import example_path, location_data_path
import location_data_pb2


def convert_to_pandas():
    return

def convert_to_pandas():
    return

if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(
        [example_path], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'timestamp': tf.FixedLenFeature([], tf.int64),
            'lat': tf.FixedLenFeature([], tf.int64),
            'lng': tf.FixedLenFeature([], tf.int64),
        })
    timestamp = features['timestamp']
    lat = features['lat']
    lng = features['lng']
    weights = tf.random_normal((3, 1))
    time_delta = tf.matmul(
        tf.to_float(tf.expand_dims(tf.stack([timestamp, lat, lng]), axis=0)),
        weights)

    with tf.Session() as sess:
        print(sess.run(lng))
