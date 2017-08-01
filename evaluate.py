from queue import Queue
import tensorflow as tf
import numpy as np

from make_protos import num_records, get_location_data

history_length = 10

def prediction_to_index():
    raise NotImplemented


def convert_to_array(proto):
    return np.array([proto.timestamp, proto.latlng.latE7, proto.latlng.lngE7])


def choose_indices(i, history, model: tf.contrib.learn.Estimator,
                   predicted_timestamp):
    if i == num_records:
        return []

    current_timestamp = history[i].timestamp
    next_timestamp = history[i + 1].timestamp if i < num_records else None
    if history.full():
        if predicted_timestamp is None:
            x = np.concatenate(map(convert_to_array, history.queue))
            time_delta = model.predict(x)
            return choose_indices(i, history, model,
                                  predicted_timestamp=current_timestamp + time_delta)
        else:  # already have a prediction
            past_predicted_timestamp = current_timestamp > predicted_timestamp
            if past_predicted_timestamp:
                history.get()
                history.put(get_location_data(i))

                time_to_current = np.abs(
                    current_timestamp - predicted_timestamp)
                time_to_previous = np.abs(
                    current_timestamp - predicted_timestamp)
                closer_to_previous = time_to_previous < time_to_current
                subsequent_indices = choose_indices(i + 1, history, model, None)
                if closer_to_previous:
                    return subsequent_indices + [i]
                else:
                    return subsequent_indices + [i + 1]
            else:  # haven't reached predicted timestamp yet.
                return choose_indices(i + 1, history, model,
                                      predicted_timestamp=predicted_timestamp)


def assess_cost(indexes, path, cost):
    return


if __name__ == '__main__':
    history = [get_location_data(i) for i in range(100)]
    with tf.Session() as sess:
        cost = choose_indices(0, history, tf.contrib.learn.LinearRegressor(),
                              None)
