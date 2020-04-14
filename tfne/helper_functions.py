from typing import Union, Callable
from configparser import ConfigParser

import tensorflow as tf


def parse_configuration(config_path):
    """"""
    config = ConfigParser()
    with open(config_path) as config_file:
        config.read_file(config_file)

    return config


def deserialize_merge_method(merge_method_str) -> Callable:
    """"""
    if merge_method_str == 'concat':
        return tf.concat
    elif merge_method_str == 'sum':
        return tf.math.reduce_sum
    else:
        raise NotImplementedError("Config supplied possible merge method '{}' not implemented".format(merge_method_str))


def round_to_nearest_multiple(value, minimum, maximum, multiple) -> Union[int, float]:
    """"""
    lower_multiple = int(value / multiple) * multiple
    if value % multiple - (multiple / 2.0) < 0:
        if minimum <= lower_multiple <= maximum:
            return lower_multiple
        if lower_multiple < minimum:
            return minimum
        if lower_multiple > maximum:
            return maximum
    else:
        higher_multiple = lower_multiple + multiple
        if minimum <= higher_multiple <= maximum:
            return higher_multiple
        if higher_multiple < minimum:
            return minimum
        if higher_multiple > maximum:
            return maximum
