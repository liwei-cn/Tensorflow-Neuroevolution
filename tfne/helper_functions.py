import ast
from typing import Union, Callable, Any
from configparser import ConfigParser

import tensorflow as tf


def parse_configuration(config_path):
    """"""
    config = ConfigParser()
    with open(config_path) as config_file:
        config.read_file(config_file)

    return config


def read_option_from_config(config, section, option) -> Any:
    """"""
    value = ast.literal_eval(config[section][option])
    if isinstance(value, list):
        value = tuple(value)
    print("Config value for '{}/{}': {}".format(section, option, value))
    return value


def deserialize_merge_method(merge_method_str) -> Callable:
    """"""
    if merge_method_str == 'concat':
        return tf.keras.layers.concatenate
    elif merge_method_str == 'add':
        return tf.keras.layers.add
    elif merge_method_str == 'average':
        return tf.keras.layers.average
    elif merge_method_str == 'subtract':
        return tf.keras.layers.subtract
    else:
        raise NotImplementedError("Config supplied possible merge method '{}' not implemented".format(merge_method_str))


def round_with_step(value, minimum, maximum, step) -> Union[int, float]:
    """"""
    lower_step = int(value / step) * step
    if value % step - (step / 2.0) < 0:
        if minimum <= lower_step <= maximum:
            return lower_step
        if lower_step < minimum:
            return minimum
        if lower_step > maximum:
            return maximum
    else:
        higher_step = lower_step + step
        if minimum <= higher_step <= maximum:
            return higher_step
        if higher_step < minimum:
            return minimum
        if higher_step > maximum:
            return maximum
