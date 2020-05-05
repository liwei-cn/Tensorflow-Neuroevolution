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
        return tf.concat
    elif merge_method_str == 'sum':
        return tf.math.reduce_sum
    else:
        raise NotImplementedError("Config supplied possible merge method '{}' not implemented".format(merge_method_str))


def round_with_step(value, min, max, step) -> Union[int, float]:
    """"""
    lower_step = int(value / step) * step
    if value % step - (step / 2.0) < 0:
        if min <= lower_step <= max:
            return lower_step
        if lower_step < min:
            return min
        if lower_step > max:
            return max
    else:
        higher_step = lower_step + step
        if min <= higher_step <= max:
            return higher_step
        if higher_step < min:
            return min
        if higher_step > max:
            return max
