import ast
from typing import Union, Any
from configparser import ConfigParser


def parse_configuration(config_path):
    """"""
    config = ConfigParser()
    with open(config_path) as config_file:
        config.read_file(config_file)

    return config


def read_option_from_config(config, section, option) -> Any:
    """"""
    value = ast.literal_eval(config[section][option])
    print("Config value for '{}/{}': {}".format(section, option, value))
    return value


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
