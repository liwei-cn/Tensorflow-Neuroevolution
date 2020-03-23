from configparser import ConfigParser


def parse_configuration(config_path):
    """"""
    config = ConfigParser()
    with open(config_path) as config_file:
        config.read_file(config_file)

    return config
