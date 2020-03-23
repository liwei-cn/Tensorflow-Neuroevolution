import os
from datetime import datetime

from absl import logging


class BackupAgentPopulation:
    """"""

    def __init__(self, periodicity, backup_dir_path):
        """"""
        # Assert requirements for parameters
        assert periodicity > 0

        # Register parameters
        self.periodicity = periodicity
        self.backup_dir_path = os.path.abspath(backup_dir_path)

        # Create directory for the backups
        run_datetime_string = datetime.now(tz=datetime.now().astimezone().tzinfo).strftime("run_%Y-%b-%d_%H-%M-%S/")
        if self.backup_dir_path[-1] != '/':
            self.backup_dir_path += '/'
        self.backup_dir_path += run_datetime_string
        os.makedirs(self.backup_dir_path)

    def __call__(self, population):
        """"""
        generation = population.get_generation_counter()
        if generation % self.periodicity == 0:
            logging.debug("\t\tToDo: Create Backups for Population")


class BackupAgentBestGenome:
    """"""

    def __init__(self, periodicity, backup_dir_path):
        """"""
        # Assert requirements for parameters
        assert periodicity > 0

        # Register parameters
        self.periodicity = periodicity
        self.backup_dir_path = os.path.abspath(backup_dir_path)

        # Create directory for the backups
        run_datetime_string = datetime.now(tz=datetime.now().astimezone().tzinfo).strftime("run_%Y-%b-%d_%H-%M-%S/")
        if self.backup_dir_path[-1] != '/':
            self.backup_dir_path += '/'
        self.backup_dir_path += run_datetime_string
        os.makedirs(self.backup_dir_path)

    def __call__(self, population):
        """"""
        generation = population.get_generation_counter()
        if generation % self.periodicity == 0:
            logging.debug("\t\tToDo: Create Backups for Best Genome")
