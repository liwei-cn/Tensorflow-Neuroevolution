import os
from datetime import datetime
from abc import ABCMeta, abstractmethod


class BaseBackupAgent(object, metaclass=ABCMeta):
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

        print("Initialized Backup directory for '{}' with periodicity of {} to: {}"
              .format(self.__class__.__name__, self.periodicity, self.backup_dir_path))

    @abstractmethod
    def __call__(self, generation_counter, ne_algorithm):
        """"""
        raise NotImplementedError("Subclass of BaseBackupAgent does not implement '__call__()'")
