from absl import logging

from .base_backup_agent import BaseBackupAgent


class VisualizePopulation(BaseBackupAgent):
    """"""

    def __init__(self, periodicity, backup_dir_path):
        """"""
        super().__init__(periodicity, backup_dir_path)

    def __call__(self, generation_counter, ne_algorithm):
        """"""
        if generation_counter % self.periodicity == 0:
            gen_backup_dir_path = self.backup_dir_path + "gen_{}/".format(ne_algorithm.get_generation_counter())
            ne_algorithm.visualize_population(save_file_path=gen_backup_dir_path)
