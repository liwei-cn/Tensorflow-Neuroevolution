from .base_backup_agent import BaseBackupAgent


class VisualizePopulation(BaseBackupAgent):
    """"""

    def __init__(self, periodicity, backup_dir_path):
        """"""
        super().__init__(periodicity, backup_dir_path)

    def __call__(self, generation_counter, ne_algorithm):
        """"""
        if generation_counter % self.periodicity == 0:
            # Append current generation count to backup dir and then visualize population
            gen_backup_dir_path = self.backup_dir_path + f"gen_{generation_counter}/"
            ne_algorithm.visualize_population(save_file_path=gen_backup_dir_path)
