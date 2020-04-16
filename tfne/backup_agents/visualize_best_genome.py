from .base_backup_agent import BaseBackupAgent


class VisualizeBestGenome(BaseBackupAgent):
    """"""

    def __init__(self, periodicity, view, backup_dir_path):
        """"""
        super().__init__(periodicity, backup_dir_path)
        self.view = view

    def __call__(self, generation_counter, ne_algorithm):
        """"""
        if generation_counter % self.periodicity == 0:
            best_genome = ne_algorithm.get_best_genome()
            best_genome.visualize(view=self.view, save_dir_path=self.backup_dir_path)
