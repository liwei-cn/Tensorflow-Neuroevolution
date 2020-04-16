from absl import app, flags, logging

import tfne

flags.DEFINE_string('logging_level',
                    default=None, help='TBD')
flags.DEFINE_string('config_path',
                    default=None, help='TBD')
flags.DEFINE_integer('num_cpus',
                     default=None, help='TBD')
flags.DEFINE_integer('num_gpus',
                     default=None, help='TBD')
flags.DEFINE_integer('max_generations',
                     default=None, help='TBD')
flags.DEFINE_float('max_fitness',
                   default=None, help='TBD')


def codeepneat_example(_):
    """"""
    # Set standard configuration specific to TFNE but not the neuroevolution process
    logging_level = logging.INFO
    config_path = './codeepneat_example_config.cfg'
    num_cpus = None
    num_gpus = None
    max_generations = 10
    max_fitness = None

    # Read in optionally supplied flags, changing the just set standard configuration
    if flags.FLAGS.logging_level is not None:
        logging_level = flags.FLAGS.logging_level
    if flags.FLAGS.config_path is not None:
        config_path = flags.FLAGS.config_path
    if flags.FLAGS.num_cpus is not None:
        num_cpus = flags.FLAGS.num_cpus
    if flags.FLAGS.num_gpus is not None:
        num_gpus = flags.FLAGS.num_gpus
    if flags.FLAGS.max_generations is not None:
        max_generations = flags.FLAGS.max_generations
    if flags.FLAGS.max_fitness is not None:
        max_fitness = flags.FLAGS.max_fitness

    # Set logging, parse config
    logging.set_verbosity(logging_level)
    config = tfne.parse_configuration(config_path)

    # Initialize the environment as well as the specific NE algorithm
    environment = tfne.environments.XOREnvironment(config, weight_training_eval=True)
    ne_algorithm = tfne.CoDeepNEAT(config)

    # Initialize and set up backup agents
    ba_save_best_genome = tfne.backup_agents.SaveBestGenome(periodicity=1,
                                                            backup_dir_path='./backups_best_genome/')
    ba_save_population = tfne.backup_agents.SavePopulation(periodicity=10,
                                                           backup_dir_path='./backups_population/')
    ba_viz_best_genome = tfne.backup_agents.VisualizeBestGenome(periodicity=1,
                                                                view=False,
                                                                backup_dir_path='./visualizations_best_genome/')
    ba_viz_population = tfne.backup_agents.VisualizePopulation(periodicity=1,
                                                               backup_dir_path='./visualizations_population/')
    backup_agents = (ba_save_best_genome, ba_save_population, ba_viz_best_genome, ba_viz_population)

    # Initialize evolution engine and supply config as well as initialized NE elements
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  num_cpus=num_cpus,
                                  num_gpus=num_gpus,
                                  max_generations=max_generations,
                                  max_fitness=max_fitness,
                                  backup_agents=backup_agents)

    # Start training process, returning the best genome when training ends
    best_genome = engine.train()

    # Show string representation of best genome, visualize it and then save it
    print("Best Genome returned by evolution:\n")
    print(best_genome)
    best_genome.visualize(view=True, save_dir_path='./')
    best_genome.save_genotype(save_dir_path='./')
    best_genome.save_model(save_dir_path='./')


if __name__ == '__main__':
    app.run(codeepneat_example)
