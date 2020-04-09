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
flags.DEFINE_integer('save_best_genome_periodicity',
                     default=None, help='TBD')
flags.DEFINE_string('save_best_genome_backup_dir_path',
                    default=None, help='TBD')
flags.DEFINE_integer('save_population_periodicity',
                     default=None, help='TBD')
flags.DEFINE_string('save_population_backup_dir_path',
                    default=None, help='TBD')
flags.DEFINE_integer('visualize_best_genome_periodicity',
                     default=None, help='TBD')
flags.DEFINE_string('visualize_best_genome_backup_dir_path',
                    default=None, help='TBD')
flags.DEFINE_integer('visualize_population_periodicity',
                     default=None, help='TBD')
flags.DEFINE_string('visualize_population_backup_dir_path',
                    default=None, help='TBD')


def codeepneat_example(argv):
    """"""
    # Set standard configuration not specific to the neuroevolutionary process
    logging_level = logging.INFO
    config_path = './codeepneat_example_config.cfg'
    num_cpus = None
    num_gpus = None
    max_generations = 1
    max_fitness = None
    save_best_genome_periodicity = 1
    save_best_genome_backup_dir_path = './backups_best_genome/'
    save_population_periodicity = 10
    save_population_backup_dir_path = './backups_population/'
    visualize_best_genome_periodicity = 1
    visualize_best_genome_backup_dir_path = './visualizations_best_genome/'
    visualize_population_periodicity = 3
    visualize_population_backup_dir_path = './visualizations_population/'

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
    if flags.FLAGS.save_best_genome_periodicity is not None:
        save_best_genome_periodicity = flags.FLAGS.save_best_genome_periodicity
    if flags.FLAGS.save_best_genome_backup_dir_path is not None:
        save_best_genome_backup_dir_path = flags.FLAGS.save_best_genome_backup_dir_path
    if flags.FLAGS.save_population_periodicity is not None:
        save_population_periodicity = flags.FLAGS.save_population_periodicity
    if flags.FLAGS.save_population_backup_dir_path is not None:
        save_population_backup_dir_path = flags.FLAGS.save_population_backup_dir_path
    if flags.FLAGS.visualize_best_genome_periodicity is not None:
        visualize_best_genome_periodicity = flags.FLAGS.visualize_best_genome_periodicity
    if flags.FLAGS.visualize_best_genome_backup_dir_path is not None:
        visualize_best_genome_backup_dir_path = flags.FLAGS.visualize_best_genome_backup_dir_path
    if flags.FLAGS.visualize_population_periodicity is not None:
        visualize_population_periodicity = flags.FLAGS.visualize_population_periodicity
    if flags.FLAGS.visualize_population_backup_dir_path is not None:
        visualize_population_backup_dir_path = flags.FLAGS.visualize_population_backup_dir_path

    # Set logging, parse config
    logging.set_verbosity(logging_level)
    config = tfne.parse_configuration(config_path)

    # Initialize the environment as well as the specific NE algorithm
    environment = tfne.environments.XOREnvironment(config, weight_training_eval=True)
    ne_algorithm = tfne.CoDeepNEAT(config)

    # Initialize and set up backup agents
    ba_save_best_genome = tfne.backup_agents.SaveBestGenome(save_best_genome_periodicity,
                                                            save_best_genome_backup_dir_path)
    ba_save_population = tfne.backup_agents.SavePopulation(save_population_periodicity,
                                                           save_population_backup_dir_path)
    ba_visualize_best_genome = tfne.backup_agents.VisualizeBestGenome(visualize_best_genome_periodicity,
                                                                      visualize_best_genome_backup_dir_path)
    ba_visualize_population = tfne.backup_agents.VisualizePopulation(visualize_population_periodicity,
                                                                     visualize_population_backup_dir_path)
    backup_agents = (ba_save_best_genome, ba_save_population, ba_visualize_best_genome, ba_visualize_population)

    # Supply configuration and initialized NE elements to the evolution engine
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
    best_genome.visualize(view=True, save_dir='./')
    best_genome.save_genotype(save_dir='./')
    best_genome.save_model(save_dir='./')


if __name__ == '__main__':
    app.run(codeepneat_example)
