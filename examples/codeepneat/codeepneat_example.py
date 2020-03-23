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
flags.DEFINE_integer('backup_periodicity_best_genome',
                     default=None, help='TBD')
flags.DEFINE_string('backup_dir_path_best_genome',
                    default=None, help='TBD')
flags.DEFINE_integer('backup_periodicity_population',
                     default=None, help='TBD')
flags.DEFINE_string('backup_dir_path_population',
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
    backup_periodicity_best_genome = 1
    backup_dir_path_best_genome = './backups_best_genome/'
    backup_periodicity_population = 5
    backup_dir_path_population = './backups_population/'

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
    if flags.FLAGS.backup_periodicity_best_genome is not None:
        backup_periodicity_best_genome = flags.FLAGS.backup_periodicity_best_genome
    if flags.FLAGS.backup_dir_path_best_genome is not None:
        backup_dir_path_best_genome = flags.FLAGS.backup_dir_path_best_genome
    if flags.FLAGS.backup_periodicity_population is not None:
        backup_periodicity_population = flags.FLAGS.backup_periodicity_population
    if flags.FLAGS.backup_dir_path_population is not None:
        backup_dir_path_population = flags.FLAGS.backup_dir_path_population

    # Set logging, parse config
    logging.set_verbosity(logging_level)
    config = tfne.parse_configuration(config_path)

    # Initialize the environment as well as the specific NE algorithm
    environment = tfne.environments.XOREnvironment()
    ne_algorithm = tfne.CoDeepNEAT(config)

    # If periodic backups of genomes or the population are desired, initialize backup agents
    backup_agent_best_genome = tfne.BackupAgentBestGenome(backup_periodicity_best_genome, backup_dir_path_best_genome)
    backup_agent_population = tfne.BackupAgentPopulation(backup_periodicity_population, backup_dir_path_population)
    backup_agents = (backup_agent_best_genome, backup_agent_population)

    # Supply configuration and initialized NE elements to the evolution engine
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  config=config,
                                  num_cpus=num_cpus,
                                  num_gpus=num_gpus,
                                  max_generations=max_generations,
                                  max_fitness=max_fitness,
                                  backup_agents=backup_agents)

    # Start training process, returning the best genome when training ends
    best_genome = engine.train()

    if best_genome is not None:
        # Show string representation of best genome, visualize it and then save it
        print("Best Genome returned by evolution:\n")
        print(best_genome)
        best_genome.visualize(view=True, save_dir='./')
        best_genome.save_genotype(save_dir='./')
        best_genome.save_model(save_dir='./')
    else:
        print("Evolutionary process was not able to return an eligible genome")


if __name__ == '__main__':
    app.run(codeepneat_example)
