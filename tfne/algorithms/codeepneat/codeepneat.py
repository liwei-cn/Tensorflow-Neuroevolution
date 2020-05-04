import sys
import math
import random
import statistics

import numpy as np
from absl import logging

import tfne
from .codeepneat_optimizer_factories import SGDFactory
from ..base_algorithm import BaseNeuroevolutionAlgorithm
from ...encodings.codeepneat.codeepneat_genome import CoDeepNEATGenome
from ...encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprintNode
from ...helper_functions import round_to_nearest_multiple, read_option_from_config


class CoDeepNEAT(BaseNeuroevolutionAlgorithm):
    """"""

    def __init__(self, config, environment_factory, initial_population_file_path=None):
        """"""

        # TODO
        return

        '''
        
    def register_environment(self, environment):
        """"""
        # Check if the registered environment is weight training
        if not environment.set_weight_training(True):
            raise AssertionError("The registered environment '{}' is not weight training, which is a requirement for "
                                 "CoDeepNEAT as CoDeepNEAT is specified to first train the weights of created genome "
                                 "phenotypes for a set amount of epochs before assigning a fitness score"
                                 .format(environment))


        if not logging.level_debug():
            verbosity = 0
        else:
            verbosity = 1

        # Register environment and its input/output shapes
        self.environment = environment
        self.input_shape = environment.get_input_shape()
        self.output_shape = environment.get_output_shape()
        
        '''

        # Read and process the supplied config and register the possibly supplied initial population
        self._process_config(config)
        self.initial_population_file_path = initial_population_file_path

        # Initialize and register the CoDeepNEAT encoding
        self.encoding = tfne.encodings.CoDeepNEATEncoding(dtype=self.dtype)

        # Declare variables of the environment and its input/output shape that will be initialized ones the environment
        # is registered
        self.environment = None
        self.input_shape = None
        self.output_shape = None

        # Declare internal variables of the population
        self.generation_counter = None
        self.best_genome = None
        self.best_fitness = None

        # Declare and initialize internal variables concerning the module population of the CoDeepNEAT algorithm
        self.modules = dict()
        self.mod_species = dict()
        self.mod_species_type = dict()
        self.mod_species_counter = 0

        # Declare and initialize internal variables concerning the blueprint population of the CoDeepNEAT algorithm
        self.blueprints = dict()
        self.bp_species = dict()
        self.bp_species_counter = 0

    def _process_config(self, config):
        """"""
        # Read and process the general config values for the CoDeepNEAT algorithm
        self.dtype = read_option_from_config(config, 'GENERAL', 'dtype')
        self.mod_pop_size = read_option_from_config(config, 'GENERAL', 'mod_pop_size')
        self.bp_pop_size = read_option_from_config(config, 'GENERAL', 'bp_pop_size')
        self.genomes_per_bp = read_option_from_config(config, 'GENERAL', 'genomes_per_bp')

        # Read and process the config values that concern the created genomes for the CoDeepNEAT algorithm
        self.available_modules = read_option_from_config(config, 'GENOME', 'available_modules')
        self.available_optimizers = read_option_from_config(config, 'GENOME', 'available_optimizers')
        self.output_activations = read_option_from_config(config, 'GENOME', 'output_activations')

        # Read and process the speciation config values for modules in the CoDeepNEAT algorithm
        self.mod_speciation_type = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_speciation_type')
        if self.mod_speciation_type == 'Basic':
            pass
        elif self.mod_speciation_type == 'k-means':
            # TODO
            raise NotImplementedError("MOD speciation type of k-means not yet implemented")
        else:
            raise NotImplementedError("MOD speciation type of '{}' not implemented".format(self.mod_speciation_type))

        # Read and process the selection config values for modules in the CoDeepNEAT algorithm
        if config.has_option('MODULE_SELECTION', 'mod_removal'):
            self.mod_removal_type = 'fixed'
            self.mod_removal = read_option_from_config(config, 'MODULE_SELECTION', 'mod_removal')
        elif config.has_option('MODULE_SELECTION', 'mod_removal_threshold'):
            self.mod_removal_type = 'threshold'
            self.mod_removal_threshold = read_option_from_config(config, 'MODULE_SELECTION', 'mod_removal_threshold')
        else:
            raise KeyError("'MODLUE_SELECTION/mod_removal' or 'MODLUE_SELECTION/mod_removal_threshold' not specified")
        if config.has_option('MODULE_SELECTION', 'mod_elitism'):
            self.mod_elitism_type = 'fixed'
            self.mod_elitism = read_option_from_config(config, 'MODULE_SELECTION', 'mod_elitism')
        elif config.has_option('MODULE_SELECTION', 'mod_elitism_threshold'):
            self.mod_elitism_type = 'threshold'
            self.mod_elitism_threshold = read_option_from_config(config, 'MODULE_SELECTION', 'mod_elitism_threshold')
        else:
            raise KeyError("'MODULE_SELECTION/mod_elitism' or 'MODULE_SELECTION/mod_elitism_threshold' not specified")

        # Read and process the evolution config values for modules in the CoDeepNEAT algorithm
        self.mod_max_mutation = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_max_mutation')
        self.mod_mutation = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_mutation')
        self.mod_crossover = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_crossover')
        if round(self.mod_mutation + self.mod_crossover, 4) != 1.0:
            raise KeyError("'MODULE_EVOLUTION/mod_mutation' and 'MODULE_EVOLUTION/mod_crossover' values "
                           "dont add up to 1")

        # Read and process the speciation config values for blueprints in the CoDeepNEAT algorithm
        self.bp_speciation_type = read_option_from_config(config, 'BP_SPECIATION', 'bp_speciation_type')
        if self.bp_speciation_type is None:
            pass
        elif self.bp_speciation_type == 'fixed-threshold':
            # TODO
            raise NotImplementedError("BP speciation type of fixed-threshold not yet implemented")
        elif self.bp_speciation_type == 'dynamic-threshold':
            # TODO
            raise NotImplementedError("BP speciation type of dynamic-threshold not yet implemented")
        elif self.bp_speciation_type == 'k-means':
            # TODO
            raise NotImplementedError("BP speciation type of k-means not yet implemented")
        else:
            raise NotImplementedError("BP speciation type of '{}' not implemented".format(self.bp_speciation_type))

        # Read and process the selection config values for blueprints in the CoDeepNEAT algorithm
        if config.has_option('BP_SELECTION', 'bp_removal'):
            self.bp_removal_type = 'fixed'
            self.bp_removal = read_option_from_config(config, 'BP_SELECTION', 'bp_removal')
        elif config.has_option('BP_SELECTION', 'bp_removal_threshold'):
            self.bp_removal_type = 'threshold'
            self.bp_removal_threshold = read_option_from_config(config, 'BP_SELECTION', 'bp_removal_threshold')
        else:
            raise KeyError("'BP_SELECTION/bp_removal' or 'BP_SELECTION/bp_removal_threshold' not specified")
        if config.has_option('BP_SELECTION', 'bp_elitism'):
            self.bp_elitism_type = 'fixed'
            self.bp_elitism = read_option_from_config(config, 'BP_SELECTION', 'bp_elitism')
        elif config.has_option('BP_SELECTION', 'bp_elitism_threshold'):
            self.bp_elitism_type = 'threshold'
            self.bp_elitism_threshold = read_option_from_config(config, 'BP_SELECTION', 'bp_elitism_threshold')
        else:
            raise KeyError("'BP_SELECTION/bp_elitism' or 'BP_SELECTION/bp_elitism_threshold' not specified")

        # Read and process the evolution config values for blueprints in the CoDeepNEAT algorithm
        self.bp_max_mutation = read_option_from_config(config, 'BP_EVOLUTION', 'bp_max_mutation')
        self.bp_mutation_add_conn = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_add_conn')
        self.bp_mutation_add_node = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_add_node')
        self.bp_mutation_remove_conn = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_remove_conn')
        self.bp_mutation_remove_node = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_remove_node')
        self.bp_mutation_node_species = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_node_species')
        self.bp_mutation_hp = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_hp')
        self.bp_crossover = read_option_from_config(config, 'BP_EVOLUTION', 'bp_crossover')
        if round(self.bp_mutation_add_conn + self.bp_mutation_add_node + self.bp_mutation_remove_conn
                 + self.bp_mutation_remove_node + self.bp_mutation_node_species + self.bp_mutation_hp
                 + self.bp_crossover, 4) != 1.0:
            raise KeyError("'BP_EVOLUTION/bp_mutation_*' and 'BP_EVOLUTION/bp_crossover' values dont add up to 1")

        # Read and process the hyperparameter config values for each optimizer
        for available_opt in self.available_optimizers:
            if available_opt == 'SGD':
                if not config.has_section('SGD_HP'):
                    raise KeyError("'SGD' optimizer set to be available in 'GENOME/available_optimizers', though "
                                   "config does not have a 'SGD_HP' section")
                # Read and process the hyperparameter config values for the SGD optimizer in the CoDeepNEAT algorithm
                self.sgd_learning_rate = read_option_from_config(config, 'SGD_HP', 'learning_rate')
                self.sgd_momentum = read_option_from_config(config, 'SGD_HP', 'momentum')
                self.sgd_nesterov = read_option_from_config(config, 'SGD_HP', 'nesterov')

                # Create Standard Deviation values for range specified global HP config values
                if len(self.sgd_learning_rate) == 4:
                    self.sgd_learning_rate_stddev = float(self.sgd_learning_rate[1] - self.sgd_learning_rate[0]) / \
                                                    self.sgd_learning_rate[3]
                else:
                    self.sgd_learning_rate_stddev = float(self.sgd_learning_rate[1] - self.sgd_learning_rate[0]) / 4
                if len(self.sgd_momentum) == 4:
                    self.sgd_momentum_stddev = float(self.sgd_momentum[1] - self.sgd_momentum[0]) / self.sgd_momentum[3]
                else:
                    self.sgd_momentum_stddev = float(self.sgd_momentum[1] - self.sgd_momentum[0]) / 4
            else:
                raise KeyError("'{}' optimizer set to be available in 'GENOME/available_optimizers', though handling "
                               "of this module has not yet been implemented".format(available_opt))

        # Read and process the hyperparameter config values for each module
        for available_mod in self.available_modules:
            if available_mod == 'DENSE':
                if not config.has_section('MODULE_DENSE_HP'):
                    raise KeyError("'DENSE' module set to be available in 'GENOME/available_modules', though config "
                                   "does not have a 'MODULE_DENSE_HP' section")
                # Read and process the hyperparameter config values for Dense modules in the CoDeepNEAT algorithm
                self.dense_merge_methods = read_option_from_config(config, 'MODULE_DENSE_HP', 'merge_methods')
                self.dense_units = read_option_from_config(config, 'MODULE_DENSE_HP', 'units')
                self.dense_activations = read_option_from_config(config, 'MODULE_DENSE_HP', 'activations')
                self.dense_kernel_initializers = read_option_from_config(config,
                                                                         'MODULE_DENSE_HP',
                                                                         'kernel_initializers')
                self.dense_bias_initializers = read_option_from_config(config, 'MODULE_DENSE_HP', 'bias_initializers')
                self.dense_dropout_probability = read_option_from_config(config,
                                                                         'MODULE_DENSE_HP',
                                                                         'dropout_probability')
                self.dense_dropout_rate = read_option_from_config(config, 'MODULE_DENSE_HP', 'dropout_rate')

                # Create Standard Deviation values for range specified Dense Module HP config values
                if len(self.dense_units) == 4:
                    self.dense_units_stddev = float(self.dense_units[1] - self.dense_units[0]) / self.dense_units[3]
                else:
                    self.dense_units_stddev = float(self.dense_units[1] - self.dense_units[0]) / 4
                if len(self.dense_dropout_rate) == 4:
                    self.dense_dropout_rate_stddev = float(self.dense_dropout_rate[1] - self.dense_dropout_rate[0]) / \
                                                     self.dense_dropout_rate[3]
                else:
                    self.dense_dropout_rate_stddev = float(self.dense_dropout_rate[1] - self.dense_dropout_rate[0]) / 4
            else:
                raise KeyError("'{}' module set to be available in 'GENOME/available_modules', though handling of "
                               "this module has not yet been implemented".format(available_mod))

    def initialize_population(self):
        """"""

        if self.initial_population_file_path is None:
            print("Initializing a new population of {} blueprints and {} modules..."
                  .format(self.bp_pop_size, self.mod_pop_size))

            # Set internal variables of the population to the initialization of a new population
            self.generation_counter = 0
            self.best_fitness = 0

            #### Initialize Module Population ####
            # Initialize module population with a basic speciation scheme, only speciating modules according to their
            # module type. Each module species (and therefore module type) is initiated with the same amount of modules
            # (or close to the same amount if module pop size not evenly divisble). When parameters of all initial
            # modules are uniform randomly chosen.

            # Set initial species counter of basic speciation, initialize module species list and map each species to
            # its type
            for mod_type in self.available_modules:
                self.mod_species_counter += 1
                self.mod_species[self.mod_species_counter] = list()
                self.mod_species_type[self.mod_species_counter] = mod_type

            for i in range(self.mod_pop_size):
                # Decide on for which species a new module is added (uniformly distributed)
                chosen_species = (i % self.mod_species_counter) + 1

                # Initialize a new module according to the module type of the chosen species
                if self.mod_species_type[chosen_species] == 'DENSE':
                    # Uniform randomly choose parameters of DENSE module
                    chosen_merge_method = random.choice(self.dense_merge_methods)
                    chosen_units_uniform = random.randint(self.dense_units[0], self.dense_units[1])
                    chosen_units = round_to_nearest_multiple(chosen_units_uniform, self.dense_units[0],
                                                             self.dense_units[1], self.dense_units[2])
                    chosen_activation = random.choice(self.dense_activations)
                    chosen_kernel_initializer = random.choice(self.dense_kernel_initializers)
                    chosen_bias_initializer = random.choice(self.dense_bias_initializers)

                    # Decide according to dropout_probabilty if there will be a dropout layer at all
                    if random.random() < self.dense_dropout_probability:
                        chosen_dropout_rate_uniform = random.uniform(self.dense_dropout_rate[0],
                                                                     self.dense_dropout_rate[1])
                        chosen_dropout_rate = round(round_to_nearest_multiple(chosen_dropout_rate_uniform,
                                                                              self.dense_dropout_rate[0],
                                                                              self.dense_dropout_rate[1],
                                                                              self.dense_dropout_rate[2]), 4)
                    else:
                        chosen_dropout_rate = None

                    # Create module
                    module_id, module = self.encoding.create_dense_module(merge_method=chosen_merge_method,
                                                                          units=chosen_units,
                                                                          activation=chosen_activation,
                                                                          kernel_initializer=chosen_kernel_initializer,
                                                                          bias_initializer=chosen_bias_initializer,
                                                                          dropout_rate=chosen_dropout_rate)
                elif self.mod_species_type[chosen_species] == 'LSTM':
                    raise NotImplementedError()
                else:
                    raise RuntimeError("Species type '{}' could not be identified during initialization"
                                       .format(self.mod_species_type[chosen_species]))

                # Append newly created module to module container and to according species
                self.modules[module_id] = module
                self.mod_species[chosen_species].append(module_id)

            #### Initialize Blueprint Population ####
            # Initialize blueprint population with a minimal blueprint graph, only consisting of an input node (with
            # not species or the 'input' species respectively) and a single output node, having a randomly assigned
            # species. All hyperparameters of the blueprint are uniform randomly chosen. All blueprints are not
            # speciated in the beginning and are assigned to species 1.

            # Initialize blueprint species list and create tuple of the possible species the output node can take on
            self.bp_species[1] = list()
            available_mod_species = tuple(self.mod_species.keys())

            for _ in range(self.bp_pop_size):
                # Determine the module species of the output (and only) node
                output_node_species = random.choice(available_mod_species)

                # Create a minimal blueprint graph with an input node, an output node with the chosen module species and
                # a connection between them
                blueprint_graph = dict()
                gene_id, gene = self.encoding.create_blueprint_node(node=1, species=None)
                blueprint_graph[gene_id] = gene
                gene_id, gene = self.encoding.create_blueprint_node(node=2, species=output_node_species)
                blueprint_graph[gene_id] = gene
                gene_id, gene = self.encoding.create_blueprint_conn(conn_start=1, conn_end=2)
                blueprint_graph[gene_id] = gene

                # Uniform randomly choose an output activation
                chosen_output_activation = random.choice(self.output_activations)

                # Uniform Randomly choose an optimizer and its hyperparameters for the blueprint. Then create a factory
                # for that optimizer.
                chosen_optimizer = random.choice(self.available_optimizers)
                if chosen_optimizer == 'SGD':
                    chosen_learning_rate_uniform = random.uniform(self.sgd_learning_rate[0], self.sgd_learning_rate[1])
                    chosen_learning_rate = round(round_to_nearest_multiple(chosen_learning_rate_uniform,
                                                                           self.sgd_learning_rate[0],
                                                                           self.sgd_learning_rate[1],
                                                                           self.sgd_learning_rate[2]), 4)
                    chosen_momentum_uniform = random.uniform(self.sgd_momentum[0], self.sgd_momentum[1])
                    chosen_momentum = round(round_to_nearest_multiple(chosen_momentum_uniform,
                                                                      self.sgd_momentum[0],
                                                                      self.sgd_momentum[1],
                                                                      self.sgd_momentum[2]), 4)
                    chosen_nesterov = random.choice(self.sgd_nesterov)

                    optimizer_factory = SGDFactory(learning_rate=chosen_learning_rate,
                                                   momentum=chosen_momentum,
                                                   nesterov=chosen_nesterov)
                elif chosen_optimizer == 'RMSprop':
                    raise NotImplementedError()
                else:
                    raise RuntimeError("Optimizer type '{}' could not be identified during initialization"
                                       .format(chosen_optimizer))

                # Create blueprint
                blueprint_id, blueprint = self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                                                         output_shape=self.output_shape,
                                                                         output_activation=chosen_output_activation,
                                                                         optimizer_factory=optimizer_factory)

                # Append newly create blueprint to blueprint container and to only initial blueprint species
                self.blueprints[blueprint_id] = blueprint
                self.bp_species[1].append(blueprint_id)

        else:
            raise NotImplementedError("Initializing population with a pre-evolved supplied initial population not yet "
                                      "implemented")

    def evaluate_population(self) -> (int, int):
        """"""
        # Create container collecting the fitness of the genomes that involve specific modules. Calculate the average
        # fitness of the genomes in which a module is involved in later and assign it as the module's fitness
        mod_genome_fitness = dict()

        # Initialize Progress counter variables for evaluate population progress bar. Print notice of evaluation start
        genome_pop_size = self.bp_pop_size * self.genomes_per_bp
        genome_eval_counter = 0
        genome_eval_counter_div = round(genome_pop_size / 40.0, 4)
        print("\nEvaluating {} genomes in generation {}...".format(genome_pop_size, self.generation_counter))

        for blueprint in self.blueprints.values():
            bp_mod_species = blueprint.get_species()

            # Create container collecting the fitness of the genomes that involve that specific blueprint.
            bp_genome_fitness = list()

            for _ in range(self.genomes_per_bp):
                # Assemble genome by first uniform randomly choosing a specific module from the species that the
                # blueprint nodes are referring to.
                bp_assigned_module_ids = dict()
                bp_assigned_modules = dict()
                for i in bp_mod_species:
                    chosen_module_id = random.choice(self.mod_species[i])
                    bp_assigned_module_ids[i] = chosen_module_id
                    bp_assigned_modules[i] = self.modules[chosen_module_id]

                # Create genome, using the specific blueprint, a dict of modules for each species and the current
                # generation
                genome_id, genome = self.encoding.create_genome(blueprint, bp_assigned_modules, self.generation_counter)

                # Now evaluate genome on registered environment
                genome_fitness = self.environment.eval_genome_fitness(genome)

                # Print population evaluation progress bar
                genome_eval_counter += 1
                progress_mult = int(round(genome_eval_counter / genome_eval_counter_div, 4))
                print_str = "\r[{:40}] {}/{} Genomes | Genome ID {} achieved fitness of {}".format("=" * progress_mult,
                                                                                                   genome_eval_counter,
                                                                                                   genome_pop_size,
                                                                                                   genome_id,
                                                                                                   genome_fitness)
                sys.stdout.write(print_str)
                sys.stdout.flush()

                # Assign the genome fitness to the blueprint and all modules used for the creation of the genome
                bp_genome_fitness.append(genome_fitness)
                for mod_id in bp_assigned_module_ids.values():
                    if mod_id in mod_genome_fitness:
                        mod_genome_fitness[mod_id].append(genome_fitness)
                    else:
                        mod_genome_fitness[mod_id] = [genome_fitness]

                # Acutally assign fitness to the genome and register it as new best if it exhibits better fitness
                if genome_fitness > self.best_fitness:
                    genome.set_fitness(genome_fitness)
                    self.best_genome = genome
                    self.best_fitness = genome_fitness

            # Average out collected fitness of genomes the blueprint was invovled in. Then assign to blueprint
            bp_genome_fitness_avg = round(statistics.mean(bp_genome_fitness), 4)
            blueprint.set_fitness(bp_genome_fitness_avg)

        # Average out collected fitness of genomes each module was invovled in. Then assign to module
        for mod_id, mod_genome_fitness_list in mod_genome_fitness.items():
            mod_genome_fitness_avg = round(statistics.mean(mod_genome_fitness_list), 4)
            self.modules[mod_id].set_fitness(mod_genome_fitness_avg)

        return self.generation_counter, self.best_fitness

    def summarize_population(self):
        """"""

        # Calculate average fitnesses of each module species and total module pop. Determine best module of each species
        mod_species_avg_fitness = dict()
        mod_species_best_id = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            spec_total_fitness = 0
            mod_species_best_id[spec_id] = None
            for mod_id in spec_mod_ids:
                mod_fitness = self.modules[mod_id].get_fitness()
                spec_total_fitness += mod_fitness
                if mod_species_best_id[spec_id] is None or mod_fitness > mod_species_best_id[spec_id]:
                    mod_species_best_id[spec_id] = mod_id
            mod_species_avg_fitness[spec_id] = round(spec_total_fitness / len(spec_mod_ids), 4)
        modules_avg_fitness = round(sum(mod_species_avg_fitness.values()) / len(mod_species_avg_fitness), 4)

        # Calculate average fitnesses of each bp species and total bp pop. Determine best bp of each species
        bp_species_avg_fitness = dict()
        bp_species_best_id = dict()
        for spec_id, spec_bp_ids in self.bp_species.items():
            spec_total_fitness = 0
            bp_species_best_id[spec_id] = None
            for bp_id in spec_bp_ids:
                bp_fitness = self.blueprints[bp_id].get_fitness()
                spec_total_fitness += bp_fitness
                if bp_species_best_id[spec_id] is None or bp_fitness > bp_species_best_id[spec_id]:
                    bp_species_best_id[spec_id] = bp_id
            bp_species_avg_fitness[spec_id] = round(spec_total_fitness / len(spec_bp_ids), 4)
        blueprints_avg_fitness = round(sum(bp_species_avg_fitness.values()) / len(bp_species_avg_fitness), 4)

        # Print summary header
        print("\n\n\n\033[1m{}  Population Summary  {}\n\n"
              "Generation: {:>4}  ||  Best Genome Fitness: {:>8}  ||  Average Blueprint Fitness: {:>8}  ||  "
              "Average Module Fitness: {:>8}\033[0m\n"
              "Best Genome: {}\n".format('#' * 60,
                                         '#' * 60,
                                         self.generation_counter,
                                         self.best_fitness,
                                         blueprints_avg_fitness,
                                         modules_avg_fitness,
                                         self.best_genome))

        # Print summary of blueprint species
        print("\033[1mBP Species                || BP Species Avg Fitness                || BP Species Size\n"
              "Best BP of Species\033[0m")
        for spec_id, spec_bp_avg_fitness in bp_species_avg_fitness.items():
            print("{:>6}                    || {:>8}                              || {:>8}\n{}"
                  .format(spec_id,
                          spec_bp_avg_fitness,
                          len(self.bp_species[spec_id]),
                          self.blueprints[bp_species_best_id[spec_id]]))

        # Print summary of module species
        print("\n\033[1mModule Species            || Module Species Avg Fitness            || "
              "Module Species Size\nBest Module of Species\033[0m")
        for spec_id, spec_mod_avg_fitness in mod_species_avg_fitness.items():
            print("{:>6}                    || {:>8}                              || {:>8}\n{}"
                  .format(spec_id,
                          spec_mod_avg_fitness,
                          len(self.mod_species[spec_id]),
                          self.modules[mod_species_best_id[spec_id]]))

        # Print summary footer
        print("\n\033[1m" + '#' * 142 + "\033[0m\n")

    def evolve_population(self) -> bool:
        """"""
        #### Speciate Modules ####
        if self.mod_speciation_type == 'Basic':
            # Population is already speciated solely according to their module type. As speciation is configured to
            # 'Basic' is therefore no further speciation necessary
            pass

            #### New Modules Species Size Calculation ####
            # Determine the intended size for the module species after evolution, depending on the average fitness of
            # the current modules in the species. This is performed before the species is selected to accurately judge
            # the fitness of the whole species and not only of the fittest members of the species.

            # Determine the specific average fitness for each species as well as the total fitness of all species
            species_fitness = dict()
            for spec_id, spec_mod_ids in self.mod_species.items():
                spec_fitness = 0
                for module_id in spec_mod_ids:
                    spec_fitness += self.modules[module_id].get_fitness()
                species_fitness[spec_id] = spec_fitness
            total_fitness = sum(species_fitness.values())

            # Calculate the new_mod_species_size depending on the species fitness share of the total fitness. Decimal
            # places are floored and a species is given an additional slot in the new size if it has the highest
            # decimal point value that was floored.
            new_mod_species_size = dict()
            new_mod_species_size_rest = dict()
            current_total_size = 0
            for spec_id, spec_fitness in species_fitness.items():
                spec_size_float = (spec_fitness / total_fitness) * self.mod_pop_size
                spec_size_floored = math.floor(spec_size_float)

                new_mod_species_size[spec_id] = spec_size_floored
                new_mod_species_size_rest[spec_id] = spec_size_float - spec_size_floored
                current_total_size += spec_size_floored

            # Sort the decimal point values that were cut and award the species with the highest floored decimal point
            # values an additional slot in the new module species size, until preset module population size is reached
            increment_order = sorted(new_mod_species_size_rest.keys(), key=new_mod_species_size_rest.get, reverse=True)
            increment_counter = 0
            while current_total_size < self.mod_pop_size:
                new_mod_species_size[increment_order[increment_counter]] += 1
                increment_counter += 1
                current_total_size += 1
        else:
            raise NotImplementedError("Other module speciation method than 'Basic' not yet implemented")

        #### Select Modules ####
        # Remove low performing modules and carry over the top modules of each species according to specified elitism
        new_mod_species = dict()
        mod_ids_to_remove_after_reproduction = []
        for spec_id, spec_mod_ids in self.mod_species.items():
            # Determine number of modules to remove and to carry over as integers
            spec_size = len(spec_mod_ids)
            if self.mod_removal_type == 'fixed':
                modules_to_remove = self.mod_removal
            else:  # if self.mod_removal_type == 'threshold':
                modules_to_remove = math.floor(spec_size * self.mod_removal_threshold)
            if self.mod_elitism_type == 'fixed':
                modules_to_carry_over = self.mod_elitism
            else:  # if self.mod_elitism_type == 'threshold':
                modules_to_carry_over = spec_size - math.ceil(spec_size * self.mod_elitism_threshold)

            # Elitism has higher precedence than removal. Therefore, if more modules are carried over than are present
            # in the species (can occur if elitism specified as absolut integer), carry over all modules without
            # removing anything. If more modules are to be removed and carried over than are present in the species,
            # decrease in modules that are to be removed.
            if modules_to_carry_over >= spec_size:
                modules_to_remove = 0
                modules_to_carry_over = spec_size
            elif modules_to_remove + modules_to_carry_over > spec_size:
                modules_to_remove = spec_size - modules_to_carry_over

            # Sort species module ids according to their fitness in order to remove/carry over the determined amount of
            # modules.
            spec_mod_ids_sorted = sorted(spec_mod_ids, key=lambda x: self.modules[x].get_fitness(), reverse=True)

            # Carry over fittest modules to new mod_species, according to elitism
            new_mod_species[spec_id] = spec_mod_ids_sorted[:modules_to_carry_over]

            # Save module ids that remain in the module species for reproduction, serving as potential parents, but have
            # to be deleted afterwards as they are also not fit enough to be carried over right away
            mod_ids_to_remove_after_reproduction += spec_mod_ids_sorted[modules_to_carry_over:-modules_to_remove]

            # Delete low performing modules from memory of the module container as well as from the module species
            # assignment
            if modules_to_remove > 0:
                # Remove modules
                module_ids_to_remove = spec_mod_ids_sorted[-modules_to_remove:]
                for mod_id_to_remove in module_ids_to_remove:
                    del self.modules[mod_id_to_remove]

                # Assign module id list without low performing modules to species
                self.mod_species[spec_id] = spec_mod_ids_sorted[:-modules_to_remove]

        #### Speciate Blueprints ####
        if self.bp_speciation_type is None:
            # Explicitely don't speciate the blueprints and leave them all assigned to species 1
            pass

            # Set the intended size of the blueprints species after evolution to the same size, as no speciation takes
            # place
            new_bp_species_size = {1: self.bp_pop_size}
        else:
            raise NotImplementedError("Other blueprint speciation method than 'None' not yet implemented")

        #### Select Blueprints ####
        # Remove low performing blueprints and carry over the top blueprints of each species according to specified
        # elitism
        new_bp_species = dict()
        bp_ids_to_remove_after_reproduction = []
        for spec_id, spec_bp_ids in self.bp_species.items():
            # Determine number of blueprints to remove and to carry over as integers
            spec_size = len(spec_bp_ids)
            if self.bp_removal_type == 'fixed':
                blueprints_to_remove = self.bp_removal
            else:  # if self.bp_removal_type == 'threshold':
                blueprints_to_remove = math.floor(spec_size * self.bp_removal_threshold)
            if self.bp_elitism_type == 'fixed':
                blueprints_to_carry_over = self.bp_elitism
            else:  # if self.bp_elitism_type == 'threshold':
                blueprints_to_carry_over = spec_size - math.ceil(spec_size * self.bp_elitism_threshold)

            # Elitism has higher precedence than removal. Therefore, if more blueprints are carried over than are
            # present in the species (can occur if elitism specified as absolut integer), carry over all blueprints
            # without removing anything. If more blueprints are to be removed and carried over than are present in the
            # species, decrease in blueprints that are to be removed.
            if blueprints_to_carry_over >= spec_size:
                blueprints_to_remove = 0
                blueprints_to_carry_over = spec_size
            elif blueprints_to_remove + blueprints_to_carry_over > spec_size:
                blueprints_to_remove = spec_size - blueprints_to_carry_over

            # Sort species blueprint ids according to their fitness in order to remove/carry over the determined amount
            # of blueprints
            spec_bp_ids_sorted = sorted(spec_bp_ids, key=lambda x: self.blueprints[x].get_fitness(), reverse=True)

            # Carry over fittest blueprints to new blueprint_species, according to elitism
            new_bp_species[spec_id] = spec_bp_ids_sorted[:blueprints_to_carry_over]

            # Save blueprint ids that remain in the blueprint species for reproduction, serving as potential parents,
            # but have to be deleted afterwards as they are also not fit enough to be carried over right away
            bp_ids_to_remove_after_reproduction += spec_bp_ids_sorted[blueprints_to_carry_over:-blueprints_to_remove]

            # Delete low performing blueprints from memory of the blueprint container as well as from the blueprint
            # species assignment
            if blueprints_to_remove > 0:
                # Remove blueprints
                blueprint_ids_to_remove = spec_bp_ids_sorted[-blueprints_to_remove:]
                for bp_id_to_remove in blueprint_ids_to_remove:
                    del self.blueprints[bp_id_to_remove]

                # Assign blueprint id list without low performing blueprints to species
                self.bp_species[spec_id] = spec_bp_ids_sorted[:-blueprints_to_remove]

        #### Evolve Modules ####
        # Traverse through the new module species and add new moduless according to the previously determined dedicated
        # offpsring
        for spec_id, carried_over_spec_mod_ids in new_mod_species.items():
            # Determine offspring and create according amount of new modules
            for _ in range(new_mod_species_size[spec_id] - len(carried_over_spec_mod_ids)):
                if random.random() < self.mod_mutation:
                    ## Create new module through mutation ##

                    # Determine chosen parent module and its parameters as well as the intensity of the mutation,
                    # meaning how many parent parameters will be perturbed.
                    parent_module = self.modules[random.choice(self.mod_species[spec_id])]
                    module_parameters = parent_module.duplicate_parameters()
                    mutation_intensity = random.uniform(0, 0.3)

                    if self.mod_species_type[spec_id] == 'DENSE':
                        # Determine explicit integer amount of parameters to be mutated, though minimum is 1
                        param_mutation_count = int(mutation_intensity * 6)
                        if param_mutation_count == 0:
                            param_mutation_count = 1

                        # Uniform randomly choose the parameters to be mutated
                        parameters_to_mutate = random.sample(range(6), k=param_mutation_count)

                        # Mutate parameters. Categorical parameters are chosen randomly from all available values.
                        # Sortable parameters are perturbed through a random normal distribution with the current value
                        # as mean and the config specified stddev
                        for param_to_mutate in parameters_to_mutate:
                            if param_to_mutate == 0:
                                module_parameters[0] = random.choice(self.dense_merge_methods)
                            elif param_to_mutate == 1:
                                perturbed_param = int(np.random.normal(loc=module_parameters[1],
                                                                       scale=self.dense_units_stddev))
                                module_parameters[1] = round_to_nearest_multiple(perturbed_param,
                                                                                 self.dense_units[0],
                                                                                 self.dense_units[1],
                                                                                 self.dense_units[2])
                            elif param_to_mutate == 2:
                                module_parameters[2] = random.choice(self.dense_activations)
                            elif param_to_mutate == 3:
                                module_parameters[3] = random.choice(self.dense_kernel_initializers)
                            elif param_to_mutate == 4:
                                module_parameters[4] = random.choice(self.dense_bias_initializers)
                            else:  # param_to_mutate == 5:
                                # If Module param 5 (dropout rate) is not None is there a 'dropout_probability' chance
                                # that the dropout parameter will be perturbed. Otherwise it will be set to None. If
                                # the dropout rate of the parent is None to begin with then there is a
                                # 'dropout_probability' chance that a new uniorm random dropout rate is created.
                                # otherweise it will remain None.
                                if module_parameters[5] is not None:
                                    if random.random() < self.dense_dropout_probability:
                                        perturbed_param = np.random.normal(loc=module_parameters[5],
                                                                           scale=self.dense_dropout_rate_stddev)
                                        module_parameters[5] = round_to_nearest_multiple(perturbed_param,
                                                                                         self.dense_dropout_rate[0],
                                                                                         self.dense_dropout_rate[1],
                                                                                         self.dense_dropout_rate[2])
                                    else:
                                        module_parameters[5] = None
                                else:
                                    if random.random() < self.dense_dropout_probability:
                                        dropout_rate_uniform = random.uniform(self.dense_dropout_rate[0],
                                                                              self.dense_dropout_rate[1])
                                        module_parameters[5] = round_to_nearest_multiple(dropout_rate_uniform,
                                                                                         self.dense_dropout_rate[0],
                                                                                         self.dense_dropout_rate[1],
                                                                                         self.dense_dropout_rate[2])

                        # Create new offpsring module with parent mutated parameters
                        new_mod_id, new_mod = self.encoding.create_dense_module(merge_method=module_parameters[0],
                                                                                units=module_parameters[1],
                                                                                activation=module_parameters[2],
                                                                                kernel_initializer=module_parameters[3],
                                                                                bias_initializer=module_parameters[4],
                                                                                dropout_rate=module_parameters[5])

                else:  # random.random() < self.mod_crossover + self.mod_mutation
                    ## Create new module through crossover ##

                    # Determine if 2 modules are available in current species, as is required for crossover
                    if len(self.mod_species[spec_id]) == 1:

                        # If Only 1 module in current species available as parent, create new module with identical
                        # parameters
                        parent_module = self.modules[random.choice(self.mod_species[spec_id])]
                        module_parameters = parent_module.duplicate_parameters()

                        if self.mod_species_type[spec_id] == 'DENSE':
                            # Create new offspring module with identical parent parameters
                            new_mod_id, new_mod = self.encoding.create_dense_module(merge_method=module_parameters[0],
                                                                                    units=module_parameters[1],
                                                                                    activation=module_parameters[2],
                                                                                    kernel_initializer=
                                                                                    module_parameters[3],
                                                                                    bias_initializer=
                                                                                    module_parameters[4],
                                                                                    dropout_rate=module_parameters[5])

                    else:
                        # Choose 2 random parent modules, both of them different
                        parent_module_1_id, parent_module_2_id = random.sample(self.mod_species[spec_id], k=2)

                        # Determine fitter parent and save parameters of 'fitter' and 'other' parent
                        if self.modules[parent_module_1_id].get_fitness() > \
                                self.modules[parent_module_2_id].get_fitness():
                            fitter_parent_params = self.modules[parent_module_1_id].duplicate_parameters()
                            other_parent_params = self.modules[parent_module_2_id].duplicate_parameters()
                        else:
                            fitter_parent_params = self.modules[parent_module_2_id].duplicate_parameters()
                            other_parent_params = self.modules[parent_module_1_id].duplicate_parameters()

                        if self.mod_species_type[spec_id] == 'DENSE':
                            # Crete offspring parameters by carrying over parameter of fitter parent for categorical
                            # parameters and calculating average parameter for sortable parameters
                            offspring_params = [None] * 6
                            offspring_params[0] = fitter_parent_params[0]
                            offspring_params[1] = round_to_nearest_multiple(int((fitter_parent_params[1] +
                                                                                 other_parent_params[1]) / 2),
                                                                            self.dense_units[0],
                                                                            self.dense_units[1],
                                                                            self.dense_units[2])
                            offspring_params[2] = fitter_parent_params[2]
                            offspring_params[3] = fitter_parent_params[3]
                            offspring_params[4] = fitter_parent_params[4]
                            if fitter_parent_params[5] is not None and other_parent_params[5] is not None:
                                # If both parents have defined a dropout rate, calculate the average
                                offspring_params[5] = round_to_nearest_multiple(((fitter_parent_params[5] +
                                                                                  other_parent_params[5]) / 2),
                                                                                self.dense_dropout_rate[0],
                                                                                self.dense_dropout_rate[1],
                                                                                self.dense_dropout_rate[2])
                            elif fitter_parent_params[5] is not None:
                                # If only fitter parent defined dropout rate, carry that dropout rate over
                                offspring_params[5] = fitter_parent_params[5]
                            else:
                                # If neither or only the lesser fit parent defined dropout rate, set dropout rate to
                                # None
                                offspring_params[5] = None

                            # Create new offspring module with crossed-over parental parameters
                            new_mod_id, new_mod = self.encoding.create_dense_module(merge_method=offspring_params[0],
                                                                                    units=offspring_params[1],
                                                                                    activation=offspring_params[2],
                                                                                    kernel_initializer=
                                                                                    offspring_params[3],
                                                                                    bias_initializer=
                                                                                    offspring_params[4],
                                                                                    dropout_rate=offspring_params[5])

                # Add newly created module to the module container and its according species
                self.modules[new_mod_id] = new_mod
                new_mod_species[spec_id].append(new_mod_id)

            # Delete all those modules remaining in the module container that were not previously removed or carried
            # over but remained in the species to serve as a potential parent for offspring
            for mod_id_to_remove in mod_ids_to_remove_after_reproduction:
                del self.modules[mod_id_to_remove]

        # As new module species dict has now been created, delete the old one and overwrite it with the new mod species
        self.mod_species = new_mod_species

        #### Evolve Blueprints ####
        # Calculate the brackets for a random_float to fall into in order for the specific evolutionary method to be
        # chosen
        bp_mutation_add_node_prob = self.bp_mutation_add_conn + self.bp_mutation_add_node
        bp_mutation_remove_conn_prob = bp_mutation_add_node_prob + self.bp_mutation_remove_conn
        bp_mutation_remove_node_prob = bp_mutation_remove_conn_prob + self.bp_mutation_remove_node
        bp_mutation_node_species_prob = bp_mutation_remove_node_prob + self.bp_mutation_node_species
        bp_mutation_hp_prob = bp_mutation_node_species_prob + self.bp_mutation_hp

        # Traverse through the new blueprint species and add new blueprints according to the previously determined
        # dedicated offpsring
        for spec_id, carried_over_spec_bp_ids in new_bp_species.items():
            # Determine offspring and create according amount of new blueprints
            for _ in range(new_bp_species_size[spec_id] - len(carried_over_spec_bp_ids)):
                random_float = random.random()
                if random_float < self.bp_mutation_add_conn:
                    ## Create new blueprint by adding connection ##

                    # Determine parent blueprint and its parameters as well as the intensity of the mutation, in this
                    # case the amount of connections added to the blueprint graph
                    parent_bp = self.blueprints[random.choice(self.bp_species[spec_id])]
                    blueprint_graph, _, output_activation, optimizer_factory = parent_bp.duplicate_parameters()
                    graph_topology = parent_bp.get_graph_topology()
                    mutation_intensity = random.uniform(0, 0.3)

                    # Traverse blueprint graph and collect tuples of connections as well as a list of all present nodes
                    bp_graph_conns = set()
                    bp_graph_nodes = list()
                    for gene in blueprint_graph.values():
                        if isinstance(gene, CoDeepNEATBlueprintNode):
                            bp_graph_nodes.append(gene.node)
                        elif gene.enabled:  # and isinstance(gene, CoDeepNEATBlueprintConn)
                            # Only consider a connection for bp_graph_conns if it is enabled
                            bp_graph_conns.add((gene.conn_start, gene.conn_end))

                    # Determine specifically how many connections will be added
                    conns_to_add_count = int(mutation_intensity * len(bp_graph_conns))
                    if conns_to_add_count == 0:
                        conns_to_add_count = 1

                    # Add connections in loop until sufficient amount added
                    added_conns_count = 0
                    while added_conns_count < conns_to_add_count:
                        # Choose random start node from all possible nodes. Remove it immediately such that the same
                        # start node is not used twice, ensuring more complex mutation and a safe loop termination in
                        # case that all connection additions are exhausted
                        start_node = random.choice(bp_graph_nodes)
                        bp_graph_nodes.remove(start_node)
                        if len(bp_graph_nodes) == 0:
                            break

                        # As graph currently only supports feedforward topologies, ensure that end node is topologically
                        # behind the start node
                        start_node_level = None
                        for i in range(len(graph_topology)):
                            if start_node in graph_topology[i]:
                                start_node_level = i
                                break

                        # Determine set of all possible end nodes that are behind the start node
                        possible_end_nodes = list(set().union(*graph_topology[start_node_level + 1:]))

                        # Traverse all possible end nodes randomly and create and add a blueprint connection to the
                        # blueprint graph if no connection tuple present yet
                        while len(possible_end_nodes) != 0:
                            end_node = random.choice(possible_end_nodes)
                            possible_end_nodes.remove(end_node)
                            if (start_node, end_node) not in bp_graph_conns:
                                gene_id, gene = self.encoding.create_blueprint_conn(conn_start=start_node,
                                                                                    conn_end=end_node)
                                blueprint_graph[gene_id] = gene
                                added_conns_count += 1

                    # Create new offpsring blueprint with parent mutated blueprint graph
                    new_bp_id, new_bp = self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                                                       output_shape=self.output_shape,
                                                                       output_activation=output_activation,
                                                                       optimizer_factory=optimizer_factory)

                elif random_float < bp_mutation_add_node_prob:
                    ## Create new blueprint by adding node ##

                    # Determine parent blueprint and its parameters as well as the intensity of the mutation, in this
                    # case the amount of nodes added to the blueprint graph
                    parent_bp = self.blueprints[random.choice(self.bp_species[spec_id])]
                    blueprint_graph, _, output_activation, optimizer_factory = parent_bp.duplicate_parameters()
                    mutation_intensity = random.uniform(0, 0.3)

                    # Identify all possible connections in blueprint graph that can be split by collecting ids. Also
                    # count nodes to determine intensity of mutation
                    node_count = 0
                    bp_graph_conn_ids = set()
                    for gene in blueprint_graph.values():
                        if isinstance(gene, CoDeepNEATBlueprintNode):
                            node_count += 1
                        elif gene.enabled:
                            bp_graph_conn_ids.add(gene.gene_id)

                    # Determine specifically how many nodes will be added
                    nodes_to_add_count = int(mutation_intensity * node_count)
                    if nodes_to_add_count == 0:
                        nodes_to_add_count = 1

                    # Uniform randomly choosen connections by ID that are to be split
                    gene_ids_to_split = random.sample(bp_graph_conn_ids, k=nodes_to_add_count)

                    # Determine possible species for new nodes
                    available_mod_species = tuple(self.mod_species.keys())

                    # Actually perform the split and adding of new node for all determined connections
                    for gene_id_to_split in gene_ids_to_split:
                        # Determine start and end node of connection and disable it
                        conn_start = blueprint_graph[gene_id_to_split].conn_start
                        conn_end = blueprint_graph[gene_id_to_split].conn_end
                        blueprint_graph[gene_id_to_split].set_enabled(False)

                        # Create a new unique node if connection has not yet been split by any other mutation. Otherwise
                        # create the same node. Choose species for new node randomly.
                        new_node = self.encoding.get_node_for_split(conn_start, conn_end)
                        new_species = random.choice(available_mod_species)

                        # Create the genes for the new node addition and add to the blueprint graph
                        gene_id, gene = self.encoding.create_blueprint_node(node=new_node, species=new_species)
                        blueprint_graph[gene_id] = gene
                        gene_id, gene = self.encoding.create_blueprint_conn(conn_start=conn_start, conn_end=new_node)
                        blueprint_graph[gene_id] = gene
                        gene_id, gene = self.encoding.create_blueprint_conn(conn_start=new_node, conn_end=conn_end)
                        blueprint_graph[gene_id] = gene

                    # Create new offpsring blueprint with parent mutated blueprint graph
                    new_bp_id, new_bp = self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                                                       output_shape=self.output_shape,
                                                                       output_activation=output_activation,
                                                                       optimizer_factory=optimizer_factory)

                elif random_float < bp_mutation_remove_conn_prob:
                    ## Create new blueprint by removing connection ##
                    raise NotImplementedError("Destructive mutations of Blueprints not yet implemented in CoDeepNEAT")

                elif random_float < bp_mutation_remove_node_prob:
                    ## Create new blueprint by removing node ##
                    raise NotImplementedError("Destructive mutations of Blueprints not yet implemented in CoDeepNEAT")

                elif random_float < bp_mutation_node_species_prob:
                    ## Create new blueprint by changing species of nodes ##

                    # Determine parent blueprint and its parameters as well as the intensity of the mutation, in this
                    # case the amount of nodes changed in the blueprint graph
                    parent_bp = self.blueprints[random.choice(self.bp_species[spec_id])]
                    blueprint_graph, _, output_activation, optimizer_factory = parent_bp.duplicate_parameters()
                    mutation_intensity = random.uniform(0, 0.3)

                    # Identify all non-Input nodes in the blueprint graph by ID
                    bp_graph_node_ids = set()
                    for gene in blueprint_graph.values():
                        if isinstance(gene, CoDeepNEATBlueprintNode) and gene.node != 1:
                            bp_graph_node_ids.add(gene.gene_id)

                    # Determine specifically how many nodes will be changed
                    nodes_to_change_count = int(mutation_intensity * len(bp_graph_node_ids))
                    if nodes_to_change_count == 0:
                        nodes_to_change_count = 1

                    # Uniform randomly choosen nodes by ID that will get a changed species
                    gene_ids_to_mutate = random.sample(bp_graph_node_ids, k=nodes_to_change_count)

                    # Determine possible species to mutate nodes into
                    available_mod_species = tuple(self.mod_species.keys())

                    # Actually perform the split and adding of new node for all determined connections
                    for gene_id_to_mutate in gene_ids_to_mutate:
                        # Randomly choose new species from available ones and assign species to blueprint graph
                        new_species = random.choice(available_mod_species)
                        blueprint_graph[gene_id_to_mutate].species = new_species

                    # Create new offpsring blueprint with parent mutated blueprint graph
                    new_bp_id, new_bp = self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                                                       output_shape=self.output_shape,
                                                                       output_activation=output_activation,
                                                                       optimizer_factory=optimizer_factory)

                elif random_float < bp_mutation_hp_prob:
                    ## Create new blueprint by mutating the hyperparameters ##

                    # Determine parent blueprint and its parameters as well as the intensity of the mutation, in this
                    # case the amount of hyperparameters to be changed
                    parent_bp = self.blueprints[random.choice(self.bp_species[spec_id])]
                    blueprint_graph, _, output_activation, optimizer_factory = parent_bp.duplicate_parameters()
                    mutation_intensity = random.uniform(0, 0.3)

                    # Uniform randomly determine optimizer to change to
                    if random.choice(self.available_optimizers) == 'SGD':
                        # Determine if current optimizer factory of same type as optimizer to change to, as in this case
                        # the variables will be perturbed instead of created completely new
                        if isinstance(optimizer_factory, SGDFactory):
                            # Get current parameters of optimizer
                            learning_rate, momentum, nesterov = optimizer_factory.get_parameters()

                            # Determine specifically how many hps will be changed
                            hps_to_change_count = int(mutation_intensity * 4)
                            if hps_to_change_count == 0:
                                hps_to_change_count = 1

                            # Create specific list of parameter ids to be changed
                            parameters_to_change = random.sample(range(4), k=hps_to_change_count)

                            # Traverse through list of parameter ids to change. Categorical hps will be uniform randomly
                            # chosen anew. Sortable hps will be perturbed with normal distribution and config specified
                            # standard deviation
                            for param_to_change in parameters_to_change:
                                if param_to_change == 0:
                                    output_activation = random.choice(self.output_activations)
                                elif param_to_change == 1:
                                    perturbed_lr = np.random.normal(loc=learning_rate,
                                                                    scale=self.sgd_learning_rate_stddev)
                                    learning_rate = round_to_nearest_multiple(perturbed_lr,
                                                                              self.sgd_learning_rate[0],
                                                                              self.sgd_learning_rate[1],
                                                                              self.sgd_learning_rate[2])
                                elif param_to_change == 2:
                                    perturbed_momentum = np.random.normal(loc=momentum, scale=self.sgd_momentum_stddev)
                                    momentum = round_to_nearest_multiple(perturbed_momentum,
                                                                         self.sgd_momentum[0],
                                                                         self.sgd_momentum[1],
                                                                         self.sgd_momentum[2])
                                elif param_to_change == 3:
                                    nesterov = random.choice(self.sgd_nesterov)

                            # Create new optimizer factory with newly mutated parameters
                            optimizer_factory = SGDFactory(learning_rate, momentum, nesterov)
                        else:
                            raise NotImplementedError("Handling of conversion to SGD optimizer while the current "
                                                      "optimizer is not SGD not yet implemented")
                    else:
                        raise RuntimeError("Optimizer other than SGD not yet implemented")

                    # Create new offpsring blueprint with parent mutated hyperparameters
                    new_bp_id, new_bp = self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                                                       output_shape=self.output_shape,
                                                                       output_activation=output_activation,
                                                                       optimizer_factory=optimizer_factory)

                else:  # random_float < self.bp_crossover + bp_mutation_hp_prob
                    ## Create new blueprint through crossover ##

                    # Determine if 2 blueprints are available in current species, as is required for crossover
                    if len(self.bp_species[spec_id]) == 1:

                        # If Only 1 blueprint in current species available as parent, create new blueprint with
                        # identical parameters
                        parent_bp = self.blueprints[random.choice(self.bp_species[spec_id])]
                        blueprint_graph, _, output_activation, optimizer_factory = parent_bp.duplicate_parameters()

                        # Create new offpsring blueprint with identical parameters
                        new_bp_id, new_bp = self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                                                           output_shape=self.output_shape,
                                                                           output_activation=output_activation,
                                                                           optimizer_factory=optimizer_factory)

                    else:
                        # Choose 2 random though different blueprint ids as parents
                        parent_bp_1_id, parent_bp_2_id = random.sample(self.bp_species[spec_id], k=2)

                        # Assign the 'base_bp_graph' variable to the fitter blueprint and the 'other_bp_graph' variable
                        # to the less fitter blueprint. Copy over output activation and optimizer factory from the
                        # fitter blueprint for the creation of the crossed over bp
                        if self.blueprints[parent_bp_1_id].get_fitness() > \
                                self.blueprints[parent_bp_2_id].get_fitness():
                            base_bp_graph, _, output_activation, optimizer_factory = \
                                self.blueprints[parent_bp_1_id].duplicate_parameters()
                            other_bp_graph, _, _, _ = self.blueprints[parent_bp_2_id].duplicate_parameters()
                        else:
                            other_bp_graph, _, _, _ = self.blueprints[parent_bp_1_id].duplicate_parameters()
                            base_bp_graph, _, output_activation, optimizer_factory = \
                                self.blueprints[parent_bp_2_id].duplicate_parameters()

                        # Create quickly searchable sets of gene ids for the ids present in both the fitter and less fit
                        # blueprint graphs
                        base_bp_graph_ids = set(other_bp_graph.keys())
                        other_bp_graph_ids = set(other_bp_graph.keys())
                        all_bp_graph_ids = base_bp_graph_ids.union(other_bp_graph_ids)

                        # For all gene ids in the blueprint graphs, choose uniform randomly which blueprint gene is
                        # carried over to the new blueprint graph if the gene id is joint (both blueprint graph have
                        # it). Carry over all gene ids to new blueprint graph that are exclusive to either parent
                        # blueprint graph.
                        for gene_id in all_bp_graph_ids:
                            if gene_id in base_bp_graph_ids and gene_id in other_bp_graph_ids:
                                if random.randint(0, 1) == 0:
                                    base_bp_graph[gene_id] = other_bp_graph[gene_id]
                            elif gene_id in other_bp_graph_ids:
                                base_bp_graph[gene_id] = other_bp_graph[gene_id]

                        # Create new offpsring blueprint with crossed over blueprint graph and hyperparameters of
                        # fitter parent blueprint
                        new_bp_id, new_bp = self.encoding.create_blueprint(blueprint_graph=base_bp_graph,
                                                                           output_shape=self.output_shape,
                                                                           output_activation=output_activation,
                                                                           optimizer_factory=optimizer_factory)

                # Add newly created blueprint to the blueprint container and its according species
                self.blueprints[new_bp_id] = new_bp
                new_bp_species[spec_id].append(new_bp_id)

            # Delete all those blueprints remaining in the blueprint container that were not previously removed or
            # carried over but remained in the species to serve as a potential parent for offspring
            for bp_id_to_remove in bp_ids_to_remove_after_reproduction:
                del self.blueprints[bp_id_to_remove]

        # As new blueprint species dict has now been created, delete the old one and overwrite it with the new blueprint
        # species
        self.bp_species = new_bp_species

        #### Return ####
        # Adjust internal variables of evolutionary process and return False, signalling that the population has not
        # gone extinct
        self.generation_counter += 1
        return False

    def save_population(self, save_dir_path):
        """"""
        logging.warning("codeepneat.save_population() NOT YET IMPLEMENTED")

    def get_best_genome(self) -> CoDeepNEATGenome:
        """"""
        return self.best_genome
