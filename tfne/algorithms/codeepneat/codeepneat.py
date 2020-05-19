import sys
import math
import random
import statistics

import numpy as np
from absl import logging

import tfne
from ..base_algorithm import BaseNeuroevolutionAlgorithm
from ...encodings.codeepneat.codeepneat_genome import CoDeepNEATGenome
from ...encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprint, CoDeepNEATBlueprintNode
from ...encodings.codeepneat.modules.codeepneat_module_base import CoDeepNEATModuleBase
from ...helper_functions import read_option_from_config, round_with_step


class CoDeepNEAT(BaseNeuroevolutionAlgorithm):
    """"""

    def __init__(self, config, environment_factory, initial_population_file_path=None):
        """"""

        # Read and process the supplied config and register the optionally supplied initial population
        self._process_config(config)
        self.initial_population_file_path = initial_population_file_path

        # Declare the variables for the environment factory and determine the input shape/dim and output shape/dim of
        # the created environments
        self.environment_factory = environment_factory
        self.input_shape = self.environment_factory.get_env_input_shape()
        self.input_dim = len(self.input_shape)
        self.output_shape = self.environment_factory.get_env_output_shape()
        self.output_dim = len(self.output_shape)

        # Initialize and register the associated CoDeepNEAT encoding
        self.encoding = tfne.encodings.CoDeepNEATEncoding(dtype=self.dtype)

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
        # Read and process the general config values for CoDeepNEAT
        self.dtype = read_option_from_config(config, 'GENERAL', 'dtype')
        self.bp_pop_size = read_option_from_config(config, 'GENERAL', 'bp_pop_size')
        self.mod_pop_size = read_option_from_config(config, 'GENERAL', 'mod_pop_size')
        self.genomes_per_bp = read_option_from_config(config, 'GENERAL', 'genomes_per_bp')
        self.eval_epochs = read_option_from_config(config, 'GENERAL', 'eval_epochs')
        self.eval_batch_size = read_option_from_config(config, 'GENERAL', 'eval_batch_size')

        # Read and process the config values that concern the genome creation for CoDeepNEAT
        self.available_modules = read_option_from_config(config, 'GENOME', 'available_modules')
        self.available_optimizers = read_option_from_config(config, 'GENOME', 'available_optimizers')
        self.preprocessing = read_option_from_config(config, 'GENOME', 'preprocessing')
        self.output_layers = read_option_from_config(config, 'GENOME', 'output_layers')

        # Adjust output_layers config to include the configured datatype
        for out_layer in self.output_layers:
            out_layer['config']['dtype'] = self.dtype

        # Read and process the config values that concern the module speciation for CoDeepNEAT
        self.mod_spec_type = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_type')
        if self.mod_spec_type == 'basic':
            self.mod_spec_min_size = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_min_size')
            self.mod_spec_max_size = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_max_size')
            self.mod_spec_elitism = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_elitism')
            self.mod_spec_reprod_thres = read_option_from_config(config, 'MODULE_SPECIATION', 'mod_spec_reprod_thres')
        else:
            raise NotImplementedError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        # Read and process the config values that concern the evolution of modules for CoDeepNEAT
        self.mod_max_mutation = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_max_mutation')
        self.mod_mutation_prob = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_mutation_prob')
        self.mod_crossover_prob = read_option_from_config(config, 'MODULE_EVOLUTION', 'mod_crossover_prob')

        # Read and process the config values that concern the blueprint speciation for CoDeepNEAT
        self.bp_spec_type = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_type')
        if self.bp_spec_type == 'basic':
            self.bp_spec_elitism = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_elitism')
            self.bp_spec_reprod_thres = read_option_from_config(config, 'BP_SPECIATION', 'bp_spec_reprod_thres')
        else:
            raise NotImplementedError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        # Read and process the config values that concern the evolution of blueprints for CoDeepNEAT
        self.bp_max_mutation = read_option_from_config(config, 'BP_EVOLUTION', 'bp_max_mutation')
        self.bp_mutation_add_conn_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_add_conn_prob')
        self.bp_mutation_add_node_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_add_node_prob')
        self.bp_mutation_rem_conn_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_rem_conn_prob')
        self.bp_mutation_rem_node_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_rem_node_prob')
        self.bp_mutation_node_spec_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_node_spec_prob')
        self.bp_mutation_optimizer_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_mutation_optimizer_prob')
        self.bp_crossover_prob = read_option_from_config(config, 'BP_EVOLUTION', 'bp_crossover_prob')

        # Read and process the config values that concern the parameter range of the modules for CoDeepNEAT
        self.available_mod_params = dict()
        for available_mod in self.available_modules:
            # Determine a dict of all supplied configuration values as literal evals
            config_section_str = 'MODULE_' + available_mod.upper()
            if not config.has_section(config_section_str):
                raise RuntimeError(f"Module '{available_mod}' marked as available in config does not have an "
                                   f"associated config section defining its parameters")
            mod_section_params = dict()
            for mod_param in config.options(config_section_str):
                mod_section_params[mod_param] = read_option_from_config(config, config_section_str, mod_param)

            # Assign that dict of all available parameters for the module to the instance variable
            self.available_mod_params[available_mod] = mod_section_params

        # Read and process the config values that concern the parameter range of the optimizers for CoDeepNEAT
        self.available_opt_params = dict()
        for available_opt in self.available_optimizers:
            # Determine a dict of all supplied configuration values as literal evals
            config_section_str = 'OPTIMIZER_' + available_opt.upper()
            if not config.has_section(config_section_str):
                raise RuntimeError(f"Optimizer '{available_opt}' marked as available in config does not have an "
                                   f"associated config section defining its parameters")
            opt_section_params = dict()
            for opt_param in config.options(config_section_str):
                opt_section_params[opt_param] = read_option_from_config(config, config_section_str, opt_param)

            # Assign that dict of all available parameters for the optimizers to the instance variable
            self.available_opt_params[available_opt] = opt_section_params

        # Perform some basic sanity checks of the configuration
        assert self.mod_spec_min_size * len(self.available_modules) <= self.mod_pop_size
        assert self.mod_spec_max_size * len(self.available_modules) >= self.mod_pop_size
        assert round(self.mod_mutation_prob + self.mod_crossover_prob, 4) == 1.0
        assert round(self.bp_mutation_add_conn_prob + self.bp_mutation_add_node_prob + self.bp_mutation_rem_conn_prob
                     + self.bp_mutation_rem_node_prob + self.bp_mutation_node_spec_prob + self.bp_crossover_prob
                     + self.bp_mutation_optimizer_prob, 4) == 1.0

    def initialize_population(self):
        """"""

        if self.initial_population_file_path is None:
            print("Initializing a new population of {} blueprints and {} modules..."
                  .format(self.bp_pop_size, self.mod_pop_size))

            # Set internal variables of the population to the initialization of a new population
            self.generation_counter = 0
            self.best_fitness = 0

            #### Initialize Module Population ####
            # Initialize module population with a basic speciation scheme, even when another speciation type is supplied
            # as config, only speciating modules according to their module type. Each module species (and therefore
            # module type) is initiated with the same amount of modules (or close to the same amount if module pop size
            # not evenly divisble). Parameters of all initial modules are uniform randomly chosen.

            # Set initial species counter of basic speciation, initialize module species list and map each species to
            # its type
            for mod_type in self.available_modules:
                self.mod_species_counter += 1
                self.mod_species[self.mod_species_counter] = list()
                self.mod_species_type[self.mod_species_counter] = mod_type

            for i in range(self.mod_pop_size):
                # Decide on for which species a new module is added (uniformly distributed)
                chosen_species = (i % self.mod_species_counter) + 1

                # Initialize a new module of the chosen species
                module_id, module = self._create_initial_module(mod_type=self.mod_species_type[chosen_species])

                # Append newly created initial module to module container and to according species
                self.modules[module_id] = module
                self.mod_species[chosen_species].append(module_id)

            #### Initialize Blueprint Population ####
            # Initialize blueprint population with a minimal blueprint graph, only consisting of an input node (with
            # None species or the 'input' species respectively) and a single output node, having a randomly assigned
            # species. All hyperparameters of the blueprint are uniform randomly chosen. All blueprints are not
            # speciated in the beginning and are assigned to species 1.

            # Initialize blueprint species list and create tuple of the possible species the output node can take on
            self.bp_species[1] = list()
            available_mod_species = tuple(self.mod_species.keys())

            for _ in range(self.bp_pop_size):
                # Determine the module species of the initial (and only) node
                initial_node_species = random.choice(available_mod_species)

                # Initialize a new blueprint with minimal graph only using initial node species
                blueprint_id, blueprint = self._create_initial_blueprint(initial_node_species)

                # Append newly create blueprint to blueprint container and to only initial blueprint species
                self.blueprints[blueprint_id] = blueprint
                self.bp_species[1].append(blueprint_id)
        else:
            raise NotImplementedError("Initializing population with pre-evolved initial population not yet implemented")

    def _create_initial_module(self, mod_type) -> (int, CoDeepNEATModuleBase):
        """"""
        # Declare container collecting the specific parameters of the module to be created
        chosen_module_params = dict()

        # Determine the specific parameter dict of the current module type
        available_module_params = self.available_mod_params[mod_type]

        # Traverse each possible parameter option and determine a uniformly random value depending on if its a
        # categorical, sortable or boolean value
        for mod_param, mod_param_val_range in available_module_params.items():
            # If the module parameter is a categorical value choose randomly from the list
            if isinstance(mod_param_val_range, list):
                chosen_module_params[mod_param] = random.choice(mod_param_val_range)
            # If the module parameter is sortable, create a random value between the min and max values adhering to the
            # configured step
            elif isinstance(mod_param_val_range, dict):
                if isinstance(mod_param_val_range['min'], int) and isinstance(mod_param_val_range['max'], int) \
                        and isinstance(mod_param_val_range['step'], int):
                    mod_param_random = random.randint(mod_param_val_range['min'],
                                                      mod_param_val_range['max'])
                    chosen_mod_param = round_with_step(mod_param_random,
                                                       mod_param_val_range['min'],
                                                       mod_param_val_range['max'],
                                                       mod_param_val_range['step'])
                elif isinstance(mod_param_val_range['min'], float) and isinstance(mod_param_val_range['max'], float) \
                        and isinstance(mod_param_val_range['step'], float):
                    mod_param_random = random.uniform(mod_param_val_range['min'],
                                                      mod_param_val_range['max'])
                    chosen_mod_param = round(round_with_step(mod_param_random,
                                                             mod_param_val_range['min'],
                                                             mod_param_val_range['max'],
                                                             mod_param_val_range['step']), 4)
                else:
                    raise NotImplementedError(f"Config parameter '{mod_param}' of the {mod_type} section is of type"
                                              f"dict though the dict values are not of type int or float")
                chosen_module_params[mod_param] = chosen_mod_param
            # If the module parameter is a binary value it is specified as a float with the probablity of that parameter
            # being set to True
            elif isinstance(mod_param_val_range, float):
                chosen_module_params[mod_param] = random.random() < mod_param_val_range
            else:
                raise NotImplementedError(f"Config parameter '{mod_param}' of the {mod_type} section is not one of the"
                                          f"valid types of list, dict or float")

        # Create new module through encoding and return its ID and module
        return self.encoding.create_module(mod_type=mod_type, module_parameters=chosen_module_params)

    def _create_initial_blueprint(self, initial_node_species) -> (int, CoDeepNEATBlueprint):
        """"""
        # Create a minimal blueprint graph with node 1 being the input node (having no species) and node 2 being the
        # random initial node species
        blueprint_graph = dict()
        gene_id, gene = self.encoding.create_blueprint_node(node=1, species=None)
        blueprint_graph[gene_id] = gene
        gene_id, gene = self.encoding.create_blueprint_node(node=2, species=initial_node_species)
        blueprint_graph[gene_id] = gene
        gene_id, gene = self.encoding.create_blueprint_conn(conn_start=1, conn_end=2)
        blueprint_graph[gene_id] = gene

        # Randomly choose an optimizer from the available optimizers and create the parameter config dict of it
        chosen_optimizer = random.choice(self.available_optimizers)
        available_optimizer_params = self.available_opt_params[chosen_optimizer]

        # Declare container collecting the specific parameters of the optimizer to be created, setting the just chosen
        # optimizer class
        chosen_optimizer_params = {'class_name': chosen_optimizer, 'config': dict()}

        # Traverse each possible parameter option and determine a uniformly random value depending on if its a
        # categorical, sortable or boolean value
        for opt_param, opt_param_val_range in available_optimizer_params.items():
            # If the optimizer parameter is a categorical value choose randomly from the list
            if isinstance(opt_param_val_range, list):
                chosen_optimizer_params['config'][opt_param] = random.choice(opt_param_val_range)
            # If the optimizer parameter is sortable, create a random value between the min and max values adhering
            # to the configured step
            elif isinstance(opt_param_val_range, dict):
                if isinstance(opt_param_val_range['min'], int) and isinstance(opt_param_val_range['max'], int) \
                        and isinstance(opt_param_val_range['step'], int):
                    opt_param_random = random.randint(opt_param_val_range['min'],
                                                      opt_param_val_range['max'])
                    chosen_opt_param = round_with_step(opt_param_random,
                                                       opt_param_val_range['min'],
                                                       opt_param_val_range['max'],
                                                       opt_param_val_range['step'])
                elif isinstance(opt_param_val_range['min'], float) and isinstance(opt_param_val_range['max'], float) \
                        and isinstance(opt_param_val_range['step'], float):
                    opt_param_random = random.uniform(opt_param_val_range['min'],
                                                      opt_param_val_range['max'])
                    chosen_opt_param = round(round_with_step(opt_param_random,
                                                             opt_param_val_range['min'],
                                                             opt_param_val_range['max'],
                                                             opt_param_val_range['step']), 4)
                else:
                    raise NotImplementedError(f"Config parameter '{opt_param}' of the {chosen_optimizer} optimizer "
                                              f"section is of type dict though the dict values are not of type int or "
                                              f"float")
                chosen_optimizer_params['config'][opt_param] = chosen_opt_param
            # If the optimizer parameter is a binary value it is specified as a float with the probablity of that
            # parameter being set to True
            elif isinstance(opt_param_val_range, float):
                chosen_optimizer_params['config'][opt_param] = random.random() < opt_param_val_range
            else:
                raise NotImplementedError(f"Config parameter '{opt_param}' of the {chosen_optimizer} optimizer section "
                                          f"is not one of the valid types of list, dict or float")

        # Create new optimizer through encoding
        optimizer_factory = self.encoding.create_optimizer_factory(optimizer_parameters=chosen_optimizer_params)

        # Create just defined initial blueprint through encoding
        return self.encoding.create_blueprint(blueprint_graph=blueprint_graph,
                                              optimizer_factory=optimizer_factory)

    def evaluate_population(self, num_cpus, num_gpus, verbosity) -> (int, int):
        """"""
        # TODO Properly implement parallelization
        logging.warning("CoDeepNEAT as of now only supports a single eval instance. Ignoring num_cpus and num_gpus.")
        environment = self.environment_factory.create_environment(verbosity=verbosity,
                                                                  weight_training=True,
                                                                  epochs=self.eval_epochs,
                                                                  batch_size=self.eval_batch_size)

        # Create container collecting the fitness of the genomes that involve specific modules. Calculate the average
        # fitness of the genomes in which a module is involved in later and assign it as the module's fitness
        mod_fitnesses_in_genomes = dict()

        # Initialize Progress counter variables for evaluate population progress bar. Print notice of evaluation start
        genome_pop_size = self.bp_pop_size * self.genomes_per_bp
        genome_eval_counter = 0
        genome_eval_counter_div = round(genome_pop_size / 40.0, 4)
        print("\nEvaluating {} genomes in generation {}...".format(genome_pop_size, self.generation_counter))

        for blueprint in self.blueprints.values():
            bp_module_species = blueprint.get_species()

            # Create container collecting the fitness of the genomes that involve that specific blueprint.
            bp_fitnesses_in_genomes = list()

            for _ in range(self.genomes_per_bp):
                # Assemble genome by first uniform randomly choosing a specific module from the species that the
                # blueprint nodes are referring to.
                bp_assigned_modules = dict()
                for i in bp_module_species:
                    chosen_module_id = random.choice(self.mod_species[i])
                    bp_assigned_modules[i] = self.modules[chosen_module_id]

                # Create genome, using the specific blueprint, a dict of modules for each species, the configured output
                # layers and input shape as well as the current generation
                genome_id, genome = self.encoding.create_genome(blueprint,
                                                                bp_assigned_modules,
                                                                self.output_layers,
                                                                self.input_shape,
                                                                self.generation_counter)

                # Now evaluate genome on registered environment and set its fitness
                genome_fitness = environment.eval_genome_fitness(genome)
                genome.set_fitness(genome_fitness)

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
                bp_fitnesses_in_genomes.append(genome_fitness)
                for assigned_module in bp_assigned_modules.values():
                    module_id = assigned_module.get_id()
                    if module_id in mod_fitnesses_in_genomes:
                        mod_fitnesses_in_genomes[module_id].append(genome_fitness)
                    else:
                        mod_fitnesses_in_genomes[module_id] = [genome_fitness]

                # Register genome as new best if it exhibits better fitness than the previous best
                if self.best_fitness is None or genome_fitness > self.best_fitness:
                    self.best_genome = genome
                    self.best_fitness = genome_fitness

            # Average out collected fitness of genomes the blueprint was invovled in. Then assign that average fitness
            # to the blueprint
            bp_fitnesses_in_genomes_avg = round(statistics.mean(bp_fitnesses_in_genomes), 4)
            blueprint.set_fitness(bp_fitnesses_in_genomes_avg)

        # Average out collected fitness of genomes each module was invovled in. Then assign that average fitness to the
        # module
        for mod_id, mod_fitness_list in mod_fitnesses_in_genomes.items():
            mod_genome_fitness_avg = round(statistics.mean(mod_fitness_list), 4)
            self.modules[mod_id].set_fitness(mod_genome_fitness_avg)

        return self.generation_counter, self.best_fitness

    def summarize_population(self):
        """"""

        # Calculate average fitnesses of each module species and total module pop. Determine best module of each species
        mod_species_avg_fitness = dict()
        mod_species_best_id = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            spec_total_fitness = 0
            for mod_id in spec_mod_ids:
                mod_fitness = self.modules[mod_id].get_fitness()
                spec_total_fitness += mod_fitness
                if spec_id not in mod_species_best_id or mod_fitness > mod_species_best_id[spec_id][1]:
                    mod_species_best_id[spec_id] = (mod_id, mod_fitness)
            mod_species_avg_fitness[spec_id] = round(spec_total_fitness / len(spec_mod_ids), 4)
        modules_avg_fitness = round(sum(mod_species_avg_fitness.values()) / len(mod_species_avg_fitness), 4)

        # Calculate average fitnesses of each bp species and total bp pop. Determine best bp of each species
        bp_species_avg_fitness = dict()
        bp_species_best_id = dict()
        for spec_id, spec_bp_ids in self.bp_species.items():
            spec_total_fitness = 0
            for bp_id in spec_bp_ids:
                bp_fitness = self.blueprints[bp_id].get_fitness()
                spec_total_fitness += bp_fitness
                if spec_id not in bp_species_best_id or bp_fitness > bp_species_best_id[spec_id][1]:
                    bp_species_best_id[spec_id] = (bp_id, bp_fitness)
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
                          self.blueprints[bp_species_best_id[spec_id][0]]))

        # Print summary of module species
        print("\n\033[1mModule Species            || Module Species Avg Fitness            || "
              "Module Species Size\nBest Module of Species\033[0m")
        for spec_id, spec_mod_avg_fitness in mod_species_avg_fitness.items():
            print("{:>6}                    || {:>8}                              || {:>8}\n{}"
                  .format(spec_id,
                          spec_mod_avg_fitness,
                          len(self.mod_species[spec_id]),
                          self.modules[mod_species_best_id[spec_id][0]]))

        # Print summary footer
        print("\n\033[1m" + '#' * 142 + "\033[0m\n")

    def evolve_population(self) -> bool:
        """"""

        #### Speciate Modules ####
        if self.mod_spec_type == 'basic':
            new_modules, new_mod_species, new_mod_species_size = self._speciate_modules_basic()
        elif self.mod_spec_type == 'param_distance':
            new_modules, new_mod_species, new_mod_species_size = self._speciate_modules_param_distance()
        else:
            raise RuntimeError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        #### Speciate Blueprints ####
        if self.bp_spec_type == 'basic':
            new_blueprints, new_bp_species, new_bp_species_size = self._speciate_blueprints_basic()
        elif self.bp_spec_type == 'gene_overlap':
            new_blueprints, new_bp_species, new_bp_species_size = self._speciate_blueprints_gene_overlap()
        else:
            raise RuntimeError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        #### Evolve Modules ####
        # Traverse through the new module species and add new modules until calculated dedicated spec size is reached
        for spec_id, carried_over_mod_ids in new_mod_species.items():
            # Determine amount of offspring and create according amount of new modules
            for _ in range(new_mod_species_size[spec_id] - len(carried_over_mod_ids)):
                # Choose randomly between mutation or crossover of module
                if random.random() < self.mod_mutation_prob:
                    ## Create new module through mutation ##
                    # Get a new module ID from the encoding, randomly determine the maximum degree of mutation and the
                    # parent module from the non removed modules of the current species. Then determine the config
                    # parameters associated with the parent module and let the internal mutation function create a new
                    # module
                    mod_offspring_id = self.encoding.get_next_module_id()
                    max_degree_of_mutation = random.uniform(0, self.mod_max_mutation)
                    parent_module = self.modules[random.choice(self.mod_species[spec_id])]
                    config_params = self.available_mod_params[self.mod_species_type[spec_id]]

                    new_mod_id, new_mod = parent_module.create_mutation(mod_offspring_id,
                                                                        config_params,
                                                                        max_degree_of_mutation)

                else:  # random.random() < self.mod_mutation_prob + self.mod_crossover_prob
                    ## Create new module through crossover ##
                    # Determine if species has at least 2 modules as required for crossover
                    if len(self.mod_species[spec_id]) >= 2:
                        # Determine the 2 parent modules used for crossover
                        parent_module_1_id, parent_module_2_id = random.sample(self.mod_species[spec_id], k=2)
                        parent_module_1 = self.modules[parent_module_1_id]
                        parent_module_2 = self.modules[parent_module_2_id]

                        # Get a new module ID from encoding, randomly determine the maximum degree of mutation and then
                        # determine the config parameters associated with the modules
                        mod_offspring_id = self.encoding.get_next_module_id()
                        max_degree_of_mutation = random.uniform(0, self.mod_max_mutation)
                        config_params = self.available_mod_params[self.mod_species_type[spec_id]]

                        # Determine the fitter parent module and let its internal crossover function create offspring
                        if parent_module_1.get_fitness() >= parent_module_2.get_fitness():
                            new_mod_id, new_mod = parent_module_1.create_crossover(mod_offspring_id,
                                                                                   parent_module_2,
                                                                                   config_params,
                                                                                   max_degree_of_mutation)
                        else:
                            new_mod_id, new_mod = parent_module_2.create_crossover(mod_offspring_id,
                                                                                   parent_module_1,
                                                                                   config_params,
                                                                                   max_degree_of_mutation)

                    else:
                        # As species does not have enough modules for crossover, perform a mutation on the remaining
                        # module
                        mod_offspring_id = self.encoding.get_next_module_id()
                        max_degree_of_mutation = random.uniform(0, self.mod_max_mutation)
                        parent_module = self.modules[random.choice(self.mod_species[spec_id])]
                        config_params = self.available_mod_params[self.mod_species_type[spec_id]]

                        new_mod_id, new_mod = parent_module.create_mutation(mod_offspring_id,
                                                                            config_params,
                                                                            max_degree_of_mutation)

                # Add newly created module to the module container and its according species
                new_modules[new_mod_id] = new_mod
                new_mod_species[spec_id].append(new_mod_id)

        # As new module container and species dict have now been fully created, overwrite the old ones
        self.modules = new_modules
        self.mod_species = new_mod_species

        # TODO Continue Here
        print("Exiting Cleanly")
        exit()

        #### Evolve Blueprints ####
        # Calculate the brackets for a random float to fall into in order to choose a specific evolutionary method
        bp_mutation_add_node_bracket = self.bp_mutation_add_conn_prob + self.bp_mutation_add_node_prob
        bp_mutation_rem_conn_bracket = bp_mutation_add_node_bracket + self.bp_mutation_rem_conn_prob
        bp_mutation_rem_node_bracket = bp_mutation_rem_conn_bracket + self.bp_mutation_rem_node_prob
        bp_mutation_node_spec_bracket = bp_mutation_rem_node_bracket + self.bp_mutation_node_spec_prob
        bp_mutation_optimizer_bracket = bp_mutation_node_spec_bracket + self.bp_mutation_optimizer_prob

        # Traverse through the new blueprint species and add new blueprints until calculated dedicated spec size reached
        for spec_id, carried_over_bp_ids in new_bp_species.items():
            # Determine amount of offspring and create according amount of new blueprints
            for _ in range(new_bp_species_size[spec_id] - len(carried_over_bp_ids)):
                # Choose random float vaue determining specific evolutionary method for blueprint
                random_choice = random.random()
                if random_choice < self.bp_mutation_add_conn_prob:
                    ## Create new blueprint by adding connection ##
                    # TODO
                    raise NotImplementedError()
                elif random_choice < bp_mutation_add_node_bracket:
                    ## Create new blueprint by adding node ##
                    # TODO
                    raise NotImplementedError()
                elif random_choice < bp_mutation_rem_conn_bracket:
                    ## Create new blueprint by removing connection ##
                    # TODO
                    raise NotImplementedError()
                elif random_choice < bp_mutation_rem_node_bracket:
                    ## Create new blueprint by removing node ##
                    # TODO
                    raise NotImplementedError()
                elif random_choice < bp_mutation_node_spec_bracket:
                    ## Create new blueprint by mutating species in nodes ##
                    # TODO
                    raise NotImplementedError()
                elif random_choice < bp_mutation_optimizer_bracket:
                    ## Create new blueprint by mutating the associated optimizer ##
                    # TODO
                    raise NotImplementedError()
                else:  # random_choice < bp_crossover_bracket:
                    ## Create new blueprint through crossover ##
                    # TODO
                    raise NotImplementedError()

                # Add newly created blueprint to the blueprint container and its according species
                new_blueprints[new_bp_id] = new_bp
                new_bp_species[spec_id].append(new_bp_id)

        # As new blueprint container and species dict have now been fully created, overwrite the old ones
        self.blueprints = new_blueprints
        self.bp_species = new_bp_species

        #### Return ####
        # Adjust generation counter and return False, signalling that the population has not gone extinct
        self.generation_counter += 1
        return False

    def _speciate_modules_basic(self) -> ({int: CoDeepNEATModuleBase}, {int: [int, ...]}, {int: int}):
        """"""

        #### Module Clustering ####
        # As module population is by default speciated by module type is further clustering not necessary
        pass

        #### New Species Size Calculation ####
        # Determine average fitness of each current species as well as the sum of each avg fitness
        mod_species_avg_fitness = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            spec_avg_fitness = statistics.mean([self.modules[mod_id].get_fitness() for mod_id in spec_mod_ids])
            mod_species_avg_fitness[spec_id] = spec_avg_fitness
        total_avg_fitness = sum(mod_species_avg_fitness.values())

        # Calculate the new_mod_species_size depending on the species fitness share of the total fitness.
        new_mod_species_size = dict()
        current_total_size = 0
        for spec_id, spec_avg_fitness in mod_species_avg_fitness.items():
            spec_size = math.floor((spec_avg_fitness / total_avg_fitness) * self.mod_pop_size)

            # If calculated species size violates config specified min/max size correct it
            if spec_size > self.mod_spec_max_size:
                spec_size = self.mod_spec_max_size
            elif spec_size < self.mod_spec_min_size:
                spec_size = self.mod_spec_min_size

            new_mod_species_size[spec_id] = spec_size
            current_total_size += spec_size

        # Flooring / Min / Max species size adjustments likely perturbed the assigned species size in that they don't
        # sum up to the desired module pop size. Decrease or increase new mod species size accordingly.
        while current_total_size < self.mod_pop_size:
            # Increase new mod species size by awarding offspring to species with the currently least assigned offspring
            min_mod_spec_id = min(new_mod_species_size.keys(), key=new_mod_species_size.get)
            new_mod_species_size[min_mod_spec_id] += 1
            current_total_size += 1
        while current_total_size > self.mod_pop_size:
            # Decrease new mod species size by removing offspring from species with currently most assigned offspring
            max_mod_spec_id = max(new_mod_species_size.keys(), key=new_mod_species_size.get)
            new_mod_species_size[max_mod_spec_id] -= 1
            current_total_size -= 1

        #### Module Selection ####
        # Declare new modules container and new module species assignment and carry over x number of best performing
        # modules of each species according to config specified elitism
        new_modules = dict()
        new_mod_species = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            # Sort module ids in species according to their fitness
            spec_mod_ids_sorted = sorted(spec_mod_ids, key=lambda x: self.modules[x].get_fitness(), reverse=True)

            # Determine carried over module ids and module ids prevented from reproduction
            spec_mod_ids_to_carry_over = spec_mod_ids_sorted[:self.mod_spec_elitism]
            removal_index_threshold = int(len(spec_mod_ids) * (1 - self.mod_spec_reprod_thres))
            # Correct removal index threshold if reproduction threshold so high that elitism modules will be removed
            if removal_index_threshold + len(spec_mod_ids_to_carry_over) < len(spec_mod_ids):
                removal_index_threshold = len(spec_mod_ids_to_carry_over)
            spec_mod_ids_to_remove = spec_mod_ids_sorted[removal_index_threshold:]

            # Carry over fittest module ids of species to new container and species assignment
            new_mod_species[spec_id] = list()
            for mod_id_to_carry_over in spec_mod_ids_to_carry_over:
                new_modules[mod_id_to_carry_over] = self.modules[mod_id_to_carry_over]
                new_mod_species[spec_id].append(mod_id_to_carry_over)

            # Delete low performing modules that will not be considered for reproduction from old species assignment
            for mod_id_to_remove in spec_mod_ids_to_remove:
                self.mod_species[spec_id].remove(mod_id_to_remove)

        return new_modules, new_mod_species, new_mod_species_size

    def _speciate_modules_param_distance(self) -> ({int: CoDeepNEATModuleBase}, {int: [int, ...]}, {int: int}):
        """"""
        raise NotImplementedError()

    def _speciate_blueprints_basic(self) -> ({int: CoDeepNEATBlueprint}, {int: [int, ...]}, {int: int}):
        """"""

        #### Blueprint Clustering ####
        # Blueprint clustering unnecessary with basic scheme as all blueprints are assigned to species 1
        pass

        #### New Species Size Calculation ####
        # Species size calculation unnecessary as only one species of blueprints will exist containing all bps
        new_bp_species_size = {1: self.bp_pop_size}

        #### Blueprint Selection ####
        # Declare new blueprints container and new blueprint species assignment and carry over x number of best
        # performing blueprints of each species according to config specified elitism
        new_blueprints = dict()
        new_bp_species = dict()
        for spec_id, spec_bp_ids in self.bp_species.items():
            # Sort blueprint ids in species according to their fitness
            spec_bp_ids_sorted = sorted(spec_bp_ids, key=lambda x: self.blueprints[x].get_fitness(), reverse=True)

            # Determine carried over blueprint ids and blueprint ids prevented from reproduction
            spec_bp_ids_to_carry_over = spec_bp_ids_sorted[:self.bp_spec_elitism]
            removal_index_threshold = int(len(spec_bp_ids) * (1 - self.bp_spec_reprod_thres))
            # Correct removal index threshold if reproduction threshold so high that elitism blueprints will be removed
            if removal_index_threshold + len(spec_bp_ids_to_carry_over) < len(spec_bp_ids):
                removal_index_threshold = len(spec_bp_ids_to_carry_over)
            spec_bp_ids_to_remove = spec_bp_ids_sorted[removal_index_threshold:]

            # Carry over fittest blueprint ids of species to new container and species assignment
            new_bp_species[spec_id] = list()
            for bp_id_to_carry_over in spec_bp_ids_to_carry_over:
                new_blueprints[bp_id_to_carry_over] = self.blueprints[bp_id_to_carry_over]
                new_bp_species[spec_id].append(bp_id_to_carry_over)

            # Delete low performing blueprints that will not be considered for reproduction from old species assignment
            for bp_id_to_remove in spec_bp_ids_to_remove:
                self.bp_species[spec_id].remove(bp_id_to_remove)

        return new_blueprints, new_bp_species, new_bp_species_size

    def _speciate_blueprints_gene_overlap(self) -> ({int: CoDeepNEATBlueprint}, {int: [int, ...]}, {int: int}):
        """"""
        raise NotImplementedError()

    def _create_mutated_blueprint_add_conn(self):
        """"""
        pass
        '''
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
        '''

    def _create_mutated_blueprint_add_node(self):
        """"""
        pass
        '''
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
        '''

    def _create_mutated_blueprint_rem_conn(self):
        """"""
        pass

    def _create_mutated_blueprint_rem_node(self):
        """"""
        pass

    def _create_mutated_blueprint_node_spec(self):
        """"""
        pass
        '''
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
        '''

    def _create_mutated_blueprint_optimizer(self):
        """"""
        pass
        '''
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
        '''

    def _create_crossed_over_blueprint(self):
        """"""
        pass
        '''
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
        '''

    def save_population(self, save_dir_path):
        """"""
        logging.warning("codeepneat.save_population() NOT YET IMPLEMENTED")

    def get_best_genome(self) -> CoDeepNEATGenome:
        """"""
        return self.best_genome
