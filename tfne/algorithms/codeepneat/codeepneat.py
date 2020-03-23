import ast
import math
import random
import statistics

import tensorflow as tf
from absl import logging

import tfne
from .codeepneat_helper import deserialize_merge_method, round_to_nearest_multiple
from ..base_algorithm import BaseNeuroevolutionAlgorithm
from ...encodings.codeepneat.codeepneat_genome import CoDeepNEATGenome


class CoDeepNEAT(BaseNeuroevolutionAlgorithm):
    """"""

    def __init__(self, config, initial_population_path=None):
        """"""
        # Read and process the supplied config
        self._process_config(config)

        # Initialize and register the CoDeepNEAT encoding
        self.encoding = tfne.encodings.CoDeepNEATEncoding(dtype=self.dtype)

        # Declare Input shape and number of output units important for blueprint creation
        self.input_shape = None
        self.output_units = None

        # Initialize internal variables of the algorithm
        self.mod_species_counter = 0

    def _process_config(self, config):
        """"""
        # Read and process the general config values for the CoDeepNEAT algorithm
        self.available_modules = ast.literal_eval(config['GENERAL']['available_modules'])
        self.dtype = tf.dtypes.as_dtype(ast.literal_eval(config['GENERAL']['dtype']))
        self.bp_pop_size = ast.literal_eval(config['GENERAL']['bp_pop_size'])
        self.mod_pop_size = ast.literal_eval(config['GENERAL']['mod_pop_size'])
        self.genomes_per_bp = ast.literal_eval(config['GENERAL']['genomes_per_bp'])

        # Read and process the speciation config values for blueprints in the CoDeepNEAT algorithm
        self.bp_spec_type = ast.literal_eval(config['BP_SPECIATION']['bp_spec_type'])
        if self.bp_spec_type is None:
            pass
        elif self.bp_spec_type == 'fixed-threshold':
            # TODO
            pass
        elif self.bp_spec_type == 'dynamic-threshold':
            # TODO
            pass
        elif self.bp_spec_type == 'k-means':
            # TODO
            pass

        # Read and process the selection config values for blueprints in the CoDeepNEAT algorithm
        if config.has_option('BP_SELECTION', 'bp_removal'):
            self.bp_removal_type = 'fixed'
            self.bp_removal = ast.literal_eval(config['BP_SELECTION']['bp_removal'])
        elif config.has_option('BP_SELECTION', 'bp_removal_threshold'):
            self.bp_removal_type = 'threshold'
            self.bp_removal_threshold = ast.literal_eval(config['BP_SELECTION']['bp_removal_threshold'])
        else:
            raise KeyError("'bp_removal' or 'bp_removal_threshold' not specified")
        if config.has_option('BP_SELECTION', 'bp_elitism'):
            self.bp_elitism_type = 'fixed'
            self.bp_elitism = ast.literal_eval(config['BP_SELECTION']['bp_elitism'])
        elif config.has_option('BP_SELECTION', 'bp_elitism_threshold'):
            self.bp_elitism_type = 'threshold'
            self.bp_elitism_threshold = ast.literal_eval(config['BP_SELECTION']['bp_elitism_threshold'])
        else:
            raise KeyError("'bp_elitism' or 'bp_elitism_threshold' not specified")

        # Read and process the evolution config values for blueprints in the CoDeepNEAT algorithm
        self.bp_max_mutation = ast.literal_eval(config['BP_EVOLUTION']['bp_max_mutation'])
        self.bp_mutation_add_conn = ast.literal_eval(config['BP_EVOLUTION']['bp_mutation_add_conn'])
        self.bp_mutation_add_node = ast.literal_eval(config['BP_EVOLUTION']['bp_mutation_add_node'])
        self.bp_mutation_remove_conn = ast.literal_eval(config['BP_EVOLUTION']['bp_mutation_remove_conn'])
        self.bp_mutation_remove_node = ast.literal_eval(config['BP_EVOLUTION']['bp_mutation_remove_node'])
        self.bp_mutation_node_species = ast.literal_eval(config['BP_EVOLUTION']['bp_mutation_node_species'])
        self.bp_mutation_hp = ast.literal_eval(config['BP_EVOLUTION']['bp_mutation_hp'])
        self.bp_crossover = ast.literal_eval(config['BP_EVOLUTION']['bp_crossover'])
        if self.bp_mutation_add_conn + self.bp_mutation_add_node + self.bp_mutation_remove_conn \
                + self.bp_mutation_remove_node + self.bp_mutation_node_species + self.bp_mutation_hp \
                + self.bp_crossover != 1.0:
            raise KeyError("'bp_mutation_*' and 'bp_crossover' values dont add up to 1")

        # Read and process the speciation config values for modules in the CoDeepNEAT algorithm
        self.mod_spec_type = ast.literal_eval(config['MODULE_SPECIATION']['mod_spec_type'])
        if self.mod_spec_type is 'Basic':
            pass
        elif self.mod_spec_type == 'k-means':
            # TODO
            pass

        # Read and process the selection config values for modules in the CoDeepNEAT algorithm
        if config.has_option('MODULE_SELECTION', 'mod_removal'):
            self.mod_removal_type = 'fixed'
            self.mod_removal = ast.literal_eval(config['MODULE_SELECTION']['mod_removal'])
        elif config.has_option('MODULE_SELECTION', 'mod_removal_threshold'):
            self.mod_removal_type = 'threshold'
            self.mod_removal_threshold = ast.literal_eval(config['MODULE_SELECTION']['mod_removal_threshold'])
        else:
            raise KeyError("'mod_removal' or 'mod_removal_threshold' not specified")
        if config.has_option('MODULE_SELECTION', 'mod_elitism'):
            self.mod_elitism_type = 'fixed'
            self.mod_elitism = ast.literal_eval(config['MODULE_SELECTION']['mod_elitism'])
        elif config.has_option('MODULE_SELECTION', 'mod_elitism_threshold'):
            self.mod_elitism_type = 'threshold'
            self.mod_elitism_threshold = ast.literal_eval(config['MODULE_SELECTION']['mod_elitism_threshold'])
        else:
            raise KeyError("'mod_elitism' or 'mod_elitism_threshold' not specified")

        # Read and process the evolution config values for modules in the CoDeepNEAT algorithm
        self.mod_max_mutation = ast.literal_eval(config['MODULE_EVOLUTION']['mod_max_mutation'])
        self.mod_mutation = ast.literal_eval(config['MODULE_EVOLUTION']['mod_mutation'])
        self.mod_crossover = ast.literal_eval(config['MODULE_EVOLUTION']['mod_crossover'])
        if self.mod_mutation + self.mod_crossover != 1.0:
            raise KeyError("'mod_mutation' and 'mod_crossover' values dont add up to 1")

        # Read and process the global hyperparameter config values for blueprints/genomes in the CoDeepNEAT algorithm
        self.optimizer = ast.literal_eval(config['GLOBAL_HP']['optimizer'])
        self.learning_rate = ast.literal_eval(config['GLOBAL_HP']['learning_rate'])
        self.momentum = ast.literal_eval(config['GLOBAL_HP']['momentum'])
        self.nesterov = ast.literal_eval(config['GLOBAL_HP']['nesterov'])
        self.output_activation = ast.literal_eval(config['GLOBAL_HP']['output_activation'])

        # Create Standard Deviation values for range specified global HP config values
        if len(self.learning_rate) == 4:
            self.learning_rate_stddev = float(self.learning_rate[1] - self.learning_rate[0]) / self.learning_rate[3]
        else:
            self.learning_rate_stddev = float(self.learning_rate[1] - self.learning_rate[0]) / 4
        if len(self.momentum) == 4:
            self.momentum_stddev = float(self.momentum[1] - self.momentum[0]) / self.momentum[3]
        else:
            self.momentum_stddev = float(self.momentum[1] - self.momentum[0]) / 4

        if 'DENSE' in self.available_modules and config.has_section('MODULE_DENSE_HP'):
            # Read and process the hyperparameter config values for Dense modules in the CoDeepNEAT algorithm
            self.merge_method = deserialize_merge_method(ast.literal_eval(config['MODULE_DENSE_HP']['merge_method']))
            self.units = ast.literal_eval(config['MODULE_DENSE_HP']['units'])
            self.activation = ast.literal_eval(config['MODULE_DENSE_HP']['activation'])
            self.kernel_initializer = ast.literal_eval(config['MODULE_DENSE_HP']['kernel_initializer'])
            self.bias_initializer = ast.literal_eval(config['MODULE_DENSE_HP']['bias_initializer'])
            self.dropout_probability = ast.literal_eval(config['MODULE_DENSE_HP']['dropout_probability'])
            self.dropout_rate = ast.literal_eval(config['MODULE_DENSE_HP']['dropout_rate'])

            # Create Standard Deviation values for range specified Dense Module HP config values
            if len(self.units) == 4:
                self.units_stddev = float(self.units[1] - self.units[0]) / self.units[3]
            else:
                self.units_stddev = float(self.units[1] - self.units[0]) / 4
            if len(self.dropout_rate) == 4:
                self.dropout_rate_stddev = float(self.dropout_rate[1] - self.dropout_rate[0]) / self.dropout_rate[3]
            else:
                self.dropout_rate_stddev = float(self.dropout_rate[1] - self.dropout_rate[0]) / 4

    def initialize_population(self) -> (dict, dict, int, dict, dict, int):
        """"""

        raise NotImplementedError("ToDo: If a pre-evolved population is supplied via 'initial_population_path' param"
                                  "then notify user accordingly and load and reg this pop."
                                  "If not, initialize minimal population as defined by CoDeepNEAT.")

        # Initialize module population. The type and the parameters of each module are uniformly randomly chosen. The
        # module population is initially specified solely by the type of the module.
        modules = dict()
        mod_species = dict()
        module_species_types = dict()
        for mod_type in self.available_modules:
            self.mod_species_counter += 1
            module_species_types[mod_type] = self.mod_species_counter
            mod_species[self.mod_species_counter] = list()
        for _ in range(self.mod_pop_size):
            module_type = random.choice(self.available_modules)
            if module_type == 'DENSE':
                chosen_merge_method = random.choice(self.merge_method)
                chosen_units_uni = random.randint(self.units[0], self.units[1])
                chosen_units = round_to_nearest_multiple(chosen_units_uni, self.units[0], self.units[1], self.units[2])
                chosen_activation = random.choice(self.activation)
                chosen_kernel_initializer = random.choice(self.kernel_initializer)
                chosen_bias_initializer = random.choice(self.bias_initializer)

                dropout_flag = random.random() < self.dropout_probability
                if dropout_flag:
                    chosen_dropout_rate_uni = random.uniform(self.dropout_rate[0], self.dropout_rate[1])
                    chosen_dropout_rate = round_to_nearest_multiple(chosen_dropout_rate_uni, self.dropout_rate[0],
                                                                    self.dropout_rate[1], self.dropout_rate[2])
                else:
                    chosen_dropout_rate = None

                new_mod_id, new_mod = self.encoding.create_dense_module(merge_method=chosen_merge_method,
                                                                        units=chosen_units,
                                                                        activation=chosen_activation,
                                                                        kernel_initializer=chosen_kernel_initializer,
                                                                        bias_initializer=chosen_bias_initializer,
                                                                        dropout_flag=dropout_flag,
                                                                        dropout_rate=chosen_dropout_rate)
                modules[new_mod_id] = new_mod
                mod_species[module_species_types[module_type]].append(new_mod_id)

            else:
                raise NotImplementedError("Module type '{}' listed as available is not implemented".format(module_type))

        # Initialize blueprints population. Initialize all graphs with the nodes 1 and 2 and a connection between them.
        # Node 1 points to the non-existent species 0, indicating it to represent the Input layer. Node 2 points to a
        # randomly chosen modules species. The genes have to be created with their according id in order to enable
        # historical marking crossover. The genome hyperparameters that are associated with the blueprints are uniformly
        # randomly chosen. All blueprints are assigned to species 1 as the first speciation for blueprints does not
        # happen until the evolution.
        blueprints = dict()
        bp_species = {1: []}
        available_mod_species = tuple(mod_species.keys())
        for _ in range(self.bp_pop_size):
            # Determine the module species of the output node of this minimal graph randomly
            output_module_species = random.choice(available_mod_species)

            # Create basic genotype
            blueprint_genotype = dict()
            gene_id, gene = self.encoding.create_blueprint_node(node=1, species=0)
            blueprint_genotype[gene_id] = gene
            gene_id, gene = self.encoding.create_blueprint_node(node=2, species=output_module_species)
            blueprint_genotype[gene_id] = gene
            gene_id, gene = self.encoding.create_blueprint_conn(conn_start=1, conn_end=2)
            blueprint_genotype[gene_id] = gene

            # Randomly chose genome hyperparameters associated for this blueprint
            chosen_optimizer = random.choice(self.optimizer)
            chosen_learning_rate_uni = random.uniform(self.learning_rate[0], self.learning_rate[1])
            chosen_learning_rate = round_to_nearest_multiple(chosen_learning_rate_uni, self.learning_rate[0],
                                                             self.learning_rate[1], self.learning_rate[2])
            chosen_momentum_uni = random.uniform(self.momentum[0], self.momentum[1])
            chosen_momentum = round_to_nearest_multiple(chosen_momentum_uni, self.momentum[0],
                                                        self.momentum[1], self.momentum[2])
            chosen_nesterov = random.choice(self.nesterov)
            chosen_output_activation = random.choice(self.output_activation)

            new_bp_id, new_bp = self.encoding.create_blueprint(blueprint_genotype=blueprint_genotype,
                                                               optimizer=chosen_optimizer,
                                                               learning_rate=chosen_learning_rate,
                                                               momentum=chosen_momentum,
                                                               nesterov=chosen_nesterov,
                                                               output_activation=chosen_output_activation)
            blueprints[new_bp_id] = new_bp
            bp_species[1].append(new_bp_id)

        return blueprints, bp_species, self.bp_pop_size, modules, mod_species, self.mod_pop_size

    def evaluate_population(self,
                            environment,
                            blueprints,
                            modules,
                            mod_species,
                            generation,
                            current_best_fitness) -> (CoDeepNEATGenome, float):
        """"""
        best_genome = None
        best_fitness = current_best_fitness

        # Create container collecting the fitness of the genomes that involve specific modules and average out the
        # collected fitness values later
        mod_fitness_dict = dict()

        for blueprint in blueprints.values():
            bp_species = blueprint.get_species()

            # Create container collecting the fitness of genomes using that specific blueprint and average it out later
            bp_fitness_list = list()

            for _ in range(self.genomes_per_bp):
                module_id_chosen_for_species = dict()
                bp_assigned_modules = dict()
                for i in bp_species:
                    module_id_chosen_for_species[i] = random.choice(mod_species[i])
                    bp_assigned_modules[i] = modules[module_id_chosen_for_species[i]]

                _, new_genome = self.encoding.create_genome(blueprint, bp_assigned_modules, generation)

                fitness = environment.eval_genome_fitness(new_genome)

                # Assign genome fitness to blueprint and all the modules used in genome
                bp_fitness_list.append(fitness)
                for mod_id in module_id_chosen_for_species.values():
                    try:
                        mod_fitness_dict[mod_id].append(fitness)
                    except:
                        mod_fitness_dict[mod_id] = [fitness]

                if fitness > best_fitness:
                    new_genome.set_fitness(fitness)
                    best_genome = new_genome
                    best_fitness = fitness

            # Average out collected blueprint fitness and assign to specific blueprint
            bp_avg_fitness = round(statistics.mean(bp_fitness_list), 3)
            blueprint.set_fitness(bp_avg_fitness)

        # Average out collected module fitness and assign to specific modules
        for mod_id, mod_fitness_list in mod_fitness_dict.items():
            mod_avg_fitness = round(statistics.mean(mod_fitness_list), 3)
            modules[mod_id].set_fitness(mod_avg_fitness)

        # TODO after each completed evaluation, deepcopy or permanently register the best genome, such that even
        # when the population goes extinct in evolve_pop there is something to return when calling get_best_genome

        return best_genome, best_fitness

    def get_best_genome(self) -> CoDeepNEATGenome:
        pass

    def evolve_population(self, blueprints, modules, bp_species, mod_species) -> (dict, dict, int, dict, dict, int):
        """"""
        #### Speciate modules ####
        if self.mod_spec_type is 'Basic':
            # Population is already speciated solely according to their module type. Therefore skip speciation as only
            # basic speciation is specified.
            pass
        else:
            raise NotImplementedError("Other module speciation method than 'Basic' not yet implemented")

        #### Select modules ####
        # Determine intended sizes for the module species
        assigned_species_size_mod = self._create_assigned_species_size(modules, mod_species, self.mod_pop_size)

        # Remove low performing modules and carry over the top modules of each species according to specified elitism
        new_mod_species = dict()
        for spec_id, spec_module_ids in mod_species.items():
            spec_size = len(spec_module_ids)
            if self.mod_removal_type == 'fixed':
                modules_to_remove = self.mod_removal
            else:  # if self.mod_removal_type == 'threshold':
                modules_to_remove = math.floor(spec_size * self.mod_removal_threshold)
            if self.mod_elitism_type == 'fixed':
                modules_to_carry_over = self.mod_elitism
            else:  # if self.mod_elitism_type == 'threshold':
                modules_to_carry_over = spec_size - math.ceil(spec_size * self.mod_elitism_threshold)
            # Elitism has higher precedence than removal, meaning if more modules are carried over then remove less
            if modules_to_carry_over >= spec_size:
                modules_to_remove = 0
                modules_to_carry_over = spec_size
            elif modules_to_remove + modules_to_carry_over > spec_size:
                modules_to_remove = spec_size - modules_to_carry_over

            # Now actually remove low performing modules of species, by first sorting them and then deleting them
            spec_module_ids_sorted = sorted(spec_module_ids, key=lambda x: modules[x].get_fitness(), reverse=True)

            # Carry fittest modules already over to new mod_species, according to elitism
            new_mod_species[spec_id] = spec_module_ids_sorted[:modules_to_carry_over]

            # Remove low performing modules
            if modules_to_remove > 0:
                # Remove modules
                module_ids_to_remove = spec_module_ids_sorted[-modules_to_remove:]
                for mod_id_to_remove in module_ids_to_remove:
                    del modules[mod_id_to_remove]

                # Assign module id list without low performing modules to species
                mod_species[spec_id] = spec_module_ids_sorted[:-modules_to_remove]

        #### Speciate blueprints ####
        if self.bp_spec_type is None:
            # Explicitely don't speciate the blueprints
            pass
        else:
            raise NotImplementedError("Other blueprint speciation method than 'None' not yet implemented")

        #### Select blueprints ####
        # Determine intended sizes for the blueprint species
        assigned_species_size_bp = self._create_assigned_species_size(blueprints, bp_species, self.bp_pop_size)

        # Remove low performing modules and carry over the top modules of each species according to specified elitism
        new_bp_species = dict()
        for spec_id, spec_blueprint_ids in bp_species.items():
            spec_size = len(spec_blueprint_ids)
            if self.bp_removal_type == 'fixed':
                blueprints_to_remove = self.bp_removal
            else:  # if self.bp_removal_type == 'threshold':
                blueprints_to_remove = math.floor(spec_size * self.bp_removal_threshold)
            if self.bp_elitism_type == 'fixed':
                blueprints_to_carry_over = self.bp_elitism
            else:  # if self.bp_elitism_type == 'threshold':
                blueprints_to_carry_over = spec_size - math.ceil(spec_size * self.bp_elitism_threshold)
            # Elitism has higher precedence than removal, meaning if more blueprints are carried over then remove less
            if blueprints_to_carry_over >= spec_size:
                blueprints_to_remove = 0
                blueprints_to_carry_over = spec_size
            elif blueprints_to_remove + blueprints_to_carry_over > spec_size:
                blueprints_to_remove = spec_size - blueprints_to_carry_over

            # Now actually remove low performing blueprints of species, by first sorting them and then deleting them
            spec_blueprint_ids_sorted = sorted(spec_blueprint_ids,
                                               key=lambda x: blueprints[x].get_fitness(),
                                               reverse=True)

            # Carry fittest blueprints already over to new bp_species, according to elitism
            new_bp_species[spec_id] = spec_blueprint_ids_sorted[:blueprints_to_carry_over]

            # Remove low performing blueprints
            if blueprints_to_remove > 0:
                # Remove blueprints
                blueprint_ids_to_remove = spec_blueprint_ids_sorted[-blueprints_to_remove:]
                for bp_id_to_remove in blueprint_ids_to_remove:
                    del blueprints[bp_id_to_remove]

                # Assign blueprint id list without low performing blueprints to species
                bp_species[spec_id] = spec_blueprint_ids_sorted[:-blueprints_to_remove]

        #### Evolve modules ####
        for spec_id in mod_species.keys():
            # Create as many modules through mutation/crossover until assigned species size is reached
            for _ in range(assigned_species_size_mod[spec_id] - len(new_mod_species[spec_id])):
                if random.random() < self.mod_mutation:
                    # TODO: Create new module through mutation
                    pass

                    new_mod_id, new_mod = -1, None
                else:
                    # TODO: Create new module through crossover
                    pass

                    new_mod_id, new_mod = -1, None

                # Add new module actually to the module container
                modules[new_mod_id] = new_mod
                new_mod_species[spec_id].append(new_mod_id)

        # As new module speciation dict now created, delete old one
        del mod_species

        #### Evolve blueprints ####
        bp_mutation_add_node_prob = self.bp_mutation_add_conn + self.bp_mutation_add_node
        bp_mutation_remove_conn_prob = bp_mutation_add_node_prob + self.bp_mutation_remove_conn
        bp_mutation_remove_node_prob = bp_mutation_remove_conn_prob + self.bp_mutation_remove_node
        bp_mutation_node_species_prob = bp_mutation_remove_node_prob + self.bp_mutation_node_species
        bp_mutation_hp_prob = bp_mutation_node_species_prob + self.bp_mutation_hp

        for spec_id in bp_species.keys():
            # Create as many blueprints through mutation/crossover until assigned species size is reached
            for _ in range(assigned_species_size_bp[spec_id] - len(new_bp_species[spec_id])):
                evolution_choice_val = random.random()
                if evolution_choice_val < self.bp_mutation_add_conn:
                    # TODO: Create new blueprint by adding conn
                    pass

                    new_bp_id, new_bp = -1, None
                elif evolution_choice_val < bp_mutation_add_node_prob:
                    # TODO: Create new blueprint by adding node
                    pass

                    new_bp_id, new_bp = -1, None
                elif evolution_choice_val < bp_mutation_remove_conn_prob:
                    # TODO: Create new blueprint by removing conn
                    pass

                    new_bp_id, new_bp = -1, None
                elif evolution_choice_val < bp_mutation_remove_node_prob:
                    # TODO: Create new blueprint by removing node
                    pass

                    new_bp_id, new_bp = -1, None
                elif evolution_choice_val < bp_mutation_node_species_prob:
                    # TODO: Create new blueprint by mutating the species assigned to the node
                    pass

                    new_bp_id, new_bp = -1, None
                elif evolution_choice_val < bp_mutation_hp_prob:
                    # TODO: Create new blueprint by mutating hyperparameters
                    pass

                    new_bp_id, new_bp = -1, None
                else:
                    # TODO: Create new blueprint through crossover
                    pass

                    new_bp_id, new_bp = -1, None

                # Add new blueprint actually to the blueprint container
                blueprints[new_bp_id] = new_bp
                new_bp_species[spec_id].append(new_bp_id)

        # TODO: Possibly remove this and refactor arch, as CoDeepNEAT currently only works on fixed pop sizes
        # Calculate new pop sizes
        self.mod_pop_size = len(modules)
        self.bp_pop_size = len(blueprints)

        return blueprints, bp_species, self.bp_pop_size, modules, new_mod_species, self.mod_pop_size

    def register_environment(self, environment):
        """"""
        raise NotImplementedError("TODO: Check if environment is set to weight training, as this is required for CDN. "
                                  "set environment as class var."
                                  "Set input and output shapes.")

    def set_input_output_shape(self, input_shape, output_units):
        """"""
        self.encoding.set_input_output_shape(input_shape, output_units)

    @staticmethod
    def _create_assigned_species_size(members, species, pop_size):
        """"""
        species_fitness = dict()
        for spec_id, member_id_list in species.items():
            spec_fitness = 0
            for member_id in member_id_list:
                spec_fitness += members[member_id].get_fitness()
            species_fitness[spec_id] = spec_fitness

        total_fitness = sum(species_fitness.values())

        assigned_species_size = dict()
        assigned_species_size_rest = dict()
        current_total_size = 0
        for spec_id, spec_fitness in species_fitness.items():
            assigned_size_float = (spec_fitness / total_fitness) * pop_size
            assigned_size_floored = math.floor(assigned_size_float)

            assigned_species_size[spec_id] = assigned_size_floored
            assigned_species_size_rest[spec_id] = assigned_size_float - assigned_size_floored
            current_total_size += assigned_size_floored

        increment_order = sorted(assigned_species_size_rest.keys(), key=assigned_species_size_rest.get, reverse=True)

        increment_counter = 0
        while current_total_size < pop_size:
            assigned_species_size[increment_order[increment_counter]] += 1
            increment_counter += 1
            current_total_size += 1

        return assigned_species_size
