from __future__ import annotations

import math
import random

import numpy as np
import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase
from ....helper_functions import round_with_step


class CoDeepNEATModuleDenseDropout(CoDeepNEATModuleBase):
    """"""

    def __init__(self,
                 module_id,
                 parent_mutation,
                 merge_method,
                 units,
                 activation,
                 kernel_init,
                 bias_init,
                 dropout_flag,
                 dropout_rate):
        """"""
        # Register parameters
        super().__init__(module_id, parent_mutation, merge_method)
        self.units = units
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dropout_flag = dropout_flag
        self.dropout_rate = dropout_rate

    def __str__(self) -> str:
        """"""
        return "CoDeepNEAT DENSE Module | ID: {:>6} | Fitness: {:>6} | Units: {:>4} | Activ: {:>6} | Dropout: {:>4}" \
            .format('#' + str(self.module_id),
                    self.fitness,
                    self.units,
                    self.activation,
                    "None" if self.dropout_flag is False else self.dropout_rate)

    def create_module_layers(self, dtype) -> (tf.keras.layers.Layer, ...):
        """"""
        # Create the basic keras Dense layer, needed in both variants of the Dense Module
        dense_layer = tf.keras.layers.Dense(units=self.units,
                                            activation=self.activation,
                                            kernel_initializer=self.kernel_init,
                                            bias_initializer=self.bias_init,
                                            dtype=dtype)
        # Determine if Dense Module also includes a dropout layer, then return appropriate layer tuple
        if self.dropout_flag:
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate, dtype=dtype)
            return dense_layer, dropout_layer
        else:
            return (dense_layer,)

    def create_mutation(self,
                        offspring_id,
                        config_params,
                        max_degree_of_mutation) -> (int, CoDeepNEATModuleDenseDropout):
        """"""
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'merge_method': self.merge_method,
                            'units': self.units,
                            'activation': self.activation,
                            'kernel_init': self.kernel_init,
                            'bias_init': self.bias_init,
                            'dropout_flag': self.dropout_flag,
                            'dropout_rate': self.dropout_rate}

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}

        # Determine exact integer amount of parameters to be mutated, though minimum is 1
        param_mutation_count = math.ceil(max_degree_of_mutation * 7)

        # Uniform randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(range(7), k=param_mutation_count)

        # Mutate offspring parameters. Categorical parameters are chosen randomly from all available values. Sortable
        # parameters are perturbed through a random normal distribution with the current value as mean and the config
        # specified stddev
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 0:
                offspring_params['merge_method'] = random.choice(config_params['merge_method'])
                parent_mutation['mutated_params']['merge_method'] = self.merge_method
            elif param_to_mutate == 1:
                perturbed_units = int(np.random.normal(loc=self.units,
                                                       scale=config_params['units']['stddev']))
                offspring_params['units'] = round_with_step(perturbed_units,
                                                            config_params['units']['min'],
                                                            config_params['units']['max'],
                                                            config_params['units']['step'])
                parent_mutation['mutated_params']['units'] = self.units
            elif param_to_mutate == 2:
                offspring_params['activation'] = random.choice(config_params['activation'])
                parent_mutation['mutated_params']['activation'] = self.activation
            elif param_to_mutate == 3:
                offspring_params['kernel_init'] = random.choice(config_params['kernel_init'])
                parent_mutation['mutated_params']['kernel_init'] = self.kernel_init
            elif param_to_mutate == 4:
                offspring_params['bias_init'] = random.choice(config_params['bias_init'])
                parent_mutation['mutated_params']['bias_init'] = self.bias_init
            elif param_to_mutate == 5:
                offspring_params['dropout_flag'] = not self.dropout_flag
                parent_mutation['mutated_params']['dropout_flag'] = self.dropout_flag
            else:  # param_to_mutate == 6:
                # Activate the dropout layer with configured probability
                offspring_params['dropout_flag'] = random.random() < config_params['dropout_flag']

                # Either way, perturb dropout_rate parameter
                perturbed_dropout_rate = np.random.normal(loc=self.dropout_rate,
                                                          scale=config_params['dropout_rate']['stddev'])
                offspring_params['dropout_rate'] = round(round_with_step(perturbed_dropout_rate,
                                                                         config_params['dropout_rate']['min'],
                                                                         config_params['dropout_rate']['max'],
                                                                         config_params['dropout_rate']['step'], ), 4)
                parent_mutation['mutated_params']['dropout_flag'] = self.dropout_flag
                parent_mutation['mutated_params']['dropout_rate'] = self.dropout_rate

        return offspring_id, CoDeepNEATModuleDenseDropout(module_id=offspring_id,
                                                          parent_mutation=parent_mutation,
                                                          **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         config_params,
                         max_degree_of_mutation) -> (int, CoDeepNEATModuleDenseDropout):
        """"""
        # Crete offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}

        offspring_params['merge_method'] = self.merge_method
        offspring_params['units'] = round_with_step(int((self.units + less_fit_module.units) / 2),
                                                    config_params['units']['min'],
                                                    config_params['units']['max'],
                                                    config_params['units']['step'])
        offspring_params['activation'] = self.activation
        offspring_params['kernel_init'] = self.kernel_init
        offspring_params['bias_init'] = self.bias_init
        offspring_params['dropout_flag'] = self.dropout_flag
        offspring_params['dropout_rate'] = round(round_with_step((self.dropout_rate + less_fit_module.dropout_rate) / 2,
                                                                 config_params['dropout_rate']['min'],
                                                                 config_params['dropout_rate']['max'],
                                                                 config_params['dropout_rate']['step'], ), 4)

        return offspring_id, CoDeepNEATModuleDenseDropout(module_id=offspring_id,
                                                          parent_mutation=parent_mutation,
                                                          **offspring_params)
