"""
Entry point for running an SDNet experiment.
"""

import argparse
import importlib
import json
import logging
import os
import matplotlib
matplotlib.use('Agg')  # environment for non-interactive environments

from easydict import EasyDict
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(1)


class Experiment(object):
    """
    Experiment class reads the configuration parameters (stored under configuration folder) and execute the experiment.
    Required command line arguments are:
        --config    the configuration file name.
        --split     split number for cross validation, e.g. 0, 1, ...

    Optional command line arguments are:
        --test          only test a model defined by the configuration file
        --l_mix         float [0, 1]. Sets the amount of labelled data.
        --ul_mix        float [0, 1]. Sets the amount of unlabelled data.
        --augment       Use data augmentation
        --modality      Set the modality to load. Used in multimodal datasets.
    """
    def __init__(self):
        self.log = None

    def init_logging(self, config):
        if not os.path.exists(config.folder):
            os.makedirs(config.folder)
        logging.basicConfig(filename=config.folder + '/logfile.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())

        self.log = logging.getLogger()
        self.log.debug(config.items())
        self.log.info('---- Setting up experiment at ' + config.folder + '----')

    def get_config(self, split, args):
        """
        Read a config file and convert it into an object for easy processing.
        :param split: the cross-validation split id
        :param args:  the command arguments
        :return: config object(namespace)
        """
        config_script = args.config
        l_mix  = None if not args.l_mix else args.l_mix
        ul_mix = None if not args.ul_mix else args.ul_mix
        modality = None if not args.modality else args.modality

        config_dict = importlib.import_module('configuration.' + config_script).get()
        config = EasyDict(config_dict)
        config.split = int(split)

        if l_mix is not None:
            config.l_mix = float(l_mix)
            if hasattr(config, 'l_mix_vol'):
                config.l_mix_vol = config.l_mix
            config.folder += '_l%s' % l_mix
        if ul_mix is not None:
            config.ul_mix = float(ul_mix)
            config.folder += '_ul%s' % ul_mix
        if modality is not None:
            config.modality = modality
            config.folder += '_' + modality

        config.folder += '_split%s' % split
        config.folder = config.folder.replace('.', '')

        if args.augment:
            config.augment = args.augment

        self.save_config(config)
        return config

    def save_config(self, config):
        if not os.path.exists(config.folder):
            os.makedirs(config.folder)
        with open(config.folder + '/experiment_configuration.json', 'w') as outfile:
            json.dump(dict(config.items()), outfile)

    def run(self):
        args = Experiment.read_console_parameters()

        if args.split == 'all':
            for split in [0, 1, 2]:
                configuration = self.get_config(split, args)
                self.init_logging(configuration)
                self.run_experiment(configuration, args.test)
        else:
            configuration = self.get_config(args.split, args)
            self.init_logging(configuration)
            self.run_experiment(configuration, args.test)

    def run_experiment(self, configuration, test):
        executor = self.get_executor(configuration)

        if test:
            executor.test()
        else:
            executor.train()
            with open(configuration.folder + '/experiment_configuration.json', 'w') as outfile:
                json.dump(vars(configuration), outfile)
            executor.test()

    @staticmethod
    def read_console_parameters():
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--config', default='', help='The experiment configuration file', required=True)
        parser.add_argument('--test', help='Evaluate the model on test data', type=bool)
        parser.add_argument('--split', help='Data split to run.', required=True)
        parser.add_argument('--l_mix', help='Percentage of labelled data')
        parser.add_argument('--ul_mix', help='Percentage of unlabelled data')
        parser.add_argument('--augment', help='Augment training data', type=bool)
        parser.add_argument('--modality', help='Modality to load', choices=['MR', 'CT', 'all', 'cine', 'BOLD'])

        return parser.parse_args()

    def get_executor(self, config):
        # Initialise model
        module_name = config.model.split('.')[0]
        model_name = config.model.split('.')[1]
        model = getattr(importlib.import_module('models.' + module_name), model_name)(config)
        model.build()
        model.compile()

        if config.l_mix == 0.015 and config.split == 1:
            config.seed = 10

        # Initialise executor
        module_name = config.executor.split('.')[0]
        model_name = config.executor.split('.')[1]
        executor = getattr(importlib.import_module('model_executors.' + module_name), model_name)(config, model)
        return executor


if __name__ == '__main__':
    exp = Experiment()
    exp.run()
