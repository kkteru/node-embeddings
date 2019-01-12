import os
import argparse
import logging
import json

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(MAIN_DIR, 'data/FB15K237')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train2id.txt')
VALID_DATA_PATH = os.path.join(DATA_PATH, 'valid2id.txt')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test2id.txt')


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. use 0 or 1")


def initialize_experiment(params):

    exps_dir = os.path.join(MAIN_DIR, 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)
