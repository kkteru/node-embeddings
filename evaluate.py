import argparse
import logging
import time

from core import *
from managers import *
from utils import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='TransE model')

parser.add_argument("--experiment_name", type=str, default="default",
                    help="Experiment folder to load model from")


parser.add_argument("--p_norm", type=int, default=1,
                    help="The norm to use for the distance metric")
parser.add_argument("--embedding_dim", type=int, default=50,
                    help="Entity and relations embedding size")
parser.add_argument("--neg_sample_size", type=int, default=100,
                    help="No. of negative samples to compare to for MRR/MR/Hit@10")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--filter', action='store_true',
                    help='Filter the samples while evaluation')
parser.add_argument('--eval_mode', type=str, default="head",
                    help='Evaluate on head and/or tail prediction?')

params = parser.parse_args()

params.device = None
if not params.disable_cuda and torch.cuda.is_available():
    params.device = torch.device('cuda')
else:
    params.device = torch.device('cpu')

logging.info(params.device)

exps_dir = os.path.join(MAIN_DIR, 'experiments')
params.exp_dir = os.path.join(exps_dir, params.experiment_name)

test_data_sampler = DataSampler(TEST_DATA_PATH, ALL_DATA_PATH)
transE = initialize_model(params)
evaluator = Evaluator(transE, test_data_sampler, params)

logging.info('Testing model %s' % os.path.join(params.exp_dir, 'best_model.pth'))


tic = time.time()
log_data = evaluator.get_log_data(params.eval_mode)
toc = time.time()

logging.info('Test performance: %s in %f' % (str(log_data), toc - tic))
