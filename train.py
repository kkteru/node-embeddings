import argparse
import logging
import time

from core import *
from managers import *
from utils import *
import torch

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='TransE model')

parser.add_argument("--experiment_name", type=str, default="default",
                    help="A folder with this name would be created to dump saved models and log files")

parser.add_argument("--nEpochs", type=int, default=1000,
                    help="Learning rate of the optimizer")
parser.add_argument("--nBatches", type=int, default=200,
                    help="Batch size")
parser.add_argument("--eval_every", type=int, default=25,
                    help="Interval of epochs to evaluate the model?")
parser.add_argument("--save_every", type=int, default=50,
                    help="Interval of epochs to save a checkpoint of the model?")

parser.add_argument("--sample_size", type=int, default=0,
                    help="No. of negative samples to compare to for MRR/MR/Hit@10")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
parser.add_argument("--margin", type=int, default=1,
                    help="The margin between positive and negative samples in the max-margin loss")
parser.add_argument("--p_norm", type=int, default=1,
                    help="The norm to use for the distance metric")
parser.add_argument("--optimizer", type=str, default="SGD",
                    help="Which optimizer to use? SGD/Adam")
parser.add_argument("--embedding_dim", type=int, default=50,
                    help="Entity and relations embedding size")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate of the optimizer")
parser.add_argument("--momentum", type=float, default=0,
                    help="Momentum of the SGD optimizer")

parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Run the code in debug mode?")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--filter', action='store_true',
                    help='Filter the samples while evaluation')

params = parser.parse_args()

initialize_experiment(params)

params.device = None
if not params.disable_cuda and torch.cuda.is_available():
    params.device = torch.device('cuda')
else:
    params.device = torch.device('cpu')

logging.info(params.device)

train_data_sampler = DataSampler(TRAIN_DATA_PATH, params.debug)
valid_data_sampler = DataSampler(VALID_DATA_PATH)
transE = initialize_model(params)
trainer = Trainer(transE, train_data_sampler, params)
evaluator = Evaluator(transE, valid_data_sampler, params.sample_size)

batch_size = int(len(train_data_sampler.data) / params.nBatches)

logging.info('Starting training with batch size = %d' % batch_size)

# tb_logger = Logger(params.exp_dir)

for e in range(params.nEpochs):
    res = 0
    tic = time.time()
    for b in range(params.nBatches):
        loss = trainer.one_step(batch_size)
        res += loss
    toc = time.time()

    # tb_logger.scalar_summary('loss', loss, e)

    logging.info('Epoch %d with loss: %f in %f'
                 % (e, res, toc - tic))

    if (e + 1) % params.eval_every == 0:
        tic = time.time()
        log_data = evaluator.get_log_data()
        toc = time.time()
        logging.info('Performance: %s in %f' % (str(log_data), (toc - tic)))

        # for tag, value in log_data.items():
        #     tb_logger.scalar_summary(tag, value, e + 1)

        # to_continue = trainer.select_model(log_data)
        # if not to_continue:
        #     break
    if (e + 1) % params.save_every == 0:
        torch.save(transE, os.path.join(params.exp_dir, 'checkpoint.pth'))
