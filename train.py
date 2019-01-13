import argparse
import logging

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

parser.add_argument("--sample_size", type=int, default=30,
                    help="No. of negative samples to compare to for MRR/MR/Hit@10")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
parser.add_argument("--margin", type=int, default=1,
                    help="The margin between positive and negative samples in the max-margin loss")
parser.add_argument("--p_norm", type=int, default=2,
                    help="The norm to use for the distance metric")
parser.add_argument("--optimizer", type=str, default="SGD",
                    help="Which optimizer to use?")
parser.add_argument("--embedding_dim", type=int, default=100,
                    help="Entity and relations embedding size")
parser.add_argument("--lr", type=int, default=0.1,
                    help="Learning rate of theoptimizer")

parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Run the code in debug mode?")

params = parser.parse_args()

initialize_experiment(params)

train_data_sampler = DataSampler(TRAIN_DATA_PATH, params.debug)
valid_data_sampler = DataSampler(VALID_DATA_PATH)
transE = initialize_model(params)
trainer = Trainer(transE, train_data_sampler, params)
evaluator = Evaluator(transE, valid_data_sampler, params.sample_size)

batch_size = int(len(train_data_sampler.data) / params.nBatches)

logging.info('Batch size = %d' % batch_size)

for e in range(params.nEpochs):
    for b in range(params.nBatches):
        loss = trainer.one_step(batch_size)

    logging.info('Epoch %d, Loss: %f Entity embeddings mean norm : %f, Relations embeddings mean norm : %f'
                 % (e, loss, torch.mean(torch.norm(transE.ent_embeddings.weight.data, dim=-1)), torch.mean(torch.norm(transE.rel_embeddings.weight.data, dim=-1))))

    if (e + 1) % params.eval_every == 0:
        log_data = evaluator.get_log_data()
        logging.info('Performance:' + str(log_data))
        to_continue = trainer.select_model(log_data)
        if not to_continue:
            break
    if (e + 1) % params.save_every == 0:
        torch.save(transE, os.path.join(params.exp_dir, 'checkpoint.pth'))
