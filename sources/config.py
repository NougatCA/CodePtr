import torch
import os
import time
import logging


# paths
dataset_dir = '../dataset/'

if not os.path.exists(dataset_dir):
    raise Exception('Dataset directory not exist.')

source_name = 'token.source'
code_name = 'token.code'
sbt_name = 'token.sbt'
nl_name = 'token.nl'


# outputs
output_root = os.path.join('../output', time.strftime('%Y%m%d_%H%M%S', time.localtime()))

model_root = os.path.join(output_root, 'models')
if not os.path.exists(model_root):
    os.makedirs(model_root)

vocab_root = os.path.join(output_root, 'vocab')
if not os.path.exists(vocab_root):
    os.makedirs(vocab_root)

source_vocab_path = 'source_vocab.pk'
code_vocab_path = 'code_vocab.pk'
ast_vocab_path = 'ast_vocab.pk'
nl_vocab_path = 'nl_vocab.pk'

source_vocab_txt_path = 'source_vocab.txt'
code_vocab_txt_path = 'code_vocab.txt'
ast_vocab_txt_path = 'ast_vocab.txt'
nl_vocab_txt_path = 'nl_vocab.txt'


# logger

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(level=logging.INFO)
logger.addHandler(console)

file = logging.FileHandler(os.path.join(output_root, 'run.log'))
file.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
file.setFormatter(formatter)
logger.addHandler(file)


# device
# use_cuda = False
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# features
use_pointer_gen = True
use_teacher_forcing = True
use_lr_decay = True
use_early_stopping = True

save_valid_model = True
save_best_model = True
save_test_outputs = True


# limitations
max_code_length = 200
max_nl_length = 30
min_nl_length = 4
max_decode_steps = 30
early_stopping_patience = 20


# hyperparameters
source_vocab_size = 30000
code_vocab_size = 30000
nl_vocab_size = 30000

embedding_dim = 256
hidden_size = 256
decoder_dropout_rate = 0.5
teacher_forcing_ratio = 0.5
batch_size = 64
learning_rate = 3e-4
lr_decay_rate = 0.9
n_epochs = 50

beam_width = 5
beam_top_sentences = 1     # number of sentences beam decoder decode for one input, must be 1 (eval.translate_indices)
valid_batch_size = 128
test_batch_size = 128

init_uniform_mag = 0.02
init_normal_std = 1e-4
eps = 1e-12

# visualization and resumes
log_state_every = 1000
