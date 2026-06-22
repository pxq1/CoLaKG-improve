import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'ml-1m', 'mind', 'fund']
all_models  = ['mf', 'lgn', 'colakg']

config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['use_drop_edge'] = args.use_drop_edge
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['neighbor_k'] = args.neighbor_k
config['dropout_i'] = args.dropout_i
config['dropout_u'] = args.dropout_u
config['dropout_n'] = args.dropout_n
config['use_semantic_gate'] = args.use_semantic_gate
config['layer_cl_reg'] = args.layer_cl_reg
config['layer_cl_temp'] = args.layer_cl_temp
config['semantic_cl_reg'] = args.semantic_cl_reg
config['semantic_cl_temp'] = args.semantic_cl_temp
config['sccf_reg'] = args.sccf_reg
config['sccf_temp'] = args.sccf_temp
config['use_layer_weight'] = args.use_layer_weight
config['use_item_bias'] = args.use_item_bias
config['neg_k'] = max(1, args.neg_k)
config['multi_neg_loss'] = args.multi_neg_loss
config['multi_neg_tau'] = args.multi_neg_tau
config['use_social_graph'] = args.use_social_graph
config['social_alpha'] = args.social_alpha
config['use_itemknn_score'] = args.use_itemknn_score
config['itemknn_alpha'] = args.itemknn_alpha
config['itemknn_train_alpha'] = args.itemknn_alpha if args.itemknn_train_alpha is None else args.itemknn_train_alpha
config['itemknn_k'] = args.itemknn_k
config['use_item_graph_prop'] = args.use_item_graph_prop
config['item_graph_alpha'] = args.item_graph_alpha
config['use_neighbor_cf_prior'] = args.use_neighbor_cf_prior
config['neighbor_cf_alpha'] = args.neighbor_cf_alpha
config['use_ema'] = args.use_ema
config['ema_decay'] = args.ema_decay
config['ema_start_epoch'] = args.ema_start_epoch


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

item_semantic_emb_file = args.item_semantic_emb_file
user_semantic_emb_file = args.user_semantic_emb_file


TRAIN_epochs = args.epochs
EVAL_FREQ = max(1, args.eval_freq)
checkpoint_tag = args.checkpoint_tag
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
