
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go Model")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--neighbor_k', type=int,default=10,
                        help="the num of neighbors")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--use_drop_edge', type=int,default=1,
                        help="using the drop_edge or not for lightgcn")
    parser.add_argument('--keepprob', type=float,default=0.7,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--dropout_i', type=float,default=0.6,
                        help="the dropout for item semantic embeddings")
    parser.add_argument('--dropout_u', type=float,default=0.6,
                        help="the dropout for user semantic embeddings")
    parser.add_argument('--dropout_n', type=float,default=0.6,
                        help="the dropout for neighbor embeddings")
    parser.add_argument('--use_semantic_gate', type=int,default=0,
                        help="whether to use adaptive gates for id-semantic and neighbor fusion")
    parser.add_argument('--layer_cl_reg', type=float,default=0.0,
                        help="regularization weight for layer-to-layer contrastive learning")
    parser.add_argument('--layer_cl_temp', type=float,default=0.2,
                        help="temperature for layer-to-layer contrastive learning")
    parser.add_argument('--semantic_cl_reg', type=float,default=0.0,
                        help="regularization weight for entity semantic contrastive alignment")
    parser.add_argument('--semantic_cl_temp', type=float,default=0.2,
                        help="temperature for entity semantic contrastive alignment")
    parser.add_argument('--sccf_reg', type=float,default=0.0,
                        help="regularization weight for SCCF-style user-item contrastive loss")
    parser.add_argument('--sccf_temp', type=float,default=0.2,
                        help="temperature for SCCF-style user-item contrastive loss")
    parser.add_argument('--use_layer_weight', type=int,default=0,
                        help="whether to learn aggregation weights over graph convolution layers")
    parser.add_argument('--use_item_bias', type=int,default=0,
                        help="whether to use learnable item bias in ranking scores")
    parser.add_argument('--neg_k', type=int,default=1,
                        help="number of sampled negative candidates per positive interaction")
    parser.add_argument('--multi_neg_loss', type=int,default=0,
                        help="whether to aggregate multiple sampled negatives with log-sum-exp BPR")
    parser.add_argument('--multi_neg_tau', type=float,default=1.0,
                        help="temperature for log-sum-exp multiple-negative aggregation")
    parser.add_argument('--use_social_graph', type=int,default=0,
                        help="whether to use LastFM user social graph for user representation enhancement")
    parser.add_argument('--social_alpha', type=float,default=0.2,
                        help="residual weight for social graph propagated user embeddings")
    parser.add_argument('--use_itemknn_score', type=int,default=0,
                        help="whether to add train-only item-item KNN residual scores")
    parser.add_argument('--itemknn_alpha', type=float,default=0.2,
                        help="weight for item-item KNN residual scores")
    parser.add_argument('--itemknn_train_alpha', type=float,default=None,
                        help="weight for item-item KNN residual scores in BPR training; defaults to itemknn_alpha")
    parser.add_argument('--itemknn_k', type=int,default=200,
                        help="top-k neighbors retained in the item-item KNN graph")
    parser.add_argument('--use_item_graph_prop', type=int,default=0,
                        help="whether to propagate item representations over semantic item-item graph after each GCN layer")
    parser.add_argument('--item_graph_alpha', type=float,default=0.1,
                        help="residual weight for semantic item graph propagation")
    parser.add_argument('--use_neighbor_cf_prior', type=int,default=0,
                        help="whether to denoise semantic neighbor attention with train-only item co-occurrence prior")
    parser.add_argument('--neighbor_cf_alpha', type=float,default=0.2,
                        help="mixture weight for collaborative prior in semantic neighbor attention")
    parser.add_argument('--use_ema', type=int,default=0,
                        help="whether to evaluate with exponential moving average model parameters")
    parser.add_argument('--ema_decay', type=float,default=0.995,
                        help="decay factor for exponential moving average parameters")
    parser.add_argument('--ema_start_epoch', type=int,default=400,
                        help="epoch number to start updating exponential moving average parameters")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='ml-1m',
                        help="available datasets")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--item_semantic_emb_file', type=str,default=" ",
                        help="the path of item_semantic_emb_file")
    parser.add_argument('--user_semantic_emb_file', type=str,default=" ",
                        help="the path of user_semantic_emb_file")
    parser.add_argument('--topks', nargs='?',default="[10,20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--eval_freq', type=int,default=5)
    parser.add_argument('--checkpoint_tag', type=str,default="")
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='colakg', help='rec-model, support [mf, lgn, colakg]')
    return parser.parse_args()
