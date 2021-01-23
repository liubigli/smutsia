import os
import yaml
import torch
import argparse
from glob import glob
import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from smutsia.nn import MLP
from smutsia.nn.models._siamese_network import FeatureExtraction, SimilarityHypHC
from smutsia.utils.data import ToyDatasets
from smutsia.utils.logger import  MyTensorBoardLogger


def load_weights(model, weights, map_location='cpu'):
    ckpt = torch.load(weights, map_location=map_location)
    model.load_state_dict(ckpt['state_dict'])
    print(weights)

    return model


def read_hparams(path):
    print(path)
    # reading parameters
    with open(path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        hparams = yaml.load(file, Loader=yaml.FullLoader)
    print(hparams)
    return hparams


def load_model(path, eval_mode=True):
    hparams = read_hparams(os.path.join(path, 'hparams.yaml'))
    embedder = hparams['embedder'] == 'True'
    hidden = hparams['hidden']
    dropout = hparams['dropout']
    negative_slope = hparams['negative_slope']
    out_features = hparams['hidden'] if embedder else 2
    if hparams['model'] == 'dgcnn':
        nn = FeatureExtraction(in_channels=2,
                               hidden_features=hparams['hidden'],
                               out_features=out_features,
                               k=hparams['k'],
                               transformer=False,
                               dropout=hparams['dropout'],
                               negative_slope=hparams['negative_slope'],
                               cosine=hparams['cosine'] == 'True')
    else:
        nn = MLP([2, hidden, hidden, hidden, hidden, out_features], dropout=dropout, negative_slope=negative_slope)

    nn_emb = MLP([hidden, hidden, 2], dropout=dropout, negative_slope=negative_slope) if embedder else None

    model = SimilarityHypHC(nn=nn,
                            embedder=nn_emb,
                            temperature=hparams['temperature'],
                            anneal=hparams['annealing'],
                            anneal_step=hparams['anneal_step'],
                            sim_distance=hparams['distance'],
                            margin=hparams['margin'])

    weights_path = glob(os.path.join(path, '*.ckpt'))[-1]
    model = load_weights(model, weights_path)
    model.hparams = hparams

    if eval_mode:
        model.eval()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='test_hyp200', type=str, help='dirname for logs')
    parser.add_argument('--path', type=str, help='path to model ckpt')
    parser.add_argument('--data', type=str, help='type of toy dataset to use for test')
    parser.add_argument('--test_samples', default=20, type=int, help='number of samples in test set')
    parser.add_argument('--min_noise', default=0.0, type=float, help='min value of noise to use')
    parser.add_argument('--max_noise', default=0.16, type=float, help='max value of noise to use')
    parser.add_argument('--cluster_std', default=0.16, type=float, help='std blobs/aniso/varied')
    parser.add_argument('--max_points', default=300, type=int, help='number of points in each sample')
    parser.add_argument('--num_blobs', default=3, type=int, help='number of blobs in blob/aniso/varied')
    parser.add_argument('--gpu', default=-1, type=int, help='use gpu')

    args = parser.parse_args()

    logdir = args.logdir
    path = args.path
    dataname = args.data
    test_samples = args.test_samples
    min_noise = args.min_noise
    max_noise = args.max_noise
    cluster_std = args.cluster_std
    max_points = args.max_points
    num_blobs = args.num_blobs
    noise = (min_noise, max_noise)
    gpu = 0 if args.gpu == -1 else [args.gpu]

    test_seed = 19
    test_dataset = ToyDatasets(name=dataname, length=test_samples, noise=noise, cluster_std=cluster_std,
                               num_blobs=num_blobs, max_samples=max_points, num_labels=0.3, seed=test_seed)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)


    model = load_model(path=path)
    logger = MyTensorBoardLogger(logdir, name=dataname)
    logger.log_hyperparams_metrics(params=model.hparams,
                                   metrics={'ari@k': 0.0, 'ari@k-std': 0.0,
                                            'acc@k':0.0, 'acc@k-std':0.0,
                                            'purity@k':0.0, 'purity@k-std':0.0,
                                            'nmi@k':0.0, 'nmi@k-std':0.0,
                                            'ari': 0.0, 'ari-std': 0.0,
                                            'best_k': 0.0, 'std_k': 0.0})

    trainer = pl.Trainer(gpus=gpu, max_epochs=1,
                         logger=logger,
                         track_grad_norm=2)

    results = trainer.test(model, test_loader)
