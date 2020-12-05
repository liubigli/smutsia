import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.data import DataLoader
from smutsia.nn.models._siamese_network import SiameseHyperbolic, FeatureExtraction
from smutsia.nn import MLP
from smutsia.utils.data import ToyDatasets

class MyTensorBoardLogger(TensorBoardLogger):
    def __init__(self, *args, **kwargs):
        super(MyTensorBoardLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, *args, **kwargs):
        pass

    @rank_zero_only
    def log_hyperparams_metrics(self, params: dict, metrics: dict) -> None:
        from torch.utils.tensorboard.summary import hparams
        params = self._convert_params(params)
        exp, ssi, sei = hparams(params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)
        # some alternative should be added
        self.hparams.update(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='moons', type=str, help='name of dataset to use')
    parser.add_argument('--train_samples', default=100, type=int, help='number of samples in training set')
    parser.add_argument('--valid_samples', default=10, type=int, help='number of samples in valid set')
    parser.add_argument('--test_samples', default=10, type=int, help='number of samples in test set')
    parser.add_argument('--max_points', default=300, type=int, help='number of points in each sample')
    parser.add_argument('--num_labels', default=0.3, type=float, help='number/ratio of labels to use in each sample')
    parser.add_argument('--min_noise', default=0.12, type=float, help='min value of noise to use')
    parser.add_argument('--max_noise', default=0.15, type=float, help='max value of noise to use')
    parser.add_argument('--cluster_std', default=0.1, type=float, help='std blobs')
    parser.add_argument('--model', default='dgcnn', type=str, help='model to use to extract features')
    parser.add_argument('--embedder', help='if True add a an embedding model from the feature space to B2', action='store_true')
    parser.add_argument('--k', default=30, type=int, help='if model dgcnn, k is the number of neigh to take into account')
    parser.add_argument('--hidden', default=64, type=int, help='number of hidden features')
    parser.add_argument('--negative_slope', default=0.2, type=float, help='negative slope for leaky relu in the feature extractor')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout in the feature extractor')
    parser.add_argument('--distance', default='cosine', type=str, help='distance to use to compute triplets')
    parser.add_argument('--margin', default=1.0, type=float, help='margin value to use in triplet loss')
    parser.add_argument('--temperature', default=0.05, type=float, help='rescale softmax value used in the hyphc loss')
    parser.add_argument('--batch', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--patience', default=50, type=int, help='patience value for early stopping')
    parser.add_argument('--plot', default=-1, type=int, help='interval in which we plot prediction on validation batch')
    parser.add_argument('--gpu', default=0, type=int, help='use gpu')
    args = parser.parse_args()

    dataname = args.data
    epochs = args.epochs
    train_samples = args.train_samples
    valid_samples = args.valid_samples
    test_samples = args.test_samples
    num_labels = args.num_labels
    max_points = args.max_points
    min_noise = args.min_noise
    max_noise = args.max_noise
    cluster_std = args.cluster_std
    model_name = args.model
    embedder = args.embedder
    k = args.k
    hidden = args.hidden
    negative_slope = args.negative_slope
    dropout = args.dropout
    distance = args.distance
    margin = args.margin
    temperature = args.temperature
    batch = args.batch
    lr = args.lr
    patience = args.patience
    plot_every = args.plot
    gpu = args.gpu

    # load dataset
    noise = (min_noise, max_noise)
    # number of points in each sample
    train_seed = 5
    valid_seed = 2
    test_seed = 19
    train_dataset = ToyDatasets(name=dataname, length=train_samples, noise=noise, cluster_std=cluster_std, max_samples=max_points, num_labels=num_labels, seed=train_seed)
    valid_dataset = ToyDatasets(name=dataname, length=valid_samples, noise=noise, cluster_std=cluster_std, max_samples=max_points, num_labels=num_labels,  seed=valid_seed)
    test_dataset = ToyDatasets(name=dataname, length=test_samples, noise=noise, cluster_std=cluster_std, max_samples=max_points, num_labels=num_labels, seed=test_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    out_features = hidden if embedder else 2
    # todo parametrize this
    if model_name == 'dgcnn':
        nn = FeatureExtraction(in_channels=2, hidden_features=hidden, out_features=out_features, k=k, transformer=False,
                               dropout=dropout, negative_slope=negative_slope, cosine=False)
    else:
        nn = MLP([2, hidden, hidden, out_features], dropout=dropout, negative_slope=negative_slope)

    nn_emb = MLP([hidden, hidden, 2], dropout=dropout, negative_slope=negative_slope) if embedder else None

    model = SiameseHyperbolic(nn=nn,
                              embedder=nn_emb,
                              sim_distance=distance,
                              margin=margin,
                              temperature=temperature,
                              plot_every=plot_every)

    logger = MyTensorBoardLogger('hyperbolic', name=dataname)
    model_params = {'dataset': dataname,
                    'num_clusters': train_dataset.num_classes,
                    'ratio_labels': num_labels,
                    'min_noise': min_noise if dataname in ['moons', 'circles'] else None,
                    'max_noise': max_noise if dataname in ['moons', 'circles'] else None,
                    'cluster_std': cluster_std if dataname in ['blobs', 'varied', 'aniso'] else None,
                    'model': model_name,
                    'embedder': 'True' if embedder else 'False',
                    'distance': distance,
                    'hidden': hidden,
                    'negative_slope': negative_slope,
                    'dropout': dropout,
                    'margin': margin,
                    'temperature': temperature,
                    'max_epochs': epochs,
                    'batch': batch,
                    'lr': lr}
    print(model_params)

    logger.log_hyperparams_metrics(params=model_params, metrics={'ari@k': 0.0, 'ari': 0.0, 'best_k': 0.0, 'std_k':0.0})
    savedir = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version), 'checkpoints')
    # call backs for trainer
    checkpoint_callback = ModelCheckpoint(filepath=savedir, verbose=True)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode='min')

    trainer = pl.Trainer(gpus=gpu, max_epochs=epochs,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback,
                         logger=logger,
                         track_grad_norm=2)

    trainer.fit(model, train_loader, valid_loader)

    print("End Training")

    results = trainer.test(model, test_loader)
