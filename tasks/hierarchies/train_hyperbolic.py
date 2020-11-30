import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.data import DataLoader
from smutsia.nn.models._siamese_network import SiameseHyperbolic, FeatureExtraction
from smutsia.nn import MLP
from smutsia.utils.data import ToyDatasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='moons', type=str, help='name of dataset to use')
    parser.add_argument('--num_samples', default=100, type=int, help='number of samples in training set')
    parser.add_argument('--max_points', default=300, type=int, help='number of points in each sample')
    parser.add_argument('--num_labels', default=0.3, type=float, help='number/ratio of labels to use in each sample')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--hidden', default=64, type=int, help='number of hidden features')
    parser.add_argument('--triplets', default=30, type=int, help='number of points to sample to generate triplets')
    parser.add_argument('--margin', default=1.0, type=float, help='margin value to use in triplet loss')
    parser.add_argument('--patience', default=50, type=int, help='patience value for early stopping')
    parser.add_argument('--plot', default=-1, type=int, help='interval in which we plot prediction on validation batch')
    parser.add_argument('--gpu', default=0, type=int, help='use gpu')
    args = parser.parse_args()

    dataname = args.data
    epochs = args.epochs
    num_samples = args.num_samples
    max_points = args.max_points
    hidden = args.hidden
    lr = args.lr
    margin = args.margin
    patience = args.patience
    plot = args.plot
    gpu = args.gpu
    num_labels = args.num_labels

    # load dataset
    noise = (0.12, 0.15)
    # number of points in each sample
    seed = 5
    train_dataset = ToyDatasets(name=dataname, length=num_samples, noise=noise, max_samples=max_points, num_labels=num_labels, seed=seed)
    valid_dataset = ToyDatasets(name=dataname, length=10, noise=noise, max_samples=max_points, seed=2, num_labels=num_labels)
    test_dataset = ToyDatasets(name=dataname, length=1, noise=noise, max_samples=max_points, seed=seed, num_labels=num_labels)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)

    # todo parametrize this
    nn = FeatureExtraction(in_channels=2, hidden_features=hidden, out_features=hidden, k=30, transformer=False)
    # embedder = MLP([hidden, hidden, 2], negative_slope=0.2)
    model = SiameseHyperbolic(nn=nn,
                              # embedder=embedder,
                              margin=margin,
                              plot_interval=plot)

    logger = TensorBoardLogger('hyperbolic', name=dataname)
    logger.log_hyperparams({'dataset': dataname,
                            'hidden': hidden,
                            'max_epochs': epochs,
                            'margin': margin})

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

    # trainer.test(model, valid_loader)