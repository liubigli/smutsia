import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DistributedSampler
from torch_geometric.datasets import ShapeNet
from torch_geometric import transforms as T
from torch_geometric.data import DataListLoader, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from smutsia.nn.models import LitDGNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Test DGCNN vs MorphoDGCNN')
    parser.add_argument('--model', type=str, default='dilate')
    parser.add_argument('--datapath', type=str, default='data/modelnet')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--k', type=int, default=40, help='number of neighboors in graph')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer; adam or SGD')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience')
    parser.add_argument('--logdir', type=str, default='./logdir/')
    parser.add_argument('--test', help='if True test a model', action='store_true')
    parser.add_argument('--distributed', help='if True run on a cluster machine', action='store_true')

    args = parser.parse_args()
    model_arch = args.model
    path = args.datapath
    gpu = args.gpu
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    k = args.k
    optim = args.optimizer
    lr = args.lr
    weight_decay = args.weight_decay
    patience = args.patience
    logdir = args.logdir
    test = args.test
    distr = args.distributed

    num_classes = 4
    categories = ['Airplane']
    include_normals = False

    model = LitDGNN(k=k, num_classes=num_classes, modelname=model_arch, optim=optim, lr=lr, weight_decay=weight_decay)

    logger = TensorBoardLogger(logdir, name=model_arch)
    logger.log_hyperparams({'Model': model_arch,
                            'k': k,
                            'num_classes': num_classes,
                            'categories': categories,
                            'include_normals': include_normals,
                            'max_epochs': epochs,
                            'batch_size': batch_size,
                            'optim': optim,
                            'lr': lr})

    savedir = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version), 'checkpoints')

    # call backs for trainer
    checkpoint_callback = ModelCheckpoint(filepath=savedir, verbose=True)
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode='max')

    # training with multiple gpu-s
    if len(gpu.split(',')) > 1:
        # check if training on a cluster or not
        distributed_backend = 'ddp' if distr else 'dp'
        replace_sampler_ddp = False if distr else True
    else:
        distributed_backend = None
        replace_sampler_ddp = True

    trainer = pl.Trainer(gpus=gpu,
                         max_epochs=epochs,
                         logger=logger,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback,
                         distributed_backend=distributed_backend,
                         replace_sampler_ddp=replace_sampler_ddp)

    # loading dataset
    train_dataset = ShapeNet(root=path,
                             categories=categories,
                             include_normals=include_normals,
                             split='train',
                             pre_transform=T.NormalizeScale()
                             )

    valid_dataset = ShapeNet(root=path,
                             categories=categories,
                             include_normals=include_normals,
                             split='val',
                             pre_transform=T.NormalizeScale()
                             )

    if len(gpu.split(',')) > 1:
        num_replicas = len(gpu.split(','))
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=num_replicas, rank=trainer.global_rank) if distr else None

        valid_sampler = DistributedSampler(valid_dataset,
                                           num_replicas=num_replicas, rank=trainer.global_rank) if distr else None

        train_loader = DataListLoader(train_dataset,
                                      batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)

        valid_loader = DataListLoader(valid_dataset,
                                      batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)

    trainer.fit(model, train_loader, valid_loader)
    print("END TRAINING")

    if test:
        test_dataset = ShapeNet(root=path,
                                categories=categories,
                                include_normals=include_normals,
                                split='test',
                                pre_transform=T.NormalizeScale()
                                )

        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        if len(gpu.split(',')) > 1:
            gpu_test = gpu.split(',')[0]
        else:
            gpu_test = gpu

        trainer = pl.Trainer(gpus=gpu_test,
                             max_epochs=epochs,
                             logger=logger,
                             checkpoint_callback=checkpoint_callback,
                             early_stop_callback=early_stop_callback
                             )

        trainer.test(model, test_dataloaders=test_loader)
