import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim import Adam, SGD, lr_scheduler
from torch_geometric.data import Batch
from pytorch_lightning.metrics.functional import accuracy


from ._morpho_models import DilateDGNN, ErodeDGNN, MorphoGradDGNN
from ._dgcnn import DGCNN

class LitDGNN(pl.LightningModule):
    def __init__(self, k, num_classes, modelname='dilate', optim='adam', lr=1e-4, weight_decay=1e-4):
        super(LitDGNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.optim = optim
        self.weight_decay = weight_decay
        self.lr = lr
        self.modelname = modelname
        if self.modelname == 'dilate':
            self.model = DilateDGNN(k=self.k, num_classes=self.num_classes)
        elif self.modelname == 'erode':
            self.model = ErodeDGNN(k=self.k, num_classes=self.num_classes)
        elif self.modelname == 'grad':
            self.model = MorphoGradDGNN(k=self.k, num_classes=self.num_classes)
        elif self.modelname == 'dgcnn':
            self.model = DGCNN(k=self.k, num_classes=self.num_classes)
        else:
            raise ValueError("modelname can be 'dilate', 'erode', 'grad' or 'dgcnn'."
                             " Value passed is {}}".format(self.modelname))

    def forward(self, x, batch=None):
        return self.model(x=x, batch=batch)

    def configure_optimizers(self):
        if self.optim == 'adam':
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        elif self.optim == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.lr * 100, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError("Optimizer not valid. Please choose between 'adam' or 'sgd'.")

        scheduler =  lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        return [optimizer], [scheduler]

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        y = data.y
        x = data.pos
        batch = data.batch

        y_hat = self(x=x, batch=batch)
        loss = F.nll_loss(y_hat, y)

        _, y_pred = y_hat.max(dim=1)

        acc = accuracy(y_pred, y, num_classes=self.num_classes)

        return {'loss': loss, 'acc': acc, 'progress_bar': {'acc': acc}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc, self.current_epoch)

        return {'loss': avg_loss, 'acc': avg_acc, }

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        y = data.y
        x = data.pos
        batch = data.batch
        y_hat = self(x=x, batch=batch)
        val_loss = F.nll_loss(y_hat, y)

        _, y_pred = y_hat.max(dim=1)
        val_acc = accuracy(y_pred, y, num_classes=self.num_classes)

        return {'val_loss': val_loss, 'val_acc': val_acc, 'progress_bar': {'val_acc': val_acc}}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss, self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def test_step(self, data, batch_idx):
        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        y = data.y
        x = data.pos
        batch = data.batch
        y_hat = self(x=x, batch=batch)

        test_loss = F.nll_loss(y_hat, y)
        _, y_pred = y_hat.max(dim=1)
        test_acc = accuracy(y_pred, y, num_classes=self.num_classes)

        self.logger.experiment.add_scalar("Loss/Test", test_loss, batch_idx)

        self.logger.experiment.add_scalar("Accuracy/Test", test_acc, batch_idx)

        return {'test_loss': test_loss, 'test_acc': test_acc, 'progress_bar': {'test_acc': test_acc}}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        return {'test_loss': avg_loss, 'test_acc': avg_acc}