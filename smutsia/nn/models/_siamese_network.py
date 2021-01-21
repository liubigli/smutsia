import torch
import math
import numpy as np
import pytorch_lightning as pl
from typing import Union

from scipy.cluster.hierarchy import fcluster, linkage

from torch.nn import functional as F, Linear, Sequential as Seq, BatchNorm1d
from torch.optim import Adam, lr_scheduler
from torch_geometric.data import Batch
from pytorch_metric_learning import losses, distances, regularizers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from sklearn.metrics.cluster import adjusted_rand_score as ri

from smutsia.utils.viz import plot_hyperbolic_eval
from smutsia.utils.scores import eval_clustering, get_optimal_k
from smutsia.nn.distances import HyperbolicLCA, HyperbolicDistance
from smutsia.nn.distances.poincare import project
from smutsia.nn.conv import DynamicEdgeConv

from smutsia.nn.optim import RAdam
from . import TransformNet
from .. import MLP


class ComplexFeatExtract(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_features: int, negative_slope: float = 0.2, bias: bool = True,
                 dropout: float = 0.0, init_gamma: float = math.pi / 2):
        super(ComplexFeatExtract, self).__init__()

        self.gamma = torch.nn.Parameter(torch.Tensor([init_gamma]), requires_grad=True)

        self.mlp1 = MLP([in_channels, hidden_features, hidden_features], negative_slope=negative_slope,
                       dropout=dropout, bias=bias)

        self.mlp2 = MLP([in_channels, hidden_features, hidden_features], negative_slope=negative_slope, dropout=dropout,
                        bias=bias)
        self.linear = Seq(
                Linear(in_features=hidden_features, out_features=1, bias=True),
                BatchNorm1d(1)
        )

    def forward(self, x):
        x = self.mlp1(x)
        # x2 = self.mlp2(x)
        # return torch.cat([x1, x2], dim=1)
        x = self.linear(x)
        # x = (x - x.min()) / (x.max() - x.min())
        return torch.cat([torch.cos(self.gamma*x), torch.sin(self.gamma*x)], dim=1)


class FeatureExtraction(torch.nn.Module):
    def __init__(self, in_channels: int, out_features: int, hidden_features: int, k: int, transformer: bool = False,
                 negative_slope: float = 0.2, dropout=0.0, cosine=False):
        super(FeatureExtraction, self).__init__()
        self.in_channels = in_channels
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.k = k
        self.negative_slope = negative_slope
        self.cosine = cosine
        self.transformer = transformer

        if self.transformer:
            self.tnet = TransformNet()

        self.conv1 = DynamicEdgeConv(
            nn=MLP([2 * in_channels, hidden_features], dropout=dropout, negative_slope=self.negative_slope),
            k=self.k,
            cosine=False,
        )
        self.conv2 = DynamicEdgeConv(
            nn=MLP([2 * hidden_features, hidden_features], dropout=dropout, negative_slope=self.negative_slope),
            k=self.k,
            cosine=False,
        )
        self.conv3 = DynamicEdgeConv(
            nn=MLP([2 * hidden_features, out_features], dropout=dropout, negative_slope=self.negative_slope),
            k=self.k,
            cosine=self.cosine,
        )

    def forward(self, x, batch=None):
        if self.transformer:
            tr = self.tnet(x, batch=batch)

            if batch is None:
                x = torch.matmul(x, tr[0])
            else:
                batch_size = batch.max().item() + 1
                x = torch.cat([torch.matmul(x[batch == i], tr[i]) for i in range(batch_size)])

        x = self.conv1(x, batch=batch)
        x = self.conv2(x, batch=batch)
        x = self.conv3(x, batch=batch)

        return x


class SiameseHyperbolic(pl.LightningModule):
    """

    Parameters
    ----------
    nn: torch.nn.Module
        model used to do feature extraction
        
    embedder: Union[torch.nn.Module, None]
        if not None, module used to embed features from initial space to Poincare's Disk
    
    sim_distance: optional {'cosine', 'hyperbolic'}
        similarity distance to use to compute the triplet loss function in the features' space
        
    temperature: float
        factor used in the HypHC loss
        
    margin: float
        margin value used in the triplet loss
         
    init_rescale: float
        scale value used to rescale leaf embeddings in the Poincare's Disk
        
    max_scale: float
        max scale value to use to rescale leaf embeddings
        
    lr: float
        learning rate
        
    patience: int
        patience value for the scheduler

    factor: float
        learning rate reduction factor

    min_lr: float
        minimum value for learning rate

    plot_every: int
        plot validation value every #plot_every epochs

    """
    def __init__(self, nn: torch.nn.Module, embedder: Union[torch.nn.Module, None],
                 sim_distance: str = 'cosine', temperature: float = 0.05, anneal: float = 0.5, anneal_step: int = 0,
                 margin: float = 1.0, init_rescale: float = 1e-3, max_scale: float = 1.-1e-3, lr: float = 1e-3,
                 patience: int = 10, factor: float = 0.5, min_lr: float = 1e-4,
                 plot_every: int = -1):
        super(SiameseHyperbolic, self).__init__()
        self.model = nn
        self.embedder = embedder
        self.rescale = torch.nn.Parameter(torch.Tensor([init_rescale]), requires_grad=True)
        self.temperature = temperature
        self.anneal = anneal
        self.anneal_step = anneal_step
        self.max_scale = max_scale
        self.margin = margin
        self.distance_lca = HyperbolicLCA()

        if sim_distance == 'cosine':
            self.distace_sim = distances.CosineSimilarity()
            # self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin)
        elif sim_distance == 'hyperbolic':
            self.distace_sim = HyperbolicDistance()
            # self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin)
        elif sim_distance == 'euclidean':
            self.distace_sim = distances.LpDistance()
            # self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin,
            #                                                  embedding_regularizer=regularizers.LpRegularizer())
        else:
            raise ValueError(f"The option {sim_distance} is not available for sim_distance. The only available are ['cosine', 'euclidean', 'hyperbolic'].")

        self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin)


        print("MARGIN", self.margin)
        # learning rate
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.plot_interval = plot_every
        self.plot_step = 0

    def _rescale_emb(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-4  # self.init_size
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.rescale.clamp_min(min_scale).clamp_max(max_scale)

    def _loss(self, x_feat, y, x_emb, labels=None):
        if labels is not None:
            x_feat_samples = x_feat[labels]
            x_emb_samples = x_emb[labels]
            y_samples = y[labels]
        else:
            x_feat_samples = x_feat
            x_emb_samples = x_emb
            y_samples = y
        # todo: change the value of parameter t_per_anchor from 'all' to a int value as for example 100
        indices_tuple = lmu.convert_to_triplets(None, y_samples, t_per_anchor='all')

        anchor_idx, positive_idx, negative_idx = indices_tuple
        # print("Len Anchor: ", len(anchor_idx))
        if len(anchor_idx) == 0:
            return self.zero_losses()
        # #
        if isinstance(self.distace_sim, distances.CosineSimilarity):
            mat_sim = 0.5 * (1 + self.distace_sim(x_feat_samples))
        else:
            # mat_sim = 0.5 * (1 + self.distace_sim(project(x_emb_samples)))
            mat_sim = torch.exp(-self.distace_sim(x_feat_samples))
            # print("euclidean", mat_sim.max(), mat_sim.min())
        #
        # mat_sim = self.distace_sim(x_feat_samples)
        mat_lca = self.distance_lca(self._rescale_emb(x_emb_samples))
        # print(f"sim values: max {mat_sim.max()}, min: {mat_sim.min()}")
        wij = mat_sim[anchor_idx, positive_idx]
        wik = mat_sim[anchor_idx, negative_idx]
        wjk = mat_sim[positive_idx, negative_idx]

        dij = mat_lca[anchor_idx, positive_idx]
        dik = mat_lca[anchor_idx, negative_idx]
        djk = mat_lca[positive_idx, negative_idx]

        # loss proposed by Chami et al.
        sim_triplet = torch.stack([wij, wik, wjk]).T
        lca_triplet = torch.stack([dij, dik, djk]).T
        weights = torch.softmax(lca_triplet / self.temperature, dim=-1)

        w_ord = torch.sum(sim_triplet * weights, dim=-1, keepdim=True)
        total = torch.sum(sim_triplet, dim=-1, keepdim=True) - w_ord
        loss_triplet_lca = torch.mean(total) + mat_sim.mean()

        loss_triplet_sim = self.loss_triplet_sim(x_feat_samples, y_samples)
        # print("Sim loss: ", loss_triplet_sim)
        return loss_triplet_sim, loss_triplet_lca

    def _decode_linkage(self, leaves_embeddings):
        """Build linkage matrix from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = self._rescale_emb(leaves_embeddings)
        sim_fn = lambda x, y: np.arccos(np.clip(np.sum(x * y, axis=-1), -1.0, 1.0))
        embeddings = F.normalize(leaves_embeddings, p=2, dim=1).detach().cpu()
        Z = linkage(embeddings, method='single', metric=sim_fn)

        return Z

    def forward(self, x, y, labels=None, batch=None, decode=False):

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)

        batch_size = batch.max() + 1

        # feature extractor
        x = self.model(x)

        if isinstance(self.embedder, torch.nn.Module):
            x_emb = self.embedder(x)
        else:
            x_emb = x

        loss_triplet = 0.0
        loss_hyphc = 0.0
        linkage_mat = []

        for i in range(batch_size):
            l_tri, l_hyphc = self._loss(x_feat=x, y=y, x_emb=x_emb, labels=labels)
            loss_triplet += l_tri
            loss_hyphc += l_hyphc
            if decode:
                Z = self._decode_linkage(x_emb[batch == i])
                linkage_mat.append(Z)

        loss_triplet = loss_triplet / batch_size
        loss_hyphc = loss_hyphc / batch_size

        return x_emb, loss_triplet, loss_hyphc, linkage_mat

    def _forward(self, data, decode=False):
        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        x = data.x
        y = data.y
        batch = data.batch
        if hasattr(data, 'labels'):
            labels = data.labels
        else:
            labels = None

        x, loss_triplet, loss_hyphc, link_mat = self(x=x, y=y, labels=labels, batch=batch, decode=decode)

        return x, loss_triplet, loss_hyphc, link_mat

    def _get_optimal_k(self, y, linkage_matrix):
        best_ri = 0.0
        n_clusters = y.max() + 1
        # min_num_clusters = max(n_clusters - 1, 1)
        best_k = 0
        best_pred = None
        for k in range(1, n_clusters+5):
            # print(k)
            y_pred = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            k_ri = ri(y, y_pred)
            if k_ri > best_ri:
                best_ri = k_ri
                best_k = k
                best_pred = y_pred

        return best_pred, best_k, best_ri

    def configure_optimizers(self):
        optim = RAdam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optim,
                                                   mode='min',
                                                   factor=self.factor,
                                                   patience=self.patience,
                                                   min_lr=self.min_lr,
                                                   verbose=True)

        return [optim], [scheduler]

    def training_step(self, data, batch_idx):
        x, loss_triplet, loss_hyphc, _ = self._forward(data)
        loss = loss_triplet + loss_hyphc
        return {'loss': loss, 'progress_bar': {'triplet': loss_triplet, 'hyphc': loss_hyphc}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # avg_ri = torch.stack([x['ri'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar("RandScore/Train", avg_ri, self.current_epoch)
        if self.current_epoch and self.anneal_step > 0 and self.current_epoch % self.anneal_step == 0:
            print(f"Annealing temperature at the end of epoch {self.current_epoch}")
            max_temp = 0.8
            min_temp = 0.01
            self.temperature = max(min(self.temperature * self.anneal, max_temp), min_temp)
            print("Temperature Value: ", self.temperature)

        return {'loss': avg_loss}

    def validation_step(self, data, batch_idx):
        maybe_plot = self.plot_interval > 0 and ((self.current_epoch + 1) % self.plot_interval == 0)
        x, val_loss_triplet, val_loss_hyphc, linkage_matrix = self._forward(data, decode=maybe_plot)
        val_loss = val_loss_triplet + val_loss_hyphc

        fig = None
        best_ri = 0.0
        if maybe_plot:
            y_pred, k, best_ri = get_optimal_k(data.y.detach().cpu().numpy(), linkage_matrix[0])
            pu_score, nmi_score, ri_score = eval_clustering(y_true=data.y.detach().cpu(), Z=linkage_matrix[0])

            fig = plot_hyperbolic_eval(x=data.x.detach().cpu(),
                                       y=data.y.detach().cpu(),
                                       labels=data.labels.detach().cpu(),
                                       y_pred=y_pred,
                                       emb=self._rescale_emb(x).detach().cpu(),
                                       linkage_matrix=linkage_matrix[0],
                                       emb_scale=self.rescale.item(),
                                       k=k,
                                       show=False)

            self.logger.experiment.add_scalar("RandScore/Validation", best_ri, self.plot_step)
            # self.logger.experiment.add_scalar("AccScore@k/Validation", acc_score, self.plot_step)
            self.logger.experiment.add_scalar("PurityScore@k/Validation", pu_score, self.plot_step)
            self.logger.experiment.add_scalar("NMIScore@k/Validation", nmi_score, self.plot_step)
            self.logger.experiment.add_scalar("RandScore@k/Validation", ri_score, self.plot_step)
            self.plot_step += 1

        return {'val_loss': val_loss, 'figures': fig, 'best_ri': torch.tensor(best_ri)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.logger.log_metrics({'val_loss': avg_loss}, step=self.current_epoch)

        figures = [x['figures'] for x in outputs if x['figures'] is not None]

        for n, fig in enumerate(figures):
            tag = n // 10
            step = n % 10
            self.logger.experiment.add_figure(f"Plots/Validation@Epoch:{self.current_epoch}:{tag}", figure=fig,
                                              global_step=step)

        return {'val_loss': avg_loss}

    def test_step(self, data, batch_idx):
        x, test_loss_triplet, test_loss_hyphc, linkage_matrix = self._forward(data, decode=True)
        test_loss = test_loss_hyphc + test_loss_triplet

        y_pred_k, k, best_ri = get_optimal_k(data.y.detach().cpu().numpy(), linkage_matrix[0])
        pu_score, nmi_score, ri_score = eval_clustering(y_true=data.y.detach().cpu(), Z=linkage_matrix[0])

        fig = plot_hyperbolic_eval(x=data.x.detach().cpu(),
                                   y=data.y.detach().cpu(),
                                   labels=data.labels.detach().cpu(),
                                   y_pred=y_pred_k,
                                   emb=self._rescale_emb(x).detach().cpu(),
                                   linkage_matrix=linkage_matrix[0],
                                   emb_scale=self.rescale.item(),
                                   k=k,
                                   show=False)

        # n_clusters = data.y.max() + 1
        # y_pred = fcluster(linkage_matrix[0], n_clusters, criterion='maxclust') - 1
        # ri_score = ri(data.y.detach().cpu().numpy(), y_pred)

        self.logger.experiment.add_scalar("Loss/Test", test_loss, batch_idx)
        self.logger.experiment.add_scalar("RandScore/Test", best_ri,  batch_idx)
        # self.logger.experiment.add_scalar("AccScore@k/Test", acc_score,  batch_idx)
        self.logger.experiment.add_scalar("PurityScore@k/Test", pu_score,  batch_idx)
        self.logger.experiment.add_scalar("NMIScore@k/Test", nmi_score,  batch_idx)
        self.logger.experiment.add_scalar("RandScore@k/Test", ri_score, batch_idx)

        tag = batch_idx // 10
        step = batch_idx % 10
        self.logger.experiment.add_figure(f"Plots/Test:{tag}", figure=fig, global_step=step)
        self.logger.log_metrics({'ari@k': ri_score, 'purity@k':pu_score, 'nmi@k':nmi_score,
                                 'ari': best_ri, 'best_k': k}, step=batch_idx)

        return {'test_loss': test_loss, 'test_ri@k': torch.tensor(ri_score),
                'test_pu@k': torch.tensor(pu_score), 'test_nmi@k': torch.tensor(nmi_score),
                'test_ri': torch.tensor(best_ri) , 'k': torch.tensor(k, dtype=torch.float)}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_ri_k = torch.stack([x['test_ri@k'] for x in outputs]).mean()
        std_ri_k = torch.stack([x['test_ri@k'] for x in outputs]).std()
        # avg_acc_k = torch.stack([x['test_acc@k'] for x in outputs]).mean()
        # std_acc_k = torch.stack([x['test_acc@k'] for x in outputs]).std()
        avg_pu_k = torch.stack([x['test_pu@k'] for x in outputs]).mean()
        std_pu_k = torch.stack([x['test_pu@k'] for x in outputs]).std()
        avg_nmi_k = torch.stack([x['test_nmi@k'] for x in outputs]).mean()
        std_nmi_k = torch.stack([x['test_nmi@k'] for x in outputs]).std()
        avg_ri = torch.stack([x['test_ri'] for x in outputs]).mean()
        std_ri = torch.stack([x['test_ri'] for x in outputs]).std()
        avg_best_k = torch.stack([x['k'] for x in outputs]).mean()
        std_best_k = torch.stack([x['k'] for x in outputs]).std()


        metrics = {'ari@k': avg_ri_k, 'ari@k-std': std_ri_k,
                   # 'acc@k': avg_acc_k, 'acc@k-std': std_acc_k,
                   'purity@k': avg_pu_k, 'purity@k-std': std_pu_k,
                   'nmi@k': avg_nmi_k, 'nmi@k-std': std_nmi_k,
                   'ari': avg_ri, 'ari-std': std_ri,
                   'best_k': avg_best_k, 'std_k': std_best_k}

        self.logger.log_metrics(metrics, step=len(outputs))

        return {'test_loss': avg_loss,
                'test_ri': avg_ri,
                'ari@k': avg_ri_k, 'ari@k-std': std_ri_k,
                # 'acc@k': avg_acc_k, 'acc@k-std': std_acc_k,
                'purity@k':avg_pu_k, 'purity@k-std': std_pu_k,
                'nmi@k': avg_nmi_k, 'nmi@k-std': std_nmi_k}
