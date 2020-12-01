import torch
import numpy as np
import higra as hg
import pytorch_lightning as pl
from typing import Union
from matplotlib import pyplot as plt
from scipy.sparse import find
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torch_geometric.nn import knn_graph
from torch_geometric.data import Batch
from pytorch_metric_learning import losses, distances, regularizers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from sklearn.metrics.cluster import adjusted_rand_score as ri

from smutsia.utils.viz import plot_graph, plot_clustering, plot_dendrogram, plot_hyperbolic_eval
from smutsia.nn.distances import HyperbolicLCA, HyperbolicDistance
from smutsia.nn.distances.poincare import project
from smutsia.nn.conv import DynamicEdgeConv

from smutsia.nn.optim import RAdam
from ..pool.ultrametric_pool import subdominant_ultrametric
from . import TransformNet
from .. import MLP


class FeatureExtraction(torch.nn.Module):
    def __init__(self, in_channels: int, out_features: int, hidden_features: int, k: int, transformer: bool = False,
                 cosine=False):
        super(FeatureExtraction, self).__init__()
        self.in_channels = in_channels
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.k = k
        self.cosine = cosine
        self.transformer = transformer

        if self.transformer:
            self.tnet = TransformNet()

        self.conv1 = DynamicEdgeConv(
            nn=MLP([2 * in_channels, hidden_features], negative_slope=0.2),
            k=self.k,
            cosine=False,
        )
        self.conv2 = DynamicEdgeConv(
            nn=MLP([2 * hidden_features, hidden_features], negative_slope=0.2),
            k=self.k,
            cosine=False,
        )
        self.conv3 = DynamicEdgeConv(
            nn=MLP([2 * hidden_features, out_features], negative_slope=0.2),
            k=self.k,
            cosine=False,
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


class SiameseUltrametric(pl.LightningModule):
    def __init__(self, in_channels: int, hidden_features: int, k: int, transformer: bool = False,
                 loss: str = 'closest+triplet', margin: float = 1.0, gamma: float = 1.0, distance: str = 'lp',
                 plot_interval: int = -1):

        super(SiameseUltrametric, self).__init__()

        self.in_channels = in_channels
        self.hidden_features = hidden_features
        self.k = k
        self.transformer = transformer

        self.model = FeatureExtraction(in_channels=self.in_channels, hidden_features=self.hidden_features, k=self.k,
                                       transformer=self.transformer)

        # auxiliary variables for losses
        self.loss = loss
        self.margin = margin
        self.gamma = gamma
        # todo: parametrize graph_type
        self.graph_type = 'knn+mst'
        self.plot_interval = plot_interval
        if distance == 'lp':
            self.distance = distances.LpDistance()
        elif distance == 'cosine':
            self.distance = distances.CosineSimilarity()
        elif distance == 'dotproduct':
            self.distance = distances.DotProductSimilarity()
        elif distance == 'snr':
            self.distance = distances.SNRDistance()

        self.loss_triplet = losses.TripletMarginLoss(distance=self.distance,
                                                     margin=self.margin,
                                                     embedding_regularizer=regularizers.LpRegularizer())

    def _build_graph(self, x, batch=None):

        if self.k == 0 and 'knn' in self.graph_type:
            raise ValueError("Class initialized with default k value")

        batch = batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long)
        batch_size = batch.max().item() + 1 if batch is not None else 1

        edge_batch = []
        edge_index = []

        for i in range(batch_size):
            if 'knn' in self.graph_type:
                knn_edges = knn_graph(x[batch == i], k=self.k)
                edge_index.append(knn_edges)
                edge_batch.append(i * torch.ones(knn_edges.size(1), dtype=torch.long).to(x.device))

            if 'mst' in self.graph_type:
                A = squareform(F.pdist(x[batch == i]).cpu())
                # minimum spanning tree is computed on the complete graph
                T = minimum_spanning_tree(A)
                T = (T + T.T) / 2
                src, dst, _ = find(T)
                mst_edges = torch.stack([torch.from_numpy(src), torch.from_numpy(dst)])
                mst_edges = mst_edges.long()
                edge_index.append(mst_edges.to(x.device))
                edge_batch.append(i * torch.ones(mst_edges.size(1), dtype=torch.long).to(x.device))

        edge_index = torch.cat(edge_index, dim=1)
        edge_batch = torch.cat(edge_batch)

        return edge_index, edge_batch

    def __compute_loss(self, x, y, edge_weights, ultrametric, labels_idx=None):
        # triplet loss
        # samples = torch.multinomial(torch.ones(len(y)), num_samples=self.triplet_samples, replacement=False)
        if labels_idx is not None:
            x_samples = x[labels_idx]
            y_samples = y[labels_idx]
        else:
            x_samples = x
            y_samples = y
        # pairs, (pos, neg) = make_triplets(y_samples, samples)
        # src, dst = pairs
        # pairs_distances = torch.norm(x[src] - x[dst], dim=1)
        # loss_triplet = torch.relu(pairs_distances[pos] - pairs_distances[neg] + self.margin)
        # loss_triplet = torch.mean(loss_triplet)
        loss_triplet = self.loss_triplet(x_samples, y_samples)

        # closest loss
        errors = (ultrametric - edge_weights) ** 2
        loss_closest = torch.mean(errors)

        return loss_closest + self.gamma * loss_triplet

    def forward(self, x, edge_index, y=None, batch=None, edge_batch=None, labels=None):
        if edge_batch is None:
            edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)

        batch_size = batch.max().item() + 1 if batch is not None else 1

        if labels is None:
            labels_idx = torch.tensor([(batch==i).sum() for i in range(batch_size)])
            labels = torch.cat([torch.arange(labels_idx[i]) for i in range(batch_size)])

        # embedding nodes to the hidden space
        x = self.model(x=x, batch=batch)

        # the learned metric is the euclidean distance in the hidden space
        # edge_weights = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1)
        edge_weights = torch.diag(self.distance(x[edge_index[0]], x[edge_index[1]]))

        linkage_matrix = []
        loss = None
        ultrametric = torch.zeros_like(edge_weights)
        y_pred = np.zeros(y.size(0), dtype=np.int)
        ri_score = torch.zeros(batch_size)

        for i in range(batch_size):
            # project the distance to ultrametric subspace
            # retrieve the underlying graph
            src, dst = edge_index[0, edge_batch == i], edge_index[1, edge_batch == i]
            # assert torch.abs(torch.unique(torch.cat([src, dst])) - torch.arange(x[batch == i].size(0))).sum() == 0
            graph = hg.UndirectedGraph((batch == i).sum().item())
            graph.add_edges(src.cpu(), dst.cpu())

            ultra = subdominant_ultrametric(graph=graph,
                                            edge_weights=edge_weights[edge_batch == i].view(-1))

            ultrametric[edge_batch == i] = ultra
            if loss is None:
                loss = self.__compute_loss(x[batch == i], y[batch == i], edge_weights[edge_batch == i], ultra,
                                           labels_idx=labels[batch==i])
            else:
                loss = loss + self.__compute_loss(x[batch == i], y[batch == i], edge_weights[edge_batch == i], ultra,
                                                  labels_idx=labels[batch==i])

            binary_hierarchy = hg.bpt_canonical(graph, ultra.detach().cpu().numpy())
            link_mat = hg.binary_hierarchy_to_scipy_linkage_matrix(*binary_hierarchy)
            linkage_matrix.append(link_mat)
            n_clusters = y[batch==i].max().item() + 1
            pred = fcluster(link_mat, n_clusters, criterion='maxclust') - 1
            y_pred[batch==i] = pred
            ri_score[i] = ri(y[batch==i].cpu(), pred)

        print(ri_score)

        loss = loss / batch_size

        return edge_weights, ultrametric, linkage_matrix, y_pred, ri_score, loss

    def configure_optimizers(self):
        # todo: parametrize learning rate
        optimizer = Adam(self.parameters(), lr=1e-2, amsgrad=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.5,
                                                   patience=10,
                                                   min_lr=1e-4, verbose=True)

        return [optimizer], [scheduler]

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        x = data.x
        y = data.y
        batch = data.batch
        if hasattr(data,'labels'):
            labels = data.labels
        else:
            labels = None

        edge_index, edge_batch = self._build_graph(x, batch)

        edge_weights, ultrametric, linkage_matrix, y_pred, ri_score, loss = self(x=x, edge_index=edge_index, y=y,
                                                               batch=batch, edge_batch=edge_batch, labels=labels)

        return {'loss': loss, 'ri': ri_score.mean(), 'progress_bar': {'ri': ri_score.mean()}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_ri = torch.stack([x['ri'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("RandScore/Train", avg_ri, self.current_epoch)

        return {'loss': avg_loss, 'ri': avg_ri}

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        x = data.x
        y = data.y
        batch = data.batch
        if hasattr(data, 'labels'):
            labels = data.labels
        else:
            labels = None
        edge_index, edge_batch = self._build_graph(x, batch)

        edge_weights, ultrametric, linkage_matrix, y_pred, ri_score, val_loss = self(x=x, edge_index=edge_index, y=y,
                                                               batch=batch, edge_batch=edge_batch, labels=labels)

        val_ri_score = ri_score.mean()
        if self.plot_interval > 0 and ((self.current_epoch + 1) % self.plot_interval == 0):
            n_clusters = y.max() + 1
            y_edges = ((y[edge_index[0]] - y[edge_index[1]]) != 0).type(torch.float)
            # plot prediction
            plt.figure(figsize=(20, 5))
            ax = plt.subplot(1, 4, 1)
            plot_graph(x, edge_index, y_edges)
            ax.set_title('Ground Truth')
            ax = plt.subplot(1, 4, 2)
            plot_graph(x, edge_index, ultrametric/ultrametric.max())
            ax.set_title(f'Prediction Sample {batch_idx}')
            ax = plt.subplot(1, 4, 3)
            plot_clustering(x.cpu(), y_pred)
            ax.set_title(f'Clustering - RI Score {val_ri_score}')
            ax = plt.subplot(1, 4, 4)
            plot_dendrogram(linkage_matrix[0], n_clusters=n_clusters)
            ax.set_title('Dendrogram')
            plt.tight_layout()
            plt.show()

        return {'val_loss': val_loss, 'val_ri': val_ri_score,
                'progress_bar': {'val_loss': val_loss, 'val_ri': val_ri_score}}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_ri = torch.stack([x['val_ri'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("RandScore/Validation", avg_ri, self.current_epoch)

        return {'val_loss': avg_loss, 'val_ri': avg_ri}

    def test_step(self, data, batch_idx):
        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        if isinstance(data, list):
            data = Batch.from_data_list(data, follow_batch=[]).to(self.device)

        x = data.x
        y = data.y
        batch = data.batch
        edge_index, edge_batch = self._build_graph(x, batch)

        edge_weights, ultrametric, linkage_matrix, y_pred, ri_score, loss = self(x=x, edge_index=edge_index, y=y,
                                                               batch=None, edge_batch=None)

        n_clusters = y.max() + 1
        y_edges = ((y[edge_index[0]] - y[edge_index[1]]) != 0).type(torch.float)
        # plot prediction
        plt.figure(figsize=(20, 5))
        ax = plt.subplot(1, 4, 1)
        plot_graph(x, edge_index, y_edges)
        ax.set_title('Ground Truth')
        ax = plt.subplot(1, 4, 2)
        plot_graph(x, edge_index, ultrametric)
        ax.set_title(f'Prediction Sample {batch_idx}')
        ax = plt.subplot(1, 4, 3)
        plot_clustering(x.cpu(), y_pred)
        ax.set_title(f'Clustering - RI Score {ri_score.mean()}')
        ax = plt.subplot(1, 4, 4)
        plot_dendrogram(linkage_matrix[0], n_clusters=n_clusters)
        ax.set_title('Dendrogram')
        plt.tight_layout()
        plt.show()

        self.logger.experiment.add_scalar("Loss/Test", loss, batch_idx)
        self.logger.experiment.add_scalar("RandScore/Test", ri_score.mean(), batch_idx)

        return {'test_loss': loss, 'test_ri': ri_score.mean(),
                'progress_bar': {'test_loss': loss, 'test_ri': ri_score.mean()}}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_ri = torch.stack([x['test_ri'] for x in outputs]).mean()

        return {'test_loss': avg_loss, 'test_ri': avg_ri}


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

    plot_every: int
        plot validation value every #plot_every epochs

    """
    def __init__(self, nn: torch.nn.Module, embedder: Union[torch.nn.Module, None], sim_distance: str ='cosine',
                 temperature: float = 0.05,
                 margin: float = 1.0, init_rescale: float = 1e-3, max_scale: float = 1.-1e-3, lr: float = 1e-3,
                 patience: int = 20, factor: float = 0.5, min_lr: float = 1e-4,
                 plot_every: int = -1):
        super(SiameseHyperbolic, self).__init__()
        self.model = nn
        self.embedder = embedder
        self.rescale = torch.nn.Parameter(torch.Tensor([init_rescale]), requires_grad=True)
        self.temperature = temperature
        self.max_scale = max_scale
        self.margin = margin
        self.distance_lca = HyperbolicLCA()
        if sim_distance == 'hyperbolic':
            self.distace_sim = HyperbolicDistance()
        else:
            self.distace_sim = distances.CosineSimilarity()

        self.loss_triplet_sim = losses.TripletMarginLoss(distance=self.distace_sim, margin=self.margin,
                                                         embedding_regularizer=regularizers.LpRegularizer())
        print("MARGIN", self.margin)
        # learning rate
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.plot_interval = plot_every

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

        indices_tuple = lmu.convert_to_triplets(None, y_samples, t_per_anchor='all')

        anchor_idx, positive_idx, negative_idx = indices_tuple
        # print("Len Anchor: ", len(anchor_idx))
        if len(anchor_idx) == 0:
            return self.zero_losses()
        # #
        mat_sim = 0.5 * (1 + self.distace_sim(project(x_feat_samples)))
        # mat_sim = self.distace_sim(project(x_feat_samples))
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
        return x, loss_triplet, loss_hyphc, link_mat

    def _get_optimal_k(self, y, linkage_matrix):
        best_ri = 0.0
        n_clusters = y.max() + 1
        best_k = 0
        best_pred = None
        for k in range(n_clusters, n_clusters+3):
            y_pred = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            k_ri = ri(y, y_pred)
            if k_ri > best_ri:
                best_ri = k_ri
                best_k = k
                best_pred = y_pred

        return best_pred, best_k

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
        loss = loss_triplet+loss_hyphc
        return {'loss': loss, 'progress_bar': {'triplet': loss_triplet, 'hyphc': loss_hyphc}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # avg_ri = torch.stack([x['ri'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar("RandScore/Train", avg_ri, self.current_epoch)

        return {'loss': avg_loss}

    def validation_step(self, data, batch_idx):
        maybe_plot = self.plot_interval > 0 and ((self.current_epoch + 1) % self.plot_interval == 0)
        x, val_loss_triplet, val_loss_hyphc, linkage_matrix = self._forward(data, decode=maybe_plot)
        val_loss = val_loss_triplet + val_loss_hyphc

        if maybe_plot:
            y_pred, k = self._get_optimal_k(data.y.detach().cpu().numpy(), linkage_matrix[0])
            plot_hyperbolic_eval(x=data.x.detach().cpu(),
                                 y=data.y.detach().cpu(),
                                 y_pred=y_pred,
                                 emb=self._rescale_emb(x).detach().cpu(),
                                 linkage_matrix=linkage_matrix[0],
                                 emb_scale=self.rescale.item(),
                                 k=k)

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_ri = torch.stack([x['val_ri'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar("RandScore/Validation", avg_ri, self.current_epoch)

        return {'val_loss': avg_loss}

    def test_step(self, data, batch_idx):
        x, test_loss_triplet, test_loss_hyphc, linkage_matrix = self._forward(data, decode=True)
        test_loss = test_loss_hyphc + test_loss_triplet

        plot_hyperbolic_eval(x=data.x.detach().cpu(),
                             y=data.y.detach().cpu(),
                             emb=self._rescale_emb(x).detach().cpu(),
                             linkage_matrix=linkage_matrix[0],
                             emb_scale=self.rescale.item())

        return {'test_loss': test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # avg_ri = torch.stack([x['test_ri'] for x in outputs]).mean()

        return {'test_loss': avg_loss}
