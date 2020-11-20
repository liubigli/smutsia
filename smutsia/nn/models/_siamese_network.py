import torch
import numpy as np
import higra as hg
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from scipy.sparse import find
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.data import Batch
from pytorch_metric_learning import losses, distances, regularizers, reducers
from sklearn.metrics.cluster import adjusted_rand_score as ri

from smutsia.utils.viz import plot_graph, plot_clustering, plot_dendrogram
from smutsia.nn.distances import HyperbolicLCA
from smutsia.graph.hierarchy.linkage import nn_merge_uf_fast_np
from smutsia.nn.optim import RAdam
from ..pool.ultrametric_pool import subdominant_ultrametric
from . import TransformNet
from .. import MLP


class FeatureExtraction(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_features: int, k: int, transformer: bool = False,):
        super(FeatureExtraction, self).__init__()
        self.in_channels = in_channels
        self.hidden_features = hidden_features
        self.k = k
        self.transformer = transformer

        if self.transformer:
            self.tnet = TransformNet()

        self.conv1 = DynamicEdgeConv(
            nn=MLP([2 * in_channels, hidden_features], negative_slope=0.2),
            k=self.k
        )
        self.conv2 = DynamicEdgeConv(
            nn=MLP([2 * hidden_features, hidden_features], negative_slope=0.2),
            k=self.k
        )
        self.conv3 = DynamicEdgeConv(
            nn=MLP([2 * hidden_features, hidden_features], negative_slope=0.2),
            k=self.k
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
    def __init__(self, nn: torch.nn.Module, margin: float = 1.0, lr=1e-3, plot_interval: int = -1):
        super(SiameseHyperbolic, self).__init__()
        self.model = nn

        self.margin = margin
        self.distance = HyperbolicLCA()
        self.loss_triplet = losses.TripletMarginLoss(distance=self.distance,
                                                     margin=self.margin)

        # learning rate
        self.lr = lr
        self.plot_interval = plot_interval

    def _loss(self, x, y, labels=None):
        if labels is not None:
            x_samples = x[labels]
            y_samples = y[labels]
        else:
            x_samples = x
            y_samples = y

        loss_triplet = self.loss_triplet(x_samples, y_samples)

        return loss_triplet

    def _decode_tree(self, leaves_embeddings):
        """Build a binary tree (nx graph) from leaves' embeddings. Assume points are normalized to same radius."""
        sim_fn = lambda x, y: np.arccos(np.clip(np.sum(x * y, axis=-1), -1.0, 1.0))
        embeddings = F.normalize(leaves_embeddings, p=2, dim=1).detach().cpu()
        Z = linkage(embeddings, metric=sim_fn)
        return Z

    def forward(self, x, y, labels=None, batch=None, decode=False):

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)

        batch_size = batch.max() + 1
        # feature extractor
        x = self.model(x)

        loss = None
        linkage_mat = []
        for i in range(batch_size):
            if loss is None:
                loss = self._loss(x[batch==i], y[batch==i], labels[batch==i])
            else:
                loss += self._loss(x, y, labels)
            if decode:
                Z = self._decode_tree(x[batch==i])
                linkage_mat.append(Z)

        loss = loss / batch_size

        return x, loss, linkage_mat

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

        x, loss, link_mat = self(x=x, y=y, labels=labels, batch=batch, decode=decode)

        return x, loss, link_mat

    def configure_optimizers(self):
        optim = RAdam(self.parameters(), lr=self.lr)
        # todo parametrize also scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optim,
                                                   mode='min',
                                                   factor=0.5,
                                                   patience=10,
                                                   min_lr=1e-4, verbose=True)

        return [optim], [scheduler]

    def training_step(self, data, batch_idx):
        x, loss, _ = self._forward(data)
        # for seq in self.model:
        #     for layer in seq:
        #         if hasattr(layer, 'weight'):
        #             if layer.weight.grad is not None:
        #                 print(layer.weight.grad.sum())

        print(loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # avg_ri = torch.stack([x['ri'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar("RandScore/Train", avg_ri, self.current_epoch)

        return {'loss': avg_loss}

    def validation_step(self, data, batch_idx):
        maybe_plot = self.plot_interval > 0 and ((self.current_epoch + 1) % self.plot_interval == 0)
        x, val_loss, linkage_matrix = self._forward(data, decode=maybe_plot)

        if maybe_plot:
            y = data.y
            batch = data.batch
            n_clusters = y.max() + 1
            y_pred = fcluster(linkage_matrix[0], n_clusters, criterion='maxclust') - 1
            val_ri_score = ri(y[batch == 0].cpu(), y_pred)
            # plot prediction
            plt.figure(figsize=(20, 5))
            ax = plt.subplot(1, 3, 1)
            plot_clustering(x, y)
            ax.set_title('Ground Truth')
            ax = plt.subplot(1, 3, 2)
            plot_clustering(x.cpu(), y_pred)
            ax.set_title(f'Clustering - RI Score {val_ri_score}')
            ax = plt.subplot(1, 3, 3)
            plot_dendrogram(linkage_matrix[0], n_clusters=n_clusters)
            ax.set_title('Dendrogram')
            plt.tight_layout()
            plt.show()

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_ri = torch.stack([x['val_ri'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar("RandScore/Validation", avg_ri, self.current_epoch)

        return {'val_loss': avg_loss}

    def test_step(self, data, batch_idx):
        return self._forward(data)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # avg_ri = torch.stack([x['test_ri'] for x in outputs]).mean()

        return {'test_loss': avg_loss}