import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from smutsia.point_cloud.laserscan import LaserScan, SemanticKittiLaserScan
from smutsia.utils.semantickitti import SemanticKittiConfig
from smutsia.point_cloud.projection import Projection

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKittiDataset(Dataset):
    def __init__(self,
                 root,
                 sequences,
                 sk_config_path,
                 sensor,
                 add_normals,
                 start=0,
                 end=-1,
                 step=1,
                 gt=True,
                 detect_ground=False):

        """
        Parameters
        ----------
        root: str
            Path to semantic kitti dataset

        sequences: list

        sk_config_path: str

        sensor: dict

        add_normals: bool

        start: int

        end: int

        step: int

        gt: bool

        detect_ground: bool
        """
        self.root = root
        self.sequences = sequences
        self.skconfig = SemanticKittiConfig(sk_config_path)
        self.add_normals = add_normals
        self.sensor = sensor
        self.proj_type = sensor.get('proj_type', 'layers')
        self.res_pitch = sensor.get('res_pitch', 64)
        self.res_yaw = sensor.get('res_yaw', 1024)
        self.nb_layers = sensor.get('nb_layers', 64)

        self.projection = Projection(proj_type=self.proj_type, res_pitch=self.res_pitch,
                                     res_yaw=self.res_yaw, nb_layers=self.nb_layers)

        self.sensor_img_means = torch.tensor(sensor.get('img_means'), dtype=torch.float)
        self.sensor_img_std = torch.tensor(sensor.get('img_stds'), dtype=torch.float)

        self.gt = gt
        self.detect_ground = detect_ground
        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure labels is a dict
        assert (isinstance(self.skconfig.config, dict))

        # make sure sequences is a list
        assert (isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []

        for seq in self.sequences:
            seq = '{0:02d}'.format(int(seq))

            print("Parsing seq {}".format(seq))

            scan_path = os.path.join(root, seq, 'velodyne')
            label_path = os.path.join(root, seq, 'labels')

            for ext_scan in EXTENSIONS_SCAN:
                scan_files = sorted(glob(os.path.join(scan_path, '*' + ext_scan)))
                s_start, s_end = self.__get_interval__(len(scan_files), start, end)
                self.scan_files.extend(scan_files[s_start:s_end:step])

            for ext_label in EXTENSIONS_LABEL:
                label_files = sorted(glob(os.path.join(label_path, '*' + ext_label)))
                s_start, s_end = self.__get_interval__(len(label_files), start, end)
                self.label_files.extend(label_files[s_start:s_end:step])

            # check all scans have labels
            if self.gt:
                assert (len(self.scan_files) == len(self.label_files))

    def __get_interval__(self, n_el, start, end):
        if isinstance(start, float):
            s_start = int(round(n_el * start))
        else:
            s_start = start
        if isinstance(end, float):
            s_end = int(round(n_el * end))
        else:
            s_end = end

        return s_start, s_end

    def __getitem__(self, item):
        scan_file = self.scan_files[item]
        label_file = self.label_files[item]

        if self.gt:
            scan = SemanticKittiLaserScan(projection=self.projection,
                                          add_normals=self.add_normals,
                                          skconfig=self.skconfig)

        else:
            scan = LaserScan(projection=self.projection,
                             add_normals=self.add_normals)

        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
            if self.detect_ground:
                scan.sem_labels = self.skconfig.labels2ground[scan.sem_labels]
                scan.proj_sem_labels = self.skconfig.labels2ground[scan.proj_sem_labels]
            else:
                scan.sem_labels = self.skconfig.labels2id[scan.sem_labels]
                scan.proj_sem_labels = self.skconfig.labels2id[scan.proj_sem_labels]

        proj_range = torch.from_numpy(scan.proj_range).clone()

        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_labels).clone()
            # todo: test if is this line is correct
            proj_labels = proj_labels.unsqueeze(0)

        else:
            proj_labels = []

        if self.add_normals:
            img_means = torch.cat([self.sensor_img_means, torch.zeros(3, dtype=torch.float)])
            img_std = torch.cat([self.sensor_img_std, torch.ones(3, dtype=torch.float)])

        else:
            img_means =  self.sensor_img_means
            img_std = self.sensor_img_std
        proj_range = proj_range.permute(2, 0, 1)

        proj_range = (proj_range - img_means[:, None, None]) / img_std[:, None, None]

        return proj_range, proj_labels

    def __len__(self):
        return len(self.scan_files)


class SKParser():
    def __init__(self,
                 root,
                 train_sequences,
                 valid_sequences,
                 test_sequences,
                 add_normals,
                 sk_config_path,
                 sensor,
                 batch_size,
                 workers,
                 start=0,
                 end=-1,
                 step=1,
                 vstart=0,
                 vend=-1,
                 vstep=1,
                 tstart=0,
                 tend=-1,
                 tstep=1,
                 gt=True,
                 shuffle_train=True,
                 detect_ground=False):

        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        # self.labels = labels
        self.add_normals = add_normals
        self.sk_config_path = sk_config_path
        self.sensor = sensor
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train
        self.detect_ground = detect_ground
        self.start = start
        self.end = end
        self.step = step
        self.vstart = vstart
        self.vend = vend
        self.vstep = vstep
        self.tstart = tstart
        self.tend = tend
        self.tstep = tstep
        # data loading code
        self.train_dataset = SemanticKittiDataset(root=self.root,
                                                  sequences=self.train_sequences,
                                                  sk_config_path=self.sk_config_path,
                                                  sensor=self.sensor,
                                                  add_normals=self.add_normals,
                                                  start=self.start,
                                                  end=self.end,
                                                  step=self.step,
                                                  gt=self.add_normals,
                                                  detect_ground=self.detect_ground)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle_train,
                                                        num_workers=self.workers,
                                                        pin_memory=True,
                                                        drop_last=True)

        assert len(self.train_loader) > 0
        self.train_iter = iter(self.train_loader)

        self.valid_dataset = SemanticKittiDataset(root=self.root,
                                                  sequences=self.valid_sequences,
                                                  sk_config_path=self.sk_config_path,
                                                  sensor=self.sensor,
                                                  add_normals=self.add_normals,
                                                  start=self.vstart,
                                                  end=self.vend,
                                                  step=self.vstep,
                                                  gt=self.gt,
                                                  detect_ground=self.detect_ground)

        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=False,
                                                        num_workers=self.workers,
                                                        pin_memory=True,
                                                        drop_last=True)

        assert len(self.valid_loader) > 0
        self.valid_iter = iter(self.valid_loader)

        if self.test_sequences:
            self.test_dataset = SemanticKittiDataset(root=self.root,
                                                     sequences=self.valid_sequences,
                                                     sk_config_path=self.sk_config_path,
                                                     sensor=self.sensor,
                                                     add_normals=self.add_normals,
                                                     start=self.tstart,
                                                     end=self.tend,
                                                     step=self.tstep,
                                                     gt=False,
                                                     detect_ground=False)

            self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=False,
                                                           num_workers=self.workers,
                                                           pin_memory=True,
                                                           drop_last=True)


            assert len(self.test_loader) > 0
            self.test_iter = iter(self.test_loader)

    def get_train_batch(self):
        scans = self.train_iter.next()
        return scans

    def get_train_set(self):
        return self.train_loader

    def get_valid_batch(self):
        scans = self.valid_iter.next()
        return scans

    def get_valid_set(self):
        return self.valid_loader

    def get_test_batch(self):
        scans = self.test_iter.next()
        return scans

    def get_test_set(self):
        return self.test_loader

    def get_train_size(self):
        return len(self.train_loader)

    def get_valid_size(self):
        return len(self.valid_loader)

    def get_test_size(self):
        return len(self.test_loader)
