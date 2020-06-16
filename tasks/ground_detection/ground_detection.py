import os
import ntpath
import argparse
import torch
import numpy as np
from glob import glob
from sklearn.metrics import precision_recall_curve
from definitions import SEMANTICKITTI_PATH, SEMANTICKITTI_CONFIG
from smutsia.utils import process_iterable, load_yaml, write_las
from smutsia.utils.semantickitti import load_pyntcloud, SemanticKittiConfig
from smutsia.point_cloud.projection import Projection
from smutsia.deep_learning.models.u_net import UNet
from smutsia.point_cloud.ground_detection import *
from smutsia.utils.scores import compute_scores, get_confusion_matrix, condense_confusion_matrix
from smutsia.utils.viz import plot_confusion_matrix, color_bool_labeling, plot_precision_recall_curve


carLikeId = [10, 13, 18]
bikeLikeId = [11, 15, 31, 32]
personId = [30]
groundLikeId = [40, 44, 48, 49, 60, 72]
buildId = [50]
discard = [1, 16, 20, 51, 52, 70, 71, 80, 81, 99]
movingLikeId = [252, 253, 254, 255, 256, 257, 258, 259]
classes = ['other', 'vehicles', 'cycles', 'person', 'ground', 'building', 'moving-objects', 'discard']
mySelect = [[0], carLikeId, bikeLikeId, personId, groundLikeId, buildId, movingLikeId, discard]
selectedId = []
condensedId = []
for elem in mySelect:
    selectedId.extend(elem)
    condensedId.append(elem)

selectedId.sort()


def recursively_add_params(key, yaml_config, params):
    """
    Function that recursively iterate over a dict and add keys to params

    Parameters
    ----------
    key: str or int
        key of dict yaml_config

    yaml_config: dict
        dictionary

    params: dict
        dictionary
    """
    if not isinstance(yaml_config[key], dict):
        params[key] = yaml_config[key]
    else:
        for k in yaml_config[key]:
            recursively_add_params(k, yaml_config[key], params)


def load_parameters(config_file, recursive_load=True):
    """
    Utils function that read a yaml config files and generate paramter dict

    Parameters
    ----------
    config_file: str
        path to config file

    recursive_load: bool
        if true the inner keys are recursively loaded

    Returns
    -------
    params: dict
        dictionary with parameters for method
    """
    yaml_config = load_yaml(config_file)
    if recursive_load:
        params = dict()
        for k in yaml_config:
            recursively_add_params(k, yaml_config, params)
    else:
        params = yaml_config

    method_name = "all"
    # remove key name
    if 'name' in params:
        method_name = params['name']
        params.pop('name')

    return params, method_name


def load_sequence(datapath, start, end, step, extension='bin'):
    """
    Load a sequence of files of a given extension

    Parameters
    ----------
    datapath: str
        path to files

    start: int
        start index of the sequence

    end: int
        end index of the sequence

    step: int
        step to use to select subsequence

    extension: str
        extension of the files to load

    Returns
    -------
    file_list: list
        list of path to files to load
    """
    # selecting all files inside a datapath of a given extension
    files = sorted(glob(os.path.join(datapath, '*.' + extension)))

    return files[start:end:step]


def aggregate_results(chunk_clouds, savedir, method_name):
    """
    Function that aggregates results obtained for each cloud
    """
    with open(os.path.join(savedir, method_name + '.txt'), 'a') as f1:
        for cloud in chunk_clouds:
            filename = cloud.filename
            if hasattr(cloud, 'sequence'):
                filename = cloud.sequence + '_' + filename

            with open(os.path.join(savedir, filename + '.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    f1.write(line)
            f.close()
            # os.remove(os.path.join(savedir, filename + '.txt'))
    f1.close()


def analyse_results(data, savedir, ground_id=40, classes=None, threshold=0.0):
    """
    Parameters
    ----------
    data: iterable
        tuple or list containing y_pred, y_true, labels, filepath

    savedir: str
        path where plot must be saved

    ground_id: int
        id of ground/road class

    classes: list
        list of class names to use as x-ticks and y-ticks in confusion matrix plot
    """
    y_pred = data[0].astype(int)
    y_true = data[1].astype(int)
    labels = data[2]
    not_other = data[3]
    cloud = data[4]
    if isinstance(cloud, str):
        filename = cloud
    else:
        filename = cloud.filename
        if hasattr(cloud, 'sequence'):
            filename = cloud.sequence + '_' + filename

    # compute scores
    scores = compute_scores(y_true=y_true[not_other], y_pred=y_pred[not_other], threshold=threshold, print_info=True,
                            sample_name=filename)

    pred = (y_pred > threshold) if threshold > 0.0 else y_pred
    cm = get_confusion_matrix(labels, ground_id * pred, selectedId)

    # cm = condense_confusion_matrix(cm[0], selectedId, condensedId)
    cm = condense_confusion_matrix(cm[0], selectedId, condensedId)

    plot_confusion_matrix(cm, classes=classes, normalize=True, savefig=os.path.join(savedir, filename + '_cm.eps'),
                          title='Confusion Matrix ' + filename.upper())

    open_type = "w" if not isinstance(cloud, str) else "a"
    f = open(os.path.join(savedir, filename + '.txt'), open_type)
    info_scores = "Scores {}: \n" \
                  "F1 -> {},\n" \
                  "Recall -> {},\n" \
                  "Precision -> {},\n" \
                  "Accuracy -> {},\n" \
                  "Jaccard -> {}.\n" \
                  "-----------------------\n".format(filename,
                                                     scores['f1'],
                                                     scores['recall'],
                                                     scores['precision'],
                                                     scores['acc'],
                                                     scores['jaccard'])
    f.write(info_scores)
    f.write('---BEGIN CONF MATRIX---\n')
    f.write(str(cm.astype(int)))
    f.write('\n-----END CONF MATRIX---\n')
    f.close()

    if threshold > 0.0:
        prec, recall, thresholds = precision_recall_curve(y_true, y_pred)
        plot_precision_recall_curve(prec, recall, xlim=[0.9, 1], ylim=[0.9, 1],
                                    savefig=os.path.join(savedir, filename + '_prc.eps'), filename=filename)

    if not isinstance(cloud, str):
        color = color_bool_labeling(y_true, y_pred)
        write_las(cloud.xyz, filepath=os.path.join(savedir, filename + '.las'), color=color)


def main(dataset, method, config_file, savedir, sequence, start=0, end=-1, step=1, chunk_size=-1):
    """
    Main function

    Parameters
    ----------
    dataset: optional {'semantickitti'}
        dataset to use to evaluate method

    method: optional {'ransac', 'hybrid', 'qfz', 'csf', 'cnn'}
        function

    config_file: str

    savedir: str
        path where to store results

    sequence: str

    start: int

    end: int

    step: int

    chunk_size: int
    """

    # load model parameters
    params, method_name = load_parameters(config_file)

    # initialise model variable
    if method == 'ransac':
        func = naive_ransac
    elif method == 'hybrid':
        func = hybrid_ground_detection
    elif method == 'qfz':
        func = dart_ground_detection
        params['savedir'] = savedir
        params['classes'] = classes
        params['condense_id'] = condensedId
        params['select_id'] = selectedId
    elif method == 'csf':
        func = cloth_simulation_filtering
    elif method == 'cnn':

        func = cnn_detect_ground
        config, _ = load_parameters(config_file, recursive_load=False)
        # initialize projector
        proj = Projection(**config['projection'])
        # initialize model
        config['model']['scale'] = tuple(config['model']['scale'])
        model = UNet(**config['model'])
        # load model weights
        model.load_state_dict(torch.load(config['weights']))
        model.eval()
        # method parameters
        img_means = np.array(config['img_means'])
        img_std = np.array(config['img_stds'])

        params = dict(model=model, proj=proj, img_means=img_means, img_std=img_std,
                      add_normals=config['add_normals'], savedir=savedir, gpu=config['gpu'])

    else:
        raise ValueError("Method {} not known.".format(method))

    # initialise dataset variables
    if dataset == 'semantickitti':
        basedir = os.path.join(SEMANTICKITTI_PATH, sequence, 'velodyne')
        db_config = SemanticKittiConfig(SEMANTICKITTI_CONFIG)
        ground_id = 40
    else:
        raise ValueError("Dataset {} not known.".format(dataset))

    # loading list of files to treat
    files = load_sequence(basedir, start, end, step)
    n = len(files)

    # setting chunk equal to the length of the list as default
    chunk_size = chunk_size if chunk_size > 0 else n

    # list containing gt values
    y_true = []
    # list containing pred values
    y_pred = []
    # list containing all label values
    labels = []
    # list containing ids for not ground labels
    not_ground = []
    threshold = 0.0 if method != 'cnn' else 0.5

    for i in range(0, n, chunk_size):
        print("Loading chunk in interval {}-{}".format(i, min(i + chunk_size, n)))

        if dataset == 'semantickitti':
            # load cloud in the chunk and store chunk labels and chunk ground truth
            clouds = process_iterable(files[i:i + chunk_size], load_pyntcloud, **{'add_label': True})
            chunk_labels = [cloud.points.labels.values.astype(int) for cloud in clouds]
            chunk_gt = [db_config.labels2ground[lbl] for lbl in chunk_labels]
        else:
            raise ValueError("Dataset {} not known.".format(dataset))

        # call func and predict ground
        chunk_pred = process_iterable(clouds, func, **params)
        not_other_idx_batch = [lbl != 0 for lbl in chunk_labels]
        print("Analysing predictions for chunk {}-{}".format(i, min(i + chunk_size, n)))
        # analyse and plot results for each file in chunk
        
        process_iterable(zip(chunk_pred, chunk_gt, chunk_labels, not_other_idx_batch, clouds),
                         analyse_results,
                         **{'savedir': savedir, 'classes': classes, 'ground_id': ground_id, 'threshold': threshold})
        aggregate_results(clouds, savedir, method_name)
        labels += chunk_labels
        y_true += chunk_gt
        y_pred += chunk_pred
        not_ground += not_other_idx_batch

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    labels = np.concatenate(labels)
    not_ground = np.concatenate(not_ground)

    print("Compute total scores")
    # analyse and plot global results
    analyse_results([y_pred, y_true, labels, not_ground, method_name], savedir=savedir, ground_id=ground_id,
                    classes=classes, threshold=threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ground Detection methods")
    parser.add_argument("--dataset", default="semantickitti", type=str, help="Dataset on which evaluate methods")
    parser.add_argument("--method", type=str, help="Method to use for ground detection")
    parser.add_argument("--params", type=str, help="Parameters for the method")
    parser.add_argument("--sequence", default="08", type=str, help="Sequence to use in semantic kitti")
    parser.add_argument("--start", default=0, type=int, help="Index of the first element of the sequence")
    parser.add_argument("--end", default=-1, type=int, help="Index of the last element of the sequence")
    parser.add_argument("--step", default=1, type=int, help="Evaluate one file each <step> in the sequence")
    parser.add_argument("--chunk_size", default=-1, type=int, help="Size of the chunk to treat at each step")

    args = parser.parse_args()
    ds = args.dataset
    gd_method = args.method
    params_path = args.params
    seq = args.sequence
    start_frame = args.start
    end_frame = args.end
    step_frames = args.step
    chunksize = args.chunk_size
    print("Evaluating performance of method {}, over Dataset {}."
          " Selected sequence {} and interval {}-{} each {}.".format(gd_method, ds, seq,
                                                                     start_frame, end_frame, step_frames))
    params_file = ntpath.basename(params_path)
    out = params_file.split('.')[:-1]
    if len(out) == 1:
        params_file = out[0]
    else:
        params_file = '.'.join(out)

    sdir = os.path.join('/home/leonardo/Dev/github/smutsia/results/ground_detection/', gd_method, params_file)
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    main(dataset=ds,
         method=gd_method,
         config_file=params_path,
         savedir=sdir,
         sequence=seq,
         start=start_frame,
         end=end_frame,
         step=step_frames, chunk_size=chunksize)
