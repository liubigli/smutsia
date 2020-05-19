import os
import argparse
import numpy as np
from glob import glob
from definitions import SEMANTICKITTI_PATH
from smutsia.utils import process_iterable, load_yaml
from smutsia.utils.semantickitti import load_pyntcloud, SemanticKittiConfig
from smutsia.point_cloud.ground_detection import *
from smutsia.utils.scores import compute_scores, get_confusion_matrix, condense_confusion_matrix
from smutsia.utils.viz import plot_confusion_matrix


carLikeId = [10, 13, 18]
bikeLikeId = [11, 15, 31, 32]
personId = [30]
groundLikeId = [40, 44, 48, 49, 60, 72]
buildId = [50]
movingLikeId = [252, 253, 254, 255, 256, 257, 258, 259]

mySelect = [[0], carLikeId, bikeLikeId, personId, groundLikeId, buildId, movingLikeId]
selectedId = []
condensedId = []
for elem in mySelect:
    selectedId.extend(elem)
    condensedId.append(elem)

print(selectedId, condensedId)

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


def load_parameters(config_file):
    """
    Utils function that read a yaml config files and generate paramter dict

    Parameters
    ----------
    config_file: str
        path to config file

    Returns
    -------
    params: dict
        dictionary with parameters for method
    """
    yaml_config = load_yaml(config_file)
    params = dict()
    for k in yaml_config:
        recursively_add_params(k, yaml_config, params)

    # remove key name
    if 'name' in params:
        params.pop('name')

    return params


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


def analyse_results(cloud, savedir):

    skconfig = SemanticKittiConfig('/home/leonardo/Dev/github/smutsia/config/labels/semantic-kitti.yaml')
    pc = cloud[0]
    y_pred = cloud[1]
    filepath = cloud[2]
    filename = filepath.split('/')[-1].split('.')[0]
    y_true = skconfig.labels2ground[pc.points.labels.values.astype(int)]

    # compute scores
    scores = compute_scores(y_true, y_pred, print_info=True)
    f = open(os.path.join(savedir, filename + '.txt'), "w")
    f.write(str(scores))
    f.close()

    cm = get_confusion_matrix(pc.points.labels.values.astype(int), 40 * y_pred, selectedId)
    cm = condense_confusion_matrix(cm[0], selectedId, condensedId)
    classes = ['other', 'vehicles', 'cycles', 'person', 'ground', 'building', 'moving-objects']
    plot_confusion_matrix(cm, classes=classes, normalize=True, savefig=os.path.join(savedir, filename + '.eps'),
                          title='Confusion Matrix ' + filename)

    return y_true, y_pred, pc.points.labels.values.astype(int)



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
    if method == 'ransac':
        func = naive_ransac
        pass
    elif method == 'hybrid':
        func = hybrid_ground_detection
        pass
    elif method == 'qfz':
        func = None
        pass
    elif method == 'csf':
        func = None
        pass
    elif method == 'cnn':
        func = None
    else:
        raise ValueError("Method {} not known.".format(method))

    params = load_parameters(config_file)

    if dataset == 'semantickitti':
        basedir = os.path.join(SEMANTICKITTI_PATH, sequence, 'velodyne')
    else:
        raise ValueError("Dataset {} not known.".format(dataset))
    # loading list of files to treat
    files = load_sequence(basedir, start, end, step)
    n = len(files)
    # setting chunk equal to the length of the list as default
    chunk_size = chunk_size if chunk_size > 0 else n
    y_true = []
    y_pred = []
    labels = []
    for i in range(0, n, chunk_size):
        print("Loading chunk in interval {}-{}".format(i, i + chunk_size))
        if dataset == 'semantickitti':
            clouds = process_iterable(files[i:i + chunk_size], load_pyntcloud, add_label=True)
        else:
            raise ValueError("Dataset {} not known.".format(dataset))

        results = process_iterable(clouds, func, **params)
        out = process_iterable(zip(clouds, results, files[i:i+chunk_size]), analyse_results, **{'savedir':savedir})
        for el in out:
            y_true.append(el[0])
            y_pred.append(el[1])
            labels.append(el[2])

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    labels = np.concatenate(labels)

    scores = compute_scores(y_true, y_pred, print_info=True)
    f = open(os.path.join(savedir, 'all' + '.txt'), "w")
    f.write(str(scores))
    f.close()

    cm = get_confusion_matrix(labels, 40 * y_pred, selectedId)
    cm = condense_confusion_matrix(cm[0], selectedId, condensedId)
    classes = ['other', 'vehicles', 'cycles', 'person', 'ground', 'building', 'moving-objects']
    plot_confusion_matrix(cm, classes=classes, normalize=True, savefig=os.path.join(savedir, 'all.eps'),
                          title='Confusion Matrix All')


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
    dataset = args.dataset
    gd_method = args.method
    params_file = args.params
    seq = args.sequence
    start_frame = args.start
    end_frame = args.end
    step_frames = args.step
    chunk_size = args.chunk_size
    print("Evaluating performance of method {}, over Dataset {}."
          " Selected sequence {} and interval {}-{} each {}.".format(gd_method, dataset, seq,
                                                                     start_frame, end_frame, step_frames))

    savedir = '/home/leonardo/Dev/github/smutsia/results/ground_detection/hybrid/'

    main(dataset=dataset,
         method=gd_method,
         config_file=params_file,
         savedir=savedir,
         sequence=seq,
         start=start_frame,
         end=end_frame,
         step=step_frames, chunk_size=chunk_size)
