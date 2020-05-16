import os
import argparse
from glob import glob
from definitions import SEMANTICKITTI_PATH
from smutsia.utils import process_iterable


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


def main(dataset, method, sequence, start, end, step, **kwargs):
    """
    Main function

    Parameters
    ----------
    dataset: optional {'semantickitti'}
        dataset to use to evaluate method

    method: func
        function
    """
    if dataset == 'semantickitti':
        basedir = os.path.join(SEMANTICKITTI_PATH, sequence, 'velodyne')
    else:
        raise ValueError("Dataset {} not known.".format(dataset))

    # loading list of files to treat
    files = load_sequence(basedir, start, end, step)

    results = process_iterable(files, method, **kwargs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ground Detection methods")
    parser.add_argument("--dataset", default="semantickitti", type=str, help="Dataset on which evaluate methods")
    parser.add_argument("--method", type=str, help="Method to use for ground detection")
    parser.add_argument("--params", type=str, help="Parameters for the method")
    parser.add_argument("--sequence", default="08", type=str, help="Sequence to use in semantic kitti")
    parser.add_argument("--start", default=0, type=int, help="Index of the first element of the sequence")
    parser.add_argument("--end", default=-1, type=int, help="Index of the last element of the sequence")
    parser.add_argument("--step", default=1, type=int, help="Evaluate one file each <step> in the sequence")


    args = parser.parse_args()
