from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp


def process_iterable(iterable, func, **kwargs):
    """
    This is an auxiliary function to call func over an iterable of any objects in multiprocess.
    Basically each element of the iterable needs to contains the parameters for the function.
    All the parameters of the function that remains constant for all the elements in the iterable can passed through
    the dict kwargs and they will be fixed at the beginning of the call.
    Basically this is a generalization of the process_list function above to the case of multi parameters to treat.

    Parameters
    ----------
    iterable: iter
        iterable of parameters to treat

    func: function
        function to call

    kwargs: dict
        dictionary of fixed elements in the call

    Return
    ------
    results: list
        list containing the results of the function func to each element in iter
    """

    n_cpu = mp.cpu_count()
    print('nCPUs = ' + repr(n_cpu))
    pool = mp.Pool(processes=n_cpu)
    from functools import partial
    if len(kwargs):
        func = partial(func, **kwargs)
    func = partial(func, **kwargs)
    result = pool.starmap(func, iterable)
    pool.close()
    return result
