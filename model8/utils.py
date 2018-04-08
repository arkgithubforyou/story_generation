import numpy as np
import dill
import os
from configurations import Config


def shuffle_data(X, Y):
    """shuffle data"""
    perm = np.random.permutation(len(Y))
    x_shuf = X[perm]
    y_shuf = Y[perm]

    return x_shuf, y_shuf


def temperature_sample(preds, temperature=1.0):
    """
    temperature softmax sampling.
    :param preds: a probability distribution
    :param temperature:
    :return: single index that is sampled
    """
    preds = np.asarray(preds).astype('float')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = np.reshape(preds, newshape=(-1, ))
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def random_parameter_generation(ranges, is_exp, num_combinations):
    """
    generates parameter combinations for random parameter search
    :param ranges:
    :param is_exp: will output 10^param instead of param
    :param num_combinations:
    :return: list of lists, each caching values of one PARAMETER
    """
    lists = list()
    for (i, (min_b, max_b)) in enumerate(ranges):
        sublist = list()
        for _ in range(num_combinations):
            rnd = np.random.uniform() * (max_b - min_b) + min_b
            if is_exp[i] is True:
                rnd = np.power(10., rnd)
            sublist.append(rnd)
        lists.append(sublist)
    return lists


def dump_combinations(ranges, is_exp, num_combinations, path):
    lists = random_parameter_generation(ranges, is_exp, num_combinations)
    dill.dump((num_combinations, lists), open(path, 'wb'))


def read_hypers(types, lists, index):
    return_hypers = list()
    for (i, hyper_list) in enumerate(lists):
        if types[i] == 'c':
            # copy hyper
            return_hypers.append(hyper_list[index])
        elif types[i] == '2e':
            # exp2 floor
            return_hypers.append(np.int(np.floor(np.power(2., np.floor(hyper_list[index])))))
        elif types[i] == 'int':
            return_hypers.append(np.floor(hyper_list[index]))
    return return_hypers


def load_hypers(path='hypers_' + Config.Run_Index, num_combinations=60):
    # dropout, lr, clipnorm, batch_size, context_length, Event_embedding, rnn_size, a loss, a weight
    # round I
    # ranges = [[.2, .8], [-5., -3.], [0., 3.], [3., 10.], [5, 100], [6., 10.99], [6., 11.99], [1., 2.], [1., 6.]]
    # round II
    ranges = [[.25, .55], [-5., -4.7], [0., 1.5], [6., 9.99], [9, 56], [9., 9.99], [8., 11.99], [1., 1.8], [3., 5.5]]
    is_exp = [False, True, True, False, False, False, False, False, False]

    if not os.path.exists(path):
        dump_combinations(ranges, is_exp, num_combinations, path)
    num_combinations, hyper_lists = dill.load(open(path, 'rb'))

    return num_combinations, hyper_lists


load_hypers()
