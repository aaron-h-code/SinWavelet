import torch

import pdb


def calculate_squared_distances(a, b):
    '''returns the squared distances between all elements in a and in b as a matrix
    of shape #a * #b'''
    na = a.shape[0]
    nb = b.shape[0]

    a = a.contiguous().view([na, 1, -1])  # (b, 1, h * w * d)
    b = b.contiguous().view([1, nb, -1])  # (1, b, h * w * d)
    d = a - b

    return (d * d).sum(2)


def plummer_kernel(a, b, dimension, epsilon):
    """
    :param a: (b, 3, h, w)
    :param b: (b, 3, h, w)
    :return: (b, b)
    """
    r = calculate_squared_distances(a, b)  # (b, b)
    r += epsilon * epsilon
    f1 = dimension - 2

    return torch.pow(r, -f1 / 2)


def get_potentials(x, y, dimension, cur_epsilon, margin=0):
    '''
    This is alsmost the same `calculate_potential`, but
        px, py = get_potentials(x, y)
    is faster than:
        px = calculate_potential(x, y, x)
        py = calculate_potential(x, y, y)
    because we calculate the cross terms only once.
    '''

    x_fixed = x.detach()  # (b, 1, h, w, d), generated
    y_fixed = y.detach()  # (1, 1, h, w, d), real
    nx = x.shape[0]
    ny = y.shape[0]

    pk_xx = plummer_kernel(x_fixed, x, dimension, cur_epsilon)  # (b, b)
    pk_yx = plummer_kernel(y, x, dimension, cur_epsilon)  # (1, b)
    pk_yy = plummer_kernel(y_fixed, y, dimension, cur_epsilon)  # (1, 1)

    # pk_xx.view(-1)[::pk_xx.size(1)+1] = 1.0
    # pk_yy.view(-1)[::pk_yy.size(1)+1] = 1.0
    # for i in range(nx):
    #    pk_xx[i, i] = 1.0
    # for i in range(ny):
    #    pk_yy[i, i] = 1.0

    kxx = pk_xx.sum(0) / nx     # (b,)
    kyx = pk_yx.sum(0) / ny     # (b,), sum over y
    kxy = pk_yx.sum(1) / nx     # (1,), sum over x
    kyy = pk_yy.sum(0) / ny     # (1,)

    # original implementation
    # pot_x = kxx - kyx  # (b,)
    # pot_y = kxy - kyy  # (b,)

    pot_x = kyx - kxx # (b,)
    pot_y = kyy - kxy # (b,)

    return pot_x, pot_y