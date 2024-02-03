import numpy as np
import torch
from pytorch_wavelets.dwt import lowlevel as lowlevel

import pdb


def prep_filt_sfb1d_torch(g0, g1, device=None):
    """
    Prepares the filters to be of the right form for the sfb1d function. In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0 (torch.Tensor): low pass filter bank
        g1 (torch.Tensor): high pass filter bank
        device: which device to put the tensors on to

    Returns:
        (g0, g1)
    """
    t = torch.get_default_dtype()

    g0 = g0.to(device=device, dtype=t).reshape((1, 1, -1))
    g1 = g1.to(device=device, dtype=t).reshape((1, 1, -1))

    return g0, g1


def prep_filt_afb1d_torch(h0, h1, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0 (torch.Tensor): low pass column filter bank
        h1 (torch.Tensor): high pass column filter bank
        device: which device to put the tensors on to

    Returns:
        (h0, h1)
    """
    h0 = torch.flip(h0, dims=[0])
    h1 = torch.flip(h1, dims=[0])

    t = torch.get_default_dtype()

    h0 = h0.to(device=device, dtype=t).reshape((1, 1, -1))
    h1 = h1.to(device=device, dtype=t).reshape((1, 1, -1))

    return h0, h1


def prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row, device):

    if isinstance(g0_dep, np.ndarray):
        g0_dep, g1_dep = lowlevel.prep_filt_sfb1d(g0_dep, g1_dep, device)
        g0_col, g1_col = lowlevel.prep_filt_sfb1d(g0_col, g1_col, device)
        g0_row, g1_row = lowlevel.prep_filt_sfb1d(g0_row, g1_row, device)
    elif isinstance(g0_dep, torch.Tensor):
        g0_dep, g1_dep = prep_filt_sfb1d_torch(g0_dep, g1_dep, device)
        g0_col, g1_col = prep_filt_sfb1d_torch(g0_col, g1_col, device)
        g0_row, g1_row = prep_filt_sfb1d_torch(g0_row, g1_row, device)

    g0_dep = g0_dep.reshape((1, 1, -1, 1, 1))
    g1_dep = g1_dep.reshape((1, 1, -1, 1, 1))
    g0_col = g0_col.reshape((1, 1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, 1, -1))

    return g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row


def prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row, device):

    if isinstance(h0_dep, np.ndarray):
        h0_dep, h1_dep = lowlevel.prep_filt_afb1d(h0_dep, h1_dep, device)
        h0_col, h1_col = lowlevel.prep_filt_afb1d(h0_col, h1_col, device)
        h0_row, h1_row = lowlevel.prep_filt_afb1d(h0_row, h1_row, device)
    elif isinstance(h0_dep, torch.Tensor):
        h0_dep, h1_dep = prep_filt_afb1d_torch(h0_dep, h1_dep, device)
        h0_col, h1_col = prep_filt_afb1d_torch(h0_col, h1_col, device)
        h0_row, h1_row = prep_filt_afb1d_torch(h0_row, h1_row, device)
    else:
        raise NotImplementedError

    h0_dep = h0_dep.reshape((1, 1, -1, 1, 1))
    h1_dep = h1_dep.reshape((1, 1, -1, 1, 1))
    h0_col = h0_col.reshape((1, 1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, 1, -1))

    return h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row