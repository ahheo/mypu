"""
>--#########################################################################--<
>------------------------------whoknows functions-----------------------------<
>--#########################################################################--<
* aggr_func_            : aggregate function over ndarray
* aw_loa_bnds_          : spherical segment areas
* ax_any_               : any upon axes other than the given one
* b2l_endian_           : return little endian copy
* bA2ind_               : transform boolean array to indices
* compressLL_           : compress 2D list
* consecutive_          : consecutive functions
* consecutiveN_         : consecutive numbers
* date_mv_mon_          : date displace by months
* date_mv_yr_           : date displace by years
* date_s_               : string to datetime.date
* day_last_             : last day: string
* day_next_             : next day: string
* dgt_                  : digits of the int part of a number
* doy2date_             : day of year to datetime
* edotm_                : end day of this month
* el_join_              : element-wise join
* ext_                  : get extension of file name
* extract_              : extraction with help of inds_ss_
* find_patt_            : extract list from items matching pattern
* flt_                  : flatten (out: generator)          --> flt_l
* flt_ndim_             : flatten number of consecutive dims
* flt_l                 : flatten (out: list)               --> (o)uniqL_
* half_grid_            : points between grids
* haversine_            : distance between geo points in radians
* iind_                 : rebuild extraction indices (n axes)
* in_loalim_            : grids if inside a rectangle
* in_polygons_          : points if inside polygon(s)
* indFront_             : move element with specified index to front
* ind_inRange_          : indices of values in a range
* ind_s_                : extraction indices (1 axis)       --> inds_ss_
* ind_shape_i_          : slice indices                     --> slice_back_
* ind_sm_               : if ind leading to sharing memory
* ind_win_              : indices of a window in cylinder axis
* inds_ss_              : extraction indices (n axes)       --> extract_
* intsect_              : intersection of lists
* is1dIter_             : if 1d Iterable
* isdecr_               : if monotonic decreasing
* isGI_                 : if Iterator
* isIter_               : if Iterable but not str or bytes
* isincr_               : if monotonic increasing
* isMonth_              : string if a month
* ismono_               : check if ismononic
* isSeason_             : string if is a season
* iterDT_               : get datetime array from a brief string
* iter_str_             : string elements
* kde_                  : kernel distribution estimate
* kde__                 : tranform before kde_
* l__                   : start logging
* l_flp_                : flip list
* l_ind_                : extract list by providing indices
* l2b_endian_           : return big endian copy
* loa2d_                : meshgrid (& transpose) longitue/latitude if necessary
* latex_unit_           : m s-1 -> m s$^{-1}$
* ll_                   : end logging
* m2s_                  : season
* m2sm_                 : season membership
* m2sya_                : season year adjust
* mmmN_                 : months in season
* mnN_                  : month order in the calendar
* mosaicAB01_           : mosaic list
* nSlice_               : total number of slices
* nanMask_              : masked array -> array with NaN
* nli_                  : if item is list flatten it
* ouniqL_               : ordered unique elements of list
* p_deoverlap_          : remove overlaping period from a list of periods
* p_least_              : extract aimed period from a list of periods
* pcorr_                : partial correlation (matrix)
* pcorr_xyz             : partial correlation (x, y, z)
* pearsonr2pV_          : p value corresponding to pearson r value
* prg_                  : string indicating progress status (e.g., '#002/999')
* pure_fn_              : filename excluding path (& also extension by default)
* pV2pearsonr_          : pearson r value corresponding to p value
* rMEAN1d_              : rolling window mean
* rMEAN2d_              : rolling window mean (2D)
* rPeriod_              : [1985, 2019] -> '1985-2019'
* rSUM1d_               : rolling window sum
* rSUM2d_               : rolling window sum (2D)
* rTime_                : a string of passing time
* rest_mns_             : rest season named with month abbreviation
* robust_bc2_           : robust alternative for numpy.broadcast_to
* rpt_                  : values in a period axis           --> extract_win_
* rpt360_               : values in an axis with period = 360
* schF_keys_            : find files by key words
* shp_drop_             : drop dims specified (and replace if desired)
* slctStrL_             : select string list include or exclude substr(s)
* sqzUnit_              : squeeze unit
* ss_fr_sl_             : subgroups that without intersections
* sub_shp_              : subgroup a shape
* timerMain_            : decorator for executable function
* tryattr_              : safely getattr
* uniqL_                : unique elements of list
* upd_                  : update a dictionary
* valid_seasons_        : if provided seasons valid
* valueEqFront_         : move elements equal specified value to front
* windds2uv_            : u v from wind speed and direction
...

###############################################################################
            Author: ahheo (Changgui Lin)
            github: https://github.com/ahheo
            E-mail: mapulynn@gmail.com
      Date created: 02.09.2019
Date last modified: 01.04.2022
          comments: add func dgt_, prg_;
                    fix rMEAN1d_ issue with mode 'full' and 'same'
"""


import numpy as np
import pandas as pd
import warnings
#import math


__all__ = ['aggr_func_',
           'aw_loa_bnds_',
           'ax_any_',
           'b2l_endian_',
           'bA2ind_',
           'compressLL_',
           'consecutive_',
           'consecutiveN_',
           'date_mv_mon_',
           'date_mv_yr_',
           'date_s_',
           'day_last_',
           'day_next_',
           'dgt_',
           'doy2date_',
           'edotm_',
           'el_join_',
           'ext_',
           'extract_',
           'find_patt_',
           'flt_',
           'flt_ndim_',
           'flt_l',
           'half_grid_',
           'haversine_',
           'iind_',
           'in_loalim_',
           'in_polygons_',
           'indFront_',
           'ind_inRange_',
           'ind_s_',
           'ind_shape_i_',
           'ind_sm_',
           'ind_win_',
           'inds_ss_',
           'intsect_',
           'is1dIter_',
           'isdecr_',
           'isGI_',
           'isIter_',
           'isincr_',
           'isMonth_',
           'ismono_',
           'isSeason_',
           'iter_str_',
           'iterDT_',
           'kde_',
           'kde__',
           'l__',
           'l_flp_',
           'l_ind_',
           'l2b_endian_',
           'loa2d_',
           'latex_unit_',
           'll_',
           'm2s_',
           'm2sm_',
           'm2sya_',
           'mmmN_',
           'mnN_',
           'mosaicAB01_',
           'nSlice_',
           'nanMask_',
           'nli_',
           'ouniqL_',
           'p_deoverlap_',
           'p_least_',
           'pcorr_',
           'pcorr_xyz',
           'pearsonr2pV_',
           'prg_',
           'pure_fn_',
           'pV2pearsonr_',
           'rMEAN1d_',
           'rMEAN2d_',
           'rPeriod_',
           'rSUM1d_',
           'rSUM2d_',
           'rTime_',
           'rest_mns_',
           'robust_bc2_',
           'rpt_',
           'rpt360_',
           'schF_keys_',
           'shp_drop_',
           'slctStrL_',
           'sqzUnit_',
           'ss_fr_sl_',
           'sub_shp_',
           'timerMain_',
           'tryattr_',
           'uniqL_',
           'upd_',
           'valid_seasons_',
           'valueEqFront_',
           'windds2uv_']


def rpt_(
        x,
        rb=2*np.pi,
        lb=0,
        ):
    """
    ... map to value(s) in a period axis ...

    Args
    ----
    x: to be mapped (numeric array_like)

    kwArgs
    ------
    rb: right bound of a period axis (default 2*pi)
    lb: left bound of a period axis (default 0)

    Returns
    -------
    normal value in a period axis

    Notes
    -----
    list, tuple transfered as np.ndarray

    Examples
    --------
    >>> rpt_(-1)
    Out: 5.283185307179586 #(2*np.pi-1)
    >>> rpt_(32, 10)
    Out: 2
    >>> rpt_(355, 180, -180)
    Out: -5
    """

    assert lb < rb, 'left bound should not greater than right bound!'

    if isIter_(x) and not isinstance(x, np.ndarray):
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.number):
            raise Exception('data not interpretable')

    return (x-lb) % (rb-lb) + lb


def rpt360_(
        x,
        rb=360,
        ):
    """
    ... map to value(s) in an axis with period = 360 ...

    Args
    ----
    x: to be mapped (numeric array_like)

    kwArgs
    ------
    rb: right bound of a period axis (default 360)

    Returns
    -------
    normal value in this period axis

    Notes
    -----
    list, tuple transfered as np.ndarray

    Examples
    --------
    >>> rpt360_(355, 180)
    Out: -5
    """

    return rpt_(x, rb=rb, lb=rb-360)


def ss_fr_sl_(sl):
    """
    ... subgroups that without intersections ...

    Examples
    --------
    >>> x = [['a'],
             ['b', 'c'],
             ['d', 'e'],
             ['b', 'c'],
             ['a', 'f'],
             ['e', 'g'],
             ['a', 'c', '0']]
    >>> ss_fr_sl_(x)
    Out: [{'d', 'e', 'g'}, {'0', 'a', 'b', 'c', 'f'}]
    >>> x += [['e', 'f']]
    >>> ss_fr_sl_(x)
    Out: [{'0', 'a', 'b', 'c', 'd', 'e', 'f', 'g'}]
    """
    uv = set(flt_l(sl))
    o = []
    def _sssl(ss):                                                             # derive a union of all sets in sl that have intersection with the given set ss
        return set(flt_l([i for i in sl if any([ii in i for ii in ss])]))
    def _ssl(vv):                                                              # get these unions
        si = set(list(vv)[:1])                                                 # starting from a single element
        while True:
            si_ = _sssl(si)
            if si_ == si:                                                      # now we got one of such a union
                break
            else:
                si = si_
        o.append(si)
        rvv = vv - si                                                          # getting another union
        if len(rvv) != 0:
            _ssl(rvv)
    _ssl(uv)
    return o


def _not_list_iter(l):
    for el in l:
        if not isinstance(el, list):
            yield el
        else:
            yield from _not_list_iter(el)


def nli_(l):
    """
    ... flatten a nested List deeply (basic item as not of List) ...
    ... like flt_l(), but work with only type List               ...

    Examples
    --------
    >>> x = [1, 2, (3, 4), [5, [6, 7], (8, [9, 10])]]
    >>> nli_(x)
    Out: [1, 2, (3, 4), 5, 6, 7, (8, [9, 10])]
    """
    return list(_not_list_iter(l))


def flt_(l, nx=None, n=0):
    """
    ... flatten a nested List deeply (output generator) ...
    """
    from typing import Iterable
    for el in l:
        if (isinstance(el, Iterable) and
            not isinstance(el, (str, bytes)) and
            (n < nx if nx is not None else True)):
            yield from flt_(el, nx=nx, n=n+1)
        else:
            yield el


def flt_l(l, nx=None, n=0):
    """
    ... flatten a nested List deeply (output a list) ...

    kwArgs
    ------
    nx: maximum levels for flattening (default to None as unlimited)
     n: flattening counter; NOTE: used within recursive call only

    Examples
    --------
    >>> x = [1, 2, (3, 4), [5, 6, [7, 8], (9, 10)]]
    >>> flt_l(x)
    Out: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> flt_l(x, nx=1)
    Out: [1, 2, 3, 4, 5, 6, [7, 8], (9, 10)]
    """
    return list(flt_(l, nx=nx, n=n))


def kde_(obs, **kde_opts):
    """
    ... derive estimated distribution using kde from statsmodels.api ...

    Notes
    -----
    default option values as documented
    >>> help(sm.nonparametric.KDEUnivariate) #for more options
    """
    import statsmodels.api as sm
    o = sm.nonparametric.KDEUnivariate(obs)
    o.fit(**kde_opts)

    return o


def kde__(
        obs,
        log_it=False,
        **kde_opts,
        ):
    """
    ...  similar to kde_ but accept log transform of observations ...

    kwArgs
    ------
    log_it: if log(obs) before fitting

    Returns
    -------
        x: true x
        y: true pdf
    kde_o: kde class for log(x) if log_it is True

    Notes
    -----
    default option values as documented
    >>> help(sm.nonparametric.KDEUnivariate) #for more options
    """
    if log_it:
        kde_o = kde_(np.log(obs[obs > 0]), **kde_opts)
        x = np.exp(kde_o.support)
        y = kde_o.density / x
    else:
        kde_o = kde_(obs, **kde_opts)
        x, y = kde_o.support, kde_o.density
    return x, y, kde_o


def ismono_(x, axis=-1):
    """
    ... check if an array is monotonic along axis (default -1) ...

    Examples
    --------
    >>> x = [[2, 3, 4], [5, 6, 7], [3, 4, 5]]
    >>> x = np.asarray(x)
    >>> ismono_(x)
    Out: True
    >>> ismono_(x, axis=0)
    Out: False
    """
    return (np.all(np.diff(x, axis=axis) > 0)
            or np.all(np.diff(x, axis=axis) < 0))


def isincr_(x, axis=-1):
    """
    ... check if an array is monotonic increasing along axis (default -1) ...
    """
    return np.all(np.diff(x, axis=axis) > 0)


def isdecr_(x, axis=-1):
    """
    ... check if an array is monotonic decreasing along axis (default -1) ...
    """
    return np.all(np.diff(x, axis=axis) < 0)


def nSlice_(shape, axis=-1):
    """
    ... get total number of slices of a CUBE/ARRAY along axis ...

    Args
    ----
    shape: shape of parent CUBE/ARRAY that has multiple dimensions
     axis: axis along which the slice is

    Returns
    -------
    total number of slices

    Examples
    --------
    >>> x = (2, 3, 5)
    >>> nSlice_(x)
    Out: 6
    >>> nSlice_(x, axis=0)
    Out: 15
    >>> nSlice_(x, axis=(0, -1))
    Out: 3
    """

    if axis is None:
        shp = shape
    else:
        axis = rpt_(axis, len(shape))
        axis = (axis,) if not isIter_(axis) else axis
        shp = [ii for i, ii in enumerate(shape) if i not in axis]
    return int(np.prod(shp))


def ind_shape_i_(
        shape,
        i,
        axis=-1,
        sl_=np.s_[:],
        ):
    """
    ... get indices of CUBE/ARRAY for #i slice along axis ...

    Args
    ----
    shape: shape of parent CUBE/ARRAY that has multiple dimensions
        i: slice # of all of parent CUBE/ARRAY in C ORDER

    kwArgs
    ------
     axis: axis along which the slice is
      sl_: slice, list, or 1d array of selected indices along axis

    Returns
    -------
    indices associated with #i slice

    Examples
    --------
    >>> x = (2, 3, 5)
    >>> ind_shape_i_(x, 0)
    Out: (0, 0, slice(None, None, None))
    >>> ind_shape_i_(x, 1, axis=(0, -1))
    Out: (slice(None, None, None), 1, slice(None, None, None))
    >>> ind_shape_i_(x, 1, sl_=[1, 2])
    Out: (0, 0, [1, 2])
    """

    if axis is None:
        return np.unravel_index(i, shape)
    else:
        axis = rpt_(axis, len(shape))
        axis = (axis,) if not isIter_(axis) else axis
        shp = [ii for i, ii in enumerate(shape) if i not in axis]
        tmp = [i for i in range(len(shape)) if i not in axis]
        shpa = {ii: i for i, ii in enumerate(tmp)}
        iikk = np.unravel_index(i, shp)
        return tuple(iikk[shpa[i]] if i in tmp else sl_
                     for i in range(len(shape)))


def ind_sm_(ind):
    """
    ... if x[ind] shares memonry of x where x is of type numpy array ...

    Examples
    --------
    >>> ind_sm_((0, 0, slice(None, None, None)))
    Out: True
    >>> ind_sm_((0, [1, 2]))
    Out: False
    """
    return (all(not isIter_(i) for i in ind) and
            any(isinstance(i, slice) for i in ind))


def ind_s_(
        ndim,
        axis,
        sl_i,
        ):
    """
    ... creat indices for extraction along axis ...

    Args
    ----
        ndim: number of dimensions in data
        axis: along which for the extraction
        sl_i: slice, list, or 1d array of selected indices along axis

    Returns
    -------
        indices of ndim datan for extraction

    Examples
    --------
    >>> ind_s_(3, 0, [3, 4])
    Out: ([3, 4], slice(None, None, None), slice(None, None, None))
    >>> ind_s_(3, 1, 3)
    Out: (slice(None, None, None), 3, slice(None, None, None))
    >>> ind_s_(3, 1, np.s_[3:4])
    Out: (slice(None, None, None), slice(3, 4, None), slice(None, None, None))
    """

    axis = rpt_(axis, ndim)
    return np.s_[:,] * axis + (sl_i,) + np.s_[:,] * (ndim - axis - 1)


def inds_ss_(
        ndim,
        axis,
        sl_i,
        *vArg,
        fancy=True,
        ):
    """
    ... creat indices for extraction, similar to ind_s_(...) but works for ...
    ... multiple axes                                                      ...

    Args
    ----
     ndim: number of dimensions in data
     axis: along which for the extraction
     sl_i: slice, list, or 1d array of selected indices along axis
     vArg: any pairs of (axis, sl_i)

    kwArgs
    ------
    fancy: secure for many complicated cases
    Returns
    -------
    indices of ndim data for extraction (tuple)

    Examples
    --------
    >>> inds_ss_(3, 0, [1, 2], -1, [2, 3, 4])
    Out:
    (array([[1],
            [2]]),
     slice(None, None, None),
     array([[2, 3, 4]]))
    >>> inds_ss_(3, 0, [3, 4, 5], -1, [2, 3, 4], fancy=False)
    Out: ([1, 2], slice(None, None, None), [2, 3, 4])
    >>> x = np.arange(3*4*5).reshape(3, 4, 5)
    >>> x
    Out:
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],

           [[20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39]],

           [[40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59]]])
    >>> x[inds_ss_(3, 0, [1, 2], -1, [2, 3, 4])]
    Out:
    array([[[22, 27, 32, 37],
            [23, 28, 33, 38],
            [24, 29, 34, 39]],

           [[42, 47, 52, 57],
            [43, 48, 53, 58],
            [44, 49, 54, 59]]])
    >>> x[inds_ss_(3, 0, [1, 2], -1, [2, 3, 4], fancy=False)]
    IndexError: shape mismatch: indexing arrays could not be broadcast
                together with shapes (2,) (3,)
    """

    assert len(vArg)%2 == 0, 'arguments not paired!'

    inds = list(ind_s_(ndim, axis, sl_i))

    if len(vArg) > 0:
        ax, sl = list(vArg[::2]), list(vArg[1::2])
        if (any(rpt_(ii, ndim) == rpt_(axis, ndim) for ii in ax)
            or len(pd.unique(rpt_(ax, ndim))) != len(ax)):
            raise ValueError('duplicated axis provided!')
        else:
            for ii, ss in zip(ax, sl):
                inds[rpt_(ii, ndim)] = ss

    return iind_(tuple(inds)) if fancy else tuple(inds)


def iind_(inds):
    """
    ... rebuild extraction indices (n axes) ...

    Examples
    --------
    >>> ind = inds_ss_(3, 0, [3, 4, 5], -1, [2, 3, 4], fancy=False)
    >>> ind
    Out: ([1, 2], slice(None, None, None), [2, 3, 4])
    >>> iind_(ind)
    Out:
    (array([[1],
            [2]]),
     slice(None, None, None),
     array([[2, 3, 4]]))
    """
    x = [ii for ii, i in enumerate(inds) if isIter_(i)]
    if len(x) < 2:
        return inds
    else:
        inds_ = list(inds)
        y = [i for ii, i in enumerate(inds) if isIter_(i)]
        z = np.ix_(*y)
        for ii, i in zip(x, z):
            inds_[ii] = i
        return tuple(inds_)


def extract_(xnd, axis, sl_i, *vArg, fancy=None):
    """
    ... extract ARRAY by providing selection along axis/axes ...

    Args
    ----
     xnd: parent ARRAY
    axis: along which for the extraction; axis name acceptable
    sl_i: slice, list, or 1d array of selected indices along axis
    vArg: any pairs of (axis, sl_i)

    Note
    ----
    >>> help(inds_ss_)
    """

    if len(vArg)%2 != 0:
        raise Exception("arguments {!r} not interpretable!".format(vArg))

    if len(vArg) > 0:
        ax, sl = list(vArg[::2]), list(vArg[1::2])
        ax.insert(0, axis)
        sl.insert(0, sl_i)
    else:
        ax = [axis]
        sl = [sl_i]

    nArg = [(i, j) for i, j in zip(ax, sl)]
    nArg = tuple(j for i in nArg for j in i)
    if fancy is None:
        if tryattr_(xnd, '__orthogonal_indexing__'):
            fancy = False
        else:
            fancy = True

    return xnd[inds_ss_(xnd.ndim, *nArg, fancy=fancy)]


def ind_inRange_(
        y,
        y0,
        y1,
        side='both',
        i_=False,
        r_=None,
        ):
    """
    ... boolen as y between y0 and y1 ...

    kwArgs
    ------
    side:
        0/'i'/'inner': exclude bounds
        -1/'l'/'left': include left bound
        1/'r'/'right': include right bound
         2/'b'/'both': include bounds
      i_:
        True: np.where(ind); False: ind
      r_: repeat period

    Examples
    --------
    >>> y = np.arange(0, 360, 60)
    >>> y
    Out: array([  0,  60, 120, 180, 240, 300])
    >>> ind_inRange_(y, 60, 180)
    Out: array([False,  True,  True,  True, False, False])
    >>> ind_inRange_(y, 60, 180, side=-1)
    Out: array([False,  True,  True, False, False, False])
    >>> ind_inRange_(y, 60, 180, i_=True)
    Out: (array([1, 2, 3]),)
    >>> ind_inRange_(y, 180, 0, r_=360)
    Out: array([ True, False, False,  True,  True,  True])
    >>> ind_inRange_(y, 120, 60, r_=360)
    Out: array([False,  True,  True, False, False, False])
    """
    if r_ is None:
        if side in (0, 'i', 'inner'):
            ind = np.logical_and((y > y0), (y < y1))
        elif side in (-1, 'l', 'left'):
            ind = np.logical_and((y >= y0), (y < y1))
        elif side in (1, 'r', 'right'):
            ind = np.logical_and((y > y0), (y <= y1))
        elif side in (2, 'b', 'both'):
            ind = np.logical_and((y >= y0), (y <= y1))
        else:
            raise ValueError("unknow value of side!")
        return np.where(ind) if i_ else ind
    else:
        if y0 > y1 and y0 - y1 < r_ / 2:
            y0, y1 = y1, y0
        else:
            y1 = rpt_(y1, y0 + r_, y0)
        return ind_inRange_(rpt_(y, y0 + r_, y0), y0, y1, side=side, i_=i_)


def ind_win_(
        doy,
        d,
        w,
        rb=366,
        lb=1,
        ):
    """
    ... indices of a window in cylinder axis ...

    Args
    ----
     d: window center
     w: window size (radius)

    kwArgs
    ------
    rb: right bound
    lb: left bound

    Examples
    --------
    >>> ind_win_(np.arange(1, 10), 2, 3)
    Out:
    array([ True,  True,  True,  True,  True, False, False, False, False])
    >>> doy = rpt_(np.arange(-4, 6), rb=366, lb=1)
    >>> doy
    Out: array([361, 362, 363, 364, 365,   1,   2,   3,   4,   5])
    >>> ind_win_(doy, 1, 3)
    Out:
    array([False, False,  True,  True,  True,  True,  True,  True,  True,
           False])
    """
    dw = rpt_(np.arange(d - w, d + 1 + w), rb=rb, lb=lb)
    return np.isin(doy, dw)


def nanMask_(data):
    """
    ... give nan where masked ...
    """
    if np.ma.isMaskedArray(data):
        o = data.data.copy()
        if np.ma.is_masked(data):
            o[np.ma.getmaskarray()] = np.nan
        return o
    else:
        return data


def rPeriod_(p_bounds, TeX_on=False):
    """
    ... return readable style of period from period bounds ...

    Examples
    --------
    >>> rPeriod_([1981, 2010])
    Out: '1981-2010'
    >>> rPeriod_([1981, 1981])
    Out: '1981'
    >>> rPeriod_([1981, 2010], TeX_on=True)
    Out: '1981$-$2010'
    """
    if p_bounds[0] != p_bounds[-1]:
        if TeX_on:
            return r'{:d}$-${:d}'.format(p_bounds[0], p_bounds[-1])
        else:
            return r'{:d}-{:d}'.format(p_bounds[0], p_bounds[-1])
    else:
        return r'{:d}'.format(p_bounds[0])


def rTime_(t):
    """
    ... return readable style of time interval ...

    Examples
    --------
    >>> rTime_(61)
    Out: 'passing :t::i:::m::::e:::::::::::::: 00:01:01'
    >>> rTime_(360000)
    Out: 'passing :t::i:::m::::e:::::41 day(s) 16:00:00'
    """
    import time
    d = t // 86400                                                             # (60 * 60 * 24)
    s = t % 86400
    _s1 = f"{d::>2g} day(s)" if d >= 1 else ":"*9
    _s2 = time.strftime(' %H:%M:%S', time.gmtime(t))
    return ''.join(('passing :t::i:::m::::e:::::', _s1, _s2))


def uniqL_(l):
    """
    ... return sorted unique elements of list l ...

    Examples
    --------
    >>> uniqL_([0, 3, 6, 3, 1])
    Out: [0, 1, 3, 6]
    >>> uniqL_([0, 3, [6, 3], 1])
    Out: [0, 1, 3, 6]
    >>> uniqL_(['c', 'a', 'cd', 'a', 'f'])
    Out: ['a', 'c', 'cd', 'f']
    """
    return list(np.unique(np.array(flt_l(l))))


def ouniqL_(l):
    """
    ... return ordered unique elements of list l ...

    Examples
    --------
    >>> ouniqL_([0, 3, 6, 3, 1])
    Out: [0, 3, 6, 1]
    >>> ouniqL_([0, 3, [6, 3], 1])
    Out: [0, 3, 6, 1]
    >>> ouniqL_(['c', 'a', 'cd', 'a', 'f'])
    Out: ['c', 'a', 'cd', 'f']
    """
    return list(dict.fromkeys(flt_l(l)).keys())


def schF_keys_(idir, *keys, s_='*',  ext='*', ordered=False, h_=False):
    """
    ... find files that contain specified keyword(s) in the directory ...

    Args
    ----
       idir: directory for file name searching
       keys: keyword(s)

    kwArgs
    ------
         s_: file name start with
        ext: extention
    ordered: should the specified keys taken as ordered or not
         h_: include hided files or not

    Returns
    -------
    file name list
    """
    import glob
    import os
    from itertools import permutations
    s = '*'
    if ordered:
        pm = [s.join(keys)]
    else:
        a = set(permutations(keys))
        pm = [s.join(i) for i in a]
    fn = []
    for i in pm:
        if h_:
            fn += glob.iglob(os.path.join(idir, '.' + s.join([s_, i, ext])))
        fn += glob.glob(os.path.join(idir, s.join([s_, i, ext])))
    fn = list(set(fn))
    fn.sort()
    return fn


def valueEqFront_(l, v):
    """
    ... move element(s) if equal to v or v() returns True ...
    ... to front in list l                                ...

    Examples
    --------
    >>> valueEqFront_([1, 2, 3, 1, 2, 3, 4, 5, 2, 6],  1)
    Out: [1, 1, 2, 3, 2, 3, 4, 5, 2, 6]
    >>> valueEqFront_([1, 2, 3, 1, 2, 3, 4, 5, 2, 6],  lambda x: x > 3)
    Out: [4, 5, 6, 1, 2, 3, 1, 2, 3, 2]
    """
    if not callable(v):
        func = lambda x: x==v
    else:
        func = v
    l0 = [i for i in l if func(i)]
    l1 = [i for i in l if not func(i)]
    return l0 + l1


def indFront_(l, v):
    """
    ... move element with specified index to front in list l ...

    Examples
    --------
    >>> indFront_([1, 2, 3, 1, 2, 3, 4, 5, 2, 6],  1)
    Out: [2, 1, 3, 1, 2, 3, 4, 5, 2, 6]
    """
    ind = valueEqFront_(list(range(len(l))), v)
    return l_ind_(l, ind)


def iter_str_(iterable):
    """
    ... transform elements to string ...

    Examples
    --------
    >>> iter_str_([1, 2, 3, 1, 2, 3, 4, 5, 2, 6])
    Out: ['1', '2', '3', '1', '2', '3', '4', '5', '2', '6']
    >>> iter_str_([1, 2, 'abc', (5, 6)])
    Out: ['1', '2', 'abc', ('5', '6')]
    >>> iter_str_([1, 2, 'abc', range(3)])
    Out: ['1', '2', 'abc', ['0', '1', '2']]
    """
    _f = _typef(iterable)
    if not isIter_(iterable):
        return str(iterable)
    else:
        return _f([iter_str_(i) for i in iterable])
    #tmp = flt_l(iterable)
    #return [str(i) for i in tmp]


def ext_(s, sub=None):
    """
    ... get extension from filename (str) ...

    Examples
    --------
    >>> ext_('a/b.nc')
    Out: '.nc'
    >>> ext_('a/b.nc', '.grb')
    Out: 'a/b.grb'
    """
    import os
    o = os.path.splitext(s)[1]
    return s.replace(o, sub) if sub else o
    #import re
    #tmp = re.search(r'(?<=\w)\.\w+$', s)
    #return tmp.group() if tmp else ''


def find_patt_(p, s):
    """
    ... return s or list of items in s that match the given pattern ...

    Examples
    --------
    >>> find_patt_('(?<!\d)\d{3}(?!\d)', ['2333db', 'a233', 'ccb000r'])
    Out: ['a233', 'ccb000r']
    """
    import re
    if isinstance(s, str):
        return s if re.search(p, s) else None
    elif isIter_(s, xi=str):
        return [i for i in s if find_patt_(p, i)]


def pure_fn_(s, no_ext=True):
    """
    ... get pure filename without path to and without extension ...

    Examples
    --------
    >>> pure_fn_('a/b/c.d')
    Out: 'c'
    >>> pure_fn_('a/b/c.d', no_ext=False)
    Out: 'c.d'
    """
    #import re
    import os
    def _rm_etc(s):
        #return re.sub(r'\.\w+$', '', s) if no_ext else s
        return os.path.splitext(s)[0] if no_ext else s
    if isinstance(s, str):
        #tmp = re.search(r'((?<=[\\/])[^\\/]*$)|(^[^\\/]+$)', s)
        #fn = tmp.group() if tmp else tmp
        fn = os.path.basename(s)
        return _rm_etc(fn) #if fn else ''
    elif isIter_(s, str):
        return [pure_fn_(i) for i in s]


def isMonth_(
        mn,
        short_=True,
        nm=3,
        ):
    """
    ...  if input string is name of a month ...

    kwArgs
    ------
    short_: if received short name of month
        nm: number of letters for short name of month

    Examples
    --------
    >>> isMonth_('Jan')
    Out: True
    >>> isMonth_('Febr')
    Out: False
    >>> isMonth_('Febr', nm=4)
    Out: True
    >>> isMonth_('May', nm=4)
    Out: True
    """
    mns = ['january', 'february', 'march', 'april', 'may', 'june',
           'july', 'august', 'september', 'october', 'november', 'december']
    n = len(mn)
    if n < 3:
        warnings.warn("month string shorter than 3 letters; return 'False'!")
        return False
    mn3s = [i[:n] for i in mns]
    if short_:
        return mn.lower() in mn3s
    else:
        return mn.lower() in mns or mn.lower() in mn3s


def mnN_(mn):
    """
    ...  month order in calendar (mn should be at least 3 letters) ...

    Examples
    --------
    >>> mnN_('nov')
    Out: 11
    """
    mns = ['january', 'february', 'march', 'april', 'may', 'june',
           'july', 'august', 'september', 'october', 'november', 'december']
    n = len(mn)
    if n < 3:
        warnings.warn("month string short than 3 letters; 1st guess used!")
    mn3s = [i[:n] for i in mns]
    return mn3s.index(mn.lower()) + 1


def isSeason_(mmm, ismmm_=True):
    """
    ...  if mmm is a season named with 1st letters of composing months ...

    Examples
    --------
    >>> isSeason_('mam')
    Out: True
    >>> isSeason_('ndjf')
    Out: True
    >>> isSeason_('spring')
    Out: False
    >>> isSeason_('spring', ismmm_=False)
    Out: True
    """
    mns = 'jfmamjjasond' * 2
    n = mns.find(mmm.lower())
    s4 = {'spring', 'summer', 'autumn', 'winter'}
    if ismmm_:
        return (1 < len(mmm) < 12 and n != -1)
    else:
        return (1 < len(mmm) < 12 and n != -1) or mmm.lower() in s4


def valid_seasons_(seasons, ismmm_=True):
    """
    ... if provided seasons valid ...

    Examples
    --------
    >>> valid_seasons_(['jja', 'son', 'djf', 'mam'])
    Out: True
    >>> valid_seasons_({'spring', 'summer', 'autumn', 'winter'})
    Out: False
    >>> valid_seasons_({'spring', 'summer', 'autumn', 'winter'}, ismmm_=False)
    Out: True
    >>> valid_seasons_(['jja', 'son', 'djfmam'])
    Out: True
    >>> valid_seasons_(['jja', 'son'])
    Out: False
    """
    o = all(isSeason_(season, ismmm_=ismmm_) for season in seasons)
    if o:
        o_ = sorted(flt_l(mmmN_(season) for season in seasons))
        return np.array_equal(o_, np.arange(12) + 1)
    else:
        return False


def _month_season_numbers(seasons):
    """
    Examples
    --------
    >>> _month_season_numbers({'spring', 'summer', 'autumn', 'winter'})
    Out: [None, 2, 2, 1, 1, 1, 0, 0, 0, 3, 3, 3, 2]
    >>> _month_season_numbers(['jja', 'son', 'djfmam'])
    Out: [None, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2]
    """
    month_season_numbers = [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for season_number, season in enumerate(seasons):
        for month in mmmN_(season):
            month_season_numbers[month] = season_number
    return month_season_numbers


def _month_year_adjust(seasons):
    """
    Examples
    --------
    >>> _month_year_adjust(['jja', 'son', 'djfmam'])
    Out: [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    month_year_adjusts = [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for season in seasons:
        months_in_season = mmmN_(season)
        for month in months_in_season:
            if month > months_in_season[-1]:
                month_year_adjusts[month] = 1
    return month_year_adjusts


def _m2sm(month, season):
    """
    ... season membership...

    Examples
    --------
    >>> m2sm_([4, 5, 6], 'jja')
    Out: array([False, False,  True])
    """
    return month in mmmN_(season)


def _m2sya(month, seasons=('djf', 'mam', 'jja', 'son')):
    """
    ... year adjust according to defined seasons ...

    Examples
    --------
    >>> m2sya_([4, 5, 6, 11, 12], seasons=['jja', 'so', 'ndjfmam'])
    Out: array([0, 0, 0, 1, 1])
    """
    sya = _month_year_adjust(seasons)
    return sya[month]


def _m2s(month, seasons=('djf', 'mam', 'jja', 'son')):
    """
    ... month to defined season ...

    Examples
    --------
    >>> m2s_([4, 5, 6, 11, 12])
    Out: array(['mam', 'mam', 'jja', 'son', 'djf'], dtype='<U3')
    >>> m2s_([4, 5, 6, 11, 12], seasons=['jja', 'so', 'ndjfmam'])
    Out:
    array(['ndjfmam', 'ndjfmam', 'jja', 'ndjfmam', 'ndjfmam'], dtype='<U7')
    """
    ssm = _month_season_numbers(seasons)
    return seasons[ssm[month]]


m2s_ = np.vectorize(_m2s, excluded=['seasons'])
m2sm_ = np.vectorize(_m2sm, excluded=['season'])
m2sya_ = np.vectorize(_m2sya, excluded=['seasons'])


def mmmN_(mmm):
    """
    ... months in season ...

    Examples
    --------
    >>> mmmN_('mam')
    Out: array([3, 4, 5])
    >>> mmmN_('djf')
    Out: array([12,  1,  2])
    >>> mmmN_('mamjja')
    Out: array([3, 4, 5, 6, 7, 8])
    """
    ss = {'spring': 'mam',
          'summer': 'jja',
          'autumn': 'son',
          'winter': 'djf'}
    mmm = ss[mmm] if mmm in ss.keys() else mmm

    mns = 'jfmamjjasond' * 2
    n = mns.find(mmm.lower())
    if n != -1:
        return rpt_(range(n+1, n+1+len(mmm)), 13, 1)
    else:
        raise ValueError("{!r} unrecognised as a season!".format(mmm))


def rest_mns_(mmm):
    """
    ... get rest season named with months' 1st letter ...

    Examples
    --------
    >>> rest_mns_('ondjfm')
    Out: 'amjjas'
    """
    mns = 'jfmamjjasond' * 2
    n = mns.find(mmm.lower())
    if n == -1:
        raise Exception('unknown season provided!')
    else:
        return mns[n + len(mmm):n + 12]


def rSUM1d_(
        y,
        n,
        mode='valid',
        ):
    """
    ... sum over a n-point rolling_window ...

    Args
    ----
       y: 1d array
       n: size of rolling window

    kwArgs
    ------
    mode: see np.convolve

    Examples
    --------
    >>> y = np.arange(6)
    >>> y = np.ma.masked_equal(y, 3)
    >>> y
    Out:
    masked_array(data=[0, 1, 2, --, 4, 5],
                 mask=[False, False, False,  True, False, False],
           fill_value=3)
    >>> rSUM1d_(y, 3)
    Out: array([3., 3., 6., 9.])
    >>> rSUM1d_(y.data, 3)
    Out: array([ 3.,  6.,  9., 12.])
    >>> rSUM1d_(y.data, 3, mode='full')
    Out: array([ 0.,  1.,  3.,  6.,  9., 12.,  9.,  5.])
    >>> rSUM1d_(y.data, 3, mode='same')
    Out: array([ 1.,  3.,  6.,  9., 12.,  9.])
    """
    if hasattr(y, 'mask'):
        msk = np.ma.getmaskarray(y)
    else:
        msk = np.isnan(y)
    return np.convolve(np.where(msk, 0, y), np.ones((n,)), mode)


def rSUM2d_(
        y,
        m,
        n,
        mode='valid',
        ):
    """
    ... sum over a n-point rolling_window ...

    Args
    ----
       y: 1d array
       m: size of rolling window along axis=0
       n: size of rolling window along axis=1

    kwArgs
    ------
    mode: see scipy.signal.convolve

    Examples
    --------
    >>> y = np.arange(20).reshape(4, -1)
    >>> y = np.ma.masked_equal(y, 3)
    >>> y
    Out:
    masked_array(
    data=[[0, 1, 2, --, 4],
          [5, 6, 7, 8, 9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19]],
    mask=[[False, False, False,  True, False],
          [False, False, False, False, False],
          [False, False, False, False, False],
          [False, False, False, False, False]],
    fill_value=3)
    >>> rSUM2d_(y, 3, 3)
    Out:
    array([[ 54.,  60.,  69.],
           [ 99., 108., 117.]])
    >>> rSUM2d_(y.data, 3, 3)
    Out:
    array([[ 54.,  63.,  72.],
           [ 99., 108., 117.]])
    >>> rSUM2d_(y.data, 3, 3, mode='full')
    Out:
    array([[  0.,   1.,   3.,   6.,   9.,   7.,   4.],
           [  5.,  12.,  21.,  27.,  33.,  24.,  13.],
           [ 15.,  33.,  54.,  63.,  72.,  51.,  27.],
           [ 30.,  63.,  99., 108., 117.,  81.,  42.],
           [ 25.,  52.,  81.,  87.,  93.,  64.,  33.],
           [ 15.,  31.,  48.,  51.,  54.,  37.,  19.]])
    >>> rSUM2d_(y.data, 3, 3, mode='same')
    Out:
    array([[ 12.,  21.,  27.,  33.,  24.],
           [ 33.,  54.,  63.,  72.,  51.],
           [ 63.,  99., 108., 117.,  81.],
           [ 52.,  81.,  87.,  93.,  64.]])
    """
    from scipy.signal import convolve2d
    if hasattr(y, 'mask'):
        msk = np.ma.getmaskarray(y)
    else:
        msk = np.isnan(y)
    return convolve2d(np.where(msk, 0, y), np.ones((m, n)), mode)


def rMEAN1d_(
        y,
        n,
        mode='valid',
        ):
    """
    ... mean over a n-point rolling_window ...

    Args
    ----
       y: 1d array
       n: size of rolling window

    kwArgs
    ------
    mode: see np.convolve

    Examples
    --------
    >>> y = np.arange(6)
    >>> y = np.ma.masked_equal(y, 3)
    >>> y
    Out[162]:
    masked_array(data=[0, 1, 2, --, 4, 5],
                 mask=[False, False, False,  True, False, False],
           fill_value=3)
    >>> rMEAN1d_(y, 3)
    Out[165]:
    masked_array(data=[1. , 1.5, 3. , 4.5],
                 mask=False,
           fill_value=1e+20)
    >>> rMEAN1d_(y.data, 3)
    Out: array([1., 2., 3., 4.])
    >>> rMEAN1d_(y.data, 3, mode='full')
    Out: array([0. , 0.5, 1. , 2. , 3. , 4. , 4.5, 5. ])
    >>> rMEAN1d_(y.data, 3, mode='same')
    Out: array([0.5, 1. , 2. , 3. , 4. , 4.5])
    """
    if hasattr(y, 'mask'):
        msk = np.ma.getmaskarray(y)
    else:
        msk = np.isnan(y)
    uu = np.convolve(np.where(msk, 0, y), np.ones((n,)), mode)
    dd = np.convolve(~msk, np.ones((n,)), mode)
    dd[dd == 0] = np.nan
    out = uu / dd
    return np.ma.masked_where(np.isnan(out), out) if hasattr(y, 'mask') else\
           out


def rMEAN2d_(
        y,
        m,
        n,
        mode='valid',
        ):
    """
    ... mean over a n-point rolling_window ...

    Args
    ----
       y: 2d array
       m: size of rolling window along axis=0
       n: size of rolling window along axis=1

    kwArgs
    ------
    mode: see np.convolve

    Examples
    --------
    >>> y = np.arange(20).reshape(4, -1)
    >>> y = np.ma.masked_equal(y, 3)
    >>> y
    Out:
    masked_array(
    data=[[0, 1, 2, --, 4],
          [5, 6, 7, 8, 9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19]],
    mask=[[False, False, False,  True, False],
          [False, False, False, False, False],
          [False, False, False, False, False],
          [False, False, False, False, False]],
    fill_value=3)
    >>> rMEAN2d_(y, 3, 3)
    Out:
    masked_array(
    data=[[ 6.   ,  7.5  ,  8.625],
          [11.   , 12.   , 13.   ]],
    mask=False,
    fill_value=1e+20)
    >>> rSUM2d_(y.data, 3, 3)
    Out:
    array([[ 6.,  7.,  8.],
           [11., 12., 13.]])
    >>> rMEAN2d_(y.data, 3, 3, mode='full')
    Out:
    array([[ 0. ,  0.5,  1. ,  2. ,  3. ,  3.5,  4. ],
           [ 2.5,  3. ,  3.5,  4.5,  5.5,  6. ,  6.5],
           [ 5. ,  5.5,  6. ,  7. ,  8. ,  8.5,  9. ],
           [10. , 10.5, 11. , 12. , 13. , 13.5, 14. ],
           [12.5, 13. , 13.5, 14.5, 15.5, 16. , 16.5],
           [15. , 15.5, 16. , 17. , 18. , 18.5, 19. ]])
    >>> rMEAN2d_(y.data, 3, 3, mode='same')
    Out:
    array([[ 3. ,  3.5,  4.5,  5.5,  6. ],
           [ 5.5,  6. ,  7. ,  8. ,  8.5],
           [10.5, 11. , 12. , 13. , 13.5],
           [13. , 13.5, 14.5, 15.5, 16. ]])
    """
    from scipy.signal import convolve2d
    if hasattr(y, 'mask'):
        msk = np.ma.getmaskarray(y)
    else:
        msk = np.isnan(y)
    uu = convolve2d(np.where(msk, 0, y), np.ones((m, n)), mode)
    dd = convolve2d(~msk, np.ones((m, n)), mode)
    dd[dd == 0] = np.nan
    out = uu / dd
    return np.ma.masked_where(np.isnan(out), out) if hasattr(y, 'mask') else\
           out


def l__(
        msg,
        out=True,
        _p=False,
        ):
    """
    ... starting logging msg giving a time stamp ...
    """
    import time
    import logging
    msg0 = ' {} -->'.format(msg)
    if _p:
        print(msg0)
    else:
        logging.info(msg0)
    if out:
        return time.time()


def ll_(
        msg,
        t0=None,
        _p=False,
        ):
    """
    ... ending logging msg giving a time lapse if starting time stamp given ...
    """
    import time
    import logging
    msg0 = ' {}{}'.format(msg, ' <--' if t0 else '')
    if _p:
        print(msg0)
    else:
        logging.info(msg0)
    if t0:
        msg1 = ' {}'.format(rTime_(time.time() - t0))
        if _p:
            print(msg1)
            print(' ')
        else:
            logging.info(msg1)
            logging.info(' ')


def slctStrL_(
        strl,
        incl=None,
        excl=None,
        ):
    """
    ... select items including/excluding sts(s) for a list of str ...

    Examples
    --------
    >>> x = ['aabbcc', 'aaccee', 'bbccdd', 'ddeeff']
    >>> slctStrL_(x, incl='aa')               # include aa
    Out: ['aabbcc', 'aaccee']
    >>> slctStrL_(x, excl='aa')               # exclude aa
    Out: ['bbccdd', 'ddeeff']
    >>> slctStrL_(x, incl=['aa', 'ee'])       # include aa & ee
    Out: ['aaccee']
    >>> slctStrL_(x, incl=[['aa', 'ee']])     # include aa | ee
    Out: ['aabbcc', 'aaccee', 'ddeeff']
    >>> slctStrL_(x, incl='aa', excl='bb')    # include aa meanwhile exclude bb
    Out: ['aaccee']
    >>> slctStrL_(x, excl=['aa', 'ee'])       # exclude aa & ee
    Out: ['aabbcc', 'bbccdd', 'ddeeff']
    >>> slctStrL_(x, excl=[['aa', 'ee']])     # exclude aa | ee
    Out: ['bbccdd']
    """
    def _in(s, L):
        if isinstance(L, str):
            return L in s
        else:
            return _inc(s, L)
    def _inc(s, L):
        return all([i in s if isinstance(i, str) else _incl(s, i) for i in L])
    def _incl(s, L):
        return any([i in s if isinstance(i, str) else _inc(s, i) for i in L])
    def _ex(s, L):
        if isinstance(L, str):
            return L not in s
        else:
            return _exc(s, L)
    def _exc(s, L):
        return any([i not in s if isinstance(i, str) else _excl(s, i)
                    for i in L])
    def _excl(s, L):
        return all([i not in s if isinstance(i, str) else _exc(s, i)
                    for i in L])
    if incl:
        strl = [i for i in strl if _in(i, incl)]
    if excl:
        strl = [i for i in strl if _ex(i, excl)]
    return strl


def latex_unit_(unit):
    """
    ... turn unit str into latex style ...

    Examples
    --------
    >>> latex_unit_('m3')
    Out: 'm$^{3}$'
    >>> latex_unit_('W m-2')
    Out: 'W m$^{-2}$'
    """
    import re
    def r__(m):
        return '$^{' + re.findall(r'-?\d+', m.group(0))[0] + '}$'
    return re.sub(r'(?<=[a-zA-Z])((\*\*)?-?\d+)', r__, unit)


def p_least_(pl, y0, y1):
    """
    ... select periods within [y0, y1] from a list of periods ...

    Examples
    --------
    >>> p_least_(['1901-1950', '1951-2000',  '2001-2050', '2051-2100'],
    ...          1981, 2010)
    Out: ['1951-2000', '2001-2050']
    """
    pl.sort()
    y0_, y1_ = str(y0), str(y1)
    def _cmp(x0, x1):
        n = min(len(x0), len(x1))
        return x0[:n] <= x1[:n]
    a = lambda x: _cmp(y0_, x) and _cmp(x, y1_)
    b = lambda x, y: a(x) or a(y)
    c = lambda x, y: _cmp(x, y0_) and _cmp(y1_, y)
    return [i for i in pl if b(i.split('-')[0], i.split('-')[-1])
            or c(i.split('-')[0], i.split('-')[-1])]


def p_deoverlap_(pl):
    """
    ... des-overlap period list ...

    Examples
    --------
    >>> p_deoverlap_(['1901-1950', '1951-2000', '1981-2000', '2001-2050',
    ...               '2051-2100'])
    Out: ['1901-1950', '1951-2000', '2001-2050', '2051-2100']
    """
    pl = np.asarray(pl)
    pi = np.arange(len(pl))
    ii_ = []
    iii_ = []
    a_ = lambda p: [int(i) for i in p.split('-')]
    b_ = lambda x, p: a_(p)[0] <= x <= a_(p)[-1]
    c_ = lambda p0, p1: b_(a_(p0)[0], p1) and b_(a_(p0)[-1], p1)
    for i in pi:
        if any([c_(pl[i], pl[ii])
                for ii in pi if ii != i and ii not in ii_]):
            ii_.append(i)
            iii_.append(False)
        else:
            iii_.append(True)
    return list(pl[iii_])


def _match2shps(
        shape0,
        shape1,
        fw=False,
        ):
    """
    ... derive dim # for shape0 inside shape1 ...

    Examples
    --------
    >>> _match2shps((4, 5), (4, 2, 3, 5))
    Out: (0, 3)
    >>> _match2shps((4, 5), (4, 4, 3, 5), fw=True)
    Out: (0, 3)
    >>> _match2shps((4, 5), (4, 4, 3, 5), fw=False)
    Out: (1, 3)
    """
    from itertools import combinations
    if len(shape1) < len(shape0):
        raise Exception('len(shape1) must >= len(shape0)!')
    cbs = list(combinations(shape1, len(shape0)))
    if shape0 not in cbs:
        raise Exception('unmatched shapes!')
    cbi = list(combinations(np.arange(len(shape1)), len(shape0)))
    if not fw:
        cbs.reverse()
        cbi.reverse()
    o = cbi[0]
    for ss, ii in zip(cbs, cbi):
        if shape0 == ss:
            o = ii
            break
    return o


def robust_bc2_(
        data,
        shape,
        axes=None,
        fw=False,
        ):
    """
    ... broadcast data to shape according to axes given or direction ...

    kwArgs
    ------
    axes: axes in target shape correspongding to the given data
      fw: when axes not specified, try mactching in the forward way or
          the backward way (default)
    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> robust_bc2_(x, (2, 2, 3))
    Out:
    array([[[0, 1, 2],
            [3, 4, 5]],

           [[0, 1, 2],
            [3, 4, 5]]])
    >>> robust_bc2_(x, (2, 2, 3), axes=(0, -1))
    Out:
    array([[[0, 1, 2],
            [0, 1, 2]],

           [[3, 4, 5],
            [3, 4, 5]]])
    >>> robust_bc2_(x, (2, 2, 3), fw=True)
    Out:
    array([[[0, 1, 2],
            [0, 1, 2]],

           [[3, 4, 5],
            [3, 4, 5]]])
    """
    data = data.squeeze()
    dshp = data.shape
    if axes:
        axes = rpt_(axes, len(shape))
        axes = (axes,) if not isIter_(axes) else axes
        if len(axes) != len(dshp):
            raise ValueError("len(axes) != len(data.squeeze().shape)")
        if (len(pd.unique(axes)) != len(axes) or
            any([i not in range(len(shape)) for i in axes])):
            raise ValueError("one or more axes exceed target shape of data!")
        if any(dshp[i] != shape[ii] for i, ii in enumerate(axes)):
            raise ValueError("unmatched shape between data and destination!")
    else:
        axes = _match2shps(dshp, shape, fw=fw)
    if not isincr_(axes):
        data = np.moveaxis(data, np.argsort(axes), np.arange(len(axes)))
    shp_ = tuple(ii if i in axes else 1 for i, ii in enumerate(shape))
    return np.broadcast_to(data.reshape(shp_), shape)


def intsect_(*l):
    """
    ... intersection of lists ...

    Examples
    --------
    >>> intsect_([1, 2, 3], [2, 3], [3, 4])
    Out: [3]
    """
    if len(l) > 1:
        ll = list(set(l[0]).intersection(*l[1:]))
        ll.sort()
        return ll
    elif len(l) == 1:
        return l[0]


def _typef(l):
    if isinstance(l, str):
        _f = ''.join
    elif isinstance(l, np.ndarray):
        _f = np.asarray
    elif isinstance(l, (tuple, set)):
        _f = type(l)
    else:
        _f = list
    return _f


def l_ind_(l, ind):
    """
    ... extract list (or other Iterable objects) by providing indices ...

    Returns
    -------
    the same type of input l if applicable (for most cases) otherwise list

    Examples
    --------
    >>> x = 'abcdefg'
    >>> l_ind_(x, [1, 2])
    Out: 'bc'
    >>> l_ind_(list(x), [1, 2])
    Out: ['b', 'c']
    >>> l_ind_(x, np.arange(len(x))%2==0)
    Out: 'aceg'
    >>> l_ind_(set(x), np.arange(len(x))%2==0)
    Out: {'a', 'c', 'e', 'g'}
    """
    _f = _typef(l)
    if isIter_(ind, xi=(bool, np.bool_)):
        return _f([i for i, ii in zip(l, ind) if ii])
    elif isIter_(ind, xi=(int, np.integer)):
        ind = rpt_(ind, len(l))
        return _f([l[i] for i in ind])


def l_flp_(l):
    """
    ... flip list (or other Iterable objects) ...

    Examples
    --------
    >>> l_flp_([(1, 2), (3, 4), (5, 6)])
    Out: [(5, 6), (3, 4), (1, 2)]
    """
    return l_ind_(l, range(len(l) - 1, -1, -1))


def dgt_(n):
    """
    ... digits of the int part of a number ...

    Examples
    --------
    >>> dgt_(143151)
    Out: 6
    >>> dgt_(1.234e5)
    Out: 6
    """
    return int(np.floor(np.log10(n)) + 1)


def prg_(i, n=None):
    """
    ... string indicating progress status ...

    Examples
    --------
    >>> prg_(89)
    Out: '#90/--'
    >>> prg_(89, n=999)
    Out: '#090/999'
    """
    ss = '#{:0' + r'{:d}'.format(dgt_(n)) + r'd}/{:d}' if n else '#{:d}/--'
    return ss.format(i + 1, n)


def b2l_endian_(x):
    """
    ... return little endian copy ...
    """
    return x.astype(np.dtype(x.dtype.str.replace('>', '<')))


def l2b_endian_(x):
    """
    ... return big endian copy ...
    """
    return x.astype(np.dtype(x.dtype.str.replace('<', '>')))


def isGI_(x):
    """
    ... if Iterator ...

    Examples
    --------
    >>> isGI_(flt_([1, 2]))
    Out: True
    >>> isGI_(flt_l([1, 2]))
    Out: False
    """
    from typing import Iterator
    return isinstance(x, Iterator)


def is1dIter_(x, XI=(str, bytes)):
    """
    ... if 1d (no-nesting) Iterable ...

    Examples
    --------
    >>> is1dIter_('abc')
    Out: True
    >>> is1dIter_(['abc'])
    Out: True
    >>> is1dIter_(['abc', []])
    Out: False
    >>> is1dIter_(range(5))
    Out: True
    >>> is1dIter_([range(5)])
    Out: False
    >>> is1dIter_([1, 2, 'abc'])
    Out: True
    """
    o = isIter_(x, XI=None)
    if o and not isGI_(x):
        o = o and all([not isIter_(i, XI=XI) for i in x])
    else:
        warnings.warn("ignored for Iterator or Generator!")
    return o


def isIter_(
        x,
        xi=None,
        XI=(str, bytes),
        ):
    """
    ... if Iterable ...

    kwArgs
    ------
    xi: specifie the type(s) of elements
    XI: specifie the type(s) that x belongs not to

    Examples
    --------
    >>> isIter_([[1, 2], 'a'])
    Out: True
    >>> isIter_([[1, 2], 'a'], xi=list)
    Out: False
    >>> isIter_([[1, 2], 'a'], xi=(list, str))
    Out: True
    >>> isIter_(np.empty((2, 1)))
    Out: True
    >>> isIter_((np.empty((2, 1)), np.arange(3)), xi=np.ndarray)
    Out: True
    >>> isIter_((np.empty((2, 1)), np.arange(3)), xi=tuple)
    Out: False
    >>> isIter_('abc')
    Out: False
    >>> isIter_('abc', XI=None)
    Out: True
    """
    from typing import Iterable
    o = isinstance(x, Iterable)
    if XI:
        o &= not isinstance(x, XI)
    if o and xi is not None:
        if not isGI_(x):
            o = o and all([isinstance(i, xi) or i is None for i in x])
        else:
            warnings.warn("xi ignored for Iterator or Generator!")
    return o


def haversine_(x0, y0, x1, y1):
    """
    ... distance between geo points in radians ...
    """
    lat0 = np.radians(y0)
    lon0 = np.radians(x0)
    lat1 = np.radians(y1)
    lon1 = np.radians(x1)

    dlon = lon1 - lon0
    dlat = lat1 - lat0

    a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lat1)*np.sin(dlon/2)**2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compressLL_(LL):
    """
    ... compress 2D list ...

    Examples
    --------
    >>> x = [[   0,    1,    2, None,    4],
    ...      [   5,    6,    7, None,    9],
    ...      [None, None, None, None, None],
    ...      [  15,   16,   17, None,   19]]
    >>> compressLL_(x)
    Out:
    ([[ 0,  1,  2,  4],
      [ 5,  6,  7,  9],
      [15, 16, 17, 19]],
     array([0, 1, 2, 4]),
     array([0, 1, 3]))
    """
    TF = np.asarray([[i is None for i in L] for L in LL])
    TFx = np.where(~np.all(TF, axis=0))[0]
    TFy = np.where(~np.all(TF, axis=1))[0]
    LL_ = l_ind_([l_ind_(L, TFx) for L in LL], TFy)
    return (LL_, TFx, TFy)


def consecutive_(
        x1d,
        func_,
        nn_=3,
        ffunc_=np.max,
        efunc_=lambda x: len(x[1:]),
        ):
    """
    ... consecutive functions ...

    Args
    ----
       x1d: 1d array (time series)
     func_: a function return booleans

    kwArgs
    ------
       nn_: least number of consecutive Trues to be taken into account
    ffunc_: method for deriving the final score (MAX default)
    efunc_: method for deriving score of each consecutive event (LENGTH
            default)

    Returnes
    --------
    final score for consecutive events
    """
    ts = np.split(np.concatenate(([0], x1d)),
                  np.concatenate(([1], np.where(func_(x1d))[0] + 1)))
    ts = [efunc_(its) for its in ts if len(its) > nn_]
    return ffunc_(ts) if ts else 0.


def consecutiveN_(
        x1d,
        func_,
        args=(),
        kargs={},
        ):
    """
    ... consecutive numbers ...

    Args
    ----
      x1d: 1d array (time series)
    func_: a function return booleans

    kwArgs
    ------
     args: args passed to func_
    kargs: kwargs passed to func_

    Examples
    --------
    >>> x = np.asarray([1, 0,
    ...                 1, 1, 0,
    ...                 1, 1, 1, 0,
    ...                 1, 1, 1, 1, 0,
    ...                 1, 1, 1, 1, 1, 0])
    >>> consecutiveN_(x, lambda y: y==1)
    Out:
    array([1, 0, 2, 2, 0, 3, 3, 3, 0, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 0])
    """
    def _f(x):
        m = np.sum(x)
        xx = np.empty(x.size, dtype=int)
        xx[x] = m
        xx[~x] = 0
        return xx
    ts = func_(x1d, *args, **kargs)
    return np.hstack(list(map(_f, np.split(ts, np.where(~ts)[0]))))


def _sz(xnd, axis=None):
    if axis is None:
        return xnd.size
    elif isinstance(axis, int):
        return xnd.shape[rpt_(axis, xnd.ndim)]
    elif isIter_(axis, xi=int):
        ind = pd.unique(rpt_(axis, xnd.ndim))
        return np.prod(np.asarray(xnd.shape)[ind])
    else:
        raise Exception(f"I don't understand axis={axis!r}")


def shp_drop_(
        shp,
        axis=None,
        replace=None,
        ):
    """
    ... drop dims specified (and replace if desired) ...
    """
    if axis is not None:
        axis = sorted(rpt_((axis,) if not isIter_(axis) else axis, len(shp)))
        if replace is None:
            return tuple(ii for i, ii in enumerate(shp) if i not in axis)
        else:
            tmp = (replace if i == axis[0] else ii for i, ii in enumerate(shp)
                   if i not in axis[1:])
            return tuple(flt_(tmp))
    else:
        return shp


def flt_ndim_(xnd, dim0, ndim):
    """
    ... flatten number of consecutive dims ...

    Examples
    --------
    >>> x = np.arange(24).reshape(2, 3, 4)
    >>> flt_ndim_(x, 1, 2)
    Out:
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
    >>> flt_ndim_(x, 0, 2)
    Out:
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]])
    >>> flt_ndim_(x, 0, 3)
    Out:
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    """
    dim0 = rpt_(dim0, xnd.ndim)
    tmp_ = tuple(-1 if i == dim0 else ii for i, ii in enumerate(xnd.shape)
                 if i not in range(dim0 + 1, dim0 + ndim))
    return xnd.reshape(tmp_)


def aggr_func_(
        xnd,
        *V,
        axis=None,
        func_=np.ma.mean,
        uniqV=False,
        fArgs=(),
        **fkwArgs,
        ):
    """
    ... aggregate function over ndarray ...

    Args
    ----
      xnd: nd array
        V: corresponding to which the data are grouped

    kwArgs
    ------
       axis: along an specified axis otherwise whole flatened data
      func_: aggregation function (MEAN default)
      uniqV: output unique Vs
      fArgs: Args passed to func
    fkwArgs: kwArgs passed to func
    """
    #--------------------------------------------------------------------------0 checking input arguments
    if not isinstance(xnd, (tuple, list)):
        arrs=[xnd]
    else:
        arrs = xnd
        msg = "all nd arrays should have a same shape!"
        assert len(set(arr.shape for arr in arrs))==1, msg
    if axis is None:
        arrs = [arr.ravel() for arr in arrs]
        axis = -1
    arr0 = arrs[0]
    if len(V) == 1:
        lbl = np.asarray(V[0]).ravel()
    elif len(V) > 1:
        lbl = np.asarray(el_join_([np.asarray(i).ravel() for i in V]))
    else:
        raise Exception("at least one label array is required, "
                        "but none is provided!")
    if lbl.size != _sz(arr0, axis=axis):
        raise Exception("input arguments not matching!")
    uV = pd.unique(lbl)
    if isIter_(axis, xi=int):                                                  # working on multiple axes
        axis = np.unique(rpt_(axis, arr0.ndim))
        if not all(np.diff(axis) == 1):
            tmp = tuple(flt_((axis if i == axis[0] else i                      # transpoing to make working axes continuos
                              for i in range(arr0.ndim)
                              if i not in axis[1:])))
            arrs = [arr.transpose(tmp) for arr in arrs]
        arrs = [flt_ndim_(arr, axis[0], len(axis)) for arr in arrs]            # flatening working axes into single axis
        naxi = axis[0]
    else:
        naxi = axis
    #--------------------------------------------------------------------------1 running func
    o = []
    arr0 = arrs[0]
    for ii in uV:
        ind_r = ind_s_(arr0.ndim, naxi, lbl==ii)
        if len(arrs) == 1:
            o.append(func_(arr0[ind_r], *fArgs, axis=naxi, **fkwArgs))
        else:
            o.append(func_([arr[ind_r] for arr in arrs], *fArgs,
                           axis=naxi, **fkwArgs))
    return np.stack(o, axis=0)                                                 # return stacked func results
    #nshp = shp_drop_(arr.shape, axis=axis, replace=uV.size)                   # deprecated method
    #o = np.empty(nshp)
    #for i, ii in enumerate(uV):
    #    ind_l = ind_s_(len(nshp), naxi, i)
    #    ind_r = ind_s_(arr.ndim, naxi, lbl==ii)
    #    o[ind_l] = func_(arr[ind_r], axis=naxi)
    #return (o, uV) if uniqV else o


def el_join_(caL, jointer='.'):
    """
    ... element-wise join ...

    Examples
    --------
    >>> el_join_(([1, 2, 3], 'abc'))
    Out: ['1.a', '2.b', '3.c']
    """
    return [jointer.join(iter_str_(i)) for i in zip(*caL)]


def windds2uv_(winds, windd):
    """
    ... u v from wind speed and direction ...
    """
    tmp = np.deg2rad(windd)
    return (- np.sin(tmp) * winds,
            - np.cos(tmp) * winds)


def pcorr_(data, rowvar=True):
    """
    ... partial correlation (matrix) ...
    """
    data = np.asarray(data)
    assert data.ndim == 2
    p = data.shape[0] if rowvar else data.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)

    corr = np.corrcoef(data, rowvar=rowvar)
    corr_inv = np.linalg.inv(corr)

    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            pcorr_ij = (
                -corr_inv[i, j] / (np.sqrt(corr_inv[i, i] * corr_inv[j, j]))
                )
            P_corr[i, j] = pcorr_ij
            P_corr[j, i] = pcorr_ij

    return P_corr


def pcorr_xyz(x, y, z):
    """
    ... partial correlation (x, y, z (controled var) ) ...
    """
    assert x.shape == y.shape == z.shape
    _corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    rxy = _corr(x, y)
    rxz = _corr(x, z)
    ryz = _corr(y, z)
    return (rxy - rxz * ryz) / (np.sqrt((1 - rxz**2) * (1 - ryz**2)))


def timerMain_(func, nm=None, lgg=None):
    """
    ... decorator to add timer for a function ...
    """
    import time
    from functools import wraps
    _nm = nm if nm is not None else func.__name__
    if lgg:
        from logging import basicConfig
        basicConfig(filename=lgg, filemode='w')
        _p = False
    else:
        _p = True
    @wraps(func)
    def func_(*args, **kwargs):
        start_time = time.time()
        ll_(' {:_^45}'.format('start of program'), _p=_p)
        if _nm:
            ll_(" {!r:_^45}".format(_nm), _p=_p)
        ll_(time.strftime(" %a, %d %b %Y %H:%M:%S %z", time.localtime()),
            _p=_p)
        ll_(' ', _p=_p)
        o = func(*args, **kwargs)
        ll_(' ', _p=_p)
        ll_(' {:_^45}'.format('end of program'), _p=_p)
        ll_(' {:_^45}'.format('TOTAL'), _p=_p)
        ll_(' ' + rTime_(time.time() - start_time), _p=_p)
        ll_(time.strftime(" %a, %d %b %Y %H:%M:%S %z", time.localtime()),
            _p=_p)
        return o
    return func_


def sub_shp_(shp, n, dims=None, _r=True, amp=1.5, N=None):
    """
    ... subgroup a shape ...

    Args
    ----
     shp: shape, array-like with dtype of int
       n: (minimum) number of groups

    kwArgs
    ------
    dims: dim# corresponding to shp, default range(shp)
      _r: give priority to high-end dims
     amp: constrain the maximum increasing rate of n during the 3rd try
       N: keeping as None as it is used during the recursively call

    Returns
    -------
    list of (dim#, points along dim#)

    Examples
    --------
    >>> sub_shp_((7, 8, 9), 6)
    Out: [(2, 3), (1, 2)]
    >>> sub_shp_((7, 888, 999), 6)
    Out: [(1, 148)]
    >>> sub_shp_((6, 3), 9)
    Out: [(1, 3), (0, 3)]
    >>> sub_shp_((6, 3), 7)
    Out: [(1, 3), (0, 3)]
    >>> sub_shp_((6, 3), 15)
    Out: [(1, 3), (0, 6)]
    >>> sub_shp_((12, 18), 6)
    Out: [(1, 3)]
    >>> sub_shp_((12, 18), 6, _r=False)
    Out: [(0, 2)]
    """
    N = n if N is None else N
    if dims is None:
        dims = tuple(range(len(shp)))
    smsg = "totally {} groups; each sub group has a size of {}."
    _smsg = "totally {} groups; each sub group has a maximum size of {}."
    emsg = "shp and dims do not match!"
    assert len(shp) == len(dims) == len(uniqL_(dims)), emsg
    _d, _s = ((tuple(reversed(dims)), tuple(reversed(shp))) if _r else
              (dims, shp))
    ############################################################################1st try group along one dimension
    for i, ii in zip(_d, _s):
        if ii%n == 0:
            print(smsg.format(n, np.prod(shp)//n))
            return [(i, ii//n)]
    ############################################################################2nd try group along two or more dimensions
    if np.prod(shp)%n == 0:
        o = []
        _n = n
        for i, ii in zip(_d, _s):
            nn = np.gcd(ii, _n)
            if nn > 1:
                o.append((i, nn))
                _n = _n//nn
                if _n == 1:
                    print(smsg.format(n, np.prod(shp)//n))
                    return o
    ############################################################################3rd try recursively increase n considering amp
    for i in range(
        n,
        (np.prod(shp) if amp is None else
         min(np.prod(shp), int(np.floor(N * amp) - 1)))
        ):
        #print(min(np.prod(shp), int(np.floor(N * amp) - 1)))
        return sub_shp_(shp, i+1, dims=dims, _r=_r, amp=amp, N=N)
    ############################################################################4th try, now OK with unequally grouped but work only along one dimension
    _step = tuple(i//N for i in shp)
    if all(i for i in _step):
        _d, _s, _tmp = list(zip(
            *((i, ii, ii/iii)
              for i, ii, iii in zip(dims, shp, _step) if iii > 0)
            ))
        if _s:
            _tmp0 = min(map(np.ceil, _tmp))                                    #search least group numbers
            _tmp2 = list(map(lambda x: _tmp0 - x if x < _tmp0 else 1, _tmp))   #calculate equality between sub groups along each available dimension
            ind = np.argmin(_tmp2)                                             #select the dimension considering least group numbers and greatest equality
            print(_smsg.format(
                int(_tmp0),
                np.prod([ii for i, ii in zip(dims, shp)
                         if i!=_d[ind]])*(_s[ind]//N)
                ))
            return [(_d[ind], _s[ind]//N)]
    ############################################################################5th try recursively increase n with amp=None
    for i in range(
        N,
        (np.prod(shp) if amp is None else
         min(np.prod(shp), int(np.floor(N * amp) - 1)))
        ):
        return sub_shp_(shp, i+1, dims=dims, _r=_r, amp=None, N=N)


def sqzUnit_(s):
    import re
    _isStarStyle = '**' in s
    def _toKD(x):
        D = re.findall(r'(?<=[a-zA-Z])((\*\*)?-?\d+)', x)
        if D:
            if re.findall('(-?\d+\.?\d?[eE]-?\d+)', x):
                return (None, x)
            else:
                K=x.replace(D[0][0], '')
            return (K, int(D[0][0].replace('**', '')))
        elif re.findall(r'^[a-zA-Z]+', x):
            return (x, 1)
        else:
            return (None, x)
    def _frKD(x):
        K, D = x
        if K:
            if D == 1:
                return K
            else:
                return f"{K}**{D:g}" if _isStarStyle else f"{K}{D:g}"
        else:
            return D
    S = []
    tmp = []
    _s = [_toKD(i) for i in s.split(' ')]
    for i, ii in _s:
        if i and i not in tmp:
            DD = [jj for j, jj in _s if j == i]
            if len(DD) > 1:
                tmp.append(i)
            DDD = sum(DD)
            if DDD != 0:
                S.append((i, DDD))
        if i is None:
            S.append((i, ii))
    S.sort(key=lambda x: x[1] if x[0] else 1e20, reverse=True)
    #S = valueEqFront_(S, lambda x: x[0] is None)
    return ' '.join([_frKD(i) for i in S])


def half_grid_(x, side='i', axis=-1, loa=None, rb=360):
    """
    ... points between grids ...

    Args
    ----
       x: nd array

    kwArgs
    ------
    side: extraploate for
          0, 'i', 'inner': off;
         -1, 'l',  'left': left;
          1, 'r', 'right': right;
          2, 'b',  'both': both
    axis: along which axis
     loa:
         None: values of x as ordinary ones; default
           lo: longitude
           la: latitude
      rb: right bound for longitude values (default 360)

    Examples
    --------
    >>> x = np.arange(6).reshape(2, -1)
    >>> half_grid_(x)
    Out:
    array([[0.5, 1.5],
           [3.5, 4.5]])
    >>> half_grid_(x, side='l')
    Out:
    array([[-0.5,  0.5,  1.5],
           [ 2.5,  3.5,  4.5]])
    >>> half_grid_(x, side='r')
    Out:
    array([[0.5, 1.5, 2.5],
           [3.5, 4.5, 5.5]])
    >>> half_grid_(x, side='b')
    Out:
    array([[-0.5,  0.5,  1.5,  2.5],
           [ 2.5,  3.5,  4.5,  5.5]])
    >>> half_grid_(half_grid_(x, side='i'), side='i', axis=0)
    Out: array([[2., 3.]])
    >>> half_grid_(half_grid_(x, side='b'), side='b', axis=0)
    Out:
    array([[-2., -1.,  0.,  1.],
           [ 1.,  2.,  3.,  4.],
           [ 4.,  5.,  6.,  7.]])
    >>> half_grid_(uu.half_grid_(x, side='b'), side='b', axis=0, loa='lo')
    Out:
    array([[358., 359.,   0.,   1.],
           [  1.,   2.,   3.,   4.],
           [  4.,   5.,   6.,   7.]])
    """
    dx = np.diff(x, axis=axis)
    if loa == 'lo':
        lb = rb - 360
        dx = rpt_(dx, 180, -180)
    tmp = extract_(x, axis, np.s_[:-1]) + dx * .5
    if side in (0, 'i', 'inner'):
        o = tmp
    elif side in (-1, 'l', 'left'):
        o = np.concatenate((extract_(x, axis, np.s_[:1]) -
                            extract_(dx, axis, np.s_[:1]) * .5,
                            tmp),
                           axis=axis)
    elif side in (1, 'r', 'right'):
        o = np.concatenate((tmp,
                            extract_(x, axis, np.s_[-1:]) +
                            extract_(dx, axis, np.s_[-1:]) * .5),
                           axis=axis)
    elif side in (2, 'b', 'both'):
        o = np.concatenate((extract_(x, axis, np.s_[:1]) -
                            extract_(dx, axis, np.s_[:1]) * .5,
                            tmp,
                            extract_(x, axis, np.s_[-1:]) +
                            extract_(dx, axis, np.s_[-1:]) * .5),
                           axis=axis)
    else:
        raise ValueError("unknow value of side!")
    if loa == 'lo':
        o = rpt_(o, rb, lb)
    if loa == 'la':
        o = np.where(o > 90, 90, o)
        o = np.where(o < -90, -90, o)
    return o


def edotm_(yyyy, mm):
    """
    ... return end day of this month ...

    Args
    ----
    yyyy: year
      mm: month

    Examples
    --------
    >>> edotm_(1988, 2)
    Out: datetime.datetime(1988, 2, 29, 0, 0)
    """
    from datetime import datetime, timedelta
    return datetime(yyyy + mm//12, mm%12 + 1, 1) - timedelta(days=1)


def date_mv_mon_(date, dmm):
    """
    ... date with increament of dmm months ...

    Args
    ----
    date: datetime object
     dmm: increament (months); should be int

    Examples
    --------
    >>> date_mv_mon_(datetime.datetime(1999, 1, 1), 18)
    Out: datetime.datetime(2000, 7, 1, 0, 0)
    >>> date_mv_mon_(datetime.datetime(1999, 1, 31), -2)
    UserWarning: day relpaced by end of this month.
    Out: datetime.datetime(1998, 11, 30, 0, 0)
    """
    if not isinstance(dmm, int):
        msg = f"'dmm' should be int!"
        raise ValueError(msg)
    day = date.day
    date = date.replace(day=1)
    if 1 <= date.month + dmm <= 12:
        o = date.replace(month=date.month+dmm)
    else:
        o = date.replace(
            year=date.year + (date.month + dmm - 1) // 12,
            month=rpt_(date.month + dmm, 13, 1)
            )
    xday = edotm_(o.year, o.month).day
    if xday < day:
        msg = 'day relpaced by end of this month.'
        warnings.warn(msg)
        return o.replace(day=xday)
    else:
        return o.replace(day=day)


def date_mv_yr_(date, dyy):
    """
    ... date with increament of dmm years ...

    Args
    ----
    date: datetime object
     dyy: increament (years); should be int

    Examples
    --------
    >>> date_mv_yr_(datetime.datetime(1999, 1, 1), 1)
    Out: datetime.datetime(2000, 1, 1, 0, 0)
    >>> date_mv_yr_(datetime.datetime(1999, 1, 31), -2)
    Out: datetime.datetime(1997, 1, 31, 0, 0)
    """
    return date.replace(year=date.year + dyy)
    


def iterDT_(datestr, delta='day'):
    """
    ... get datetime array from a brief string ...

    Args
    ----
    datestr: a date string with format of YYYY | YYYYMM | YYYYMMDD
             | YYYYMMDD:hh | YYYYMMDD:hhmm | YYYYMMDD:hhmmss, can also
             be two dates joined by a dash indicating START and END

    kwArgs
    ------
      delta: one of the following: second, minute, hour, day (default),
             week, month, year, can also add a integer as prefix

    Examples
    --------
    >>> iterDT_('2000', 'day')
    Out:
    array([datetime.datetime(2000, 1, 1, 0, 0),
           datetime.datetime(2000, 1, 2, 0, 0),
           datetime.datetime(2000, 1, 3, 0, 0), ...
           datetime.datetime(2000, 12, 31, 0, 0)], dtype=object)
    >>> iterDT_('2000', '3month')
    Out:
    array([datetime.datetime(2000, 1, 1, 0, 0),
           datetime.datetime(2000, 4, 1, 0, 0),
           datetime.datetime(2000, 7, 1, 0, 0),
           datetime.datetime(2000, 10, 1, 0, 0)], dtype=object)
    >>> iterDT_('20000201:000000-20000201:235959', '30minute')
    Out:
    array([datetime.datetime(2000, 2, 1, 0, 0),
           datetime.datetime(2000, 2, 1, 0, 30),
           datetime.datetime(2000, 2, 1, 1, 0),
           datetime.datetime(2000, 2, 1, 1, 30), ...
           datetime.datetime(2000, 2, 1, 23, 0),
           datetime.datetime(2000, 2, 1, 23, 30)], dtype=object)
    """
    import re
    from datetime import datetime, timedelta
    def _get_ymdhms(s):
        if ((len(s) < 4) or
            (':' in s and s.index(":") != 8) or
            (':' not in s and len(s) > 8)):
            raise Exception("unknown date string!")
        ss = ((0, 4), (4, 6), (6, 8), (9, 11), (11, 13), (13, 15))
        def _int(sss):
            if s[slice(*sss)]:
                return int(s[slice(*sss)])
        return tuple(_int(_s) for _s in ss if _int(_s))

    def _guess_max(*datetuple):
        x = [9999, 12, 31, 23, 59, 59]
        for i, v in enumerate(datetuple):
            x[i] = min(x[i], v)
        x[2] = min(edotm_(*x[:2]).day, x[2])
        return datetime(*x)

    def _guess_min(*datetuple):
        x = [0, 1, 1, 0, 0, 0]
        for i, v in enumerate(datetuple):
            x[i] = max(x[i], v)
        return datetime(*x)

    if '-' not in datestr:
        ymd = _get_ymdhms(datestr)
        date0 = _guess_min(*ymd)
        date1 = _guess_max(*ymd)
    else:
        date_ = datestr.split('-')
        if len(date_[0]) != len(date_[1]) or date_[0] >= date_[1]:
            emsg = "I don't know what to do with the specipfied DATE!"
            raise Exception(emsg)
        else:
            ymd0 = _get_ymdhms(date_[0])
            ymd1 = _get_ymdhms(date_[1])
            date0 = _guess_min(*ymd0)
            date1 = _guess_max(*ymd1)

    o = []
    while (date0 <= date1):
        emsg = "I don't know the specifiled delta {!r}".format(delta)
        o.append(date0)
        if delta == 'year':
            date0 = date_mv_yr_(date0, 1)
        elif delta == 'month':
            date0 = date_mv_mon_(date0, 1)
        elif delta in ('second', 'minute', 'hour', 'day', 'week',):
            incr = timedelta(**{delta+'s': 1})
            date0 += incr
        elif re.match('(\d+)(\w+)', delta):
            _n, _delta = re.findall('(\d+)(\w+)', delta)[0]
            if _delta in ('second', 'minute', 'hour', 'day', 'week',):
                incr = timedelta(**{_delta+'s': int(_n)})
                date0 += incr
            elif _delta == 'year':
                date0 = date_mv_yr_(date0, int(_n))
            elif _delta == 'month':
                date0 = date_mv_mon_(date0, int(_n))
            else:
                raise Exception(emsg)
        else:
            raise Exception(emsg)
    return np.asarray(o)


def doy2date_(*args):
    """
    ... day of year to date ...

    Args
    ----
    yyyydoy: str: 'YYYYdoy' | int: YYYYdoy | two int: YYYY, doy
    """
    from datetime import datetime, timedelta
    msg = "passed args should only be single str or one or two int"
    if len(args) == 2:
        yyyy, doy = args
    elif len(args) == 1:
        arg = args[0]
        if isinstance(arg, int):
            yyyy, doy = arg//1000, arg%1000
        elif isinstance(arg, str):
            yyyy, doy = int(arg[:4]), int(arg[4:])
        else:
            raise ValueError(msg)
    else:
        raise ValueError(msg)
    dt0 = datetime(yyyy - 1, 12, 31)
    delta = timedelta(days=1)
    return dt0 + doy * delta


def loa2d_(lo, la, isYX=True):
    """
    ... meshgrid (& transpose) longitue/latitude if necessary ...

    Args
    ----
      lo: longitude
      la: latitude

    kwArgs
    ------
    isYX: if 'x' axis is after 'y' axis
    """
    if lo.ndim != 2:
        if isYX:
            x, y = np.meshgrid(lo, la)
        else:
            y, x = np.meshgrid(la, lo)
    else:
        if isYX:
            x, y = lo, la
        else:
            x, y = lo.T, la.T
    return (x, y)


def in_loalim_(lo, la, shp, axXY=None, lolim=None, lalim=None, isYX=True):
    """
    ... grids if inside a rectangle specified by longitude/latitude limits ...

    Args
    ----
       lo: longitude
       la: latitude
      shp: shape of grided data

    kwArgs
    ------
     axXY: axes for 'x' and 'y'
    lolim: longitude limits
    lalim: latitude limits
     isYX: if 'x' axis is after 'y' axis
    """
    axXY = tuple(range(len(shp))[-2:]) if axXY is None else axXY
    lo2d, la2d = loa2d_(lo, la, isYX)
    booL = []
    if lolim is not None:
        booL.append(ind_inRange_(lo2d, *lolim, r_=360))
    if lalim is not None:
        booL.append(ind_inRange_(la2d, *lalim))
    if booL:
        return robust_bc2_(np.logical_and.reduce(booL), shp, axXY)
    else:
        return np.full(shape=shp, fill_value=True)


def in_polygons_(poly, points, **kwArgs):
    """
    ... if points within polygon(s) ...

    Args
    ----
      poly: mpl.path.Path
    points: series of (x, y)
    """
    if not isIter_(poly):
        ind = poly.contains_points(points, **kwArgs)
    elif len(poly) < 2:
        ind = poly[0].contains_points(points, **kwArgs)
    else:
        inds = [i.contains_points(points, **kwArgs) for i in poly]
        ind = np.logical_or.reduce(inds)
    return ind


def tryattr_(x, attr, ifno=None):
    """
    ... safely get attr ...
    """
    return getattr(x, attr) if hasattr(x, attr) else ifno


def ax_any_(x, axis=-1, _where=False):
    """
    ... any upon axes other than the given one ...
    """
    _x = x.swapaxes(axis, -1)
    _n = _x.shape[-1]
    o = np.any(_x.reshape(-1, _n), axis=0)
    return np.where(o) if _where else o


def bA2ind_(x, fancy=False, _con=True):
    """
    ... transform boolean array to indices ...
    """
    def _f(ax):
        ind1ax = ax_any_(x, axis=ax)
        if np.all(ind1ax):
            return slice(None)
        else:
            ii = np.where(ind1ax)[0]
            if _con:
                return slice(ii.min(), ii.max()+1)
            else:
                return ii 
    o = []
    for i in range(x.ndim):
        o.append(_f(i))
    return iind_(tuple(o)) if fancy else tuple(o)


def aw_loa_bnds_(lob, lab, rEARTH=6367470):
    """
    ... spherical segment areas ...

    Args
    ----
     lob: longitude bounds (radians)
     lab: latitude bounds (radians)

    kwArgs
    ------
    rEARTH: radius of the earth (currently assumed spherical)
    """
    if (lob.shape[-1] != 2 or lab.shape[-1] != 2):                             # ensure pairs of bounds
        raise ValueError("Bounds must be [...,2] array")

    radius_sqr = rEARTH**2                                                     # fill in a new array of areas
    lob_64 = lob.astype(np.float64)
    lab_64 = lab.astype(np.float64)
    lob0, lob1 = np.moveaxis(lob_64, -1, 0)
    lab0, lab1 = np.moveaxis(lab_64, -1, 0)
    ylen = np.sin(lab1) - np.sin(lab0)
    xlen = lob1 - lob0

    if lob.ndim == lab.ndim == 2:
        areas = radius_sqr * np.outer(ylen, xlen)
    elif lob.shape == lab.shape:
        areas = radius_sqr * ylen * xlen

    return np.abs(areas)                                                       # we use abs because backwards bounds (min > max) give negative areas.


def mosaicAB01_(*args):
    """
    ... get mosaic listoflist for subplots ...
    """
    if len(args) == 2:
        m, n = args
    elif len(args) == 1:
        m, n = args[0]
    else:
        raise ValueError("check if as 'm, n' or '(m, n)'!")
    import string
    y = np.array([i for i in string.ascii_letters[:m]])
    x = np.arange(n)
    xx, yy = np.meshgrid(x, y)
    o = np.array(el_join_((yy.ravel(), xx.ravel()), ''))
    return o.reshape((m, n)).tolist()


def upd_(d, **kwargs):
    """
    ... update a dictionary ...
    """
    if kwargs:
        d.update(kwargs)


def date_s_(s):
    """
    ... string to datetime.date ...
    """
    from datetime import date
    if '-' in s:
        return np.datetime64(s).tolist()
    elif len(s)==8:
        return np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:]}").tolist()
    else:
        emsg = f"{s!r} not recognised as a date string"
        raise ValueError(emsg)


def day_last_(s):
    """
    ... last day: string ...
    """
    from datetime import timedelta
    return (date_s_(s) - timedelta(days=1)).strftime(r"%Y-%m-%d")


def day_next_(s):
    """
    ... next day: string ...
    """
    from datetime import timedelta
    return (date_s_(s) + timedelta(days=1)).strftime(r"%Y-%m-%d")


def pV2pearsonr_(N, alpha):
    """
    ... pearson r value corresponding to p value ...
    """
    from scipy.stats import beta
    dist = beta(N/2 - 1, N/2 - 1, -1, 2)
    return dist.isf(alpha)


def pearsonr2pV_(N, r):
    """
    ... p value corresponding to pearson r value ...
    """
    from scipy.stats import beta
    dist = beta(N/2 - 1, N/2 - 1, -1, 2)
    return dist.sf(np.abs(r))
