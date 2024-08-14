"""
>--#########################################################################--<
>-------------------------functions operating on cube-------------------------<
>--#########################################################################--<
* alng_axis_            : apply along axis
* ax_fn_mp_             : apply along axis mp
* concat_cube_          : robust cube concatenator
* corr_cube_            : modified version of iris.analysis.stats.pearsonr
* cubesv_               : save cube to nc with dim_t unlimitted
* curl_cube             : curl of (ucube, vcube)
* cut_as_cube           : cut into the domain of another cube
* div_cube              : divergence of (ucube, vcube)
* doy_f_cube            : f for each doy
* en_iqr_               : ensemble interquartile range
* en_max_               : ensemble max
* en_mean_              : ensemble mean
* en_min_               : ensemble min
* en_mxn_               : ensemble spread
* en_mm_cubeL_          : make ensemble cube for multimodels
* en_rip_               : ensemble (rxixpx) cube
* extract_month_cube    : extraction cube of month
* extract_period_cube   : extraction cube within [y0, y1]
* extract_season_cube   : extraction cube of season
* extract_win_cube      : extraction within a window (daily)
* f_allD_cube           : iris analysis func over all dims of cube(L) (each)
* getGridAL_cube        : grid_land_area
* getGridA_cube         : grid_area from file or calc with basic assumption
* get_gwl_y0_           : first year of 30-year window of global warming level
* guessBnds_cube        : bounds of dims points
* initAnnualCube_       : initiate annual cube
* intersection_         : cube intersection with lon/lat range
* isMyIter_             : Iterable with items as cube/ndarray
* kde_cube              : kernal distribution estimation over all cube data
* lccs_m2km_            : change LambfortComfort unit
* maskLS_cube           : mask land or sea area
* maskNaN_cube          : mask nan points in a cube
* maskPOLY_cube         : mask area in respect of polygons
* max_cube              : max of cube(L) data (each)
* max_cube_             : max of cube(L) data (all)
* merge_cube_           : robust cube merger
* min_cube              : min of cube(L) data (each)
* min_cube_             : min of cube(L) data (all)
* minmax_cube           : minmax of cube(L) data (each)
* minmax_cube_          : minmax of cube(L) data (all)
* myAuxTime_            : create single value time auxcoord
* myDimTime_            : create time dimcoord
* nTslice_cube          : slices along a no-time axis
* nearest_point_cube    : extract 1 point cube
* nine_points_cube      : extract nine points cube centered at a given point
* pSTAT_cube            : period statistic (month, season, year)
* pcorr_cube            : partial correlation (cube_0, cube_1, cube_cntr)
* pp_cube               : pth and 100-pth of cube(L) data (each)
* pst_                  : post-rename/reunits cube(L)
* purefy_cubeL_         : prepare for concat or merge
* repair_cs_            : bug fix for save cube to nc
* repair_lccs_          : bug fix for save cube to nc (LambertConformal)
* replace_coord_        : replace coord data according to coord.name()
* rgCount_cube          : regional count
* rgCount_poly_cube     : regional count over in polygon only
* rgF_cube              : regional function
* rgF_poly_cube         : regional function over in polygon only
* rgMean_cube           : regional mean
* rgMean_poly_cube      : regional mean over in polygon only
* ri_cube               : recurrence period for each grid space
* rm_sc_cube            : remove scalar coords
* rm_t_aux_cube         : remove time-related aux_coords
* rm_yr_doy_cube        : opposite action of yr_doy_cube
* seasonyr_cube         : season_year auxcoord
* slice_back_           : slice back to parent (1D)
* smth_cube             : smoothing cube over XY axes
* unique_yrs_of_cube    : unique year points of cube
* y0y1_of_cube          : starting and ending year of cube
* yr_doy_cube           : year and day-of-year auxcoord
...

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 06.09.2019
Date last modified: 11.11.2020
"""

import iris
from iris.cube import Cube as _Cube, CubeList as _CubeList
import iris.coord_categorisation as _ica
from iris.coords import AuxCoord as _iAuxC, DimCoord as _iDimC
from iris.util import equalise_attributes

import numpy as np
import warnings
import re
from datetime import datetime, timedelta

from .ffff import *
from .uuuu import *


__all__ = [
        'alng_axis_',
        'ax_fn_mp_',
        'concat_cube_',
        'corr_cube_',
        'cubesv_',
        'curl_cube',
        'cut_as_cube',
        'div_cube',
        'doy_f_cube',
        'en_iqr_',
        'en_max_',
        'en_mean_',
        'en_min_',
        'en_mxn_',
        'en_mm_cubeL_',
        'en_rip_',
        'extract_month_cube',
        'extract_period_cube',
        'extract_season_cube',
        'extract_win_cube',
        'f_allD_cube',
        'getGridAL_cube',
        'getGridA_cube',
        'get_gwl_y0_',
        'guessBnds_cube',
        'initAnnualCube_',
        'intersection_',
        'isMyIter_',
        'kde_cube',
        'lccs_m2km_',
        'maskLS_cube',
        'maskNaN_cube',
        'maskPOLY_cube',
        'max_cube',
        'max_cube_',
        'merge_cube_',
        'min_cube',
        'min_cube_',
        'minmax_cube',
        'minmax_cube_',
        'myAuxTime_',
        'myDimTime_',
        'nTslice_cube',
        'nearest_point_cube',
        'nine_points_cube',
        'pSTAT_cube',
        'pcorr_cube',
        'pp_cube',
        'pst_',
        'purefy_cubeL_',
        'repair_cs_',
        'repair_lccs_',
        'replace_coord_',
        'rgCount_cube',
        'rgCount_poly_cube',
        'rgF_cube',
        'rgF_poly_cube',
        'rgMean_cube',
        'rgMean_poly_cube',
        'ri_cube',
        'rm_sc_cube',
        'rm_t_aux_cube',
        'rm_yr_doy_cube',
        'seasonyr_cube',
        'slice_back_',
        'smth_cube',
        'unique_yrs_of_cube',
        'y0y1_of_cube',
        'yr_doy_cube',
        ]


def slice_back_(cnd, c1d, ii, axis):
    """
    ... put slice back to its parent CUBE/ARRAY ...

    Parsed arguments:
         cnd: parent CUBE/ARRAY that has multiple dimensions
         c1d: CUBE/ARRAY slice
          ii: slice # of c1d in iteration
        axis: axis of cnd to place c1d
    Returns:
        revised cnd
    """

    if isinstance(c1d, _Cube):
        c1d = c1d.data
    if not isinstance(c1d, np.ndarray):
        c1d = np.asarray(c1d)
    if ((isinstance(cnd, _Cube) or np.ma.isMaskedArray(cnd)) and
        not np.ma.isMaskedArray(c1d)):
        c1d = np.ma.masked_array(c1d, np.isnan(c1d))
    emsg = "slice NOT matched its parent along axis({!r})."
    if axis is None:
        if c1d.size != 1:
            raise Exception(emsg.format(axis))
    else:
        axis = rpt_(axis, cnd.ndim)
        axis = sorted(axis) if isIter_(axis, xi=int) else axis
        if not np.all(np.asarray(cnd.shape)[axis] == np.asarray(c1d.shape)):
            raise Exception(emsg.format(axis))
    ind = ind_shape_i_(cnd.shape, ii, axis)
    if isinstance(cnd, _Cube):
        cnd.data[ind] = c1d
    elif np.ma.isMaskedArray(cnd):
        cnd[ind] = c1d
    else:
        cnd[ind_shape_i_(cnd.shape, ii, axis)] = c1d


def isMyIter_(x):
    """
    ... Iterable with items as CUBE/ARRAY ...
    """
    return isIter_(x,
                   xi=(np.ndarray, _Cube),
                   XI=(np.ndarray, _Cube, str, bytes))


def pst_(
        c,
        name=None,
        units=None,
        var_name=None,
        attrU=None,
        ):
    """
    ... post-rename/reunits c(L) ...
    """
    if isinstance(c, _Cube):
        if name:
            c.rename(name)
        if units:
            c.units = units
        if var_name:
            c.var_name = var_name
        if attrU:
            c.attributes.update(attrU)
    elif isMyIter_(c):
        for i in c:
            pst_(i, name=name, units=units, var_name=var_name, attrU=attrU)


def nTslice_cube(c, n):
    """
    ... slices along a no-time axis ...

    Args:
        c: CUBE
        n: maximum size of a slice
    """
    nd = c.ndim
    tc = _dimcT(c)
    ax_nT = [i for i in range(nd) if i not in c.coord_dims(tc)]
    shp = tuple(c.shape[i] for i in ax_nT)
    if np.prod(shp) < n:
        return [c]
    else:
        ss = sub_shp_(shp, n, dims=ax_nT)
        oo = [c]
        for ax, step in ss:
            oo = nli_(
                    [[_extract_byAxes(o, ax, np.s_[i:(i + step)])
                      for i in range(0, c.shape[ax], step)] for o in oo]
                    )
        return _CubeList(oo)


def unique_yrs_of_cube(
        c,
        coord='year',
        mmm=None,
        ):
    """
    ... unique year points of c(L) ...
    """
    if isinstance(c, _Cube):
        _c = c.copy()
        _coords = [i.name() for i in _c.coords()]
        if coord not in _coords:
            if 'season' in coord:
                if mmm:
                    seasonyr_cube(_c, mmm, name=coord)
                else:
                    emsg = "'mmm' must not be None for adding coord {!r}!"
                    raise ValueError(emsg.format(coord))
            else:
                _ica.add_year(_c, _dimcT(_c), name=coord)
        return np.unique(_c.coord(coord).points)
    elif isIter_(c, xi=_Cube):
        return [unique_yrs_of_cube(i) for i in c]
    else:
        raise TypeError("unknown type for the first argument!")


def y0y1_of_cube(
        c,
        coord='year',
        mmm=None
        ):
    if isinstance(c, _Cube):
        _c = c.copy()
        _coords = [i.name() for i in _c.coords()]
        if coord not in _coords:
            if 'season' in coord:
                if mmm:
                    seasonyr_cube(_c, mmm, name=coord)
                else:
                    emsg = "'mmm' must not be None for adding coord {!r}!"
                    raise ValueError(emsg.format(coord))
            else:
                _ica.add_year(_c, _dimcT(_c), name=coord)
        elif mmm and 'season' in coord:
            _c.remove_coord(coord)
            seasonyr_cube(_c, mmm, name=coord)
        return list(_c.coord(coord).points[[0, -1]])
    elif isIter_(c, xi=_Cube):
        yy = np.array([y0y1_of_cube(i) for i in c])
        return [np.max(yy[:, 0]), np.min(yy[:, 1])]
    else:
        raise TypeError("unknown type for the first argument!")


def extract_period_cube(
        c,
        y0,
        y1,
        yy=False,
        coord='year',
        mmm=None,
        ):
    """
    ... extract CUBE within the period from year y0 to year y1 ...

    Args:
            c: CUBE with coord('time')
           y0: starting year
           y1: end year
    kwArgs:
           yy: True forcing output strictly starting/end at y0/y1
        coord: name of coord with year points for extraction
          mmm: defined season using 1st letters of composing months, should not
               be None when seasonyr_cube is called

    useful info:
        >>> help(seasonyr_cube)
    """
    _c = c.copy()
    _coords = [i.name() for i in _c.coords()]
    if coord not in _coords:
        if 'season' in coord:
            if mmm:
                seasonyr_cube(_c, mmm, name=coord)
            else:
                emsg = "'mmm' must not be None for adding coord {!r}!"
                raise ValueError(emsg.format(coord))
        else:
            _ica.add_year(_c, _dimcT(_c), name=coord)
    elif mmm and 'season' in coord:
        _c.remove_coord(coord)
        seasonyr_cube(_c, mmm, name=coord)
    cstrD = {coord: lambda x: y0 <= x <= y1}
    cstr = iris.Constraint(**cstrD)
    o = _c.extract(cstr)
    if not (
            yy and (
                y0y1_of_cube(o) != [y0, y1] or
                not np.all(np.diff(unique_yrs_of_cube(o)) == 1)
                )
            ):
        return o


def extract_win_cube(c, d, r=15):
    """
    ... extract CUBE within a window centered at doy==d ...

    Args:
        c: CUBE containing coord('time')
        d: center of the window
    kwArgs:
        r: half window width (defaul 15)
    """
    _c = c.copy()
    try:
        _ica.add_day_of_year(_c, _dimcT(_c), name='doy')
    except ValueError:
        pass
    x1, x2 = rpt_(d - r, 365), rpt_(d + r, 365)
    cstr = iris.Constraint(
            doy=lambda x: x1 <= x <= x2 if x1 < x2 else not (x2 < x < x1)
            )
    o = _c.extract(cstr)
    return o


def extract_season_cube(c, mmm, valid_season=True):
    """
    ... extract a CUBE of season named with continuous-months' 1st letters ...

    Args:
                   c: CUBE/CubeList containing coord('time')
                 mmm: continuous-months' 1st letters (for example, 'djf')
    kwArgs:
        valid_season: if True and season mmm is crossing years, the 1st & end
                      year will be excluded
    """
    if isinstance(c, _Cube):
        if 'season' in (i.name() for i in c.coords()):
            o = c.extract(iris.Constraint(season=mmm))
        else:
            _c = c.copy() # to avoid changing metadata of original CUBE
            try:
                _ica.add_season_membership(_c, _dimcT(_c), mmm,
                                           name=mmm)
            except ValueError:
                _c.remove_coord(mmm)
                _ica.add_season_membership(_c, _dimcT(_c), mmm,
                                           name=mmm)
            _c.coord(mmm).points = _c.coord(mmm).points.astype(np.int32)
            o = _c.extract(iris.Constraint(**{mmm: True}))
        if valid_season and not ismono_(mmmN_(mmm)):
            y0, y1 = y0y1_of_cube(o, ccsn='seasonyr', mmm=mmm)
            o = extract_period_cube(
                    o, y0 + 1, y1 - 1,
                    ccsn='seasonyr',
                    mmm=mmm,
                    )
        return o
    elif isMyIter_(c):
        cl = [extract_season_cube(i, mmm, valid_season=valid_season)
              for i in c]
        return _CubeList(cl)
    else:
        raise TypeError("unknown type for the first argument!")


def extract_month_cube(c, Mmm):
    """
    ... extract CUBE of month Mmm ...
    """
    _c = c.copy()
    try:
        _ica.add_month(_c, _dimcT(_c), name='month')
    except ValueError:
        pass
    return _c.extract(iris.Constraint(month=Mmm[:3].capitalize()))


def f_allD_cube(c, rg=None, func='MAX', **fK_):
    """
    ... iris analysis func(unc) over all dims of c(L) (each) ...
    """
    #warnings.filterwarnings("ignore", category=UserWarning)
    if isinstance(c, _Cube):
        if rg:
            c = intersection_(c, **rg)
        _f = getattr(iris.analysis, func.upper())
        return c.collapsed(c.dim_coords, _f, **fK_).data
    elif isMyIter_(c):
        return np.asarray([f_allD_cube(i, rg=rg, func=func, **fK_) for i in c])
    else:
        return np.nan


def min_cube(c, rg=None):
    return f_allD_cube(c, rg=rg, func='MIN')


def max_cube(c, rg=None):
    return f_allD_cube(c, rg=rg)


def minmax_cube(c, rg=None):
    return np.asarray([min_cube(c, rg=rg), max_cube(c, rg=rg)])


def pp_cube(c, rg=None, p=10):
    return np.asarray(
        [f_allD_cube(c, rg=rg, func='PERCENTILE', percent=p),
         f_allD_cube(c, rg=rg, func='PERCENTILE', percent=100 - p)]
        )


def min_cube_(c, rg=None):
    return np.nanmin(min_cube(c, rg=rg))


def max_cube_(c, rg=None):
    return np.nanmax(max_cube(c, rg=rg))


def minmax_cube_(c, rg=None, p=None):
    if p:
        mms = pp_cube(c, rg=rg, p=p)
    else:
        mms = minmax_cube(c, rg=rg)
    return (np.nanmin(mms), np.nanmax(mms))


def seasonyr_cube(c, mmm, name='seasonyr'):
    """
    ... add season_year auxcoords to CUBE especially regarding ...
    ... specified season                                       ...
    """
    if isinstance(c, _Cube):
        if isinstance(mmm, str):
            seasons = (mmm, rest_mns_(mmm))
        elif (isinstance(mmm, (list, tuple)) and
              sorted(''.join(mmm)) == sorted('djfmamjjason')):
            seasons = mmm
        else:
            raise Exception(f"unknown seasons {mmm!r}!")
        try:
            _ica.add_season_year(c, _dimcT(c),
                                 name=name,
                                 seasons=seasons)
        except ValueError:
            c.remove_coord(name)
            _ica.add_season_year(c, _dimcT(c),
                                 name=name,
                                 seasons=seasons)
    elif isIter_(c, xi=(_Cube, _CubeList, tuple, list)):
        for i in c:
            seasonyr_cube(i, mmm, name=name)


def yr_doy_cube(c):
    """
    ... add year, day-of-year AuxCoords to CUBE ...
    """
    if isinstance(c, _Cube):
        try:
            _ica.add_year(c, _dimcT(c), name='year')
        except ValueError:
            pass
        else:
            c.coord('year').attributes = {}
        try:
            _ica.add_day_of_year(c, _dimcT(c), name='doy')
        except ValueError:
            pass
        else:
            c.coord('doy').attributes = {}
    elif isIter_(c, xi=(_Cube, _CubeList, tuple, list)):
        for i in c:
            yr_doy_cube(i)


def rm_yr_doy_cube(c):
    """
    ... remove year, day-of-year auxcoords from CUBE ...
    """
    try:
        c.remove_coord('year')
    except iris.exceptions.CoordinateNotFoundError:
        pass
    try:
        c.remove_coord('doy')
    except iris.exceptions.CoordinateNotFoundError:
        pass


def rm_t_aux_cube(c, keep=None):
    """
    ... remove time-related auxcoords from CUBE or a list of cubes ...
    """
    tauxL = ['year', 'month', 'season', 'day', 'doy', 'hour', 'yr']
    def _isTCoord(x):
        return any((i in x.name() for i in tauxL)) or isSeason_(x.name())
    if isinstance(c, _Cube):
        for i in c.aux_coords:
            if keep is None:
                isTaux = _isTCoord(i)
            elif isIter_(keep):
                isTaux = _isTCoord(i) and i.name() not in keep
            else:
                isTaux = _isTCoord(i) and i.name() != keep
            if isTaux:
                c.remove_coord(i)
    elif isMyIter_(c):
        for i in c:
            rm_t_aux_cube(i)
    else:
        raise TypeError('Input should be CUBE or iterable CUBEs!')


def rm_sc_cube(c):
    if isinstance(c, _Cube):
        for i in c.coords():
            if len(c.coord_dims(i)) == 0:
                c.remove_coord(i)
    elif isMyIter_(c):
        for i in c:
            rm_sc_cube(i)
    else:
        raise TypeError('Input should be CUBE or Iterable CUBEs!')


def guessBnds_cube(c):
    """
    ... guess bounds of dims of CUBE if not exist ...
    """
    for i in c.dim_coords:
        try:
            i.guess_bounds()
        except ValueError:
            pass


def cut_as_cube(c0, c1):
    """
    ... cut CUBE1 with the domain of CUBE0 ...
    """
    xc1, yc1 = _dimcXY(c1)
    xc0, yc0 = _dimcXY(c0)
    xn, yn = xc1.name(), yc1.name()
    xe = np.min(np.abs(np.diff(xc1.points))) / 2
    ye = np.min(np.abs(np.diff(yc1.points))) / 2
    x0, x1 = np.min(xc0.points), np.max(xc0.points)
    y0, y1 = np.min(yc0.points), np.max(yc0.points)
    return _extract_byAxes(
            c1,
            xn,
            ind_inRange_(xc1.points, x0 - xe, x1 + xe, side=0),
            yn,
            ind_inRange_(yc1.points, y0 - ye, y1 + ye, side=0)
            )


def maskLS_cube(c, sftlf, LorS='S', thr=0):
    """
    ... mask sea/land area ...

    Parsed arguments:
            c: DATA CUBE to be masked
        sftlf: land area fraction; at least covering entire CUBE
         LorS: 'land' or 'sea' to be masked (default 'sea')
          thr: sftlf value <= thr as not land area (default 0)
    """
    LList = ['L', 'LAND']
    SList = ['S', 'O', 'W', 'SEA', 'OCEAN', 'WATER']
    if LorS.upper() not in (LList + SList):
        raise ValueError("Variable 'LorS' not interpretable!")
    sftlf_ = cut_as_cube(c, sftlf)
    ma_0 = sftlf_.data <= thr
    if LorS.upper() in LList:
        ma_0 = ~ma_0
    ma_ = np.broadcast_to(ma_0, c.shape)
    c = iris.util.mask_cube(c, ma_)


def getGridA_cube(c, areacella=None):
    """
    ... get grid_area of CUBE ...
    """
    if areacella:
        ga_ = iris.util.squeeze(areacella)
        if ga_.ndim != 2:
            return getGridA_cube(c)
        ga = cut_as_cube(c, ga_).data
        try:
            ga = robust_bc2_(ga, c.shape, _axXY(c))
            return ga
        except:
            return getGridA_cube(c)
    else:
        try:
            guessBnds_cube(c)
            ga = _area_weights(c)
        except:
            ga = None
        return ga


def getGridAL_cube(c, sftlf=None, areacella=None):
    """
    ... return grid_land_area of CUBE if sftlf provided ...
    ... else return grid_area of CUBE                   ...
    """
    ga = getGridA_cube(c, areacella)
    if sftlf is not None:
        sf_sqz = iris.util.squeeze(sftlf)
        if sf_sqz.ndim != 2:
            raise Exception('NOT 2D area-c!')
        sf = cut_as_cube(c, sf_sqz).data
        sf = robust_bc2_(sf, c.shape, _axXY(c))
        if ga is None:
            return np.ones(c.shape) * sf / 100
        else:
            return ga * sf / 100.
    else:
        return ga


def rgF_cube(c, func, rg=None, **funcD):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if rg:
        ind = _ind_loalim(c, **rg)
        tmp = iris.util.mask_cube(c.copy(), ~ind)
    else:
        tmp = c
    xc, yc = _dimcXY(tmp)
    return tmp.collapsed([xc, yc], func, **funcD)


def rgF_poly_cube(c, poly, func, inpolyKA={}, **funcD):
    #warnings.filterwarnings("ignore", category=UserWarning)
    ind = _ind_poly(c, poly, **inpolyKA)
    tmp = iris.util.mask_cube(c.copy(), ~ind)
    xc, yc = _dimcXY(tmp)
    return tmp.collapsed([xc, yc], func, **funcD)


def rgCount_cube(c, sftlf=None, areacella=None, rg=None, func=None):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if func is None:
        func = lambda values: values > 0
        warnings.warn("'func' not provided; count values greater than 0.")
    xyd = _axXY(c)
    ga0 = getGridA_cube(c, areacella)
    ga0 = np.ones(c.shape) if ga0 is None else ga0
    ga = getGridAL_cube(c, sftlf, areacella)
    ga = np.ones(c.shape) if ga is None else ga
    if rg:
        ind = _ind_loalim(c, **rg)
        ga0 = ga0 * ind
        ga = ga * ind
    umsk = ~c.data.mask if (np.ma.isMaskedArray(c.data) and
                            np.ma.is_masked(c.data)) else 1
    sum0 = np.sum(ga0 * umsk, axis=xyd)
    if np.any(sum0 == 0):
        raise Exception("empty slice encountered.")
    data = np.sum(func(c.data) * ga, axis=xyd) * 100 / sum0
    xc, yc = _dimcXY(c)
    tmp = c.collapsed([xc, yc], iris.analysis.MEAN)
    return tmp.copy(data)


def rgCount_poly_cube(c, poly, sftlf=None, areacella=None, func=None,
                      **kwArgs):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if func is None:
        func = lambda values: values > 0
        warnings.warn("'func' not provided; count values greater than 0.")
    xyd = _axXY(c)
    ga0 = getGridA_cube(c, areacella)
    ga0 = np.ones(c.shape) if ga0 is None else ga0
    ga = getGridAL_cube(c, sftlf, areacella)
    ga = np.ones(c.shape) if ga is None else ga
    ind = _ind_poly(c, poly, **kwArgs)
    ga0 = ga0 * ind
    ga = ga * ind
    sum0 = np.sum(ga0, axis=xyd)
    if np.any(sum0 == 0):
        raise Exception("empty slice encountered.")
    data = np.sum(func(c.data) * ga, axis=xyd) * 100 / sum0
    xc, yc = _dimcXY(c)
    tmp = c.collapsed([xc, yc], iris.analysis.MEAN)
    return tmp.copy(data)


def rgMean_cube(c, sftlf=None, areacella=None, rg=None):
    """
    ... regional mean; try weighted if available ...
    """
    #warnings.filterwarnings("ignore", category=UserWarning)
    ga = getGridAL_cube(c, sftlf, areacella)
    if rg:
        ind = _ind_loalim(c, **rg)
        if ga is None:
            ga = ind * np.ones(ind.shape)
        else:
            ga = ga * ind
    xc, yc = _dimcXY(c)
    if ga is None:
        return c.collapsed([xc, yc], iris.analysis.MEAN)
    else:
        return c.collapsed([xc, yc], iris.analysis.MEAN, weights=ga)


def get_gwl_y0_(c, gwl, pref=[1861, 1890]):
    """
    ... first year of 30-year window of global warming level ...

    Args:
          c: CUBE of global surface temperature
        gwl: warming level compared to reference period pref
    """
    _c = pSTAT_cube(c if c.ndim == 1 else rgMean_cube(c), 'year')
    tref = extract_period_cube(_c, *pref)
    tref = tref.collapsed(_dimcT(tref), iris.analysis.MEAN).data

    def _G_tR(G, tR):
        if not isIter_(G):
            ind = np.where(rMEAN1d_(_c.data, 30) >= G + tR)[0][0]
            return _c.coord('year').points[ind]
        else:
            return [_G_tR(i, tR) for i in G]

    if _c.ndim == 1:
        return _G_tR(gwl, tref)
    else:
        o = np.empty(tref.shape + np.array(gwl).shape)
        ax = _c.coord_dims('year')[0]
        for i in range(nSlice_(_c.shape, ax)):
            ind = ind_shape_i_(_c.shape, i, ax)
            ind_ = ind_shape_i_(tref.shape, i, axis=None)
            ind__ = ind_shape_i_(o.shape, i,
                                 axis=-1 if np.array(gwl).shape else None)
            o[ind__] = np.array(_G_tR(gwl, tref[ind]))
        return o


def maskNaN_cube(c):
    ind = np.isnan(c.data)
    iris.util.mask_cube(c, ind, in_place=True)


def maskPOLY_cube(c, poly, masked_out=True, **kwArgs):
    ind = _ind_poly(c, poly, **kwArgs)
    ind = ~ind if masked_out else ind
    iris.util.mask_cube(c, ind, in_place=True)


def rgMean_poly_cube(c, poly, sftlf=None, areacella=None, **kwArgs):
    #warnings.filterwarnings("ignore", category=UserWarning)
    ga = getGridAL_cube(c, sftlf, areacella)
    ind = _ind_poly(c, poly, **kwArgs)
    xc, yc = _dimcXY(c)
    if ga is None:
        ga = ind * np.ones(ind.shape)
    else:
        ga = ga * ind
    return c.collapsed([xc, yc], iris.analysis.MEAN, weights=ga)


def _rm_extra_coords_cubeL(cL):
    l0 = [[ii.name() for ii in i.aux_coords] for i in cL]
    l1 = ouniqL_(flt_l(l0))
    l2 = [i for i in l1 if sum(np.array(flt_l(l0))==i) < len(cL)]
    if len(l2) != 0:
        for i, ii in zip(cL, l0):
            for iii in l2:
                if iii in ii:
                    i.remove_coord(iii)


def _get_xycoords(c):
    """
    ... get xy (spatial) coords ...
    """
    xycoord_names = ['lon', 'x_coord', 'x-coord', 'x coord',
                     'lat', 'y_coord', 'y-coord', 'y coord']
    xycoords = [coord for coord in c.coords()
                if any([i in coord.name() for i in xycoord_names])]
    return xycoords


def _unify_1coord_points(cL, coord_name, **close_kwArgs):
    epochs = {}
    emsg = "COORD {!r} can't be unified!".format(coord_name)
    emsg_ = "Bounds of COORD {!r} can't be unified!".format(coord_name)
    for c in cL:
        cc = c.coord(coord_name)
        d0 = epochs.setdefault('points', cc.points)
        if np.allclose(cc.points, d0, **close_kwArgs):
            cc.points = d0
        else:
            raise Exception(emsg)
        if cc.has_bounds():
            d1 = epochs.setdefault('bounds', cc.bounds)
            if np.allclose(cc.bounds, d1, **close_kwArgs):
                cc.bounds = d1
            else:
                raise Exception(emsg_)


def _unify_xycoord_points(cL, **close_kwArgs):
    ll_('cccc: _unify_xycoord_points() called')
    if len(cL) > 1:
        coord_names = [i.name() for i in _get_xycoords(cL[0])]
        for coord_name in coord_names:
            _unify_1coord_points(cL, coord_name, **close_kwArgs)


def _unify_1coord_attrs(cL, coord_name):
    attrs = ['long_name', 'var_name', 'attributes', 'coord_system']
    epochs = {}
    for c in cL:
        cc = c.coord(coord_name)
        tp = cc.points.dtype
        tp_ = np.dtype(tp.str.replace('>', '<')) if '>' in tp.str else tp
        tmp = epochs.setdefault('dtype', tp_)
        if tp != tmp:
            cc.points = cc.points.astype(tmp)
        #if hasattr(cc, 'bounds') and cc.has_bounds():
        #    tmp_b = epochs.setdefault('bounds', cc.bounds)
        try:
            cc.guess_bounds()
        except ValueError:
            pass
        if cc.has_bounds() and cc.bounds.dtype != tmp:
            cc.bounds = cc.bounds.astype(tmp)
        for i in attrs:
            tmp = epochs.setdefault(i, cc.__getattribute__(i))
            cc.__setattr__(i, tmp)
    if 'bounds' in epochs:
        for c in cL:
            cc = c.coord(coord_name)


def _unify_coord_attrs(cL, coord_names=None):
    ll_('cccc: _unify_coord_attrs() called')
    if len(cL) > 1:
        coord_names = coord_names if coord_names else\
                      [i.name() for i in cL[0].coords()]
        for coord_name in coord_names:
            _unify_1coord_attrs(cL, coord_name)


def _unify_time_units(cL):
    CLD0 = 'proleptic_gregorian'
    CLD = 'gregorian'
    _cT = lambda x: x.coord(axis='T')
    clds = [_cT(c).units.calendar for c in cL]
    if len(ouniqL_(clds)) > 1:
        for c in cL:
            ctu = _cT(c).units
            if ctu.calendar == CLD0:
                _cT(c).units = cUnit(ctu.origin, CLD)
    iris.util.unify_time_units(cL)


def _unify_dtype(cL, first=False):
    ll_('cccc: _unify_dtype() called')
    tps = [c.dtype for c in cL]
    if first:
        tp = tps[0]
    else:
        utps = np.unique(tps)
        tpi = [np.sum(np.asarray(tps) == i) for i in utps]
        tp = utps[np.argmax(tpi)]
    for c in cL:
        if c.dtype != tp:
            c.data = c.data.astype(tp)


def _unify_cellmethods(cL, first=True):
    ll_('cccc: _unify_cellmethods() called')
    cms = [c.cell_methods for c in cL]
    if first:
        cm = cms[0]
    else:
        ucms = np.unique(cms)
        cmi = [np.sum(np.asarray(cms) == i) for i in ucms]
        cm = utps[np.argmax(cmi)]
    for c in cL:
        if c.cell_methods != cm:
            c.cell_methods = cm


def purefy_cubeL_(cL):
    """
    ... helpful when merge or concatenate CubeList ...
    """
    _rm_extra_coords_cubeL(cL)
    equalise_attributes(cL)
    _unify_time_units(cL)


def _collect_errCC(x):
    tmp = re.findall(r'(?<=\!\= ).+$', x)
    return tmp[0].split(', ') if tmp else tmp


def concat_cube_(cL, **close_kwArgs):
    """
    ... robust cube concatenator ...
    """
    purefy_cubeL_(cL)
    try:
        o = cL.concatenate_cube()
    except iris.exceptions.ConcatenateError as ce_:
        if any(['Data types' in i for i in ce_.args[0]]):
            _unify_dtype(cL)
        if any(['Cube metadata' in i for i in ce_.args[0]]):
            _unify_cellmethods(cL)
        if any(['coordinates metadata differ' in i for i in ce_.args[0]]):
            tmp = flt_l([_collect_errCC(i) for i in ce_.args[0]
                         if 'coordinates metadata differ' in i])
            if 'height' in tmp:
                ll_("cccc: set COORD 'height' points to those of first CUBE")
                _unify_1coord_points(cL, 'height', atol=10)
                tmp.remove('height')
            if len(tmp) > 0:
                _unify_coord_attrs(cL, tmp)
        try:
            o = cL.concatenate_cube()
        except iris.exceptions.ConcatenateError as ce_:
            if any(['Expected only a single cube' in i for i in ce_.args[0]]):
                _unify_xycoord_points(cL, **close_kwArgs)
            o = cL.concatenate_cube()
    return o


def merge_cube_(cL, **close_kwArgs):
    purefy_cubeL_(cL)
    try:
        o = cL.merge_cube()
    except iris.exceptions.MergeError as ce_:
        if any(['Data types' in i for i in ce_.args[0]]):
            _unify_dtype(cL)
        if any(['Cube metadata' in i for i in ce_.args[0]]):
            _unify_cellmethods(cL)
        if any(['coordinates metadata differ' in i for i in ce_.args[0]]):
            tmp = flt_l([_collect_errCC(i) for i in ce_.args[0]
                         if 'coordinates metadata differ' in i])
            if 'height' in tmp:
                ll_("cccc: set COORD 'height' points to those of cL[0]")
                _unify_1coord_points(cL, 'height', atol=10)
                tmp.remove('height')
            if len(tmp) > 0:
                _unify_coord_attrs(cL, tmp)
        try:
            o = cL.merge_cube()
        except iris.exceptions.MergeError as ce_:
            if any(['Expected only a single cube' in i for i in ce_.args[0]]):
                _unify_xycoord_points(cL, **close_kwArgs)
            o = cL.merge_cube()
    return o


def en_mxn_(c):
    """
    ... ensemble max of CUBE (along dimcoord 'realization') ...
    """
    if c.coord_dims('realization'):
        a = en_max_(c)
        b = en_min_(c)
        o = a - b
        o.rename(a.name())
        return o


def en_min_(c):
    """
    ... ensemble max of CUBE (along dimcoord 'realization') ...
    """
    if c.coord_dims('realization'):
        return c.collapsed('realization', iris.analysis.MIN)


def en_max_(c):
    """
    ... ensemble max of CUBE (along dimcoord 'realization') ...
    """
    if c.coord_dims('realization'):
        return c.collapsed('realization', iris.analysis.MAX)


def en_mean_(c, **kwArgs):
    """
    ... ensemble mean of CUBE (along dimcoord 'realization') ...
    """
    if c.coord_dims('realization'):
        return c.collapsed('realization', iris.analysis.MEAN, **kwArgs)


def en_iqr_(c):
    """
    ... ensemble interquartile range (IQR) of CUBE (along dimcoord
        'realization') ...
    """
    if c.coord_dims('realization'):
        a = c.collapsed('realization', iris.analysis.PERCENTILE, percent=75)
        b = c.collapsed('realization', iris.analysis.PERCENTILE, percent=25)
        o = a - b
        o.rename(a.name())
        return o


def kde_cube(c, **kde_opts):
    """
    ... kernal distribution estimate over all nomasked data ...
    """
    data = nanMask_(c.data).flatten()
    data = data[~np.isnan(data)]
    data = data.astype(np.float64)
    return kde_(data, **kde_opts)


def _rip(c):
    """
    ... get rxixpx from CUBE metadata ...
    """
    if 'parent_experiment_rip' in c.attributes:
        return c.attributes['parent_experiment_rip']
    elif 'driving_model_ensemble_member' in c.attributes:
        return c.attributes['driving_model_ensemble_member']
    else:
        return None


def en_rip_(cL):
    """
    ... ensemble CUBE over rxixpxs (along dimcoord 'realization') ...
    """
    for i, c in enumerate(cL):
        rip = _rip(c)
        rip = str(i) if rip is None else rip
        new_coord = _iAuxC(rip,
                           long_name='realization',
                           units='no_unit')
        c.add_aux_coord(new_coord)
        c.attributes = {}
    return cL.merge_cube()


def en_mm_cubeL_(cL, opt=0, cref=None):
    """
    ... make ensemble cube for multimodels ...

    kwArgs:
         opt:
             0: rgd_li_opt0_; try rgd_iris_ first then rgd_scipy_
             1: rgd_iris_
             2: rgd_scipy_
        cref: reference CUBE (default 1st CUBE in CubeList)

    useful info:
        >>> help(rgd_li_opt0_)
        >>> help(rgd_iris_)
        >>> help(rgd_scipy_)
    """
    from .rgd import rgd_scipy_, rgd_iris_, rgd_li_opt0_
    tmpD = {}
    cl = []
    for i, c in enumerate(cL):
        c.attributes = {}
        if cref is None:
            cref = tmpD.setdefault('ref', c.copy())
        else:
            cref.attributes = {}
        if opt == 0:
            tmp = rgd_li_opt0_(c, cref)
        elif opt == 1:
            tmp = rgd_iris_(c, cref)
        elif opt == 2:
            tmp = rgd_scipy_(c, cref)
        else:
            raise ValueError('opt should be one of (0, 1, 2)!')
        a0 = tmpD.setdefault('a0', tmp)
        a = a0.copy(tmp.data)
        a.add_aux_coord(_iAuxC(np.int32(i),
                               long_name='realization',
                               units='no_unit'))
        cl.append(a)
    return _CubeList(cl).merge_cube()


def _func(func, ak_):
    arr, o0, args, kwargs = ak_
    try:
        if o0.data.mask or o0.mask:
            return None
    except AttributeError:
        pass
    return func(*arr, *args, **kwargs)


def ax_fn_ray_(arr, ax, func, out, *args, npr=32, **kwargs):
    """
    ... apply func on arrs along ax and output out ...
    ... work with multiprocessing for speeding up  ...

    Args:
        arrs: NDARRAY or CUBE (may you check func input)
          ax: axis along which func is applied
        func: function work with 1D ARRAY or CUBE
         out: NDARRAY or CUBE for storing output

    kwArgs:
         npr: number of processes

    args, kwargs:
        along with the arrs to be passed to func
    """
    import ray
    import psutil
    nproc = min(psutil.cpu_count(logical=False), npr)
    ray.init(num_cpus=nproc)

    if isMyIter_(out):
        o0 = out[0]
    else:
        o0 = out
    if not isinstance(o0, _Cube):
        raise Exception("type of 'out' should be CUBE!")
    if not isinstance(arr, (tuple, list)):
        arr = (arr,)

    @ray.remote
    def f(i, arr):
        ind = ind_shape_i_(arr[0].shape, i, ax)
        try:
            if o0[ind][0].data.mask:
                return None
        except AttributeError:
            pass
        aaa = tuple([ii[ind] for ii in arr])
        return func(*aaa, *args, **kwargs)

    arr_id = ray.put(arr)
    tmp = [f.remote(i, arr_id) for i in range(nSlice_(arr[0].shape, ax))]
    XX = ray.get(tmp)

    _sb(XX, out, ax)


def ax_fn_mp_(arr, ax, func, out, *args, npr=32, **kwargs):
    """
    ... apply func on arrs along ax and output out ...
    ... work with multiprocessing for speeding up  ...

    Args:
        arrs: NDARRAY or CUBE (may you check func input)
          ax: axis along which func is applied
        func: function work with 1D ARRAY or CUBE
         out: NDARRAY or CUBE for storing output

    kwArgs:
         npr: number of processes

    args, kwargs:
        along with the arrs to be passed to func
    """
    import multiprocessing as mp
    nproc = min(mp.cpu_count(), npr)

    if isMyIter_(out):
        o0 = out[0]
    else:
        o0 = out
    if not isinstance(o0, (np.ndarray, _Cube)):
        raise Exception("type of 'out' should be NDARRAY or CUBE!")
    if not isinstance(arr, (tuple, list)):
        arr = (arr,)

    P = mp.Pool(nproc)
    def _i(i, sl_=np.s_[:]):
        return ind_shape_i_(arr[0].shape, i, ax, sl_)
    X = P.starmap_async(
            _func,
            [(func,
              (tuple(ii[_i(i)] for ii in arr),
               arr[0][_i(i, sl_=0)],
               args,
               kwargs)) for i in range(nSlice_(arr[0].shape, ax))])
    XX = X.get()
    P.close()

    _sb(XX, out, ax)


def _sb(XX, out, ax):
    for i, o in enumerate(XX):
        if o is not None:
            _isb(i, o, out, ax)


def _isb(i, o, out, ax):
    def _get_nax(x):
        if x.ndim == 0:
            nax = None
        else:
            ax_ = tuple(ax) if isIter_(ax) else (ax,)
            nax = tuple(np.arange(ax_[0], ax_[0] + x.ndim))
        return nax
    if isMyIter_(out):
        for j, k in zip(out, o):
            if not isinstance(k, np.ndarray):
                k = np.asarray(k)
            slice_back_(j, k, i, _get_nax(k))
    else:
        if not isinstance(o, np.ndarray):
            o = np.asarray(o)
        slice_back_(out, o, i, _get_nax(o))


def alng_axis_(arrs, ax, func, out, *args, **kwargs):
    """
    ... apply func on arrs along ax and output out ...

    Args:
        arrs: NDARRAY or CUBE (may you check func input)
          ax: axis along which func is applied
        func: function work with 1D ARRAY or CUBE
         out: NDARRAY or CUBE for storing output

    args, kwargs:
        along with the arrs to be passed to func
    """
    if isMyIter_(out):
        o0 = out[0]
    else:
        o0 = out
    if not isinstance(o0, (np.ndarray, _Cube)):
        raise Exception("type of 'out' should be NDARRAY or CUBE!")
    if not isinstance(arrs, (list, tuple)):
        arrs = (arrs,)
    for i in range(nSlice_(arrs[0].shape, ax)):
        ind = ind_shape_i_(arrs[0].shape, i, ax)
        try:
            if o0[ind][0].data.mask:
                continue
        except AttributeError:
            pass
        aaa = [xxx[ind] for xxx in arrs]
        tmp = func(*aaa, *args, **kwargs)
        _isb(i, tmp, out, ax)


def initAnnualCube_(
        c0, y0y1,
        name=None,
        units=None,
        var_name=None,
        long_name=None,
        attrU=None,
        mmm='j-d',
        ):
    """
    ... initiate annual cube ...

    Args:
               c0: CUBE source for initialization
             y0y1: [y0, y1] range of year
    kwArgs:
             name: name of cube
            units: units of variable
         var_name: name of variable
        long_name: long name of variable
            attrU: a dict for updating cube attribution
              mmm: season/month str specified for CUBE to be initialized
    """
    mmm = 'jfmamjjasond' if mmm == 'j-d' else mmm
    y0, y1 = y0y1
    ny = y1 - y0 + 1
    c = _extract_byAxes(c0, _axT(c0), np.s_[:ny])
    rm_t_aux_cube(c)

    def _mm01():
        if isMonth_(mmm):
            m0 = mnN_(mmm)
            m1 = rpt_(m0 + 1, 13, 1)
            y0_ = y0
            y0__ = y0 + 1 if m1 < m0 else y0
        else:
            tmp = mmmN_(mmm)
            m0, m1 = tmp[0], rpt_(tmp[-1] + 1, 13, 1)
            y0_ = y0 if ismono_(tmp) else y0 - 1
            y0__ = y0_ + 1 if m1 <= m0 else y0
        return (m0, m1, y0_, y0__)

    ##data and mask
    if isinstance(c.data, np.ma.MaskedArray):
        if ~np.ma.is_masked(c.data):
                c.data.data[:] = 0.
        #if c.data.mask.ndim == 0:
        #    if ~c.data.mask:
        #        c.data.data[:] = 0.
        else:
            c.data.data[~c.data.mask] = 0.
    else:
        c.data = np.zeros(c.shape)
    ##coord('time')
    ct = _dimcT(c)
    ct.units = TSC_(1900)
    m0, m1, y0_, y0__ = _mm01()
    y0_h = [datetime(i, m0, 1) for i in range(y0_, y0_ + ny)]
    y1_h = [datetime(i, m1, 1) for i in range(y0__, y0__ + ny)]
    tbnds = np.empty((ny, 2))
    tbnds[:, 0] = ct.units.date2num(y0_h)
    tbnds[:, 1] = ct.units.date2num(y1_h)
    tdata = np.mean(tbnds, axis=-1)
    ct.points = tdata
    ct.bounds = tbnds
    ##var_name ...
    if name:
        c.rename(name)
    if units:
        c.units = units
    if var_name:
        c.var_name = var_name
    if long_name:
        c.long_name = long_name
    if attrU:
        c.attributes.update(attrU)
    return c


def pSTAT_cube(
        c0,
        *freq,
        stat='MEAN',
        valid_season=True,
        with_year=True,
        list_out=False,
        **stat_opts):
    """
    ... period statistic ...

    Args:
                  c0: CUBE to be analyzed
                stat: getattr(iris.analysis, stat)
                freq: frequency for statistic
    kwArgs:
        valid_season: if True and season mmm is crossing years, the results for
                      1st & end year will be excluded
           with_year: if year points to be taken into account during aggreation
           stat_opts: options to be passed to stat
    """
    ef0 = "stat {!r} unreconnigsed!"
    ef1 = "freq {!r} unreconigsed!"
    stat = stat.upper()
    if stat not in (
        'MEAN', 'MAX', 'MIN', 'MEDIAN', 'SUM', 'PERCENTILE', 'PROPORTION',
        'STD_DEV', 'RMS', 'VARIANCE', 'HMEAN', 'COUNT', 'PEAK'
        ):
        raise Exception(ef0.format(stat))

    s4 = ('djf', 'mam', 'jja', 'son')
    ax_t = _dimT(c0)

    d_y = dict(year=('year',),
               season=('season', 'seasonyr'),
               month=('month', 'year'),
               day=('doy', 'year'),
               hour=('hour', 'year'))

    d_ = dict(year=('year',),
              season=('season',),
              month=('month',),
              day=('doy',),
              hour=('hour',))

    dd = dict(hour=(_ica.add_hour, (ax_t,), dict(name='hour')),
              day=(_ica.add_day_of_year, (ax_t,), dict(name='doy')),
              month=(_ica.add_month, (ax_t,), dict(name='month')),
              year=(_ica.add_year, (ax_t,), dict(name='year')),
              season=(_ica.add_season, (ax_t,),
                      dict(name='season', seasons=s4)),
              seasonyr=(seasonyr_cube, (s4,), dict(name='seasonyr')))

    def _x(f0):
        d = d_y if with_year else d_
        if f0 in d.keys():
            return d[f0]
        elif isSeason_(f0):
            return (f0, 'seasonyr') if with_year else (f0,)
        elif isMonth_(f0):
            return d['month']
        else:
            raise Exception(ef1.format(f0))

    def _xx(x):
        if isinstance(x, str):
            if x in d_.keys():
                return (dd.copy(), None)
            elif isSeason_(x):
                tmp = {
                        x: (_ica.add_season_membership,
                            (ax_t, x),
                            dict(name=x)),
                        'seasonyr': (seasonyr_cube,
                                     (x,),
                                     dict(name='seasonyr'))
                        }
                dd_ = dd.copy()
                dd_.update(tmp)
                return (dd_, x)
            elif isMonth_(x):
                return (dd.copy(), x.capitalize())
            else:
                raise Exception(ef1.format(x))
        else:
            if all((f_ in d_.keys() for f_ in x)):
                return (dd.copy(), None)
            else:
                f_ = [f_ for f_ in x if f_ not in d_.keys()]
                if len(f_) == 1 and isSeason_(f_[0]):
                    tmp = {
                            f_[0]: (_ica.add_season_membership,
                                    (ax_t, f_[0]),
                                    dict(name=f_[0])),
                            'seasonyr': (seasonyr_cube,
                                         (f_[0],),
                                         dict(name='seasonyr'))
                            }
                    dd_ = dd.copy()
                    dd_.update(tmp)
                    return (dd_, f_[0])
                elif len(f_) == 1 and isMonth_(f_[0]):
                    return (dd.copy(), x.capitalize())
                else:
                    raise Exception(ef1.format('-'.join(x)))

    def _xxx(c, x):
        dd_, mmm = _xx(x)
        if isinstance(x, str):
            dff = _x(x)
        else:
            dff = uniqL_(flt_l([_x(i) for i in x]))
            if 'seasonyr' in dff:
                dff.remove('year')
        #dff = (dff, ) if isinstance(dff, str) else dff
        for i in dff:
            _f, fA, fK = dd_[i]
            try:
                _f(c, *fA, **fK)
            except ValueError:
                c.remove_coord(fK['name'])
                _f(c, *fA, **fK)
        if mmm:
            if isSeason_(mmm):
                c.coord(mmm).points = c.coord(mmm).points.astype(np.int32)
                cstr = iris.Constraint(**{mmm: True})
            elif isMonth_(mmm):
                cstr = iris.Constraint(**{'month': mmm})
            c = c.extract(cstr)
            tmp = c.aggregated_by(dff, getattr(iris.analysis, stat),
                                  **stat_opts)
            if (isSeason_(mmm) and not ismono_(mmmN_(mmm)) and valid_season and
                with_year):
                tmp = _extract_byAxes(tmp, ax_t, np.s_[1:-1])
        else:
            tmp = c.aggregated_by(dff, getattr(iris.analysis, stat),
                                  **stat_opts)
            if x == 'season' and valid_season:
                tmp = _extract_byAxes(tmp, ax_t, np.s_[1:-1])
        rm_t_aux_cube(tmp, keep=dff)
        return tmp

    freqs = ('year',) if len(freq) == 0 else freq
    o = ()
    for ff in [i.split('-') if '-' in i else i for i in freqs]:
        tmp = _xxx(c0.copy(), ff)
        o += (tmp,)
    return o[0] if len(o) == 1 and not list_out else o


def repair_cs_(c0):
    def _repair_cs_cube(c):
        cs = c.coord_system('CoordSystem')
        if cs is not None:
            for k in cs.__dict__.keys():
                if getattr(cs, k) is None:
                    setattr(cs, k, "")
        for coord in c.coords():
            if coord.coord_system is not None:
                coord.coord_system = cs
    if isinstance(c0, _Cube):
        _repair_cs_cube(c0)
    elif isMyIter_(c0):
        for i in c0:
            if isinstance(i, _Cube):
                _repair_cs_cube(i)


def _repair_lccs_cube(c, out=False):
    cs = c.coord_system('CoordSystem')
    o_ = 0
    if isinstance(cs, iris.coord_systems.LambertConformal):
        if (cs.false_easting is None or
            isinstance(cs.false_easting, np.ndarray)):
            cs.false_easting = ''
            o_ += 1
        if (cs.false_northing is None or
            isinstance(cs.false_northing, np.ndarray)):
            cs.false_northing = ''
            o_ += 1
        for coord in c.coords():
            if coord.coord_system is not None:
                coord.coord_system = cs
                coord.convert_units('m')
    if out:
        return o_


def repair_lccs_(c):
    if isinstance(c, _Cube):
        _repair_lccs_cube(c)
    elif isMyIter_(c):
        for i in c:
            _repair_lccs_cube(i)


def lccs_m2km_(c):
    if not isMyIter_(c):
        if isinstance(c, _Cube):
            cs = c.coord_system('CoordSystem')
            if isinstance(cs, iris.coord_systems.LambertConformal):
                for coord in c.coords():
                    if coord.coord_system is not None:
                        coord.convert_units('km')
    else:
        for i in c:
            lccs_m2km_(i)


def cubesv_(c, filename,
            netcdf_format='NETCDF4',
            local_keys=None,
            zlib=True,
            complevel=4,
            shuffle=True,
            fletcher32=False,
            contiguous=False,
            chunksizes=None,
            endian='native',
            least_significant_digit=None,
            packing=None,
            fill_value=None):
    """
    ... save CUBE to nc with dim_t unlimitted ...
    """
    if isinstance(c, _Cube):
        #repair_lccs_(c) # execute before if necessary
        udm = _dimcT(c)
        iris.save(c, filename,
                  netcdf_format=netcdf_format,
                  local_keys=local_keys,
                  zlib=zlib,
                  complevel=complevel,
                  shuffle=shuffle,
                  fletcher32=fletcher32,
                  contiguous=contiguous,
                  chunksizes=chunksizes,
                  endian=endian,
                  least_significant_digit=least_significant_digit,
                  packing=packing,
                  fill_value=fill_value,
                  unlimited_dimensions=udm)
    elif isMyIter_(c):
        for i, ii in enumerate(c):
            ext = ext_(filename)
            cubesv_(ii, filename.replace(ext, '_{}{}'.format(i, ext)))


def _ri1d(c1d, v):
    from skextremes.models.classic import GEV
    data = c1d.data
    data = data.compressed() if np.ma.isMaskedArray(data) else data
    if data.size:
        _gev = GEV(data)
        return _gev.return_periods(v)
    else:
        return np.nan


def ri_cube(c, v, nmin=10):
    ax_t = _axT(c)
    if ax_t is None or c.shape[ax_t] < nmin:
        emsg = "too few data for estimation!"
        raise Exception(emsg)
    o = _extract_byAxes(c, ax_t, 0)
    rm_sc_cube(o)
    pst_(o, 'recurrence interval', units='year')
    ax_fn_mp_(c, ax, _ri1d, o, v)
    return o


def nearest_point_cube(c, longitude, latitude):
    x, y = _loa_pnts_2d(c)
    d_ = ind_shape_i_(x.shape,
                      np.argmin(haversine_(longitude, latitude, x, y)),
                      axis=None)
    xyd = _axXY(c)
    ind = list(np.s_[:,] * c.ndim)
    for i, ii in zip(xyd, d_):
        ind[i] = ii
    return c[tuple(ind)]


def nine_points_cube(c, longitude, latitude):
    x, y = _loa_pnts_2d(c)
    d_ = ind_shape_i_(x.shape,
                      np.argmin(haversine_(longitude, latitude, x, y)),
                      axis=None)
    xyd = _axXY(c)
    ind_ = np.arange(-1, 2, dtype=np.int32)
    ind = list(np.s_[:,] * c.ndim)
    wmsg = ("Causious that center point may be given outside (or at the "
            "boundary of) the geo domain of the input CUBE!")
    for i, ii in zip(xyd, d_):
        if ii == c.shape[i] - 1:
            warnings.warn(wmsg)
            ii -= 1
        elif ii == 0:
            warnings.warn(wmsg)
            ii += 1
        ind[i] = ind_ + ii
    return c[tuple(ind)]


def replace_coord_(c, new_coord):
    """
    Replace the coordinate whose metadata matches the given coordinate.

    """
    old_coord = c.coord(new_coord.name())
    dims = c.coord_dims(old_coord)
    was_dimensioned = old_coord in c.dim_coords
    c._remove_coord(old_coord)
    if was_dimensioned and isinstance(new_coord, _iDimC):
        c.add_dim_coord(new_coord, dims[0])
    else:
        c.add_aux_coord(new_coord, dims)

    for factory in c.aux_factories:
        factory.update(old_coord, new_coord)


def doy_f_cube(c,
               func,
               fA_=(),
               fK_={},
               ws=None,
               mF=None,
               out=None,
               pp=False,
               ):
    """
    ... func(tion) for each doy ...

    kwArgs:
       fA_, fK_: Args, kwArgs along with CUBE data to be passed to func
             ws: window size
             mF: for replacing missing value
            out: CUBE for storing output (default derived from input CUBE)
             pp: print process status
    """

    ax_t = _axT(c)
    yr_doy_cube(c)
    doy_data = c.coord('doy').points

    doy_ = np.unique(doy_data)
    if len(doy_) < 360:
        raise Exception('doy less than 360!')
    doy = np.arange(1, 367, dtype=np.int32)

    if out is None:
        out = _extract_byAxes(c, ax_t, doy - 1)
        cT = _dimcT(out)
        #select 2000 as it is a leap year...
        cT.units = TSC_(1900)
        d0 = cT.units.date2num(datetime(2000, 1, 1))
        dimT = cT.copy(doy - 1 + d0)
        out.replace_coord(dimT)

    if pp:
        t0 = l__('0', _p=True)

    data_ = np.ma.filled(c.data, mF) if mF is not None else c.data
    if pp:
        ll_('releazing', t0=t0, _p=True)

    for i in doy:
        indw = ind_win_(doy_data, i, 15) if ws else np.isin(doy_data, i)
        ind = ind_s_(c.ndim, ax_t, indw)
        f_kArgs.update(dict(axis=ax_t, keepdims=True))
        tmp = func(data_[ind], *fA_, **fK_)
        out.data[ind_s_(out.ndim, _axT(out), doy == i)] = tmp
        if pp:
            ll_('{}'.format(i), t0=t0, _p=True)

    return out


def pcorr_cube(x, y, z, **cck):
    assert x.shape == y.shape == z.shape
    if 'corr_coords' not in cck:
        cck.update(dict(corr_coords=_dimT(x)))
    if 'common_mask' not in cck:
        cck.update(dict(common_mask=True))
    from iris.analysis.maths import apply_ufunc as _apply
    def _corr(cube_0, cube_1):
        return corr_cube_(
            cube_0, cube_1,
            alpha=None,
            **cck
            )
    rxy = _corr(x, y)
    rxz = _corr(x, z)
    ryz = _corr(y, z)
    covar = rxy - rxz * ryz
    denom = _apply(np.sqrt, (1 - rxz**2) * (1 - ryz**2), new_unit=covar.units)
    corr_cube = covar / denom
    corr_cube.rename("Pearson's partial r")
    return corr_cube


def corr_cube_(cube_a, cube_b,
               corr_coords=None,
               weights=None,
               mdtol=1.0,
               common_mask=False,
               alpha=None
               ):
    """
    Calculate the Pearson's r correlation coefficient over specified
    dimensions, modified from iris.analysis.stats.pearsonr.

    Args:

    * cube_a, cube_b (cubes):
        Cubes between which the correlation will be calculated.  The cubes
        should either be the same shape and have the same dimension coordinates
        or one CUBE should be broadcastable to the other.
    * corr_coords (str or list of str):
        The CUBE coordinate name(s) over which to calculate correlations. If no
        names are provided then correlation will be calculated over all common
        CUBE dimensions.
    * weights (numpy.ndarray, optional):
        Weights array of same shape as (the smaller of) cube_a and cube_b. Note
        that latitude/longitude area weights can be calculated using
        :func:`iris.analysis.cartography.area_weights`.
    * mdtol (float, optional):
        Tolerance of missing data. The missing data fraction is calculated
        based on the number of grid cells masked in both cube_a and cube_b. If
        this fraction exceed mdtol, the returned value in the corresponding
        cell is masked. mdtol=0 means no missing data is tolerated while
        mdtol=1 means the resulting element will be masked if and only if all
        contributing elements are masked in cube_a or cube_b. Defaults to 1.
    * common_mask (bool):
        If True, applies a common mask to cube_a and cube_b so only cells which
        are unmasked in both cubes contribute to the calculation. If False, the
        variance for each CUBE is calculated from all available cells. Defaults
    * alpha (float, optional):
        If specified, a critical coorelation value (p=alpha) will be given
        along with the output CUBE
    Returns:
        A CUBE of the correlation between the two input cubes along the
        specified dimensions, at each point in the remaining dimensions of the
        cubes.

        For example providing two time/altitude/latitude/longitude cubes and
        corr_coords of 'latitude' and 'longitude' will result in a
        time/altitude CUBE describing the latitude/longitude (i.e. pattern)
        correlation at each time/altitude point.

    Reference:
        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    This operation is non-lazy.

    """
    from iris.util import broadcast_to_shape
    from iris.analysis.maths import apply_ufunc as _apply

    # Assign larger cube to cube_1
    if cube_b.ndim > cube_a.ndim:
        cube_1 = cube_b
        cube_2 = cube_a
    else:
        cube_1 = cube_a
        cube_2 = cube_b

    smaller_shape = cube_2.shape

    dim_coords_1 = [coord.name() for coord in cube_1.dim_coords]
    dim_coords_2 = [coord.name() for coord in cube_2.dim_coords]
    common_dim_coords = list(set(dim_coords_1) & set(dim_coords_2))
    for i in common_dim_coords:
        cube_1.replace_coord(cube_2.coord(i))
    # If no coords passed then set to all common dimcoords of cubes.
    if corr_coords is None:
        corr_coords = common_dim_coords

    def _ones_like(cube):
        # Return a copy of cube with the same mask, but all data values set to 1.
        # The operation is non-lazy.
        # For safety we also discard any cell-measures and ancillary-variables, to
        # avoid cube arithmetic possibly objecting to them, or inadvertently retaining
        # them in the result where they might be inappropriate.
        ones_cube = cube.copy()
        ones_cube.data = np.ones_like(cube.data)
        ones_cube.rename("unknown")
        ones_cube.units = 1
        for cm in ones_cube.cell_measures():
            ones_cube.remove_cell_measure(cm)
        for av in ones_cube.ancillary_variables():
            ones_cube.remove_ancillary_variable(av)
        return ones_cube

    # Match up data masks if required.
    if common_mask:
        # Create CUBE of 1's with a common mask.
        if ma.is_masked(cube_2.data):
            mask_cube = _ones_like(cube_2)
        else:
            mask_cube = 1.0
        if ma.is_masked(cube_1.data):
            # Take a slice to avoid unnecessary broadcasting of cube_2.
            slice_coords = [
                dim_coords_1[i]
                for i in range(cube_1.ndim)
                if dim_coords_1[i] not in common_dim_coords
                and np.array_equal(
                    cube_1.data.mask.any(axis=i), cube_1.data.mask.all(axis=i)
                    )
                ]
            cube_1_slice = next(cube_1.slices_over(slice_coords))
            mask_cube = _ones_like(cube_1_slice) * mask_cube
        # Apply common mask to data.
        if isinstance(mask_cube, _Cube):
            cube_1 = cube_1 * mask_cube
            cube_2 = mask_cube * cube_2
            dim_coords_2 = [coord.name() for coord in cube_2.dim_coords]
    # Broadcast weights to shape of cubes if necessary.
    if weights is None or cube_1.shape == smaller_shape:
        weights_1 = weights
        weights_2 = weights
    else:
        if weights.shape != smaller_shape:
            raise ValueError(
                "weights array should have dimensions {}".format(smaller_shape)
                )

        dims_1_common = [
            i
            for i in range(cube_1.ndim)
            if dim_coords_1[i] in common_dim_coords
            ]
        weights_1 = broadcast_to_shape(weights, cube_1.shape, dims_1_common)
        if cube_2.shape != smaller_shape:
            dims_2_common = [
                i
                for i in range(cube_2.ndim)
                if dim_coords_2[i] in common_dim_coords
                ]
            weights_2 = broadcast_to_shape(
                weights, cube_2.shape, dims_2_common
                )
        else:
            weights_2 = weights

    # Calculate correlations.
    s1 = cube_1 - cube_1.collapsed(
        corr_coords, iris.analysis.MEAN, weights=weights_1
        )
    s2 = cube_2 - cube_2.collapsed(
        corr_coords, iris.analysis.MEAN, weights=weights_2
        )

    covar = (s1 * s2).collapsed(
        corr_coords, iris.analysis.SUM, weights=weights_1, mdtol=mdtol
        )
    var_1 = (s1**2).collapsed(
        corr_coords, iris.analysis.SUM, weights=weights_1
        )
    var_2 = (s2**2).collapsed(
        corr_coords, iris.analysis.SUM, weights=weights_2
        )

    denom = _apply(
        np.sqrt, var_1 * var_2, new_unit=covar.units
        )
    corr_cube = covar / denom
    corr_cube.rename("Pearson's r")

    if alpha is not None:
        from scipy.stats import beta
        if isinstance(corr_coords, str):
            n = cube_1.coord(corr_coords).shape[0]
        else:
            n = np.prod([cube_1.coord(i).shape[0] for i in corr_coords])
        dist = beta(n/2. - 1, n/2. - 1, -1, 2)

    return (corr_cube, dist.isf(alpha/2.)) if alpha is not None else corr_cube


def myAuxTime_(
        ymdhms,
        delta='day',
        unit='days since 1900-1-1',
        calendar='standard',
        ):
    """
    ... create single value time auxcoord ...

    Args:
         year, month, day (int): as the names
            hms (int, optional): hour, minute, second

    kwArgs:
        unit, calendar (string): see cf_unit
    """
    _unit = cUnit(unit, calendar=calendar)
    if isinstance(ymdhms, str):
        dnum = _unit.date2num(iterDT_(ymdhms, delta=delta))
    elif len(ymdhms) > 3:
        dnum = _unit.date2num(datetime(*ymdhms))
    elif len(ymdhms) == 2:
        dnum = _unit.date2num(datetime(*ymdhms, 1))
    elif len(ymdhms) == 1:
        dnum = _unit.date2num(datetime(*ymdhms, 1, 1))
    return _iAuxC(dnum, units=_unit, standard_name='time')


def myDimTime_(
        datestr,
        delta='day',
        unit='days since 1900-1-1',
        calendar='standard',
        ):
    """
    ... create time dimcoord ...

    Args:
        datestr (str): date string
          delta (str): time delta (default 1 day)

    kwArgs:
        unit, calendar (str): see cf_unit
    """
    _unit = cUnit(unit, calendar=calendar)
    dnum = _unit.date2num(iterDT_(datestr, delta=delta))
    return _iDimC(dnum, units=_unit, standard_name='time')


def div_cube(uc, vc):
    """
    ... divergence of ucube and v cube ...
    """
    from iris.analysis.cartography import DEFAULT_SPHERICAL_EARTH_RADIUS as _r
    ucxy = _dimcXY(uc)
    vcxy = _dimcXY(vc)
    amsg = "xycoords error!"
    assert ucxy == vcxy and all(i is not None for i in ucxy), amsg
    ucx, ucy = ucxy
    if ucx.units == ucy.units == cUnit('m'):
        du = np.gradient(uc.data, ucx.points, axis=uc.coord_dims(ucx))
        dv = np.gradient(vc.data, ucy.points, axis=uc.coord_dims(ucy))
    elif ucx.units == ucy.units == cUnit('degree'):
        du = np.gradient(uc.data, axis=uc.coord_dims(ucx))
        dv = np.gradient(vc.data, axis=uc.coord_dims(ucy))
        _ucx, _ucy = ucx.copy(), ucy.copy()
        if uc.coord_dims(ucx) > uc.coord_dims(ucy):
            x2d, y2d = np.meshgrid(_ucx.points, _ucy.points)
            xdim, ydim = 1, 0
        else:
            y2d, x2d = np.meshgrid(_ucy.points, _ucx.points)
            xdim, ydim = 0, 1
        wx = (np.gradient(np.deg2rad(x2d.astype(np.float64)), axis=xdim) *
              np.cos(np.deg2rad(y2d.astype(np.float64))) *
              _r)
        wy = np.gradient(np.deg2rad(y2d.astype(np.float64)), axis=ydim) * _r
        du, dv = du / wx, dv / wy
    else:
        emsg = ("check the units in xycoords. We accept "
                f"{cUnit('m')!r} or {cUnit('degree')!r}!")
        raise(emsg)
    o = uc.copy(du + dv)
    o.rename('divergence')
    o.units = sqzUnit_(f"{uc.units.origin} m**-1")
    return o


def curl_cube(uc, vc):
    """
    ... curl of ucube and v cube ...
    """
    from iris.analysis.cartography import DEFAULT_SPHERICAL_EARTH_RADIUS as _r
    ucxy = _dimcXY(uc)
    vcxy = _dimcXY(vc)
    amsg = "xycoords error!"
    assert ucxy == vcxy and all(i is not None for i in ucxy), amsg
    ucx, ucy = ucxy
    if ucx.units == ucy.units == cUnit('m'):
        du = np.gradient(uc.data, ucy.points, axis=uc.coord_dims(ucy))
        dv = np.gradient(vc.data, ucx.points, axis=uc.coord_dims(ucx))
    elif ucx.units == ucy.units == cUnit('degree'):
        du = np.gradient(uc.data, axis=uc.coord_dims(ucy))
        dv = np.gradient(vc.data, axis=uc.coord_dims(ucx))
        _ucx, _ucy = ucx.copy(), ucy.copy()
        if uc.coord_dims(ucx) > uc.coord_dims(ucy):
            x2d, y2d = np.meshgrid(_ucx.points, _ucy.points)
            xdim, ydim = 1, 0
        else:
            y2d, x2d = np.meshgrid(_ucy.points, _ucx.points)
            xdim, ydim = 0, 1
        wx = (np.gradient(np.deg2rad(x2d.astype(np.float64)), axis=xdim) *
              np.cos(np.deg2rad(y2d.astype(np.float64))) *
              _r)
        wy = np.gradient(np.deg2rad(y2d.astype(np.float64)), axis=ydim) * _r
        du, dv = du / wy, dv / wx
    else:
        emsg = ("check the units in xycoords. We accept "
                f"{cUnit('m')!r} or {cUnit('degree')!r}!")
        raise(emsg)
    o = uc.copy(dv - du)
    o.rename('curl')
    o.units = sqzUnit_(f"{uc.units.origin} m**-1")
    return o


def smth_cube(c, m=9, n=9):
    """
    ... smoothing 2d cube ...
    """
    amsg = "xycoords error!"
    ucxy = _dimcXY(c)
    o = []
    for _sl in c.slices(ucxy):
        data = rMEAN2d_(_sl.data, m, n, mode='same')
        _sl.data = data
        o.append(_sl)
    return _CubeList(o).merge_cube()


def _any2dim(c, **kwargs):
    _ind = _ind_clim(c, **kwargs)
    axs, inds = _ind[0::2], _ind[1::2]
    def _isok(ind):
        o = True
        if not isinstance(ind, slice):
            o &= np.all(np.diff(ind) == 1)
        return o
    def _allisok():
        return all(_isok(i) for i in inds)

    def _r(coord):
        if tryattr_(coord, 'circular'):
            if tryattr_(coord.units, 'modulus'):
                return coord.units.modulus

    def _f(ax, ind):
        coord = _dimc(c, ax)
        msg = f"no dim_coord found along dimension {ax!r}"
        if coord is None:
            raise Exception(msg)
        value = coord.points[ind]
        if not _isok(ind):
            if _r(coord) and ind[-1] == coord.points.size - 1:
                rb = coord.points[ind[np.where(np.diff(ind)!=1)[0]]]
                rb = _r(coord) if rb < 0  else _r(coord) / 2
                value = rpt_(value, rb, rb - _r(coord))
            else:
                msg = (f"coord {coord.name()} is not circular, "
                       "or units.modulus is missing, "
                       "or too many slices for the given index!")
                raise Exception(msg)
        lim = [value.min(), value.max()]
        return {coord.name(): lim}

    if _allisok():
        return _ind
    else:
        o = {}
        for ax, ind in zip(axs, inds):
            o.update(_f(ax, ind))
        return o


def intersection_(c, **kwargs):
    """
    ... intersection by range of coords ...
    """
    xxx = _any2dim(c, **kwargs)
    if isinstance(xxx, dict):
        return c.intersection(**xxx)
    else:
        return _extract_byAxes(c, *xxx)


#-- ccxx ----------------------------------------------------------------------
#-- _dimc ---------------------------------------------------------------------
nmXs = ('x-coord', 'x_coord', 'x coord', 'lon', 'west_east')
nmYs = ('y-coord', 'y_coord', 'y coord', 'lat', 'south_north')
nmZs = ('z-coord', 'z_coord', 'z coord', 'hgt', 'height', 'bottom_top',
        'press')
nmTs = ('date', 'time', 'day', 'month', 'season', 'year', 'second', 'minute')

def _dimcT(c):
    try:
        return c.coord(axis='T', dim_coords=True)
    except:
        return None

def _dimcX(c):
    return _dimc(c, 'X')

def _dimcY(c):
    return _dimc(c, 'Y')

def _dimcZ(c):
    return _dimc(c, 'Z')

def _dimcXY(c):
    return (_dimcX(c), _dimcY(c))

def _dimc(c, axis):
    if isinstance(axis, str):
        if len(axis) == 1:
            if axis.upper() == 'T':
                return _dimcT(c)
            elif axis.upper() in 'XYZ':
                nms = eval(f"nm{axis.upper()}s")
                for i in c.dim_coords:
                    if (any(ii in i.name().lower() for ii in nms)
                        or i.name().upper() == axis.upper()):
                        return i
            else:
                return None
        else:
            return tuple(_dimc(c, i) for i in axis)
    else:
        try:
            return c.coord(dimensions=rpt_(axis, c.ndim), dim_coords=True)
        except:
            return None

#-- _dim ----------------------------------------------------------------------
def _dim(c, axis):
    o = eval(f"_dimc(c, {axis!r})")
    return o.name() if o else None

def _dimT(c):
    return _dim(c, 'T')

def _dimX(c):
    return _dim(c, 'X')

def _dimY(c):
    return _dim(c, 'Y')

def _dimZ(c):
    return _dim(c, 'Z')

#-- _ax -----------------------------------------------------------------------
def _ax(c, axis):
    if len(axis) == 1:
        o = eval(f"_dimc{axis.upper()}(c)")
        return c.coord_dims(o)[0] if o else None
    else:
        return (c.coord_dims(axis) if axis in (i.name() for i in c.coords())
                else None)

def _axT(c):
    return _ax(c, 'T')

def _axX(c):
    return _ax(c, 'X')

def _axY(c):
    return _ax(c, 'Y')

def _axZ(c):
    return _ax(c, 'Z')

def _axXY(c): #_axXY
    axXY = [_axX(c), _axY(c)]
    if all(i is not None for i in axXY):
        return tuple(sorted(axXY))

def _isyx(c):
    if _axXY(c) is not None:
        return _axY(c) < _axX(c)
    else:
        msg = f"Dimension 'X' or 'Y' does not exist in Cube {c.name()!r}"
        raise Exception(msg)

#-- _guessbnds ----------------------------------------------------------------
def _guessXYZT(coord):
    for i in 'XYZT':
        nms = eval(f"nm{i}s")
        if any(ii in coord.name().lower() for ii in nms):
            return i

def _guessLOA(coord):
    for i in ('lo', 'la'):
        if i in coord.name().lower():
            return i

def _guessbnds(c, coord, **kwargs):
    hgKA = dict(loa=_guessLOA(coord))
    hgKA.update(kwargs)
    if coord.ndim == 1:
        coord.guess_bounds()
        return coord.bounds
    else:
        _xyzt = _guessXYZT(coord)
        ax = _ax(c, _xyzt)
        axs = c.coord_dims(coord)
        axincoord = axs.index(ax)
        lb = half_grid_(coord.points, side='l', axis=axincoord, **hgKA)
        rb = half_grid_(coord.points, side='r', axis=axincoord, **hgKA)
        return np.stack((lb, rb), axis=-1)

#-- _loa ----------------------------------------------------------------------
def _loa(c):
    def _f(s):
        for i in c.coords():
            if s in i.name().lower():
                return i
    return (_f('lon'), _f('lat'))

def _axLOA(c):
    return tuple(c.coord_dims(i) for i in _loa(c) if i)

def _loa_pnts(c):
    def _f(s):
        for i in c.coords():
            if s in i.name().lower():
                return i.points
    return (_f('lon'), _f('lat'))

def _loa_bnds(c):
    def _f(s):
        for i in c.coords():
            if s in i.name().lower():
                if i.has_bounds():
                    return i.bounds
                else:
                    return _guessbnds(c, i, loa=s[:2])
    return (_f('lon'), _f('lat'))

def _loa_pnts_2d(c):
    lo, la = _loa_pnts(c)
    if lo is None or la is None:
        emsg = f"Cube {c.name()!r} must have longitude/latidute coords!"
        raise Exception(emsg)
    return loa2d_(lo, la, isYX=_isyx(c))

#-- _msk ----------------------------------------------------------------------
def _ind_clim(c, **kwargs):
    shp = c.shape
    cnms = [i.name() for i in c.coords()]
    _tmp = [c.coord_dims(k) for k in kwargs.keys() if k in cnms]
    uds = [tuple(i) for i in ss_fr_sl_(_tmp)]
    def _d(_k):
        return c.coord_dims(_k)
    def _r(coord):                                                             # right bounds for coord that is circular
        if tryattr_(coord, 'circular'):
            if tryattr_(coord.units, 'modulus'):
                return coord.units.modulus
    def _f(_k, ud):                                                            # derive ind for a given coord and an unique dimension
        _shp = tuple(shp[i] for i in ud)
        udD = {ii:i for i, ii in enumerate(ud)}
        _coord = c.coord(_k)
        _dims = _d(_k)
        ax = tuple(udD[i] for i in _dims)
        r_ = _r(_coord)
        if _coord.has_bounds():
            b0, b1 = _coord.bounds.T
        else:
            b0, b1 = np.moveaxis(_guessbnds(c, _coord), -1, 0)
        _b0 = ind_inRange_(b0, *kwargs[_k], r_=r_)
        _b1 = ind_inRange_(b1, *kwargs[_k], r_=r_)
        return robust_bc2_(
                np.logical_and(_b0, _b1),
                _shp,
                ax,
                )
    def _ff(ud):                                                               # derive ind for an unique dimension
        udd = {i:ii for i, ii in enumerate(ud)}
        _ma = ()
        for k in kwargs.keys():
            if k in cnms and any(i in ud for i in _d(k)):
                _ma += (_f(k, ud),)
        _ind = bA2ind_(np.logical_and.reduce(_ma))
        o = ()
        for i, ii in zip(ud, _ind):
            if not(isinstance(ii, slice) and ii == slice(None)):
                o += (i, ii)
        return o
    o = ()                                                                     # collect ind along each dimension, to be passed to extract_
    for i in uds:
        o += _ff(i)
    return o

def _omsk_clim(c, to_ind=False, **kwargs):
    shp = c.shape
    def _r(coord):
        if tryattr_(coord, 'circular'):
            if tryattr_(coord.units, 'modulus'):
                return coord.units.modulus
    def _f1(_k):
        _coord = c.coord(_k)
        _dims = c.coord_dims(_coord)
        r_ = _r(_coord)
        if _coord.has_bounds():
            b0, b1 = _coord.bounds.T
        else:
            b0, b1 = np.moveaxis(_guessbnds(c, _coord), -1, 0)
        _b0 = ind_inRange_(b0, *kwargs[_k], r_=r_)
        _b1 = ind_inRange_(b1, *kwargs[_k], r_=r_)
        return robust_bc2_(
                np.logical_and(_b0, _b1),
                shp,
                _dims,
                )
    booL = []
    for k in kwargs.keys():
        if k in (i.name() for i in c.coords()):
            booL.append(_f1(k))
    o = np.logical_and.reduce(booL)
    return bA2ind_(o) if to_ind else o

def _ind_loalim(c, longitude=None, latitude=None):
#-- in_loalim_(lo, la, shp, axXY=None, lolim=None, lalim=None, isYX=True)
    return in_loalim_(*_loa_pnts(c),
                      c.shape,
                      axXY=_axXY(c),
                      lolim=longitude,
                      lalim=latitude,
                      isYX=_isyx(c),
                      )

def _ind_poly(c, poly, **kwArgs):
    x, y = _loa_pnts_2d(c)
    ind = in_polygons_(poly, np.vstack((x.ravel(), y.ravel())).T, **kwArgs)
    ind = robust_bc2_(ind.reshape(x.shape), c.shape, _axXY(c))
    return ind

def _where_not_msk(c, omsk, **kwargs):
    return iris.util.mask_cube(c, ~omsk, in_place=False, **kwargs)

#-- _extract ------------------------------------------------------------------
def _extract_byAxes(c, axis, sl_i, *vArg):
    """
    ... extract CUBE/ARRAY by providing selection along axis/axes ...

    Args:
           c: parent CUBE/ARRAY
        axis: along which for the extraction; axis name acceptable
        sl_i: slice, list, or 1d array of selected indices along axis
        vArg: any pairs of (axis, sl_i)

    useful info:
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

    if isinstance(c, _Cube):
        ax = [c.coord_dims(i)[0] if isinstance(i, (str, _iDimC)) else i
              for i in ax]

    return extract_(c, *(i for ii in zip(ax, sl) for i in ii), fancy=False)

#-- _aw -----------------------------------------------------------------------
def _rEARTH(c):
    """
    ... Get the radius of the earth ...
    """
    from iris.analysis.cartography import DEFAULT_SPHERICAL_EARTH_RADIUS
    cs = c.coord_system("CoordSystem")
    if isinstance(cs, iris.coord_systems.GeogCS):
        if cs.inverse_flattening != 0.0:
            warnings.warn("Assuming spherical earth from ellipsoid.")
        radius_of_earth = cs.semi_major_axis
    elif (isinstance(cs, iris.coord_systems.RotatedGeogCS) and
            (cs.ellipsoid is not None)):
        if cs.ellipsoid.inverse_flattening != 0.0:
            warnings.warn("Assuming spherical earth from ellipsoid.")
        radius_of_earth = cs.ellipsoid.semi_major_axis
    else:
        warnings.warn("Using DEFAULT_SPHERICAL_EARTH_RADIUS.")
        radius_of_earth = DEFAULT_SPHERICAL_EARTH_RADIUS
    return radius_of_earth

def _area_weights(c, normalize=False):
    """
    ... revised iris.analysis.cartography.area_weights to ignore lon/lat in
        auxcoords ...
    """
    radius_of_earth = _rEARTH(c)                                               # Get the radius of the earth

    lon, lat = _loa(c)                                                         # Get the lon and lat coords and axes
    if any(i is None for i in (lon, lat)):
        msg = "Cannot get latitude/longitude coordinates from CUBE {!r}!"
        raise ValueError(msg.format(c.name()))

    if lon.ndim == lat.ndim == 1:                                              # axes for the weights to be broadcasted
        axes = (_axY(c), _axX(c))
    elif lon.shape == lat.shape:
        axes = c.coord_dims(lat)

    for coord in (lat, lon):                                                   # check units
        if coord.units not in (cUnit('degree'), cUnit('radian')):
            msg = ("Units of degrees or radians required, coordinate "
                   f"{coord.name()!r} has units: {coord.units.name!r}")
            raise ValueError(msg)

    lob, lab = _loa_bnds(c)                                                    # Create 2D weights from bounds
    lob = cUnit(lon.units).convert(lob, 'radian')
    lab = cUnit(lat.units).convert(lab, 'radian')
    ll_weights = aw_loa_bnds_(lob, lab, radius_of_earth)                       # Use the geographical area as the weight for each cell

    if normalize:                                                              # Normalize the weights if necessary
        ll_weights /= ll_weights.sum()

                                                                               # Now we create an array of weights for each cell. This process will
                                                                               # handle adding the required extra dimensions and also take care of
                                                                               # the order of dimensions.
    return robust_bc2_(ll_weights, c.shape, axes=axes)

#-- _rg -----------------------------------------------------------------------
def _rg_func(c, func, rg=None, inv=False, **funcD):
    if rg:
        ind = _ind_loalim(c, **rg)
        tmp = _where_not_msk(c, ~ind) if inv else _where_not_msk(c, ind)
    else:
        tmp = c.copy()
    return tmp.collapsed(_dimcXY(tmp), func, **funcD)

def _rg_mean(c, rg=None, inv=False, **funcD):
    aw = _area_weights(c)
    return _rg_func(c, iris.analysis.MEAN, rg=rg, inv=inv, weights=aw, **funcD)

def _poly_func(c, poly, func, inpolyKA={}, inv=False, **funcD):
    ind = _ind_poly(c, poly, **inpolyKA)
    tmp = _where_not_msk(c, ~ind) if inv else _where_not_msk(c, ind)
    return tmp.collapsed(_dimcXY(tmp), func, **funcD)

def _poly_mean(c, poly, inpolyKA={}, inv=False, **funcD):
    aw = _area_weights(c)
    return _poly_func(c, poly, iris.analysis.MEAN,
                      inpolyKA=inpolyKA,
                      inv=inv,
                      weights=aw,
                      **funcD,
                      )

#-- _xy_slice -----------------------------------------------------------------
def _xy_slice(c, i=0):
    ax_xy = _axXY(c)
    ind = ind_shape_i_(c.shape, i, axis=ax_xy)
    return c[ind]

#-- ccxx ----------------------------------------------------------------------
