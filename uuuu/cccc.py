"""
>--#########################################################################--<
>-------------------------functions operating on cube-------------------------<
>--#########################################################################--<
* alng_axis_            : apply along axis
* area_weights_         : modified area_weights from iris
* ax_fn_mp_             : apply along axis mp
* axT_cube              : time axis of cube
* concat_cube_          : robust cube concatenator
* corr_cube_            : modified version of iris.analysis.stats.pearsonr
* cubesv_               : save cube to nc with dim_t unlimitted
* cut_as_cube           : cut into the domain of another cube
* doy_f_cube            : f for each doy
* en_iqr_               : ensemble interquartile range
* en_max_               : ensemble max
* en_mean_              : ensemble mean
* en_min_               : ensemble min
* en_mxn_               : ensemble spread
* en_mm_cubeL_          : make ensemble cube for multimodels
* en_rip_               : ensemble (rxixpx) cube
* extract_byAxes_       : extraction with help of inds_ss_
* extract_month_cube    : extraction cube of month
* extract_period_cube   : extraction cube within [y0, y1]
* extract_season_cube   : extraction cube of season
* extract_win_cube      : extraction within a window (daily)
* f_allD_cube           : iris analysis func over all dims of cube(L) (each)
* getGridAL_cube        : grid_land_area
* getGridA_cube         : grid_area from file or calc with basic assumption
* get_gwl_y0_           : first year of 30-year window of global warming level
* get_loa_              : longitude/latitude coords of cube
* get_loa_dim_          : modified _get_lon_lat_coords from iris
* get_loa_pts_2d_       : 2d longitude/latitude points (from coord or meshed)
* get_xy_dim_           : horizontal spatial dim coords
* get_xyd_cube          : cube axes of xy dims
* guessBnds_cube        : bounds of dims points
* half_grid_            : points between grids
* initAnnualCube_       : initiate annual cube
* inpolygons_cube       : points if inside polygons
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
* nTslice_cube          : slices along a no-time axis
* nearest_point_cube    : extract 1 point cube
* nine_points_cube      : extract nine points cube centered at a given point
* pSTAT_cube            : period statistic (month, season, year)
* pcorr_cube            : partial correlation (cube_0, cube_1, cube_cntr)
* pp_cube               : pth and 100-pth of cube(L) data (each)
* pst_                  : post-rename/reunits cube(L)
* purefy_cubeL_         : prepare for concat or merge
* repair_cs_            : bug fix for save cube to nc
* repair_lccs_          : bug fix for save cube to nc (LamgfortComfort)
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
* unique_yrs_of_cube    : unique year points of cube
* y0y1_of_cube          : starting and ending year of cube
* yr_doy_cube           : year and day-of-year auxcoord
...

###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 06.09.2019
Date last modified: 11.11.2020
           comment: add function half_grid_, move remaping functions to rgd.py
"""

import iris
from iris.cube import Cube as _Cube
from iris.cube import CubeList as _CubeList
import iris.coord_categorisation as _ica
from iris.coords import AuxCoord as _iAuxC
from iris.util import equalise_attributes

import numpy as np
import cf_units
import warnings
import re
from datetime import datetime

from .ffff import *


__all__ = ['alng_axis_',
           'area_weights_',
           'ax_fn_mp_',
           'axT_cube',
           'concat_cube_',
           'corr_cube_',
           'cubesv_',
           'cut_as_cube',
           'doy_f_cube',
           'en_iqr_',
           'en_max_',
           'en_mean_',
           'en_min_',
           'en_mxn_',
           'en_mm_cubeL_',
           'en_rip_',
           'extract_byAxes_',
           'extract_month_cube',
           'extract_period_cube',
           'extract_season_cube',
           'extract_win_cube',
           'f_allD_cube',
           'getGridAL_cube',
           'getGridA_cube',
           'get_gwl_y0_',
           'get_loa_',
           'get_loa_dim_',
           'get_loa_pts_2d_',
           'get_xy_dim_',
           'get_xyd_cube',
           'guessBnds_cube',
           'half_grid_',
           'initAnnualCube_',
           'inpolygons_cube',
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
           'unique_yrs_of_cube',
           'y0y1_of_cube',
           'yr_doy_cube']


def slice_back_(cnd, c1d, ii, axis):
    """
    ... put 1D slice back to its parent CUBE/ARRAY ...

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
    if ((isinstance(cnd, _Cube) or np.ma.isMaskedArray(cnd))
        and not np.ma.isMaskedArray(c1d)):
        c1d = np.ma.masked_array(c1d, np.isnan(c1d))
    emsg = "slice NOT matched its parent along axis({})."
    if axis is None:
        if c1d.size != 1:
            raise Exception(emsg.format(axis))
    else:
        axis = rpt_(axis, cnd.ndim)
        axis = sorted(axis) if isIter_(axis, xi=(int, np.integer)) else axis
        if not np.all(np.asarray(cnd.shape)[axis] == np.asarray(c1d.shape)):
            raise Exception(emsg.format(axis))
    ind = ind_shape_i_(cnd.shape, ii, axis)
    if isinstance(cnd, _Cube):
        cnd.data[ind] = c1d
    elif np.ma.isMaskedArray(cnd):
        cnd[ind] = c1d
    else:
        cnd[ind_shape_i_(cnd.shape, ii, axis)] = c1d


def extract_byAxes_(cnd, axis, sl_i, *vArg):
    """
    ... extract CUBE/ARRAY by providing selection along axis/axes ...

    Args:
         cnd: parent CUBE/ARRAY
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

    if isinstance(cnd, _Cube):
        ax = [cnd.coord_dims(i)[0]
              if isinstance(i, (str, iris.coords.DimCoord)) else i
              for i in ax]

    nArg = [(i, j) for i, j in zip(ax, sl)]
    nArg = tuple(j for i in nArg for j in i)
    if (hasattr(cnd, '__orthogonal_indexing__') and
        cnd.__orthogonal_indexing__):
        inds = inds_ss_(cnd.ndim, *nArg, fancy=False)
    else:
        inds = inds_ss_(cnd.ndim, *nArg, fancy=True)

    return cnd[inds]


def isMyIter_(x):
    """
    ... Iterable with items as cube/ndarray ...
    """
    return isIter_(x,
                   xi=(np.ndarray, _Cube),
                   XI=(np.ndarray, _Cube, str, bytes))


def pst_(
        cube,
        name=None,
        units=None,
        var_name=None,
        attrU=None,
        ):
    """
    ... post-rename/reunits cube(L) ...
    """
    if isinstance(cube, _Cube):
        if name:
            cube.rename(name)
        if units:
            cube.units = units
        if var_name:
            cube.var_name = var_name
        if attrU:
            cube.attributes.update(attrU)
    elif isMyIter_(cube):
        for i in cube:
            pst_(i, name=name, units=units, var_name=var_name, attrU=attrU)


def axT_cube(cube):
    """
    ... dimesion of time axis in CUBE ...
    """
    try:
        tc = cube.coord(axis='T', dim_coords=True)
        return cube.coord_dims(tc)[0]
    except:
        return None


def nTslice_cube(cube, n):
    """
    ... slices along a no-time axis ...
    
    Args:
        cube: CUBE or iterable CUBEs
           n: maximum size of a slice
    """
    nd = cube.ndim
    ax_nT = [i for i in range(nd) if i not in cube.coord_dims('time')]
    shp = tuple(cube.shape[i] for i in ax_nT)
    if np.prod(shp) < n:
        return [cube]
    else:
        ss = sub_shp_(shp, n, dims=ax_nT)
        oo = [cube]
        for s in ss:
            oo = nli_(
                    [
                        [
                            extract_byAxes_(o, s[0], np.s_[i:(i + s[1])])
                            for i in range(0, cube.shape[s[0]], s[1])
                            ]
                        for o in oo
                        ]
                    )
        return _CubeList(oo)
    #    for i in reversed(ax_nT):
    #        if shp[i] > n:
    #            step = int(np.ceil(shp[i] / n))
    #            return [extract_byAxes_(cube, i, np.s_[ii:(ii + step)])
    #                    for ii in range(0, shp[i], step)]


def unique_yrs_of_cube(
        cube,
        ccsn='year',
        mmm=None,
        ):
    """
    ... unique year points of cube(L) ...
    """
    if isinstance(cube, _Cube):
        c = cube.copy()
        ccs = [i.name() for i in c.coords()]
        if ccsn not in ccs:
            if 'season' in ccsn:
                if mmm:
                    seasonyr_cube(c, mmm, name=ccsn)
                else:
                    emsg = "'mmm' must not be None for adding coord {!r}!"
                    raise ValueError(emsg.format(ccsn))
            else:
                _ica.add_year(c, 'time', name=ccsn)
        return np.unique(c.coord(ccsn).points)
    elif isIter_(cube, xi=_Cube):
        return [unique_yrs_of_cube(i) for i in cube]
    else:
        raise TypeError("unknown type for the first argument!")


def y0y1_of_cube(cube, ccsn='year', mmm=None):
    if isinstance(cube, _Cube):
        c = cube.copy()
        ccs = [i.name() for i in c.coords()]
        if ccsn not in ccs:
            if 'season' in ccsn:
                if mmm:
                    seasonyr_cube(c, mmm, name=ccsn)
                else:
                    emsg = "'mmm' must not be None for adding coord {!r}!"
                    raise ValueError(emsg.format(ccsn))
            else:
                _ica.add_year(c, 'time', name=ccsn)
        elif mmm and 'season' in ccsn:
            c.remove_coord(ccsn)
            seasonyr_cube(c, mmm, name=ccsn)
        return list(c.coord(ccsn).points[[0, -1]])
    elif isIter_(cube, xi=_Cube):
        yy = np.array([y0y1_of_cube(i) for i in cube])
        return [np.max(yy[:, 0]), np.min(yy[:, 1])]
    else:
        raise TypeError("unknown type for the first argument!")


def extract_period_cube(cube, y0, y1, yy=False, ccsn='year', mmm=None):
    """
    ... extract cube within the period from year y0 to year y1 ...

    Args:
        cube: CUBE with coord('time')
          y0: starting year
          y1: end year
    kwArgs:
          yy: True forcing output strictly starting/end at y0/y1
        ccsn: name of coord with year points for extraction
         mmm: defined season using 1st letters of composing months, should not
              be None when seasonyr_cube is called

    useful info:
        >>> help(seasonyr_cube)
    """
    c = cube.copy()
    ccs = [i.name() for i in c.coords()]
    if ccsn not in ccs:
        if 'season' in ccsn:
            if mmm:
                seasonyr_cube(c, mmm, name=ccsn)
            else:
                emsg = "'mmm' must not be None for adding coord {!r}!"
                raise ValueError(emsg.format(ccsn))
        else:
            _ica.add_year(c, 'time', name=ccsn)
    elif mmm and 'season' in ccsn:
        c.remove_coord(ccsn)
        seasonyr_cube(c, mmm, name=ccsn)
    cstrD = {ccsn: lambda x: y0 <= x <= y1}
    cstr = iris.Constraint(**cstrD)
    o = c.extract(cstr)
    if not (yy and (y0y1_of_cube(o) != [y0, y1] or
               not np.all(np.diff(unique_yrs_of_cube(o)) == 1))):
        return o


def extract_win_cube(cube, d, r=15):
    """
    ... extract cube within a window centered at doy==d ...

    Args:
        cube: CUBE containing coord('time')
           d: center of the window
    kwArgs:
           r: half window width (defaul 15)
    """
    c = cube.copy()
    try:
        _ica.add_day_of_year(c, 'time', name='doy')
    except ValueError:
        pass
    x1, x2 = rpt_(d - r, 365), rpt_(d + r, 365)
    cstr = iris.Constraint(
            doy=lambda x: x1 <= x <= x2 if x1 < x2 else not (x2 < x < x1)
            )
    o = c.extract(cstr)
    return o


def extract_season_cube(cube, mmm, valid_season=True):
    """
    ... extract a cube of season named with continuous-months' 1st letters ...

    Args:
                cube: CUBE/CubeList containing coord('time')
                 mmm: continuous-months' 1st letters (for example, 'djf')
    kwArgs:
        valid_season: if True and season mmm is crossing years, the 1st & end
                      year will be excluded 
    """
    if isinstance(cube, _Cube):
        if 'season' in (i.name() for i in cube.coords()):
            o = cube.extract(iris.Constraint(season=mmm))
        else:
            c = cube.copy() # to avoid changing metadata of original cube
            try:
                _ica.add_season_membership(c, 'time', mmm, name=mmm)
            except ValueError:
                c.remove_coord(mmm)
                _ica.add_season_membership(c, 'time', mmm, name=mmm)
            c.coord(mmm).points = c.coord(mmm).points.astype(np.int32)
            o = c.extract(iris.Constraint(**{mmm: True}))
        if valid_season and not ismono_(mmmN_(mmm)):
            y0, y1 = y0y1_of_cube(o, ccsn='seasonyr', mmm=mmm)
            o = extract_period_cube(
                    o, y0 + 1, y1 - 1,
                    ccsn='seasonyr', mmm=mmm
                    )
        return o
    elif isMyIter_(cube):
        cl = [extract_season_cube(i, mmm) for i in cubeL]
        return _CubeList(cl)
    else:
        raise TypeError("unknown type for the first argument!")


def extract_month_cube(cube, Mmm):
    """
    ... extraction cube of month Mmm ...
    """
    c = cube.copy()
    try:
        _ica.add_month(c, 'time', name='month')
    except ValueError:
        pass
    o = c.extract(iris.Constraint(month=Mmm[:3].capitalize()))
    return o


def f_allD_cube(cube, rg=None, f='MAX', **fK_):
    """
    ... iris analysis f(unc) over all dims of cube(L) (each) ...
    """
    #warnings.filterwarnings("ignore", category=UserWarning)
    if isinstance(cube, _Cube):
        if rg:
            cube = intersection_(cube, **rg)
        _f = eval('iris.analysis.{}'.format(f.upper()))
        c = cube.collapsed(cube.dim_coords, _f, **fK_)
        return c.data
    elif isMyIter_(cube):
        return np.asarray(
                [f_allD_cube(i, rg=rg, f=f, **fK_) for i in cube]
                )
    else:
        return np.nan


def min_cube(cube, rg=None):
    return f_allD_cube(cube, rg=rg, f='MIN')


def max_cube(cube, rg=None):
    return f_allD_cube(cube, rg=rg)


def minmax_cube(cube, rg=None):
    return np.asarray([min_cube(cube, rg=rg), max_cube(cube, rg=rg)])


def pp_cube(cube, rg=None, p=10):
    return np.asarray(
        [f_allD_cube(cube, rg=rg, f='PERCENTILE', percent=p),
         f_allD_cube(cube, rg=rg, f='PERCENTILE', percent=100 - p)]
        )


def min_cube_(cube, rg=None):
    return np.nanmin(min_cube(cube, rg=rg))


def max_cube_(cube, rg=None):
    return np.nanmax(max_cube(cube, rg=rg))


def minmax_cube_(cube, rg=None, p=None):
    if p:
        mms = pp_cube(cube, rg=rg, p=p)
    else:
        mms = minmax_cube(cube, rg=rg)
    return (np.nanmin(mms), np.nanmax(mms))


def get_xyd_cube(cube, guess_lst2=True):
    """
    ... cube axes of xy dims  ...
    ... see help(get_xy_dim_) ...
    """
    xc, yc = get_xy_dim_(cube)
    if xc is None:
        if guess_lst2:
            warnings.warn("missing 'x' or 'y' dimcoord in input cube; "
                          "guess last two as xyd.")
            return tuple(rpt_([-2, -1], cube.ndim))
        else:
            raise Exception("missing 'x' or 'y' dimcoord in input cube!")
    else:
        xyd = list(cube.coord_dims(yc) + cube.coord_dims(xc))
        xyd.sort()
        return tuple(xyd)


def _get_xy_lim(cube, longitude=None, latitude=None):
    xc, yc = get_xy_dim_(cube)
    if xc is None or yc is None:
        raise Exception("missing 'x' or 'y' dimcoord in input cube!")
    lo, la = cube.coord('longitude'), cube.coord('latitude')
    if longitude is None:
        longitude = [lo.points.min(), lo.points.max()]
    if latitude is None:
        latitude = [la.points.min(), la.points.max()]
    if xc == lo and yc == la:
        xyl = {xc.name(): longitude, yc.name(): latitude}
    else:
        xd = cube.coord_dims(lo).index(np.intersect1d(cube.coord_dims(lo),
                                                      cube.coord_dims(xc)))
        yd = cube.coord_dims(lo).index(np.intersect1d(cube.coord_dims(lo),
                                                      cube.coord_dims(yc)))
        a_ = ind_inRange_(lo.points, *longitude, r_=360)
        b_ = ind_inRange_(la.points, *latitude)
        c_ = np.logical_and(a_, b_)
        xi, yi = np.where(np.any(c_, axis=yd)), np.where(np.any(c_, axis=xd))
        xv, yv = xc.points[xi], yc.points[yi]
        if np.any(np.diff(xi[0]) != 1):
            if xc.circular and xc.units.modulus:
                rb = xc.points[np.where(np.diff(xi[0]) != 1)[0] + 1]
                xv = rpt_(xv, rb, rb - xc.units.modulus)
            else:
                raise Exception("limits given outside data!")
        xll = [xv.min(), xv.max()]
        yll = [yv.min(), yv.max()]
        xyl = {xc.name(): xll, yc.name(): yll}
    if xc.units.modulus:
        return xyl
    else:
        return (xc.name(), ind_inRange_(xc.points, *xll),
                yc.name(), ind_inRange_(yc.points, *yll))


def _get_ind_lolalim(cube, longitude=None, latitude=None):
    xyd = get_xyd_cube(cube)
    lo, la = get_loa_pts_2d_(cube)
    if longitude is None:
        longitude = [lo.min(), lo.max()]
    if latitude is None:
        latitude = [la.min(), la.max()]
    a_ = ind_inRange_(lo, *longitude, r_=360)
    b_ = ind_inRange_(la, *latitude)
    c_ = np.logical_and(a_, b_)
    return robust_bc2_(c_, cube.shape, xyd)


def intersection_(cube, longitude=None, latitude=None):
    """
    ... intersection by range of longitude/latitude ...
    """
    kwArgs=dict()
    if longitude is not None:
        kwArgs.update(dict(longitude=longitude))
    if latitude is not None:
        kwArgs.update(dict(latitude=latitude))
    if cube.coord('latitude').ndim == 1:
        return cube.intersection(**kwArgs)
    else:
        xyl = _get_xy_lim(cube, **kwArgs)
        if isinstance(xyl, dict):
            return cube.intersection(**xyl)
        else:
            return extract_byAxes_(cube, *xyl)


def seasonyr_cube(cube, mmm, name='seasonyr'):
    """
    ... add season_year auxcoords to a cube especially regarding
        specified season ...
    """
    if isinstance(cube, _Cube):
        if isinstance(mmm, str):
            seasons = (mmm, rest_mns_(mmm))
        elif (isinstance(mmm, (list, tuple)) and
              sorted(''.join(mmm)) == sorted('djfmamjjason')):
            seasons = mmm
        else:
            raise Exception("unknown seasons {!r}!".format(mmm))
        try:
            _ica.add_season_year(cube, 'time', name=name, seasons=seasons)
        except ValueError:
            cube.remove_coord(name)
            _ica.add_season_year(cube, 'time', name=name, seasons=seasons)
    elif isIter_(cube, xi=(_Cube, _CubeList, tuple, list)):
        for c in cube:
            seasonyr_cube(c, mmm, name=name)


def yr_doy_cube(cube):
    """
    ... add year, day-of-year auxcoords to a cube ...
    """
    if isinstance(cube, _Cube):
        try:
            _ica.add_year(cube, 'time', name='year')
        except ValueError:
            pass
        else:
            cube.coord('year').attributes = {}
        try:
            _ica.add_day_of_year(cube, 'time', name='doy')
        except ValueError:
            pass
        else:
            cube.coord('doy').attributes = {}
    elif isIter_(cube, xi=(_Cube, _CubeList, tuple, list)):
        for c in cube:
            yr_doy_cube(c)


def rm_yr_doy_cube(cube):
    """
    ... remove year, day-of-year auxcoords from a cube ...
    """
    try:
        cube.remove_coord('year')
    except iris.exceptions.CoordinateNotFoundError:
        pass
    try:
        cube.remove_coord('doy')
    except iris.exceptions.CoordinateNotFoundError:
        pass


def rm_t_aux_cube(cube, keep=None):
    """
    ... remove time-related auxcoords from a cube or a list of cubes ...
    """
    tauxL = ['year', 'month', 'season', 'day', 'doy', 'hour', 'yr']
    def _isTCoord(x):
        return any((i in x.name() for i in tauxL)) or isSeason_(x.name())
    if isinstance(cube, _Cube):
        for i in cube.aux_coords:
            if keep is None:
                isTaux = _isTCoord(i)
            elif isIter_(keep):
                isTaux = _isTCoord(i) and i.name() not in keep
            else:
                isTaux = _isTCoord(i) and i.name() != keep
            if isTaux:
                cube.remove_coord(i)
    elif isMyIter_(cube):
        for c in cube:
            rm_t_aux_cube(c)
    else:
        raise TypeError('Input should be CUBE or iterable CUBEs!')


def rm_sc_cube(cube):
    if isinstance(cube, _Cube):
        for i in cube.coords():
            if len(cube.coord_dims(i)) == 0:
                cube.remove_coord(i)
    elif isMyIter_(cube):
        for c in cube:
            rm_sc_cube(c)
    else:
        raise TypeError('Input should be CUBE or Iterable CUBEs!')


def guessBnds_cube(cube):
    """
    ... guess bounds of dims of cube if not exist ...
    """
    for i in cube.dim_coords:
        try:
            i.guess_bounds()
        except ValueError:
            pass


def get_loa_dim_(cube):
    """
    ... get lon and lat coords (dimcoords only) ...
    """
    lat_coords = [coord for coord in cube.dim_coords
                  if "latitude" in coord.name()]
    lon_coords = [coord for coord in cube.dim_coords
                  if "longitude" in coord.name()]
    if len(lat_coords) > 1 or len(lon_coords) > 1:
        raise ValueError(
            "Calling `get_loa_dim_` with multiple lat or lon coords"
            " is currently disallowed")
    lat_coord = lat_coords[0]
    lon_coord = lon_coords[0]
    return (lon_coord, lat_coord)


def get_xy_dim_(cube, guess_lst2=True):
    """
    ... horizontal spatial dim coords                    ...
    ... return last 2 dimcoords if failed in 'XY' method ...
    """
    try:
        return (cube.coord(axis='X', dim_coords=True),
                cube.coord(axis='Y', dim_coords=True))
    except:
        if guess_lst2 and cube.ndim > 1:
            return(cube.coord(dimensions=rpt_(-1, cube.ndim), dim_coords=True),
                   cube.coord(dimensions=rpt_(-2, cube.ndim), dim_coords=True))
        else:
            return (None, None)


def get_loa_(cube):
    """
    ... longitude/latitude coords of cube ...
    """
    try:
        lo, la = cube.coord('longitude'), cube.coord('latitude')
        lo.convert_units('degrees')
        la.convert_units('degrees')
        return (lo, la)
    except:
        return (None, None)


def area_weights_(cube, normalize=False):
    """
    ... revised iris.analysis.cartography.area_weights to ignore lon/lat in
        auxcoords ...
    """
    from iris.analysis.cartography import (
            DEFAULT_SPHERICAL_EARTH_RADIUS,
            DEFAULT_SPHERICAL_EARTH_RADIUS_UNIT,
            _quadrant_area
            )
    # Get the radius of the earth
    cs = cube.coord_system("CoordSystem")
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

    # Get the lon and lat coords and axes
    try:
        lon, lat = get_loa_dim_(cube)
    except IndexError:
        raise ValueError('Cannot get latitude/longitude '
                         'coordinates from cube {!r}.'.format(cube.name()))

    if lat.ndim > 1:
        raise iris.exceptions.CoordinateMultiDimError(lat)
    if lon.ndim > 1:
        raise iris.exceptions.CoordinateMultiDimError(lon)

    lat_dim = cube.coord_dims(lat)
    lat_dim = lat_dim[0] if lat_dim else None

    lon_dim = cube.coord_dims(lon)
    lon_dim = lon_dim[0] if lon_dim else None

    if not (lat.has_bounds() and lon.has_bounds()):
        msg = ("Coordinates {!r} and {!r} must have bounds to determine "
               "the area weights.".format(lat.name(), lon.name()))
        raise ValueError(msg)

    # Convert from degrees to radians
    lat = lat.copy()
    lon = lon.copy()

    for coord in (lat, lon):
        if coord.units in (cf_units.Unit('degrees'),
                           cf_units.Unit('radians')):
            coord.convert_units('radians')
        else:
            msg = ("Units of degrees or radians required, coordinate "
                   "{!r} has units: {!r}".format(coord.name(),
                                                 coord.units.name))
            raise ValueError(msg)
    # Create 2D weights from bounds.
    # Use the geographical area as the weight for each cell
    ll_weights = _quadrant_area(lat.bounds,
                                lon.bounds, radius_of_earth)

    # Normalize the weights if necessary.
    if normalize:
        ll_weights /= ll_weights.sum()

    # Now we create an array of weights for each cell. This process will
    # handle adding the required extra dimensions and also take care of
    # the order of dimensions.
    broadcast_dims = [x for x in (lat_dim, lon_dim) if x is not None]
    wshape = []
    for idim, dim in zip((0, 1), (lat_dim, lon_dim)):
        if dim is not None:
            wshape.append(ll_weights.shape[idim])
    ll_weights = ll_weights.reshape(wshape)
    broad_weights = iris.util.broadcast_to_shape(
            ll_weights, cube.shape, broadcast_dims
            )

    return broad_weights


def cut_as_cube(cube0, cube1):
    """
    ... cut cube1 with the domain of cube0 ...
    """
    xc1, yc1 = get_xy_dim_(cube1)
    xc0, yc0 = get_xy_dim_(cube0)
    xn, yn = xc1.name(), yc1.name()
    xe = np.min(np.abs(np.diff(xc1.points))) / 2
    ye = np.min(np.abs(np.diff(yc1.points))) / 2
    x0, x1 = np.min(xc0.points), np.max(xc0.points)
    y0, y1 = np.min(yc0.points), np.max(yc0.points)
    return extract_byAxes_(
            cube1,
            xn,
            ind_inRange_(xc1.points, x0 - xe, x1 + xe, side=0),
            yn,
            ind_inRange_(yc1.points, y0 - ye, y1 + ye, side=0)
            )


def maskLS_cube(cube, sftlf, LorS='S', thr=0):
    """
    ... mask sea/land area ...

    Parsed arguments:
         cube: DATA cube to be masked
        sftlf: land area fraction; at least covering entire cube
         LorS: 'land' or 'sea' to be masked (default 'sea')
          thr: sftlf value <= thr as not land area (default 0)
    """
    LList = ['L', 'LAND']
    SList = ['S', 'O', 'W', 'SEA', 'OCEAN', 'WATER']
    if LorS.upper() not in (LList + SList):
        raise ValueError("Variable 'LorS' not interpretable!")
    sftlf_ = cut_as_cube(cube, sftlf)
    ma_0 = sftlf_.data <= thr
    if LorS.upper() in LList:
        ma_0 = ~ma_0
    ma_ = np.broadcast_to(ma_0, cube.shape)
    cube = iris.util.mask_cube(cube, ma_)


def getGridA_cube(cube, areacella=None):
    """
    ... get grid_area of cube ...
    """
    if areacella:
        ga_ = iris.util.squeeze(areacella)
        if ga_.ndim != 2:
            return getGridA_cube(cube)
        ga = cut_as_cube(cube, ga_).data
        try:
            ga = robust_bc2_(ga, cube.shape, get_xyd_cube(cube))
            return ga
        except:
            return getGridA_cube(cube)
    else:
        try:
            guessBnds_cube(cube)
            ga = area_weights_(cube)
        except:
            ga = None
        return ga


def getGridAL_cube(cube, sftlf=None, areacella=None):
    """
    ... return grid_land_area of cube if sftlf provided ...
    ... else return grid_area of cube                   ...
    """
    ga = getGridA_cube(cube, areacella)
    if sftlf is not None:
        sf_sqz = iris.util.squeeze(sftlf)
        if sf_sqz.ndim != 2:
            raise Exception('NOT 2D area-cube!')
        sf = cut_as_cube(cube, sf_sqz).data
        sf = robust_bc2_(sf, cube.shape, get_xyd_cube(cube))
        if ga is None:
            return np.ones(cube.shape) * sf / 100
        else:
            return ga * sf / 100.
    else:
        return ga


def rgF_cube(cube, function, rgD=None, **functionD):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if rgD:
        ind = _get_ind_lolalim(cube, **rgD)
        tmp = iris.util.mask_cube(cube.copy(), ~ind)
    else:
        tmp = cube
    xc, yc = get_xy_dim_(tmp)
    return tmp.collapsed([xc, yc], function, **functionD)


def rgF_poly_cube(cubeD, poly, function, **functionD):
    #warnings.filterwarnings("ignore", category=UserWarning)
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    tmp = iris.util.mask_cube(cubeD.copy(), ~ind)
    xc, yc = get_xy_dim_(tmp)
    return tmp.collapsed([xc, yc], function, **functionD)


def rgCount_cube(cubeD, sftlf=None, areacella=None, rgD=None, function=None):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if function is None:
        function = lambda values: values > 0
        warnings.warn("'function' not provided; count values greater than 0.")
    xyd = get_xyd_cube(cubeD)
    ga0 = getGridA_cube(cubeD, areacella)
    ga0 = np.ones(cubeD.shape) if ga0 is None else ga0
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    ga = np.ones(cubeD.shape) if ga is None else ga
    if rgD:
        ind = _get_ind_lolalim(cubeD, **rgD)
        ga0 *= ind
        ga *= ind
    umsk = ~cubeD.data.mask if (np.ma.isMaskedArray(cubeD.data) and
                                np.ma.is_masked(cubeD.data)) else 1
    sum0 = np.sum(ga0 * umsk, axis=xyd)
    if np.any(sum0 == 0):
        raise Exception("empty slice encountered.")
    data = np.sum(function(cubeD.data) * ga, axis=xyd) * 100 / sum0
    xc, yc = get_xy_dim_(cubeD)
    tmp = cubeD.collapsed([xc, yc], iris.analysis.MEAN)
    return tmp.copy(data)


def rgCount_poly_cube(cubeD, poly, sftlf=None, areacella=None, function=None,
                      **kwArgs):
    #warnings.filterwarnings("ignore", category=UserWarning)
    if function is None:
        function = lambda values: values > 0
        warnings.warn("'function' not provided; count values greater than 0.")
    xyd = get_xyd_cube(cubeD)
    ga0 = getGridA_cube(cubeD, areacella)
    ga0 = np.ones(cubeD.shape) if ga0 is None else ga0
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    ga = np.ones(cubeD.shape) if ga is None else ga
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    ga0 *= ind
    ga *= ind
    sum0 = np.sum(ga0, axis=xyd)
    if np.any(sum0 == 0):
        raise Exception("empty slice encountered.")
    data = np.sum(function(cubeD.data) * ga, axis=xyd) * 100 / sum0
    xc, yc = get_xy_dim_(cubeD)
    tmp = cubeD.collapsed([xc, yc], iris.analysis.MEAN)
    return tmp.copy(data)


def rgMean_cube(cubeD, sftlf=None, areacella=None, rgD=None):
    """
    ... regional mean; try weighted if available ...
    """
    #warnings.filterwarnings("ignore", category=UserWarning)
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    if rgD:
        ind = _get_ind_lolalim(cubeD, **rgD)
        if ga is None:
            ga = ind * np.ones(ind.shape)
        else:
            ga *= ind
    xc, yc = get_xy_dim_(cubeD)
    if ga is None:
        return cubeD.collapsed([xc, yc], iris.analysis.MEAN)
    else:
        return cubeD.collapsed([xc, yc], iris.analysis.MEAN, weights=ga)


def get_gwl_y0_(cube, gwl, pref=[1861, 1890]):
    """
    ... first year of 30-year window of global warming level ...

    Args:
        cube: CUBE of global surface temperature
         gwl: warming level compared to reference period pref
    """
    c = pSTAT_cube(cube if cube.ndim == 1 else rgMean_cube(cube), 'year')
    tref = extract_period_cube(c, *pref)
    tref = tref.collapsed('time', iris.analysis.MEAN).data

    def _G_tR(G, tR):
        if not isIter_(G):
            ind = np.where(rMEAN1d_(c.data, 30) >= G + tR)[0][0]
            return c.coord('year').points[ind]
        else:
            return [_G_tR(i, tR) for i in G]

    if c.ndim == 1:
        return _G_tR(gwl, tref)
    else:
        o = np.empty(tref.shape + np.array(gwl).shape)
        ax = c.coord_dims('year')[0]
        for i in range(nSlice_(c.shape, ax)):
            ind = ind_shape_i_(c.shape, i, ax)
            ind_ = ind_shape_i_(tref.shape, i, axis=None)
            ind__ = ind_shape_i_(o.shape, i,
                                 axis=-1 if np.array(gwl).shape else None)
            o[ind__] = np.array(_G_tR(gwl, tref[ind]))
        return o


def _inpolygons(poly, points, **kwArgs):
    if not isIter_(poly):
        ind = poly.contains_points(points, **kwArgs)
    elif len(poly) < 2:
        ind = poly[0].contains_points(points, **kwArgs)
    else:
        inds = [i.contains_points(points, **kwArgs) for i in poly]
        ind = np.logical_or.reduce(inds)
    return ind


def _isyx(cube):
    xc, yc = get_xy_dim_(cube)
    if xc is None:
        raise Exception("cube missing 'x' or 'y' coord")
    xcD, ycD = cube.coord_dims(xc)[0], cube.coord_dims(yc)[0]
    return ycD < xcD


def get_loa_pts_2d_(cube):
    """
    ... 2d longitude/latitude points (from coord or meshed) ...
    """
    lo_, la_ = get_loa_(cube)
    if lo_ is None or la_ is None:
        raise Exception("input cube(s) must have "
                        "longitude/latidute coords!")
    yx_ = _isyx(cube)
    if lo_.ndim != 2:
        if yx_:
            x, y = np.meshgrid(lo_.points, la_.points)
        else:
            y, x = np.meshgrid(la_.points, lo_.points)
    else:
        if yx_:
            x, y = lo_.points, la_.points
        else:
            x, y = lo_.points.T, la_.points.T
    return (x, y)


def inpolygons_cube(poly, cube, **kwArgs):
    x, y = get_loa_pts_2d_(cube)
    ind = _inpolygons(poly, np.vstack((x.ravel(), y.ravel())).T, **kwArgs)
    ind = robust_bc2_(ind.reshape(x.shape), cube.shape, get_xyd_cube(cube))
    return ind


def maskNaN_cube(cube):
    ind = np.isnan(cube.data)
    cube = iris.util.mask_cube(cube, ind)


def maskPOLY_cube(poly, cube, masked_out=True, **kwArgs):
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    ind = ~ind if masked_out else ind
    cube = iris.util.mask_cube(cube, ind)


def rgMean_poly_cube(cubeD, poly, sftlf=None, areacella=None, **kwArgs):
    #warnings.filterwarnings("ignore", category=UserWarning)
    ga = getGridAL_cube(cubeD, sftlf, areacella)
    ind = inpolygons_cube(poly, cubeD, **kwArgs)
    xc, yc = get_xy_dim_(cubeD)
    if ga is None:
        ga = ind * np.ones(ind.shape)
    else:
        ga = ga * ind
    return cubeD.collapsed([xc, yc],
                           iris.analysis.MEAN,
                           weights=ga)


def _rm_extra_coords_cubeL(cubeL):
    l0 = [[ii.name() for ii in i.aux_coords] for i in cubeL]
    l1 = ouniqL_(flt_l(l0))
    l2 = [i for i in l1 if sum(np.array(flt_l(l0))==i) < len(cubeL)]
    if len(l2) != 0:
        for i, ii in zip(cubeL, l0):
            for iii in l2:
                if iii in ii:
                    i.remove_coord(iii)


def _get_xycoords(cube):
    """
    ... get xy (spatial) coords ...
    """
    xycn = ['lon', 'x_coord', 'x-coord', 'x coord',
            'lat', 'y_coord', 'y-coord', 'y coord']
    xycoords = [coord for coord in cube.coords()
                if any([i in coord.name() for i in xycn])]
    return xycoords


def _unify_1coord_points(cubeL, coord_name, **close_kwArgs):
    epochs = {}
    emsg = "COORD {!r} can't be unified!".format(coord_name)
    emsg_ = "Bounds of COORD {!r} can't be unified!".format(coord_name)
    for c in cubeL:
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


def _unify_xycoord_points(cubeL, **close_kwArgs):
    ll_('cccc: _unify_xycoord_points() called')
    if len(cubeL) > 1:
        coord_names = [i.name() for i in _get_xycoords(cubeL[0])]
        for coord_name in coord_names:
            _unify_1coord_points(cubeL, coord_name, **close_kwArgs)


def _unify_1coord_attrs(cubeL, coord_name):
    attrs = ['long_name', 'var_name', 'attributes', 'coord_system']
    epochs = {}
    for c in cubeL:
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
        for c in cubeL:
            cc = c.coord(coord_name)


def _unify_coord_attrs(cubeL, coord_names=None):
    ll_('cccc: _unify_coord_attrs() called')
    if len(cubeL) > 1:
        coord_names = coord_names if coord_names else\
                      [i.name() for i in cubeL[0].coords()]
        for coord_name in coord_names:
            _unify_1coord_attrs(cubeL, coord_name)


def _unify_time_units(cubeL):
    CLD0 = 'proleptic_gregorian'
    CLD = 'gregorian'
    clds = [c.coord('time').units.calendar for c in cubeL]
    if len(ouniqL_(clds)) > 1:
        for c in cubeL:
            ctu = c.coord('time').units
            if ctu.calendar == CLD0:
                c.coord('time').units = cf_units.Unit(ctu.origin, CLD)
    iris.util.unify_time_units(cubeL)


def _unify_dtype(cubeL, fst=False):
    ll_('cccc: _unify_dtype() called')
    tps = [c.dtype for c in cubeL]
    if fst:
        tp = tps[0]
    else:
        utps = np.unique(tps)
        tpi = [np.sum(np.asarray(tps) == i) for i in utps]
        tp = utps[np.argmax(tpi)]
    for c in cubeL:
        if c.dtype != tp:
            c.data = c.data.astype(tp)


def _unify_cellmethods(cubeL, fst=True):
    ll_('cccc: _unify_cellmethods() called')
    cms = [c.cell_methods for c in cubeL]
    if fst:
        cm = cms[0]
    else:
        ucms = np.unique(cms)
        cmi = [np.sum(np.asarray(cms) == i) for i in ucms]
        cm = utps[np.argmax(cmi)]
    for c in cubeL:
        if c.cell_methods != cm:
            c.cell_methods = cm


def purefy_cubeL_(cubeL):
    """
    ... helpful when merge or concatenate CubeList ...
    """
    _rm_extra_coords_cubeL(cubeL)
    equalise_attributes(cubeL)
    _unify_time_units(cubeL)


def _collect_errCC(x):
    tmp = re.findall(r'(?<=\!\= ).+$', x)
    return tmp[0].split(', ') if tmp else tmp


def concat_cube_(cubeL, **close_kwArgs):
    """
    ... robust cube concatenator ...
    """
    purefy_cubeL_(cubeL)
    try:
        o = cubeL.concatenate_cube()
    except iris.exceptions.ConcatenateError as ce_:
        if any(['Data types' in i for i in ce_.args[0]]):
            _unify_dtype(cubeL)
        if any(['Cube metadata' in i for i in ce_.args[0]]):
            _unify_cellmethods(cubeL)
        if any(['coordinates metadata differ' in i for i in ce_.args[0]]):
            tmp = flt_l([_collect_errCC(i) for i in ce_.args[0]
                         if 'coordinates metadata differ' in i])
            if 'height' in tmp:
                ll_("cccc: set COORD 'height' points to those of cubeL[0]")
                _unify_1coord_points(cubeL, 'height', atol=10)
                tmp.remove('height')
            if len(tmp) > 0:
                _unify_coord_attrs(cubeL, tmp)
        try:
            o = cubeL.concatenate_cube()
        except iris.exceptions.ConcatenateError as ce_:
            if any(['Expected only a single cube' in i for i in ce_.args[0]]):
                _unify_xycoord_points(cubeL, **close_kwArgs)
            o = cubeL.concatenate_cube()
    return o


def merge_cube_(cubeL, **close_kwArgs):
    purefy_cubeL_(cubeL)
    try:
        o = cubeL.merge_cube()
    except iris.exceptions.MergeError as ce_:
        if any(['Data types' in i for i in ce_.args[0]]):
            _unify_dtype(cubeL)
        if any(['Cube metadata' in i for i in ce_.args[0]]):
            _unify_cellmethods(cubeL)
        if any(['coordinates metadata differ' in i for i in ce_.args[0]]):
            tmp = flt_l([_collect_errCC(i) for i in ce_.args[0]
                         if 'coordinates metadata differ' in i])
            if 'height' in tmp:
                ll_("cccc: set COORD 'height' points to those of cubeL[0]")
                _unify_1coord_points(cubeL, 'height', atol=10)
                tmp.remove('height')
            if len(tmp) > 0:
                _unify_coord_attrs(cubeL, tmp)
        try:
            o = cubeL.merge_cube()
        except iris.exceptions.MergeError as ce_:
            if any(['Expected only a single cube' in i for i in ce_.args[0]]):
                _unify_xycoord_points(cubeL, **close_kwArgs)
            o = cubeL.merge_cube()
    return o


def en_mxn_(eCube):
    """
    ... ensemble max of a cube (along dimcoord 'realization') ...
    """
    if eCube.coord_dims('realization'):
        a = en_max_(eCube)
        b = en_min_(eCube)
        o = a - b
        o.rename(a.name())
        return o


def en_min_(eCube):
    """
    ... ensemble max of a cube (along dimcoord 'realization') ...
    """
    if eCube.coord_dims('realization'):
        return eCube.collapsed('realization', iris.analysis.MIN)


def en_max_(eCube):
    """
    ... ensemble max of a cube (along dimcoord 'realization') ...
    """
    if eCube.coord_dims('realization'):
        return eCube.collapsed('realization', iris.analysis.MAX)


def en_mean_(eCube, **kwArgs):
    """
    ... ensemble mean of a cube (along dimcoord 'realization') ...
    """
    if eCube.coord_dims('realization'):
        return eCube.collapsed('realization', iris.analysis.MEAN, **kwArgs)


def en_iqr_(eCube):
    """
    ... ensemble interquartile range (IQR) of a cube (along dimcoord
        'realization') ...
    """
    if eCube.coord_dims('realization'):
        a = eCube.collapsed('realization', iris.analysis.PERCENTILE,
                            percent=75)
        b = eCube.collapsed('realization', iris.analysis.PERCENTILE,
                            percent=25)
        o = a - b
        o.rename(a.name())
        return o


def kde_cube(cube, **kde_opts):
    """
    ... kernal distribution estimate over all nomasked data ...
    """
    data = nanMask_(cube.data).flatten()
    data = data[~np.isnan(data)]
    data = data.astype(np.float64)
    return kde_(data, **kde_opts)


def _rip(cube):
    """
    ... get rxixpx from cube metadata ...
    """
    if 'parent_experiment_rip' in cube.attributes:
        return cube.attributes['parent_experiment_rip']
    elif 'driving_model_ensemble_member' in cube.attributes:
        return cube.attributes['driving_model_ensemble_member']
    else:
        return None


def en_rip_(cubeL):
    """
    ... ensemble cube over rxixpxs (along dimcoord 'realization') ...
    """
    for i, c in enumerate(cubeL):
        rip = _rip(c)
        rip = str(i) if rip is None else rip
        new_coord = _iAuxC(rip,
                           long_name='realization',
                           units='no_unit')
        c.add_aux_coord(new_coord)
        c.attributes = {}
    return cubeL.merge_cube()


def en_mm_cubeL_(cubeL, opt=0, cref=None):
    """
    ... make ensemble cube for multimodels ...
    
    kwArgs:
         opt:
             0: rgd_li_opt0_; try rgd_iris_ first then rgd_scipy_
             1: rgd_iris_
             2: rgd_scipy_
        cref: reference CUBE (default 1st CUBE in cubeL)

    useful info:
        >>> help(rgd_li_opt0_)
        >>> help(rgd_iris_)
        >>> help(rgd_scipy_)
    """
    from .rgd import rgd_scipy_, rgd_iris_, rgd_li_opt0_
    tmpD = {}
    cl = []
    for i, c in enumerate(cubeL):
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
    cl = _CubeList(cl)
    eCube = cl.merge_cube()
    return eCube


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


def initAnnualCube_(c0, y0y1,
                    name=None, units=None, var_name=None, long_name=None,
                    attrU=None, mmm='j-d'):
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
    c = extract_byAxes_(c0, 'time', np.s_[:ny])
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
    c.coord('time').units = cf_units.Unit('days since 1850-1-1',
                                           calendar='gregorian')
    m0, m1, y0_, y0__ = _mm01()
    y0_h = [datetime(i, m0, 1) for i in range(y0_, y0_ + ny)]
    y1_h = [datetime(i, m1, 1) for i in range(y0__, y0__ + ny)]
    tbnds = np.empty((ny, 2))
    tbnds[:, 0] = cf_units.date2num(y0_h,
                                    c.coord('time').units.origin,
                                    c.coord('time').units.calendar)
    tbnds[:, 1] = cf_units.date2num(y1_h,
                                    c.coord('time').units.origin,
                                    c.coord('time').units.calendar)
    tdata = np.mean(tbnds, axis=-1)
    c.coord('time').points = tdata
    c.coord('time').bounds = tbnds
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
        cube,
        *freq,
        stat='MEAN',
        valid_season=True,
        with_year=True,
        **stat_opts):
    """
    ... period statistic ...

    Args:
                cube: CUBE to be analyzed
                stat: eval('iris.analysis.{}'.format(stat))
                freq: frequency for statistic
    kwArgs:
        valid_season: if True and season mmm is crossing years, the 1st & end
                      year will be excluded 
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

    dd = dict(hour=(_ica.add_hour, ('time',), dict(name='hour')),
              day=(_ica.add_day_of_year, ('time',), dict(name='doy')),
              month=(_ica.add_month, ('time',), dict(name='month')),
              year=(_ica.add_year, ('time',), dict(name='year')),
              season=(_ica.add_season, ('time',), dict(name='season',
                                                      seasons=s4)),
              seasonyr=(seasonyr_cube, (s4,), dict(name='seasonyr')))

    def _x(f0):
        d = d_y if with_year else d_
        if f0 in d.keys():
            return d[f0]
        elif isSeason_(f0):
            return (f0, 'seasonyr')
        elif isMonth_(f0):
            return d['month']
        else:
            raise Exception(ef1.format(f0))

    def _xx(x):
        if isinstance(x, str):
            if x in d_.keys():
                return (dd.copy(), None)
            elif isSeason_(x):
                tmp = {x: (_ica.add_season_membership, ('time', x),
                           dict(name=x)),
                       'seasonyr': (seasonyr_cube, (x,),
                                    dict(name='seasonyr'))}
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
                    tmp = {f_[0]: (_ica.add_season_membership, ('time', f_[0]),
                                   dict(name=f_[0])),
                           'seasonyr': (seasonyr_cube, (f_[0],),
                                        dict(name='seasonyr'))}
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
            tmp = c.aggregated_by(dff, eval('iris.analysis.' + stat),
                                  **stat_opts)
            if isSeason_(mmm) and not ismono_(mmmN_(mmm)):
                tmp = extract_byAxes_(tmp, 'time', np.s_[1:-1])
        else:
            tmp = c.aggregated_by(dff, eval('iris.analysis.' + stat),
                                  **stat_opts)
            if x == 'season' and valid_season:
                tmp = extract_byAxes_(tmp, 'time', np.s_[1:-1])
        rm_t_aux_cube(tmp, keep=dff)
        return tmp

    freqs = ('year',) if len(freq) == 0 else freq
    o = ()
    for ff in [i.split('-') if '-' in i else i for i in freqs]:
        tmp = _xxx(cube.copy(), ff)
        o += (tmp,)
    return o[0] if len(o) == 1 else o


def repair_cs_(cube):
    def _repair_cs_cube(c):
        cs = c.coord_system('CoordSystem')
        if cs is not None:
            for k in cs.__dict__.keys():
                if eval('cs.' + k) is None:
                    exec('cs.' + k + ' = ""')
        for coord in c.coords():
            if coord.coord_system is not None:
                coord.coord_system = cs
    if isinstance(cube, _Cube):
        _repair_cs_cube(cube)
    elif isMyIter_(cube):
        for i in cube:
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


def repair_lccs_(cube):
    if isinstance(cube, _Cube):
        _repair_lccs_cube(cube)
    elif isMyIter_(cube):
        for i in cube:
            _repair_lccs_cube(i)


def lccs_m2km_(cube):
    if not isMyIter_(cube):
        if isinstance(cube, _Cube):
            cs = cube.coord_system('CoordSystem')
            if isinstance(cs, iris.coord_systems.LambertConformal):
                for coord in cube.coords():
                    if coord.coord_system is not None:
                        coord.convert_units('km')
    else:
        for i in cube:
            lccs_m2km_(i)


def cubesv_(cube, filename,
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
    ... save cube to nc with dim_t unlimitted ...
    """
    if isinstance(cube, _Cube):
        #repair_lccs_(cube) # pls. execute outside the function if necessary
        dms = [i.name() for i in cube.dim_coords]
        udm = ('time',) if 'time' in dms else None
        iris.save(cube, filename,
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
    elif isMyIter_(cube):
        for i, ii in enumerate(cube):
            ext = ext_(filename)
            cubesv_(ii, filename.replace(ext, '_{}{}'.format(i, ext)))


def half_grid_(x, side='i', axis=-1, loa=None, rb=360):
    """
    ... points between grids ...
    """
    dx = np.diff(x, axis=axis)
    if loa == 'lo':
        lb = rb - 360
        dx = rpt_(dx, 180, -180)
    tmp = extract_byAxes_(x, axis, np.s_[:-1]) + dx * .5
    if side in (0, 'i', 'inner'):
        o = tmp
    elif side in (-1, 'l', 'left'):
        o = np.concatenate((extract_byAxes_(x, axis, np.s_[:1]) -
                            extract_byAxes_(dx, axis, np.s_[:1]) * .5,
                            tmp),
                           axis=axis)
    elif side in (1, 'r', 'right'):
        o = np.concatenate((tmp,
                            extract_byAxes_(x, axis, np.s_[-1:]) +
                            extract_byAxes_(dx, axis, np.s_[-1:]) * .5),
                           axis=axis)
    elif side in (2, 'b', 'both'):
        o = np.concatenate((extract_byAxes_(x, axis, np.s_[:1]) -
                            extract_byAxes_(dx, axis, np.s_[:1]) * .5,
                            tmp,
                            extract_byAxes_(x, axis, np.s_[-1:]) +
                            extract_byAxes_(dx, axis, np.s_[-1:]) * .5),
                           axis=axis)
    else:
        raise ValueError("unknow value of side!")
    if loa == 'lo':
        o = rpt_(o, rb, lb)
    if loa == 'la':
        o = np.where(o > 90, 90, o)
        o = np.where(o < -90, -90, o)
    return o


def _ri1d(c1d, v):
    from skextremes.models.classic import GEV
    data = c1d.data
    data = data.compressed() if np.ma.isMaskedArray(data) else data
    if data.size:
        _gev = GEV(data)
        return _gev.return_periods(v)
    else:
        return np.nan


def ri_cube(cube, v, nmin=10):
    c = extract_byAxes_(cube, 'time', 0)
    rm_sc_cube(c)
    pst_(c, 'recurrence interval', units='year')
    ax = axT_cube(cube)
    if ax is None or cube.shape[ax] < nmin:
        emsg = "too few data for estimation!"
        raise Exception(emsg)
    ax_fn_mp_(cube, ax, _ri1d, c, v)
    return c


def nearest_point_cube(cube, longitude, latitude):
    x, y = get_loa_pts_2d_(cube)
    d_ = ind_shape_i_(x.shape,
                      np.argmin(haversine_(longitude, latitude, x, y)),
                      axis=None)
    xyd = get_xyd_cube(cube)
    ind = list(np.s_[:,] * cube.ndim)
    for i, ii in zip(xyd, d_):
        ind[i] = ii
    return cube[tuple(ind)]


def nine_points_cube(cube, longitude, latitude):
    x, y = get_loa_pts_2d_(cube)
    d_ = ind_shape_i_(x.shape,
                      np.argmin(haversine_(longitude, latitude, x, y)),
                      axis=None)
    xyd = get_xyd_cube(cube)
    ind_ = np.arange(-1, 2, dtype=np.int32)
    ind = list(np.s_[:,] * cube.ndim)
    wmsg = ("Causious that center point may be given outside (or at the "
            "boundary of) the geo domain of the input CUBE!")
    for i, ii in zip(xyd, d_):
        if ii == cube.shape[i] - 1:
            warnings.warn(wmsg)
            ii -= 1
        elif ii == 0:
            warnings.warn(wmsg)
            ii += 1
        ind[i] = ind_ + ii
    return cube[tuple(ind)]


def replace_coord_(cube, new_coord):
    """
    Replace the coordinate whose metadata matches the given coordinate.

    """
    old_coord = cube.coord(new_coord.name())
    dims = cube.coord_dims(old_coord)
    was_dimensioned = old_coord in cube.dim_coords
    cube._remove_coord(old_coord)
    if was_dimensioned and isinstance(new_coord, iris.coords.DimCoord):
        cube.add_dim_coord(new_coord, dims[0])
    else:
        cube.add_aux_coord(new_coord, dims)

    for factory in cube.aux_factories:
        factory.update(old_coord, new_coord)


def doy_f_cube(cube,
               f, fA_=(), fK_={},
               ws=None,
               mF=None,
               out=None,
               pp=False):
    """
    ... f(unction) for each doy ...

    kwArgs:
       fA_, fK_: Args, kwArgs along with CUBE data to be passed to f
             ws: window size
             mF: for replacing missing value
            out: CUBE for storing output (default derived from input CUBE)
             pp: print process status
    """

    ax_t = axT_cube(cube)
    yr_doy_cube(cube)
    doy_data = cube.coord('doy').points

    doy_ = np.unique(doy_data)
    if len(doy_) < 360:
        raise Exception('doy less than 360!')
    doy = np.arange(1, 367, dtype=np.int32)

    if out is None:
        out = extract_byAxes_(cube, ax_t, doy - 1)
        #select 2000 as it is a leap year...
        out.coord('time').units = cf_units.Unit(
                'days since 1850-1-1',
                calendar='gregorian'
                )
        d0 = cf_units.date2num(
                datetime(2000, 1, 1),
                out.coord('time').units.origin,
                out.coord('time').units.calendar
                )
        dimT = out.coord('time').copy(doy - 1 + d0)
        out.replace_coord(dimT)

    if pp:
        t0 = l__('0', _p=True)

    data_ = np.ma.filled(cube.data, mF) if mF is not None else cube.data
    if pp:
        ll_('releazing', t0=t0, _p=True)

    for i in doy:
        indw = ind_win_(doy_data, i, 15) if ws else np.isin(doy_data, i)
        ind = ind_s_(cube.ndim, ax_t, indw)
        f_kArgs.update(dict(axis=ax_t, keepdims=True))
        tmp = f(data_[ind], *fA_, **fK_)
        out.data[ind_s_(out.ndim, axT_cube(out), doy == i)] = tmp
        if pp:
            ll_('{}'.format(i), t0=t0, _p=True)

    return out


def pcorr_cube(x, y, z, **cck):                                                         
    assert x.shape == y.shape == z.shape                                        
    if 'corr_coords' not in cck:
        cck.update(dict(corr_coords='time'))
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
        or one cube should be broadcastable to the other.
    * corr_coords (str or list of str):
        The cube coordinate name(s) over which to calculate correlations. If no
        names are provided then correlation will be calculated over all common
        cube dimensions.
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
        variance for each cube is calculated from all available cells. Defaults
    * alpha (float, optional):
        If specified, a critical coorelation value (p=alpha) will be given
        along with the output cube
    Returns:
        A cube of the correlation between the two input cubes along the
        specified dimensions, at each point in the remaining dimensions of the
        cubes.

        For example providing two time/altitude/latitude/longitude cubes and
        corr_coords of 'latitude' and 'longitude' will result in a
        time/altitude cube describing the latitude/longitude (i.e. pattern)
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
        # Create a cube of 1's with a common mask.
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
        if isinstance(mask_cube, iris.cube.Cube):
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
        year,
        month,
        day,
        *hms,
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
    _unit_ = cf_units.Unit(unit, calendar=calendar)
    dnum = cf_units.date2num(datetime(year, month, day, *hms), unit, calendar)
    return iris.coords.AuxCoord(dnum, units=_unit_, standard_name='time')
