"""
>--#########################################################################--<
>--------------------------------xr functions---------------------------------<
>--#########################################################################--<
*
*
###############################################################################
            Author: Changgui Lin
            E-mail: changgui.lin@smhi.se
      Date created: 06.09.2019
Date last modified: 11.11.2020
"""

from xarray import (open_dataset, open_mfdataset, merge as xmerge,
                    Dataset as XDS, DataArray as XDA)

from wrf import ALL_TIMES, getvar, interplevel
from netCDF4 import Dataset as DS

import pandas as pd
import numpy as np
import os
import warnings

from .ffff import *
from .uuuu import *


__all__ = [
#------------------------------------------------------------------------------- general
        'x2iris_',
        'x2nc_',
        'xarea_weighted_',
        'xds_',
        'xmean_',
        'xmeanT_',
        'xsum_',
        'xsumT_',
        'xres_daily_',
        'xres_monthly_',
        'xres_seasonal_',
        'xres_annual_',
        'xresample_',
        'xu2u_',
        'xsmth_',
        'xdiv_',
        'xcurl_',
        'XDA',
        'XDS',
#------------------------------------------------------------------------------- gsod
        'gsod_uvds_',
        'gsod_attrs_',
        'gsod_single_',
        'gsod_',
#------------------------------------------------------------------------------- wrf, e5
        'ds_',
        'ds_close_',
        'DS',
        'getvar_',
        'cmg_read_',
        'interplevel'
        ]


#------------------------------------------------------------------------------- general
def x2iris_(da):
    o = da.assign_coords({i: getattr(da, i)
                          for i in da.dims if i not in da.coords})
    for i in _loa(o):
        try:
            i.attrs.update(units='degree')
        except:
            pass
    return o.to_iris()


def x2nc_(da, fn, **kwargs):
    try:
        da.to_netcdf(fn, **kwargs)
    except TypeError as _TE:
        try:
            import re
            _nm = re.findall(r"(?<=attr ')\w+(?=')", _TE.args[0])[0]
            da.attrs.update({_nm: str(da.attrs[_nm])})
            da.to_netcdf(fn, **kwargs)
        except:
            raise _TE


def xsmth_(da, m=9, n=9):
    ax_xy = _axXY(da)
    nS = nSlice_(da.shape, axis=ax_xy)
    data = da.data.copy()
    for i in range(nS):
        ind = ind_shape_i_(da.shape, i, axis=ax_xy)
        data[ind] = rMEAN2d_(_xy_slice(da, i).data, m, n, mode='same')
    return da.copy(data=data)


def xcurl_(uda, vda, rEARTH=6367470):
    xc, yc = _dimcXY(uda)
    if (hasattr(xc, 'units') and hasattr(yc, 'units') and
        cUnit(xc.units) == cUnit(yc.units) == cUnit('m')):
        du = np.gradient(uda.data, yc.data, axis=_axY(uda))
        dv = np.gradient(vda.data, xc.data, axis=_axX(vda))
    else:
        lo, la = _loa_pnts_2d(uda)
        ax_x, ax_y = (1, 0) if _isyx(uda) else (0, 1)
        du = np.gradient(uda.data, axis=_axY(uda))
        dv = np.gradient(vda.data, axis=_axX(vda))
        wx = (np.gradient(np.deg2rad(lo.astype(np.float64)), axis=ax_x) *
              np.cos(np.deg2rad(la.astype(np.float64))) *
              rEARTH)
        wy = np.gradient(np.deg2rad(la.astype(np.float64)), axis=ax_y) * rEARTH
        du, dv = du / wy, dv / wx
    o = uda.copy(data=dv - du)
    o = o.rename('curl')
    o.attrs.update(units=sqzUnit_(f"{uda.units} m-1"))
    return o


def xdiv_(uda, vda, rEARTH=6367470):
    xc, yc = _dimcXY(uda)
    if (hasattr(xc, 'units') and hasattr(yc, 'units') and
        cUnit(xc.units) == cUnit(yc.units) == cUnit('m')):
        du = np.gradient(uda.data, xc.data, axis=_axX(uda))
        dv = np.gradient(vda.data, yc.data, axis=_axY(vda))
    else:
        lo, la = _loa_pnts_2d(uda)
        ax_x, ax_y = (1, 0) if _isyx(uda) else (0, 1)
        du = np.gradient(uda.data, axis=_axX(uda))
        dv = np.gradient(vda.data, axis=_axY(vda))
        wx = (np.gradient(np.deg2rad(lo.astype(np.float64)), axis=ax_x) *
              np.cos(np.deg2rad(la.astype(np.float64))) *
              rEARTH)
        wy = np.gradient(np.deg2rad(la.astype(np.float64)), axis=ax_y) * rEARTH
        du, dv = du / wx, dv / wy
    o = uda.copy(data=du + dv)
    o = o.rename('divergence')
    o.attrs.update(units=sqzUnit_(f"{uda.units} m-1"))
    return o


def xds_(fn, **kwargs):
    if isIter_(fn):
        axnm = _dimT(open_dataset(fn[0], **kwargs))
        return open_mfdataset(fn, concat_dim=axnm, combine='nested', **kwargs)
    else:
        return open_dataset(fn, **kwargs)


def xarea_weighted_(da, normalize=False, rEARTH=6367470, llonly=True):
    pA_ = None
    if llonly:
        pA_ = flt_l(
                [(i, 0) for i in da.dims if i not in(_dimX(da), _dimY(da))],
                nx=1,
                )
    if pA_:
        tmp = _extract_byAxes(da, *pA_)
    else:
        tmp = da
    data=_area_weights(tmp,
                       normalize=normalize,
                       rEARTH=rEARTH,
                       llonly=llonly)
    aw = XDA(data=data,
             coords=tmp.coords,
             dims=tmp.dims,
             name='weights')
    return da.weighted(aw)


#-- ccxx ----------------------------------------------------------------------
#-- _dim ----------------------------------------------------------------------
nmXs = ('x-coord', 'x_coord', 'x coord', 'lon', 'west_east')
nmYs = ('y-coord', 'y_coord', 'y coord', 'lat', 'south_north')
nmZs = ('z-coord', 'z_coord', 'z coord', 'hgt', 'height', 'bottom_top',
                'press')
nmTs = ('date', 'time', 'day', 'month', 'season', 'year', 'second', 'minute')

def _dimT(da):
    return _dim(da, 'T')

def _dimX(da):
    return _dim(da, 'X')

def _dimY(da):
    return _dim(da, 'Y')

def _dimZ(da):
    return _dim(da, 'Z')

def _dim(da, axis):
    if isinstance(axis, str):
        if len(axis) == 1 and axis.upper() in 'TXYZ':
            nms = eval(f"nm{axis.upper()}s")
            for i in da.dims:
                if (any(ii in i.lower() for ii in nms)
                    or i.upper() == axis.upper()):
                    return i
        else:
            return tuple(_dim(da, i) for i in axis)
    else:
        return da.dims[rpt_(axis, da.ndim)]

#-- _dimc ---------------------------------------------------------------------
def _dimc(da, axis):
    if not isinstance(da, XDA):
        emsg = "'da' is not instance of 'xarray.DataArray'"
        raise TypeError(emsg)
    if isinstance(axis, str):
        if len(axis) == 1:
            if axis.upper() in 'TXYZ':
                o = eval(f"_dim{axis.upper()}(da)")
                return getattr(da, o) if o and hasattr(da, o) else None
            else:
                return None
        else:
            return tuple(_dimc(da, i) for i in axis)
    else:
        axis = rpt_(axis, da.ndim)
        return tryattr_(da, da.dims[axis])

def _dimcT(da):
    return _dimc(da, 'T')

def _dimcX(da):
    return _dimc(da, 'X')

def _dimcY(da):
    return _dimc(da, 'Y')

def _dimcZ(da):
    return _dimc(da, 'Z')

def _dimcXY(da):
    return (_dimcX(da), _dimcY(da))

#-- _ax -----------------------------------------------------------------------
def _ax(da, axis):
    if not isinstance(da, XDA):
        msg = "'da' is not instance of 'xarray.DataArray'"
        raise TypeError(msg)
    if not isinstance(axis, str):
        msg = "'axis' should be a string"
        raise TypeError(msg)
    if len(axis) == 1:
        o = eval(f"_dim{axis.upper()}(da)")
        return da.dims.index(o) if o else None
    elif axis in da.coords:
        cdims = getattr(da, axis).dims
        return tuple(da.dims.index(i) for i in cdims)

def _axT(da):
    return _ax(da, 'T')

def _axX(da):
    return _ax(da, 'X')

def _axY(da):
    return _ax(da, 'Y')

def _axZ(da):
    return _ax(da, 'Z')

def _axXY(da):
    axXY = [_axX(da), _axY(da)]
    if all(i is not None for i in axXY):
        return tuple(sorted(axXY))

def _isyx(da):
    if _axXY(da) is not None:
        return _axY(da) < _axX(da)
    else:
        msg = f"Dimension 'X' or 'Y' does not exist in DataArray {da.name!r}"
        raise Exception(msg)

#-- _guessbnds ----------------------------------------------------------------
def _guessXYZT(coord):
    for i in 'XYZT':
        nms = eval(f"nm{i}s")
        if any(ii in coord.name.lower() for ii in nms):
            return i

def _guessLOA(coord):
    for i in ('lo', 'la'):
        if i in coord.name.lower():
            return i

def _guessbnds(da, coord, **kwargs):
    hgKA = dict(loa=_guessLOA(coord))
    hgKA.update(kwargs)
    data = coord.data
    data = data.compute() if hasattr(data, 'compute') else data
    if coord.ndim == 1:
        lb = half_grid_(data, side='l', **hgKA)
        rb = half_grid_(data, side='r', **hgKA)
    else:
        _xyzt = _guessXYZT(coord)
        ax = _ax(da, _xyzt)
        axs = _ax(da, coord.name)
        axincoord = axs.index(ax)
        lb = half_grid_(data, side='l', axis=axincoord, **hgKA)
        rb = half_grid_(data, side='r', axis=axincoord, **hgKA)
    return np.stack((lb, rb), axis=-1)

#-- _loa ----------------------------------------------------------------------
def _loa(da):
    def _f(s):
        for i in da.coords:
            if s in i.lower():
                return getattr(da, i)
    return (_f('lon'), _f('lat'))

def _axLOA(da):
    _f = lambda x: tuple(da.dims.index(i) for i in x.dims)
    return tuple(_f(i) for i in _loa(da))

def _loa_pnts(da):
    def _f(s):
        for i in da.coords:
            if s in i.lower():
                data = getattr(da, i).data
                return data.compute() if hasattr(data, 'compute') else data
    return (_f('lon'), _f('lat'))

def _loa_bnds(da):
    def _f(s):
        for i in da.coords:
            if s in i.lower():
                return _guessbnds(da, getattr(da, i), loa=s[:2])
    return (_f('lon'), _f('lat'))

def _loa_pnts_2d(da):
    lo, la = _loa_pnts(da)
    if lo is None or la is None:
        emsg = f"DataArray {da.name!r} must have longitude/latidute coords!"
        raise Exception(emsg)
    return loa2d_(lo, la, isYX=_isyx(da))

#-- _msk ----------------------------------------------------------------------
def _ind_clim(da, **kwargs):
    shp = da.shape
    cnms = [i for i in da.coords]
    _d = lambda x: _ax(da, x)
    _tmp = [_d(k) for k in kwargs.keys() if k in cnms]
    uds = [tuple(i) for i in ss_fr_sl_(_tmp)]
    def _r(_k):                                                                # right bounds for longitude
        if ('lo' in _k and
            any(i in getattr(da, _k).units) for i in ('degree', 'radian')):
            return 360
    def _f(_k, ud):                                                            # derive ind for a given coord and an unique dimension
        _shp = tuple(shp[i] for i in ud)
        udD = {ii:i for i, ii in enumerate(ud)}
        _coord = getattr(da, _k)
        _dims = _d(_k)
        ax = tuple(udD[i] for i in _dims)
        r_ = _r(_coord)
        b0, b1 = np.moveaxis(_guessbnds(da, _coord), -1, 0)
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

def _omsk_clim(da, to_ind=False, **kwargs):
    shp = da.shape
    cnms = [i for i in da.coords]
    _d = lambda x: _ax(da, x)
    def _r(_k):
        if ('lo' in _k and
            any(i in getattr(da, _k).units) for i in ('degree', 'radian')):
            return 360
    def _f1(_k):
        _coord = getattr(da, _k)
        _dims = _d(_k)
        r_ = _r(_coord)
        b0, b1 = np.moveaxis(_guessbnds(da, _coord), -1, 0)
        _b0 = ind_inRange_(b0, *kwargs[_k], r_=r_)
        _b1 = ind_inRange_(b1, *kwargs[_k], r_=r_)
        return robust_bc2_(
                np.logical_and(_b0, _b1),
                shp,
                _dims,
                )
    booL = []
    for k in kwargs.keys():
        if k in cnms:
            booL.append(_f1(k))
    o = np.logical_and.reduce(booL)
    return bA2ind_(o) if to_ind else o

def _ind_loalim(da, longitude=None, latitude=None):
#-- in_loalim_(lo, la, shp, axXY=None, lolim=None, lalim=None, isYX=True)
    return in_loalim_(*_loa_pnts(da),
                      da.shape,
                      axXY=_axXY(da),
                      lolim=longitude,
                      lalim=latitude,
                      isYX=_isyx(da),
                      )

def _ind_poly(da, poly, **kwArgs):
    x, y = _loa_pnts_2d(da)
    ind = in_polygons_(poly, np.vstack((x.ravel(), y.ravel())).T, **kwArgs)
    ind = robust_bc2_(ind.reshape(x.shape), da.shape, _axXY(da))
    return ind

def _where_not_msk(da, omsk, **kwargs):
    return da.where(omsk, **kwargs)

#-- _extract_byAxes -----------------------------------------------------------
def _extract_byAxes(da, axis, sl_i, *vArg):
    """
    ... extract by providing selection along axis/axes ...

    Args:
          da: parent XDA
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

    if isinstance(da, XDA):
        ax = [_ax(da, i)[0] if isinstance(i, str) else i for i in ax]
    return extract_(da, *(i for ii in zip(ax, sl) for i in ii), fancy=False)

#-- _aw -----------------------------------------------------------------------
def _area_weights(da, normalize=False, rEARTH=6367470, llonly=True):
    """
    ... revised iris.analysis.cartography.area_weights to ignore lon/lat in
        auxcoords ...
    """
    lon, lat = _loa(da)                                                        # Get the lon and lat coords and axes
    if any(i is None for i in (lon, lat)):
        msg = "Cannot get latitude/longitude coordinates from CUBE {!r}!"
        raise ValueError(msg.format(da.name))
    if lon.ndim == lat.ndim == 1:                                              # axes for the weights to be broadcasted
        axes = (_axY(da), _axX(da))
    elif lon.shape == lat.shape:
        axes = _ax(da, lat.name)

    for coord in (lat, lon):                                                   # check units
        if not any(i in coord.units.lower() for i in ('degree', 'radian')):
            msg = ("Units of degrees or radians required, coordinate "
                   f"{coord.name()!r} has units: {coord.units.name!r}")
            raise ValueError(msg)

    lob, lab = _loa_bnds(da)                                                   # Create 2D weights from bounds
    lob = cUnit(lon.units).convert(lob, 'radian')
    lab = cUnit(lat.units).convert(lab, 'radian')
    ll_weights = aw_loa_bnds_(lob, lab, rEARTH=rEARTH)                         # Use the geographical area as the weight for each cell

    if normalize:                                                              # Normalize the weights if necessary
        ll_weights /= ll_weights.sum()

                                                                               # Now we create an array of weights for each cell. This process will
                                                                               # handle adding the required extra dimensions and also take care of
                                                                               # the order of dimensions.
    if llonly:
        return ll_weights if isincr_(axes) else ll_weights.T
    else:
        return robust_bc2_(ll_weights, da.shape, axes=axes)

#-- _rg -----------------------------------------------------------------------
def _rg_func(da, func, rg=None, inv=False, **funcD):
    tmp = ((_where_not_msk(da, ~_ind_loalim(da, **rg)) if inv else
            _where_not_msk(da, _ind_loalim(da, **rg)))
           if rg else da.copy())
    tmp = xarea_weighted_(tmp)
    return tmp.reduce(func, dim=_dim(da, 'XY'), **funcD)

def _rg_mean(da, rg=None, inv=False, **funcD):
    tmp = ((_where_not_msk(da, ~_ind_loalim(da, **rg)) if inv else
            _where_not_msk(da, _ind_loalim(da, **rg)))
           if rg else da.copy())
    tmp = xarea_weighted_(tmp)
    return tmp.mean(dim=_dim(da, 'XY'), **funcD)

def _poly_func(da, poly, func, inpolyKA={}, inv=False, **funcD):
    ind = _ind_poly(da, poly, **inpolyKA)
    tmp = _where_not_msk(da, ~ind) if inv else _where_not_msk(da, ind)
    tmp = xarea_weighted_(tmp)
    return tmp.reduce(func, dim=_dim(da, 'XY'), **funcD)

def _poly_mean(da, poly, inpolyKA={}, **funcD):
    ind = _ind_poly(da, poly, **inpolyKA)
    tmp = _where_not_msk(da, ~ind) if inv else _where_not_msk(da, ind)
    tmp = xarea_weighted_(tmp)
    return tmp.mean(dim=_dim(da, 'XY'), **funcD)

#-- _xy_slice -----------------------------------------------------------------
def _xy_slice(da, i=0):
    ax_xy = _axXY(da)
    ind = ind_shape_i_(da.shape, i, axis=ax_xy)
    return da[ind]

#-- ccxx ----------------------------------------------------------------------


def xresample_(ds, ff='D', nn=None):
    '''
    ... resample xarray dataset (temporally) ...
    inputs:
        ds: xarray.Dataset with a 'date' dim
        ff: target frequency
            'D': daily, equal to xres_daily_
            'MS': monthly, equal to xres_monthly_
            'QS-DEC':, seasonal, equal to xres_seasonal_
            'YS': annual, equal to xres_annual_
            and others
        nn: min number of samples for output valid values
    output:
        same type as ds
    '''
    tD = {_dimT(ds): ff}
    dsr = ds.resample(**tD)
    o = dsr.mean()
    if nn:
        tmp = dsr.count()
        o = o.where(tmp >= nn)
    return o


def xres_daily_(ds, nn=None):
    return xresample_(ds, ff='D', nn=nn)


def xres_monthly_(ds, nn=None):
    return xresample_(ds, ff='MS', nn=nn)


def xres_seasonal_(ds, nn=None):
    return xresample_(ds, ff='QS-DEC', nn=nn)


def xres_annual_(ds, nn=None):
    return xresample_(ds, ff='YS', nn=nn)


def xmean_(da, dim=None, add_bounds=False, **kwargs):
    if dim is None:
        return da.mean(**kwargs)
    else:
        o = da.mean(dim=dim, **kwargs)
        if isinstance(dim, str):
            dim = [dim]
        _coords = {}
        for _c in da.coords:
            _dims = [i for i in dim if i in da[_c].dims]
            if _dims:
                _cda = da[_c].mean(dim=_dims, **kwargs)
                if add_bounds:
                    _cda = _cda.assign_attrs(dict(
                        bounds=np.asarray(
                            [da[_c].data.min(), da[_c].data.max()],
                            dtype=da[_c].data.dtype,
                            )))
                _coords.update({_c: _cda})
        for _k in _coords:
            if _k in o.coords:
                o[_k] = _coords[_k]
            else:
                o = o.assign_coords({_k: _coords[_k]})
        return o


def xsum_(da, dim=None, add_bounds=False, **kwargs):
    if dim is None:
        return da.sum(**kwargs)
    else:
        o = da.sum(dim=dim, **kwargs)
        if isinstance(dim, str):
            dim = [dim]
        _coords = {}
        for _c in da.coords:
            _dims = [i for i in dim if i in da[_c].dims]
            if _dims:
                _cda = da[_c].mean(dim=_dims, **kwargs)
                if add_bounds:
                    _cda = _cda.assign_attrs(dict(
                        bounds=np.asarray(
                            [da[_c].data.min(), da[_c].data.max()],
                            dtype=da[_c].data.dtype,
                            )))
                _coords.update({_c: _cda})
        for _k in _coords:
            if _k in o.coords:
                o[_k] = _coords[_k]
            else:
                o = o.assign_coords({_k: _coords[_k]})
        return o


def xmeanT_(da,
            add_bounds=False,
            keep_attrs=True,
            keepdims=True,
            **kwargs):
    kwargs.update(dict(keep_attrs=keep_attrs, keepdims=keepdims))
    return xmean_(da, dim=_dimT(da), add_bounds=add_bounds, **kwargs)


def xsumT_(da,
            add_bounds=False,
            keep_attrs=True,
            keepdims=True,
            **kwargs):
    kwargs.update(dict(keep_attrs=keep_attrs, keepdims=keepdims))
    return xsum_(da, dim=_dimT(da), add_bounds=add_bounds, **kwargs)


def xu2u_(da, other, **kwargs):
    uold = cUnit(da.units)
    if not uold.is_convertible(other):
        wmsg = f"{uold!r} is not convertible to {other!r}; Nothing changed!"
        warnings.warn(wmsg)
    else:
        data_old = da.data
        if hasattr(da.data, 'compute'):
            import dask.array as _da
            data = _da.map_blocks(u2u_, data_old, uold, other, **kwargs)
        else:
            data = u2u_(data_old, uold, other, **kwargs)
        da_new = da.copy(data=data)
        da_new.attrs.update(dict(units=other))
        return da_new


#------------------------------------------------------------------------------- gsod
def _missing_to_nan_column(df, name, value):
    df[name] = df[name].replace(value, np.nan)

GSOD_MV = dict(
        temp=(9999.9, dict(units='K')),
        dewp=(9999.9, dict(units='K')),
        slp=(9999.9, dict(units='mb')),
        stp=(9999.9, dict(units='mb')),
        visib=(999.9, dict(units='km')),
        wdsp=(999.9, dict(units='m/s')),
        mxspd=(999.9, dict(units='m/s')),
        gust=(999.9, dict(units='m/s')),
        max=(9999.9, dict(units='K')),
        min=(9999.9, dict(units='K')),
        prcp=(99.99, dict(units='mm')),
        sndp=(999.9, dict(units='mm'))
        )

GSOD_VAR = [
        'temp', 'temp_attributes',
        'dewp', 'dewp_attributes',
        'slp', 'slp_attributes',
        'stp', 'stp_attributes',
        'visib', 'visib_attributes',
        'wdsp', 'wdsp_attributes',
        'mxspd',
        'gust',
        'max', 'max_attributes',
        'min', 'min_attributes',
        'prcp', 'prcp_attributes',
        'sndp',
        'frshtt']

def _stp(x, y):
    z = x.copy()
    z.loc[x<100] += 1000
    z.loc[y==0] = np.nan
    return z

def _f2k(x):
    return (x - 32) * 5 / 9 + 273.15

def _knot2ms(x):
    return x * 0.5144444444444445

def _mi2km(x):
    return x * 1.609344

def _in2mm(x):
    return x * 25.400000000000002


def gsod_uvds_(ds):
    uwind, vwind = windds2uv_(ds.winds, ds.windd)
    uwind.assign_attrs(units='m/s', standard_name='u-component wind')
    vwind.assign_attrs(units='m/s', standard_name='v-component wind')
    o = ds.assign(uwind=uwind, vwind=vwind)
    return o.drop_vars('windd')


def gsod_attrs_(csvfile):
    o = pd.read_csv(csvfile, dtype={'STATION': str}).rename(
            str.lower, axis='columns')
    attrs = dict(
            usaf=o.station[0][:6],
            wban=o.station[0][-5:],
            name=o.name[0],
            lat=o.latitude[0],
            lon=o.longitude[0],
            elv=o.elevation[0]
            )
    return attrs


def gsod_single_(csvfile, variables=None, _attrs=False):
    o = pd.read_csv(csvfile, dtype={'STATION': str}).rename(
            str.lower, axis='columns')
    o['date'] = pd.to_datetime(o.date)
    o = o.set_index('date')
    if _attrs:
        attrs = dict(
                usaf=o.station[0][:6],
                wban=o.station[0][-5:],
                name=o.name[0],
                lat=o.latitude[0],
                lon=o.longitude[0],
                elv=o.elevation[0]
                )
    o = o.drop(
            columns=['station', 'latitude', 'longitude', 'elevation', 'name']
            )
    if variables is None:
        variables = set(o.columns)
    else:
        assert set(variables).issubset(o.columns)
    for cc in variables:
        if cc in GSOD_MV.keys():
            _missing_to_nan_column(o, cc, GSOD_MV[cc][0])
    for cc in ['temp', 'dewp', 'max', 'min']:
        if cc in o.columns:
            o[cc] = _f2k(o[cc])
    for cc in ['wdsp', 'mxspd', 'gust']:
        if cc in o.columns:
            o[cc] = _knot2ms(o[cc])
    if 'visib' in o.columns:
        o['visib'] = _mi2km(o['visib'])
    if 'stp' in o.columns:
        o['stp'] = _stp(o['stp'], o['stp_attributes'])
    for cc in ['prcp', 'sndp']:
        if cc in o.columns:
            o[cc] = _in2mm(o[cc])
    o = o[variables]
    return (o, attrs) if _attrs else o


def gsod_(csvfiles=None,
          root=None, usaf=None, wban='99999', ys=None, ye=None,
          variables=None, _attrs=False):
    import glob
    import re
    if csvfiles is not None:
        if isinstance(csvfiles, str):
            o = gsod_single_(csvfiles, variables=variables)
            if _attrs:
                attrs = gsod_attrs_(csvfiles)
        elif isIter_(csvfiles, xi=str):
            o = [gsod_single_(i, variables=variables) for i in csvfiles]
            o = pd.concat(o)
            if _attrs:
                attrs = gsod_attrs_(csvfiles[0])
        else:
            raise TypeError("unknown 1st argin!!!")
        o = o.to_xarray()
        if _attrs:
            o = o.assign_attrs(attrs)
        for cc in o.data_vars:
            if cc in GSOD_MV.keys():
                o[cc] = o[cc].assign_attrs(**GSOD_MV[cc][1])
        return o
    elif all([i is not None for i in (root, usaf)]):
        ggstr = os.path.join(root, '*', '{:0>6}{:0>5}.csv'.format(usaf, wban))
        fns = sorted(glob.glob(ggstr))
        def _year_fr_fn(fn):
            return int(os.path.basename(fn)[:4])
        if ys is not None and ye is not None:
            assert ys <= ye
            fns = [i for i in fns if _year_fr_fn(i) in range(ys, ye + 1)]
        if fns:
            return gsod_(csvfiles=fns, variables=variables, _attrs=_attrs)
        else:
            raise Exception("no file found!!!")
    else:
        raise Exception("You may know more what is going wrong than me!!!")


#------------------------------------------------------------------------------- wrf, e5, modis
def ds_(fn):
    if not isIter_(fn):
        return DS(fn)
    else:
        return [ds_(i) for i in fn]


def ds_close_(ds):
    if isinstance(ds, DS):
        ds.close()
    elif isIter_(ds):
        for i in ds:
            ds_close_(i)


def getvar_(fn, var, _close=False):
    _fn = fn[0] if isIter_(fn) else fn
    if isinstance(_fn, str):
        arg0 = ds_(fn)
    elif isinstance(_fn, DS):
        arg0 = fn
    else:
        raise ValueError("check 1st input argument!")
    o = getvar(arg0, var, timeidx=ALL_TIMES, method='cat')
    if _close:
        ds_close_(arg0)
    for i in ['XLONG', 'XLAT', 'XLONG_M', 'XLAT_M']:
        coord = tryattr_(o, i)
        if coord is not None:
            coord.attrs.update(dict(units='degree'))
    return o


def cmg_read_(fn, rg={}):
    def _lo(ds):
        n = ds.x.size
        dx = 180/n
        b0 = tryattr_(ds, 'WESTHBOUNDINGCOORDINATE')
        b0 = b0 if b0 else -180
        return XDA(
                data=np.linspace(dx + b0, b0 + 360 - dx, n),
                name='lon',
                dims=('x',),
                attrs=dict(long_name='longitude', units='degree'),
                )
    def _la(ds):
        n = ds.y.size
        dy = 90/n
        b0 = tryattr_(ds, 'SOUTHBOUNDINGCOORDINATE')
        b0 = b0 if b0 else -90
        return XDA(
                data=np.linspace(b0 + 180 - dy, dy + b0, n),
                name='lay',
                dims=('y',),
                attrs=dict(long_name='latitude', units='degree'),
                )
    def _time(fn):
        import re
        tmp = re.findall("A\d{7}", fn)
        if tmp:
            yyyydoy = tmp[0][1:]
            data = np.array([np.datetime64(doy2date_(yyyydoy))],
                            dtype='datetime64[ns]')
            return XDA(
                    data=data,
                    name='time',
                    dims=('time',),
                    )
    if isinstance(fn, str):
        ds = xds_(fn, engine='rasterio')
        KA = dict(
                lon=_lo(ds),
                lat=_la(ds),
                )
        if _time(fn):
            KA.update(dict(time=_time(fn)))
        #o = ds.expand_dims(dim="time", axis=0)
        o = ds.drop('band')
        o = o.rename_dims(dict(band='time'))
        o = o.assign_coords(**KA)
        if rg:
            iselKA={}
            if 'longitude' in rg:
                iselKA.update(dict(
                    x=np.where(ind_inRange_(
                        o.lon.data,
                        *rg['longitude'],
                        r_=360,
                        ))[0]
                    ))
            if 'latitude' in rg:
                iselKA.update(dict(
                    y=np.where(ind_inRange_(
                        o.lat.data,
                        *rg['latitude'],
                        ))[0]
                    ))
            o = o.isel(**iselKA)
        return o
    elif isIter_(fn, xi=str):
        return xmerge([cmg_read_(i, rg=rg) for i in fn])
