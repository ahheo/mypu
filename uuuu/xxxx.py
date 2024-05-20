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

from xarray import (open_dataset, open_mfdataset,
                    Dataset as XDS, DataArray as XDA)

from wrf import ALL_TIMES, getvar
from netCDF4 import Dataset as DS

import pandas as pd
import numpy as np
import os
import warnings

from .ffff import *
from .uuuu import *


__all__ = [
#------------------------------------------------------------------------------- general
        'x2nc_',
        'xax_',
        'xaxT_',
        'xaxX_',
        'xaxY_',
        'xaxZ_',
        'xdimT_',
        'xdimX_',
        'xdimY_',
        'xdimZ_',
        'xds_',
        'xmean_',
        'xmeanT_',
        'xres_daily_',
        'xres_monthly_',
        'xres_seasonal_',
        'xres_annual_',
        'xresample_',
        'xu2u_',
        'XDA',
        'XDS',
#------------------------------------------------------------------------------- gsod
        'gsod_uvds_',
        'gsod_attrs_',
        'gsod_single_',
        'gsod_',
#------------------------------------------------------------------------------- wrf, e5
        'ds_',
        'DS',
        'getvar_',
        'hgt_wrf_',
        'hgt_e5_',
        'lu_wrf_',
        'tp_wrf_',
        'tp_e5_',
        's2a_',
        'swtp_',
        'rho_wrf_',
        'zstag_wrf_',
        'wvt_wrf_',
        'loa_',
        ]


#------------------------------------------------------------------------------- general
def x2nc_(da, fn):
    try:
        da.to_netcdf(fn)
    except TypeError as _TE:
        try:
            _nm = re.findall(r"(?<=attr ')\w+(?=')", _TE.args[0])[0]
            da.attrs.update({_nm: str(da.attrs[_nm])})
            da.to_netcdf(fn)
        except:
            raise _TE


def xds_(fn):
    if isIter_(fn):
        _dimT = xdimT_(open_dataset(fn[0]))
        return open_mfdataset(fn, concat_dim=_dimT, combine='nested')
    else:
        return open_dataset(fn)


def xdimT_(da):
    for i in da.dims:
        if i.lower() in ('time', 'date', 'datetime'):
            return i

def xdimX_(da):
    for i in da.dims:
        if any(ii in i.lower() for ii in ('x', 'lon', 'west_east')):
            return i


def xdimY_(da):
    for i in da.dims:
        if any(ii in i.lower() for ii in ('y', 'lat', 'south_north')):
            return i


def xdimZ_(da):
    for i in da.dims:
        if any(ii in i.lower() for ii in ('z', 'hgt', 'height', 'bottom_top',
                                          'press')):
            return i


def xax_(da, axis):
    if not isinstance(da, XDA):
        emsg = "'da' is not instance of 'xarray.DataArray'"
        raise TypeError(emsg)
    _dim = eval(f"xdim{axis}_(da)")
    return da.dims.index(_dim) if _dim else None


xaxT_ = lambda da: xax_(da, 'T')

xaxX_ = lambda da: xax_(da, 'X')

xaxY_ = lambda da: xax_(da, 'Y')

xaxZ_ = lambda da: xax_(da, 'Z')


def xresample_(ds, ff='D', nn=None):
    '''
    ... resample xarray dataset (temporally) ...
    inputs:
        ds: xarray.Dataset with a 'date' dim
        ff: target frequency
            'D': daily, equal to xres_daily_
            'M': monthly, equal to xres_monthly_
            'QS-DEC':, seasonal, equal to xres_seasonal_
            'A': annual, equal to xres_annual_
            and others
        nn: min number of samples for output valid values
    output:
        same type as ds
    '''
    tD = {xdimT_(da): ff}
    dsr = ds.resample(**tD)
    o = dsr.mean()
    if nn:
        tmp = dsr.count()
        o = o.where(tmp >= nn)
    return o


def xres_daily_(ds, nn=None):
    return xresample_(ds, ff='D', nn=nn)


def xres_monthly_(ds, nn=None):
    return xresample_(ds, ff='M', nn=nn)


def xres_seasonal_(ds, nn=None):
    return xresample_(ds, ff='QS-DEC', nn=nn)


def xres_annual_(ds, nn=None):
    return xresample_(ds, ff='A', nn=nn)


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


def xmeanT_(da,
            add_bounds=False,
            keep_attrs=True,
            keepdims=True,
            **kwargs):
    kwargs.update(dict(keep_attrs=keep_attrs, keepdims=keepdims))
    return xmean_(da, dim=xdimT_(da), add_bounds=add_bounds, **kwargs)


def xu2u_(da, other, **kwargs):
    uold = cUnit(da.units)
    if not uold.is_convertible(other):
        wmsg = f"{uold!r} is not convertible to {other!r}; Nothing changed!"
        warnings.warn(wmsg)
    else:
        data_old = da.data
        if hasattr(da.data, 'compute'):
            import dask.array as _da
            da.data = _da.map_blocks(u2u_, data_old, uold, other, **kwargs)
        else:
            da.data = u2u_(data_old, uold, other, **kwargs)
        da.attrs.update(dict(units=other))


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


#------------------------------------------------------------------------------- wrf, e5
def ds_(fn):
    if not isIter_(fn):
        return DS(fn)
    else:
        return [ds_(i) for i in fn]


def getvar_(fn, var):
    _fn = fn[0] if isIter_(fn) else fn
    if isinstance(_fn, str):
        arg0 = ds_(fn)
    elif isinstance(_fn, DS):
        arg0 = fn
    else:
        raise ValueError("check 1st input argument!")
    return getvar(arg0, var, timeidx=ALL_TIMES, method='cat')


def hgt_wrf_(fn):
    return getvar_(fn, 'HGT_M')


def lu_wrf_(fn):
    return getvar_(fn, 'LU_INDEX')


def hgt_e5_(fn):
    o = xds_(fn).z
    o.data /= 9.81
    o.attrs.update(dict(units='m', long_name='height', standard_name='height'))
    return o


def tp_wrf_(fn0, fn1, ind=np.s_[:]):
    ds0 = ds_(fn0)
    ds1 = ds_(fn1)
    o = getvar_(ds1, 'RAINC')[ind]
    o.data += getvar_(ds1, 'RAINNC')[ind].data
    o.data += getvar_(ds1, 'RAINSH')[ind].data
    o.data -= getvar_(ds0, 'RAINC')[ind].data
    o.data -= getvar_(ds0, 'RAINNC')[ind].data
    o.data -= getvar_(ds0, 'RAINSH')[ind].data
    o.name = 'tp'
    o.attrs.update(dict(description='total precipitation'))
    return o


def tp_e5_(fn, ind=np.s_[:]):
    o = xds_(fn).tp[ind]
    xu2u_(o, 'mm')
    return o


def s2a_(tp):
    axnm = xdimT_(tp)
    if axnm is None:
        raise Exception("dimension of time not found!")
    grp = tp[axnm].dt.month.copy(data=m2sm_(tp[axnm].dt.month, 'jja'))
    og = tp.groupby(grp)
    o0, o1 = og[True].mean(dim=axnm), og[False].mean(dim=axnm)
    o = o0 * 92 / (o0 * 92 + o1 * 273)
    o.name = 's2a'
    o.attrs.update(dict(units='1',
                        description='ratio of jja to annual precipitation'))
    return o


def swtp_(tp):
    axnm = xdimT_(tp)
    if axnm is None:
        raise Exception("dimension of time not found!")
    grp = tp[axnm].dt.month.copy(data=m2sm_(tp[axnm].dt.month, 'jja'))
    og = tp.groupby(grp)
    return (og[True].mean(dim=axnm), og[False].mean(dim=axnm))


def rho_wrf_(ds, ind=np.s_[:]):
    Gas_C = 287.04
    o = getvar_(ds, 'p')[ind]
    o.data /= (Gas_C * getvar_(ds, 'tv')[ind].data)
    o.name = 'rho'
    o.attrs.update(dict(units='kg m-3', description='air density'))
    return o


def zstag_wrf_(ds, ind=np.s_[:]):
    o = getvar_(ds, 'PH')[ind]
    o.data += getvar_(ds, 'PHB')[ind].data
    o.data /= 9.81
    o.name = 'height'
    o.attrs.update(dict(units='m', description='height over msl'))
    return o


def wvt_wrf_(*zrhoqv):
    z, rho, q = zrhoqv[:3]
    v = zrhoqv[3] if len(zrhoqv)>3 else None
    axZ = axZ_(z)
    _z = (z[ind_s_(z.ndim, axZ, np.s_[1:])] -
          z[ind_s_(z.ndim, axZ, np.s_[:-1])])
    o = q.copy()
    o.data *= _z.data*rho.data if v is None else _z.data*rho.data*v.data
    o = o.integrate(o.dims[axZ])
    o.name = ''.join(
        ['z', '' if v is None else v.name.lower(), 'rho', q.name.lower()])
    o.attrs.update(dict(
        units=sqzUnit_(' '.join(
            [z.units, rho.units, q.units, '' if v is None else v.units])),
        description="vertical integral of{0}{2}{1}".format(
            ' ' if v is None else (' easterward ' if 'u' in v.name else
                                   ' northward '),
            '' if v is None else ' flux',
            ' '.join(q.description.lower().split(' ')[:2]),
            )))
    return o


def loa_(da):
    try:
        return (da.XLONG, da.XLAT) # wrfout
    except:
        try:
            return (da.XLONG_M, da.XLAT_M) # wrf geo
        except:
            return (da.longitude, da.latitude) # era5
