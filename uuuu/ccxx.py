from xarray import DataArray as _XDA
from iris.cube import Cube as _Cube

from .ffff import *
from . import cccc as _cc
from . import xxxx as _xx


__f1dict__ = {
        ### dimc_ -------------------------------------------------------------
        'dimc_': "dimension coord",
        'dimcT_': "dimension 'T' coord",
        'dimcX_': "dimension 'X' coord",
        'dimcY_': "dimension 'Y' coord",
        'dimcZ_': "dimension 'Z' coord",
        'dimcXY_': "dimension coords of 'X' and 'Y'",
        ### dim_ --------------------------------------------------------------
        'dim_': "name of dimension coord for a given axis",
        'dimT_': "name of dimension 'T' coord",
        'dimX_': "name of dimension 'X' coord",
        'dimY_': "name of dimension 'Y' coord",
        'dimZ_': "name of dimension 'Z' coord",
        ### ax_ ---------------------------------------------------------------
        'ax_': "axi(e)s of a given dimension or coord",
        'axT_': "axis of dimension 'T' coord",
        'axX_': "axis of dimension 'X' coord",
        'axY_': "axis of dimension 'Y' coord",
        'axZ_': "axis of dimension 'Z' coord",
        'axXY_': "axes of 'X' and 'Y'",
        'isyx_': "if axis 'Y' before axis 'X'",
        ### loa_ -------------------------------------------------------------- 
        'loa_': "return coords of longitude/latitude",
        'axLOA_': "return axes of longitude/latitude coords",
        'loa_pnts_': "longitude/latitude points from coords",
        'loa_bnds_': "longitude/latitude bounds",
        'loa_pnts_2d_': "2d longitude/latitude points (from coords or meshed)",
        ### msk_ -------------------------------------------------------------- 
        'ind_clim_': "derive args for extract_ acoording to coord limits",
        'omsk_clim_': "derive a maskarray or ind acoording to coord limit",
        'ind_loalim_': "wrapped in_loalim_ for XDA/Cube",
        'ind_poly_': "wrapped in_poly_ for XDA/Cube",
        'where_not_msk_': "a copy of XDA/Cube mask where omsk is False",
        ### extract_byAxes_ ---------------------------------------------------
        'extract_byAxes_': "extraction with help of inds_ss_ for XDA/Cube",
        ### area_weights_ -----------------------------------------------------
        'area_weights_': "area weights for grid cells",
        ### rg_ --------------------------------------------------------------- 
        'rg_func_': "func upon area within a rectangle",
        'rg_mean_': "area-weighted mean upon area within a rectangle",
        'poly_func_': "func upon area within polygon(s)",
        'poly_mean_': "area-weighted mean upon area within a rectangle",
        }

__all__ = list(__f1dict__.keys())


def _func(func, arg0):
    """
    ... derive the function depending on the type of 1st argument ...
    """
    mod = _module(arg0)
    _fn = f"_{func}"
    if not hasattr(mod, _fn):
        msg = f"{mod!r} has not attribute {_fn!r}!"
        raise Exception(msg)
    return getattr(mod, _fn)


def _dstr(func):
    """
    ... making the __doc__ string ...
    """
    f_ = f"{func}_"
    _f = f"'_{func}'"
    o0 = f"... {'':.^71} ..."
    o1 = f"... {__f1dict__[f_]: ^71} ..."
    o2 = f"... {'': ^71} ..."
    o3 = f"see {_f} in either {_cc!r} or {_xx!r}"
    o3 = '\n'.join([f"... {o3[i:i+71]: <71} ..."
                    for i in range(0, len(o3), 71)])
    return '\n'.join([o0, o1, o2, o3, o0])


def _module(x):
    """
    ... determine the module depending on the type of 1st argument ...
    """
    if isinstance(x, _Cube):
        return _cc
    elif isinstance(x, _XDA):
        return _xx
    else:
        msg = ("have no idea about which module should be used! "
               "I determine the module where the wanted function defined "
               "according to the instance of first argument, which should be "
               f"iris.cube.Cube or xarray.DataArray, but it is {type(x)}"
               )
        raise Exception(msg)


_f1_src = """
def {0}_(*args, **kwargs):
    assert len(args) > 0, 'at least one positional argument required!'
    x = args[0]
    return _func({0!r}, x)(*args, **kwargs)
{0}_.__doc__ = _dstr({0!r})""".format


### EXEC ----------------------------------------------------------------------
for i in __f1dict__.keys():
    exec(_f1_src(i[:-1]))
