import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
from osgeo import gdal
from matplotlib.path import Path

from .ffff import isIter_, l_flp_


__all__ = ['gpd_read',
           'gTiff_read_',
           'poly_to_path_']


gpd_read = gpd.GeoDataFrame.from_file


def _i_poly_to_path(poly, flp=False):
    def _is_closed(coords):
        _lcoords = list(coords)
        return np.array_equal(_lcoords[0], _lcoords[-1])
    _2pntsL = lambda x: [tuple(i[:2]) for i in x]
    _verts = lambda x: l_flp_(_2pntsL(x)) if flp else _2pntsL(x)
    _codes = lambda n: [Path.MOVETO] + [Path.LINETO]*(n - 2) + [Path.CLOSEPOLY]
    _c2p = lambda x: Path(_verts(x), codes=_codes(len(list(x))))
    
    if (hasattr(poly, 'exterior') and hasattr(poly.exterior, 'coords') and
          _is_closed(poly.exterior.coords)):
        return _c2p(poly.exterior.coords)
    elif hasattr(poly, 'coords') and _is_closed(poly.coords):
        return _c2p(poly.coords)
    elif hasattr(poly, 'geoms'):
        tmp = list(poly.geoms) 
        return [_i_poly_to_path(i, flp=flp) for i in tmp]
    elif hasattr(poly, 'geometry'):
        tmp = list(poly.geometry)
        return [_i_poly_to_path(i, flp=flp) for i in tmp]


def poly_to_path_(poly, flp=False):
    if isinstance(poly, gpd.GeoDataFrame):
        return _i_poly_to_path(poly, flp=flp)
    elif isIter_(poly, xi=gpd.GeoDataFrame):
        return [_i_poly_to_path(i, flp=flp) for i in poly]


def gTiff_read_(filename):
    ds = gdal.Open(filename)
    dsGT = ds.GetGeoTransform()
    xS, yS = ds.RasterXSize, ds.RasterYSize
    band = ds.RasterCount
    data = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(band)]
    noData = [ds.GetRasterBand(i + 1).GetNoDataValue() for i in range(band)]
    data = [np.ma.masked_equal(i, ii) if ii else i
            for i, ii in zip(data, noData)]
    lon = np.arange(xS) * dsGT[1] + dsGT[0]
    lat = np.arange(yS) * dsGT[5] + dsGT[3]
    return lon, lat, data
