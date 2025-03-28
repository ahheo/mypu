import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from iris.cube import Cube as _Cube
import iris.coord_systems as _ics
import iris.plot as iplt
import cartopy.crs as ccrs
import os
import warnings

from .ffff import (nanMask_, kde__, flt_, flt_l, isIter_, rpt_, ind_inRange_,
                   robust_bc2_, extract_, upd_, schF_keys_)
from .cccc import y0y1_of_cube, extract_period_cube
from .ccxx import loa_, isyx_


__all__ = [
        #----------------------------------------------------------------------- general
        'init_fig_',
        'aligned_cb_',
        'aligned_qk_',
        'aligned_tx_',
        'annotate_heatmap',
        'heatmap',
        'axColor_',
        'axVisibleOff_',
        'ax_move_',
        'axs_abc_',
        'axs_move_',
        'axs_rct_',
        'axs_shrink_',
        'hspace_ax_',
        'hspace_axs_',
        'wspace_ax_',
        'wspace_axs_',
        'fg_ax_',
        'frame_lw_',
        'spine_c_',
        'get_1st_mappable_obj_',
        'get_1st_patchCollection_',
        'geoTkLbl_',
        'pstGeoAx_',
        'mycm',
        'cmnm_lsc_',
        #----------------------------------------------------------------------- pd
        'distri_swe_',
        #----------------------------------------------------------------------- cube
        'bp_cubeL_eval_',
        'bp_dataLL0_',
        'bp_dataLL1_',
        'bp_dataLL_',
        'cdf_iANDe_',
        'hatch_cube',
        'imp_',
        'imp_eur_',
        'imp_ll_',
        'imp_swe_',
        'pdf_iANDe_',
        'ts_eCube_',
        #----------------------------------------------------------------------- xarray
        'da_map_',
        'da_hatch_'
        'das_map_',
        'uv_map_',
        ]


def init_fig_(
        fx=12,
        fy=6,
        h=0.075,
        w=0.075,
        t=0.98,
        b=0.075,
        l=0.075,
        r=0.98,
        ):
    fig = plt.figure(figsize=(fx, fy))
    fig.subplots_adjust(
        hspace=h,
        wspace=w,
        top=t,
        bottom=b,
        left=l,
        right=r,
        )
    return fig


def axVisibleOff_(ax, which='all'):
    which_ = 'tbrl' if which == 'all' else which
    tbrl = dict(t='top', b='bottom', r='right', l='left')
    for i in which_:
        ax.spines[tbrl[i]].set(
                visible=False,
                zorder=0,
                )


def axColor_(ax, color):
    for child in ax.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color(color)


def _get_clo(cube):
    cs = cube.coord_system()
    if isinstance(cs, (_ics.LambertConformal,
                       _ics.Stereographic)):
        clo = cs.central_lon
    elif isinstance(cs, _ics.RotatedGeogCS):
        clo = rpt_(180 + cs.grid_north_pole_longitude, 180, -180)
    elif isinstance(cs, (_ics.Orthographic,
                         _ics.VerticalPerspective)):
        clo = cs.longitude_of_projection_origin
    elif isinstance(cs, _ics.TransverseMercator):
        clo = cs.longitude_of_central_meridian
    else:
        clo = np.floor(np.mean(cube.coord('longitude').points) / 5) * 5
    return clo


def imp2_swe_(
        cube0,
        cube1,
        *subplotspec,
        fig=None,
        func="pcolormesh",
        rg=None,
        sc=1,
        axK_={},
        pK_={},
        ):
    ext = _mapext(rg=rg, cube=cube0)
    if isinstance(clo_, (int, float)):
        clo = clo_
    elif clo_ == 'cs':
        clo = _get_clo(cube0)
    else:
        clo = _clo_ext(ext, h_=clo_)
    proj = ccrs.NorthPolarStereo(central_longitude=clo)
    return imp2_(cube0, cube1, *subplotspec,
                fig=fig,
                func=func,
                proj=proj,
                ext=ext,
                sc=sc,
                axK_=axK_,
                pK_=pK_,
                )


def imp_swe_(
        cube,
        *subplotspec,
        fig=None,
        func="pcolormesh",
        rg=None,
        sc=1,
        axK_={},
        pK_={},
        ):
    ext = _mapext(rg=rg, cube=cube)
    if isinstance(clo_, (int, float)):
        clo = clo_
    elif clo_ == 'cs':
        clo = _get_clo(cube)
    else:
        clo = _clo_ext(ext, h_=clo_)
    proj = ccrs.NorthPolarStereo(central_longitude=clo)
    return imp_(cube, *subplotspec,
                fig=fig,
                func=func,
                proj=proj,
                ext=ext,
                sc=sc,
                axK_=axK_,
                pK_=pK_,
                )


def imp2_eur_(
        cube0,
        cube1,
        *subplotspec,
        fig=None,
        func="pcolormesh",
        rg=None,
        sc=1,
        axK_={},
        pK_={},
        ):
    return imp2_(cube0, cube1, *subplotspec,
                fig=fig,
                func=func,
                proj=ccrs.EuroPP(),
                ext=_mapext(rg=rg, cube=cube0),
                sc=sc,
                axK_=axK_,
                pK_=pK_,
                )


def imp_eur_(
        cube,
        *subplotspec,
        fig=None,
        func="pcolormesh",
        rg=None,
        sc=1,
        axK_={},
        pK_={},
        ):
    return imp_(cube, *subplotspec,
                fig=fig,
                func=func,
                proj=ccrs.EuroPP(),
                ext=_mapext(rg=rg, cube=cube),
                sc=sc,
                axK_=axK_,
                pK_=pK_,
                )


def imp2_ll_(
        cube0,
        cube1,
        *subplotspec,
        fig=None,
        func="pcolormesh",
        rg=None,
        sc=1,
        axK_={},
        pK_={},
        ):
    return imp2_(cube0, cube1, *subplotspec,
                fig=fig,
                func=func,
                proj=ccrs.PlateCarree(),
                ext=_mapext(rg=rg, cube=cube0),
                sc=sc,
                axK_=axK_,
                pK_=pK_,
                )


def imp_ll_(
        cube,
        *subplotspec,
        fig=None,
        func="pcolormesh",
        rg=None,
        sc=1,
        axK_={},
        pK_={},
        ):
    return imp_(cube, *subplotspec,
                fig=fig,
                func=func,
                proj=ccrs.PlateCarree(),
                ext=_mapext(rg=rg, cube=cube),
                sc=sc,
                axK_=axK_,
                pK_=pK_,
                )


def imp2_(
        cube0,
        cube1,
        *subplotspec,
        fig=None,
        func="quiver",
        proj=None,
        **kwargs,
        ):
    fig = plt.gcf() if fig is None else fig
    ax = fig.add_subplot(*subplotspec, projection=proj)
    o = cube_uv_map_(cube0, cube1, axes=ax, func=func, **kwargs)
    return (ax, o)


def imp_(
        cube,
        *subplotspec,
        fig=None,
        func="pcolormesh",
        proj=None,
        **kwargs,
        ):
    fig = plt.gcf() if fig is None else fig
    ax = fig.add_subplot(*subplotspec, projection=proj)
    o = cube_map_(cube, axes=ax, func=func, **kwargs)
    return (ax, o)


def _clo_ext(ext, h_=None):
    if h_ == 'human':
        clo = np.floor(np.mean(ext[:2]) / 5) * 5
    else:
        clo = np.floor(np.mean(ext[:2]))
    return clo


def _mapext(rg={}, cube=None):
    o = {}
    if cube:
        lo0 = cube.coord('longitude').points
        la0 = cube.coord('latitude').points
        o.update(dict(longitude=[lo0.min(), lo0.max()],
                      latitude=[la0.min(), la0.max()]))
    if isinstance(rg, dict):
        o.update(**rg)
    if 'longitude' in o and 'latitude' in o:
        return flt_l([o['longitude'], o['latitude']])


def hatch_cube(cube, **kwargs):
    if 'pK_' in kwargs:
        kwargs['pK_'].setdefault('zorder', 5)
        kwargs['pK_'].setdefault('colors', 'none')
    return cube_map_(cube, func='contourf', **kwargs)


def cube_uv_map_(
        cube0,
        cube1,
        axes=None,
        func='quiver',
        ext=None,
        sc=1,
        axK_={},
        pK_={},
        ):
    axes = plt.gca() if axes is None else axes
    if ext:
        axes.set_extent(ext, crs=ccrs.PlateCarree())
    axK_.setdefault("frame_on", False)
    axes.set(**axK_)
    support = ['quiver', 'barbs', 'streamplot']
    assert func in support, f"func {func!r} not supported!"
    _func = getattr(axes, func)
    lo0, la0 = cube0.coord('longitude'), cube0.coord('latitude')
    if lo0.ndim == 2:
        o = _func(lo0.points, la0.points, cube0.data*sc, cube1.data*sc,
                  transform=ccrs.PlateCarree(),
                  **pK_)
    else:
        if cube0.coord_dims(lo0)[0] > cube0.coord_dims(la0)[0]:
            x, y = np.meshgrid(lo0.points, la0.points)
        else:
            y, x = np.meshgrid(la0.points, lo0.points)
        o = _func(x, y, cube0.data*sc, cube1.data*sc,
                  transform=ccrs.PlateCarree(),
                  **pK_)
    return o


def cube_map_(
        cube,
        axes=None,
        func='pcolormesh',
        ext=None,
        sc=1,
        axK_={},
        pK_={},
        ):
    axes = plt.gca() if axes is None else axes
    if ext:
        axes.set_extent(ext, crs=ccrs.PlateCarree())
    axK_.setdefault("frame_on", False)
    axes.set(**axK_)
    support = ['pcolor', 'pcolormesh', 'contour', 'contourf']
    assert func in support, f"func {func!r} not supported!"
    if func in support[-2:]:
        _func = getattr(iplt, func)
        o = _func(cube.copy(cube.data*sc), axes=axes, **pK_)
    else:
        lo0, la0 = cube.coord('longitude'), cube.coord('latitude')
        if lo0.ndim == 1:
            _func = getattr(iplt, func)
            o = _func(cube.copy(cube.data*sc), axes=axes, **pK_)
        else:
            if hasattr(lo0, 'has_bounds') and lo0.has_bounds():
                x, y = lo0.contiguous_bounds(), la0.contiguous_bounds()
            else:
                x, y = _2d_bounds(lo0.points, la0.points)
            _func = getattr(axes, func)
            o = _func(x, y, cube.data*sc,
                      transform=ccrs.PlateCarree(),
                      **pK_)
    return o


def _2d_bounds(x, y):
    def _extx(x2d):
        dx2d = np.diff(x2d, axis=-1)
        return np.hstack((x2d, x2d[:, -1:] + dx2d[:, -1:]))
    def _exty(y2d):
        dy2d = np.diff(y2d, axis=0)
        return np.vstack((y2d, y2d[-1:, :] + dy2d[-1:, :]))
    dx0 = _extx(np.diff(x, axis=-1))
    dx1 = _exty(np.diff(x, axis=0))
    x00 = x - .5 * dx0 - .5 * dx1
    x01 = _extx(x00)
    xx = _exty(x01)
    dy0 = _extx(np.diff(y, axis=-1))
    dy1 = _exty(np.diff(y, axis=0))
    y00 = y - .5 * dy0 - .5 * dy1
    y01 = _extx(y00)
    yy = _exty(y01)
    return (xx, yy)


def ax_move_(
        ax,
        dx=0.,
        dy=0.,
        ):
    axp = ax.get_position()
    axp.x0 += dx
    axp.x1 += dx
    axp.y0 += dy
    axp.y1 += dy
    ax.set_position(axp)


def axs_move_(
        axs,
        dx,
        d_='x',
        ):
    for i, ax in enumerate(axs):
        if 'x' in d_:
            ax_move_(ax, dx=dx * i)
        elif 'y' in d_:
            ax_move_(ax, dy=dx * i)


def axs_shrink_(
        axs,
        rx=1.,
        ry=1.,
        anc='tl',
        ):
    if anc[0] not in 'tbm':
        raise ValueError("anc[0] must be one of 't' ,'m', 'b'!")
    if anc[1] not in 'lcr':
        raise ValueError("anc[1] must be one of 'l' ,'c', 'r'!")
    x0, x1, y0, y1 = _minmaxXYlm(axs)
    for i in axs:
        x00, x11, y00, y11 = _minmaxXYlm(i)
        if anc[1] == 'l':
            dx = (x0 - x11) * (1 - rx) if x0 != x00 else 0.
        elif anc[1] == 'c':
            dx = (x0 + x1 - x00 - x11) * .5 * (1 - rx)
        else:
            dx = (x1 - x00) * (1 - rx) if x1 != x11 else 0.
        if anc[0] == 't':
            dy = (y1 - y00) * (1 - ry) if y1 != y11 else 0.
        elif anc[0] == 'm':
            dy = (y0 + y1 - y00 - y11) * .5 * (1 - ry)
        else:
            dy = (y0 - y11) * (1 - ry) if y0 != y00 else 0.
        ax_move_(i, dx, dy)


def _minmaxXYlm(ax):
    if isIter_(ax):
        x0 = min([i.get_position().x0 for i in flt_(ax)])
        y0 = min([i.get_position().y0 for i in flt_(ax)])
        x1 = max([i.get_position().x1 for i in flt_(ax)])
        y1 = max([i.get_position().y1 for i in flt_(ax)])
    else:
        x0, y0 = ax.get_position().p0
        x1, y1 = ax.get_position().p1
    return (x0, x1, y0, y1)


def axs_rct_(
        ax,
        dx=.005,
        dy=None,
        **kwargs,
        ):
    fig = _fig_ax(ax)
    x0, x1, y0, y1 = _minmaxXYlm(ax)
    kD = dict(fill=False, color='k', zorder=1000,
              transform=fig.transFigure, figure=fig)
    kD.update(kwargs)
    fx, fy = fig.get_size_inches()
    dy = dx * fx / fy if dy is None else dy
    fig.patches.extend(
        [plt.Rectangle(
            (x0 - dx, y0 -dy),
            x1 - x0 + 2*dx,
            y1 - y0 + 2*dy,
            **kD,
            )])


def wspace_ax_(ax0, ax1):
    return ax1.get_position().x0 - ax0.get_position().x1


def hspace_ax_(ax0, ax1):
    return ax0.get_position().y0 - ax1.get_position().y1


def wspace_axs_(axs, sub=.01):
    from itertools import combinations as comb
    o = [wspace_ax_(*i) for i in comb(flt_l(axs), 2)]
    o = np.array([i for i in o if i > 0])
    return o.min() if o.size > 0  else sub


def hspace_axs_(axs, sub=.01):
    from itertools import combinations as comb
    o = [hspace_ax_(*i) for i in comb(flt_l(axs), 2)]
    o = np.array([i for i in o if i > 0])
    return o.min() if o.size > 0 else sub


def get_1st_mappable_obj_(ax):
    for i in ax.children():
        if hasattr(i, 'get_cmap'):
            return i


def _guess_cbiw(ax, lrbt='r'):
    apo = ax.get_position()
    _aD = dict(l=apo.x0, r=1-apo.x1, b=apo.y0, t=1-apo.y1)
    return (_aD[lrbt]/4, _aD[lrbt]/3)


def _fig_ax(ax):
    if isinstance(ax, (mpl.axes.Axes, )):
        return ax.figure
    elif isIter_(ax):
        for i in flt_(ax):
            return _fig_ax(i)
    else:
        emsg = "failed to get Parent figure"
        raise TypeError(emsg)


def aligned_cb_(
        ax=None,
        ppp=None,
        iw=None,
        orientation='vertical',
        shrink=1.,
        side=True,
        ncx='c',
        ti=None,
        **kwargs,
        ):
    ax = plt.gca() if ax is None else ax
    fig = _fig_ax(ax)
    ppp = get_1st_mappable_obj_(ax) if ppp is None else ppp
    vh = 'rl' if orientation == 'vertical' else 'bt'
    lrbt = vh[0] if side else vh[1]
    _i, _w = _guess_cbiw(ax, lrbt) if iw is None else iw

    cD = dict(orientation=orientation, **kwargs)
    x0, x1, y0, y1 = _minmaxXYlm(ax)
    shrink_ = 0 if ncx == 'n' else (1 if ncx=='x' else .5)

    if lrbt == 'r':
        caxb = [x1 + _i,
                y0 + (y1 - y0) * (1. - shrink) * shrink_,
                _w,
                (y1 - y0) * shrink]
    elif lrbt == 'l':
        caxb = [x0 - _i -_w,
                y0 + (y1 - y0) * (1. - shrink) * shrink_,
                _w,
                (y1 - y0) * shrink]
    elif lrbt == 'b':
        caxb = [x0 + (x1 - x0) * (1. - shrink) * shrink_,
                y0 - _i - _w,
                (x1 - x0) * shrink,
                _w]
    elif lrbt == 't':
        caxb = [x0 + (x1 - x0) * (1. - shrink) * shrink_,
                y1 + _i,
                (x1 - x0) * shrink,
                _w]
    cax = fig.add_axes(caxb)
    cb = plt.colorbar(ppp, cax=cax, **cD)
    if lrbt == 'l':
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position('left')
    elif lrbt == 't':
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
    if ti:
        cb.set_label(ti)
    return cb


def aligned_qk_(
        ax,
        q,
        U,
        s,
        pad=.02,
        rPos='NE',
        coordinates='figure',
        **kwargs,
        ):
    # _get_xy():
    x0, x1, y0, y1 = _minmaxXYlm(ax)
    if isIter_(pad) and len(pad) == 2:
        padx, pady = pad
    elif not isIter_(pad):
        padx = pady =pad
    else:
        raise("'pad' should be scalar (padx=pady) or arraylike (padx, pady)!")
    if 'N' in rPos:
        y = y1 + pady
    elif 'n' in rPos:
        y = y1 - pady
    elif 'S' in rPos:
        y = y0 - pady
    elif 's' in rPos:
        y = y0 + pady
    else:
        y = (y0 + y1) * .5
    if 'E' in rPos:
        x = x1 + padx
    elif 'e' in rPos:
        x = x1 - padx
    elif 'W' in rPos:
        x = x0 - padx
    elif 'w' in rPos:
        x = x0 + padx
    else:
        x = (x0 + x1) * .5
    #print(f"padx:{padx}; pady:{pady}")
    #print(f"x0:{x0}; x1:{x1}; y0:{y0}; y1:{y1};")
    #print(f"x:{x}, y:{y}")
    qk = plt.quiverkey(
            q, x, y, U, s,
            coordinates=coordinates,
            **kwargs
            )
    return qk


def axs_abc_(
        ax,
        s='(a)',
        dx=.001,
        dy=None,
        fontdict=dict(fontweight='bold'),
        **kwargs,
        ):
    fig = _fig_ax(ax)
    if isinstance(fig, mpl.figure.SubFigure):
        fra = 1
    else:
        fra = fig.get_figwidth() / fig.get_figheight()
    dy = abs(dx * fra) if dy is None else dy
    x0, _, _, y1 = _minmaxXYlm(ax)
    kD = dict(ha='left') if dx >= 0 else dict(ha='right')
    kD.update(dict(va='bottom') if dy >= 0 else dict(va='top'))
    kD.update(kwargs)
    fig.text(x0 + dx, y1 + dy, s, fontdict=fontdict, **kD)


def aligned_tx_(
        ax=None,
        s='',
        rpo='tl',
        itv=0.005,
        fontdict=None,
        **kwargs,
        ):

    ax = plt.gca() if ax is None else ax
    fig = _fig_ax(ax)
    x0, x1, y0, y1 = _minmaxXYlm(ax)

    if rpo[0].upper() in 'TB':
        xlm = [x0, x1]
    elif rpo[0].upper() in 'LR':
        xlm = [y0, y1]
    else:
        raise Exception('uninterpretable rpo!')

    if rpo[0].upper() == 'T':
        y = y1 + itv
        if itv >= 0:
            kwargs.update({'va': 'bottom'})
        else:
            kwargs.update({'va': 'top'})
    elif rpo[0].upper() == 'B':
        y = y0 - itv
        if itv >= 0:
            kwargs.update({'va': 'top'})
        else:
            kwargs.update({'va': 'bottom'})
    elif rpo[0].upper() == 'R':
        y = x1 + itv
        if itv >= 0:
            kwargs.update({'va': 'top'})
        else:
            kwargs.update({'va': 'bottom'})
    elif rpo[0].upper() == 'L':
        y = x0 - itv
        if itv >= 0:
            kwargs.update({'va': 'bottom'})
        else:
            kwargs.update({'va': 'top'})

    if rpo[1].upper() == 'L':
        x = xlm[0] + abs(itv)
        kwargs.update({'ha': 'left'})
    elif rpo[1].upper() == 'C':
        x = np.mean(xlm)
        kwargs.update({'ha': 'center'})
    elif rpo[1].upper() == 'R':
        x = xlm[1] - abs(itv)
        kwargs.update({'ha': 'right'})
    else:
        raise Exception('uninterpretable rpo!')

    if rpo[0].upper() in 'LR':
       x, y = y, x
       kwargs.update({'rotation': 'vertical', 'rotation_mode': 'anchor'})

    tx = fig.text(x, y, s, fontdict=fontdict, **kwargs)
    return tx


def _flt_cube(cube):
    data = nanMask_(cube.data).flatten()
    return data[~np.isnan(data)]


def pdf_iANDe_(
        ax,
        eCube,
        color,
        log_it=False,
        kopt={},
        ):
    if 'clip' in kopt:
        clip = np.array(kopt['clip'], dtype=np.float64)
        kopt.update({'clip': clip})
    ils = []
    if 'realization' in (i.name() for i in eCube.dim_coords):
        for c in eCube.slices_over('realization'):
            obs = _flt_cube(c)
            _, _, kdeo = kde__(obs.astype(np.float64), log_it=log_it, **kopt)
            #plot
            il, = ax.plot(kdeo.support, kdeo.density, lw=0.75, color=color,
                          alpha=.25)
            ils.append(il)
    obs = _flt_cube(eCube)
    _, _, kdeo = kde__(obs.astype(np.float64), log_it=log_it, **kopt)
    el, = ax.plot(kdeo.support, kdeo.density, lw=1.5, color=color, alpha=.85)
    return (ils, el)


def cdf_iANDe_(
        ax,
        eCube,
        color,
        log_it=False,
        kopt={},
        ):
    if 'clip' in kopt:
        clip = np.array(kopt['clip'], dtype=np.float64)
        kopt.update({'clip': clip})
    ils = []
    if 'realization' in (i.name() for i in eCube.dim_coords):
        for c in eCube.slices_over('realization'):
            obs = _flt_cube(c)
            x, _, kdeo = kde__(obs.astype(np.float64), log_it=log_it, **kopt)
            #plot
            il, = ax.plot(x, kdeo.cdf, lw=0.75, color=color, alpha=.25)
            ils.append(il)
    obs = _flt_cube(eCube)
    x, _, kdeo = kde__(obs.astype(np.float64), log_it=log_it, **kopt)
    el, = ax.plot(x, kdeo.cdf, lw=1.5, color=color, alpha=.85)
    return (ils, el)


def ts_eCube_(ax, eCube, color):
    y0y1 = y0y1_of_cube(eCube)
    cl = []
    ils = []
    if isinstance(eCube, _Cube):
        if 'realization' in (i.name() for i in eCube.coords()):
            cubes = eCube.slices_over('realization')
            #ax_r = eCube.coord_dims('realization')[0]
            #crd_r = eCube.coord('realization').points
            #cubes = [extract_byAxes_(eCube, ax_r, np.where(crd_r == i)[0][0])
            #         for i in crd_r]
            cut = False
        else:
            cubes = []
    else:
        cubes, cut = eCube, True
    for c in cubes:
        tmp = extract_period_cube(c, *y0y1) if cut else c
        cl.append(tmp.data)
        #plot
        il, = iplt.plot(tmp, axes=ax, lw=.5, color=color, alpha=.25, zorder=0)
        ils.append(il)
    ets = tmp.copy(np.mean(np.array(cl), axis=0)) if cubes else eCube
    el, = iplt.plot(ets, axes=ax, lw=1.75, color=color, alpha=.8, zorder=9)
    return (ils, el)


def bp_dataLL_(ax, dataLL, labels=None):
    gn = len(dataLL)
    ng = len(dataLL[0])
    ax.set_xlim(.5, ng + .5)
    ww = .001
    wd = (.6 - (gn - 1) * ww) / gn
    p0s = np.arange(ng) + .7 + wd / 2

    cs = plt.get_cmap('Set2').colors
    bp_dict = {'notch': True,
               'sym': '+',
               'positions': p0s,
               'widths': wd,
               'patch_artist': True,
               'medianprops': {'color': 'lightgray',
                               'linewidth': 1.5}}

    hgn = []
    for i, ii in enumerate(dataLL):
        ts_ = [np.ma.compressed(iii) for iii in ii]
        h_ = ax.boxplot(ts_, **bp_dict)
        for patch in h_['boxes']:
            patch.set_facecolor(cs[rpt_(i, len(cs))] + (.667,))
        hgn.append(h_['boxes'][0])
        p0s += ww + wd
    ax.set_xticks(np.arange(ng) + 1)
    if labels:
        ax.set_xticklabels(labels, rotation=60, ha='right',
                           rotation_mode='anchor')
    else:
        ax.set_xticklabels([None] * ng)
    return hgn


def bp_dataLL0_(ax, dataLL, labels=None):
    gn = len(dataLL)
    ng = len(dataLL[0])
    dd0 = [np.ma.compressed(i) for i in dataLL]
    dd1 = [[np.ma.compressed(dd[i]) for dd in dataLL] for i in range(ng)]
    ax.set_xlim(.5, gn + .5)
    ww = .001
    wd = (.6 - (ng - 1) * ww) / ng
    p0s = np.arange(gn) + .7 + wd / 2
    wd0 = .667
    p0 = np.arange(gn) + 1.

    cs = plt.get_cmap('Set2').colors
    if gn <= 3:
        cs0 = ['b', 'g', 'r']
    else:
        cs0 = plt.get_cmap('tab10').colors
    bp_dict = {'notch': True,
               'sym': '+',
               'zorder': 15,
               'positions': p0s,
               'widths': wd,
               'patch_artist': True,
               'medianprops': {'color': 'lightgrey',
                               'linewidth': 1.5}}
    bp0_dict= {'positions': p0,
               'widths': wd0,
               'sym': '',
               'zorder': 2,
               'capprops': {'color': '#555555dd',
                            'linewidth': 3},
               'boxprops': {'color': '#555555dd'},
               'whiskerprops': {'color': '#555555dd',
                                'linewidth': 3},
               'flierprops': {'color': '#555555dd'}}

    hgn = []
    for i, ii in enumerate(dd1):
        h_ = ax.boxplot(ii, **bp_dict)
        for patch in h_['boxes']:
            patch.set_facecolor(cs[rpt_(i, len(cs))] + (.667,))
        hgn.append(h_['boxes'][0])
        p0s += ww + wd

    bp_dict.update(bp0_dict)
    h_ = ax.boxplot(dd0, **bp_dict)

    for i, patch in enumerate(h_['boxes']):
        patch.set_facecolor('#555555dd')
        patch.set_zorder(bp_dict['zorder'] + 2 * i)
    hgn.append(h_['boxes'][0])
    eps = {}

    for i, md in enumerate(h_['medians']):
        if i == 0:
            y0 = md.get_ydata()[1]
        xd = md.get_xdata()
        xd[1] = ax.get_xlim()[1]
        md.set_xdata(xd)
        md.set_color(cs0[i])
        md.set_zorder(bp_dict['zorder'] + 1 + 2 * i)
        if i > 0:
            s = '${:+.2g}$'.format(md.get_ydata()[1] - y0)
            ax.text(xd[1], md.get_ydata()[1], s, va='center', color=cs0[i])

    ax.set_xticks(p0)
    if labels is not None:
        ax.set_xticklabels(labels, ha='center')
    for i, xtl in enumerate(ax.get_xticklabels()):
        xtl.set_color(cs0[i])

    return hgn


def bp_dataLL1_(ax, dataLL, labels=None):
    gn = len(dataLL)
    ng = len(dataLL[0])
    dd0 = [np.ma.compressed(i) for i in dataLL]
    dd1 = [[dd[i] for dd in dataLL] for i in range(ng)]
    ax.set_xlim(.5, gn + .5)
    wd0 = .667
    p0 = np.arange(gn) + 1.

    if gn <= 3:
        cs0 = ['b', 'g', 'r']
    else:
        cs0 = plt.get_cmap('tab10').colors
    bp_dict = {'notch': True,
               'positions': p0,
               'widths': wd0,
               'sym': '',
               'zorder': 5,
               'patch_artist': True,
               'capprops': {'color': '#555555dd',
                            'linewidth': 3},
               'boxprops': {'color': '#555555dd'},
               'whiskerprops': {'color': '#555555dd',
                                'linewidth': 3},
               'flierprops': {'color': '#555555dd'},
               'medianprops': {'color': 'lightgray',
                               'linewidth': 1.5}}

    hgn = []

    h_ = ax.boxplot(dd0, **bp_dict)
    for i, patch in enumerate(h_['boxes']):
        patch.set_facecolor('#555555dd')
        patch.set_zorder(bp_dict['zorder'] + 2 * i)
    hgn.append(h_['boxes'][0])

    for i, md in enumerate(h_['medians']):
        if i == 0:
            y0 = md.get_ydata()[1]
        xd = md.get_xdata()
        xd[1] = ax.get_xlim()[1]
        md.set_xdata(xd)
        md.set_color(cs0[i])
        md.set_zorder(bp_dict['zorder'] + 1 + 2 * i)
        if i > 0:
            s = '{:+.2g}'.format(md.get_ydata()[1] - y0)
            ax.text(xd[1], md.get_ydata()[1], s, va='center', color=cs0[i])

    ax.set_xticks(p0)
    if labels is not None:
        ax.set_xticklabels(labels, ha='center')
    for i, xtl in enumerate(ax.get_xticklabels()):
        xtl.set_color(cs0[i])

    return hgn


def bp_cubeL_eval_(ax, cubeL):
    XL = ['Simulations']
    dd0 = [flt_l([np.ma.compressed(i.data) for i in cubeL[:-2]])]
    if cubeL[-2]:
        XL.append('EOBS')
        dd0.append(np.ma.compressed(cubeL[-2].data))
    if cubeL[-1]:
        XL.append('ERA-Interim')
        dd0.append(np.ma.compressed(cubeL[-1].data))
    gn = len(XL)
    ng = len(cubeL) - 2
    dd1 = [i.data for i in cubeL[:-2]]
    ax.set_xlim(.5, gn + .5)
    ww = .001
    wd = (.6 - (ng - 1) * ww) / ng
    p0s = np.asarray([.7]) + wd / 2
    wd0 = .667
    p0 = np.arange(gn) + 1.

    cs = plt.get_cmap('Set2').colors
    bp_dict = {'notch': True,
               'sym': '+',
               'zorder': 15,
               'positions': p0s,
               'widths': wd,
               'patch_artist': True,
               'medianprops': {'color': 'lightgrey',
                               'linewidth': 1.5}}
    bp0_dict= {'positions': p0,
               'widths': wd0,
               'sym': '',
               'zorder': 2,
               'capprops': {'color': '#555555dd',
                            'linewidth': 3},
               'boxprops': {'color': '#555555dd'},
               'whiskerprops': {'color': '#555555dd',
                                'linewidth': 3},
               'flierprops': {'color': '#555555dd'}}

    hgn = []
    for i, ii in enumerate(dd1):
        h_ = ax.boxplot(ii, **bp_dict)
        for patch in h_['boxes']:
            patch.set_facecolor(cs[rpt_(i, len(cs))] + (.667,))
        hgn.append(h_['boxes'][0])
        p0s += ww + wd

    bp_dict.update(bp0_dict)
    h_ = ax.boxplot(dd0, **bp_dict)

    for i, patch in enumerate(h_['boxes']):
        patch.set_facecolor('#555555dd')
        patch.set_hatch('x')
        patch.set_zorder(bp_dict['zorder'] + 2 * i)
    hgn.append(h_['boxes'][0])

    ax.set_xticks(p0)
    ax.set_xticklabels(XL, ha='center')

    return hgn


def distri_swe_(
        df,
        *subplotspec,
        fig=None,
        pK_={},
        **kwargs,
        ):
    fig = plt.gcf() if fig is None else fig
    ax = fig.add_subplot(**subplotspec)
    df.plot(ax=ax, **kwargs, **pK_)
    ax.set_axis_off()
    return ax


def get_1st_patchCollection_(ax):
    for i in ax.get_children():
        if isinstance(i, mpl.collections.PatchCollection):
            return i


#def heatmap(data, row_labels, col_labels, ax=None,
#            cbar_kw={}, cbarlabel="", **kwargs):
def heatmap(
        data,
        row_labels,
        col_labels,
        ax=None,
        tkD=None,
        **kwargs,
        ):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwArgs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    if tkD:
        ax.tick_params(**tkD)
        rot=-45
    else:
        rot=45
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=rot,
             ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    #return im, cbar
    return im


def annotate_heatmap(
        im,
        data=None,
        valfmt="{:.2f}",
        data_=None,
        textcolors=("black", "white"),
        threshold=None,
        middle_0=False,
        **textkw,
        ):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwArgs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(ha="center",
              va="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    #if isinstance(valfmt, str):
    #    valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    def _ccc(v):
        if middle_0:
            kw.update(color=textcolors[int(im.norm(abs(v)) > threshold)])
        else:
            kw.update(color=textcolors[int(im.norm(v) > threshold)])
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            _ccc(data[i, j])
            #kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if data_ is None:
                #text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                text = im.axes.text(j, i, valfmt.format(data[i, j]), **kw)
            else:
                text = im.axes.text(j, i,
                                    valfmt.format(data[i, j], data_[i, j]),
                                    **kw)
            texts.append(text)

    return texts


def geoTkLbl_(ax):
    _lat = lambda x: rpt_(x, 180, -180)
    ax.xaxis.set_major_formatter(
            lambda x, pos: '{}\xb0E'.format(_lat(x)) if _lat(x) > 0 else
            ('{}\xb0W'.format(abs(_lat(x))) if _lat(x) < 0 else '0')
            )
    ax.yaxis.set_major_formatter(
            lambda x, pos: '{}\xb0N'.format(x) if x > 0 else
            ('{}\xb0S'.format(abs(x)) if x < 0 else 'Eq.')
            )


def pstGeoAx_(
        ax,
        delta=(30, 20),
        coastline=True,
        pad=-.1,
        xpad=None,
        ypad=None,
        c="dimgray",
        xc=None,
        yc=None,
        fsz=7,
        xfsz=None,
        yfsz=None,
        **kwargs):
    xpad = pad if xpad is None else xpad
    ypad = pad if ypad is None else ypad
    xc = c if xc is None else xc
    yc = c if yc is None else yc
    xfsz = fsz if xfsz is None else xfsz
    yfsz = fsz if yfsz is None else yfsz
    proj = ccrs.PlateCarree()
    if coastline:
        ax.coastlines(linewidth=0.2, color="darkgray")
    glD = dict(
            crs=ccrs.PlateCarree(),
            draw_labels={"bottom": "x", "left": "y"},
            xpadding=xpad,
            ypadding=ypad,
            xlabel_style=dict(color=xc, fontsize=xfsz),
            ylabel_style=dict(color=yc, fontsize=yfsz),
            dms=True,
            lw=.5,
            color="darkgray",
            alpha=.5,
            )
    if delta:
        if delta[0] < .5:
            ix = .1
        elif delta[0] < 1:
            ix = .5
        elif delta[0] < 5:
            ix = 1
        else:
            ix = 5
        if delta[1] < .5:
            iy = .1
        elif delta[1] < 1:
            iy = .5
        elif delta[1] < 5:
            iy = 1
        else:
            iy = 5
        x0, x1, y0, y1 = ax.get_extent(crs=proj)
        if any(i > 180 for i in (x0, x1)):
            _xtks = np.arange(0, 360, ix)
        else:
            _xtks = np.arange(-180, 180, ix)
        _xind = ind_inRange_(_xtks, x0, x1)
        _ytks = np.arange(-90, 90, iy)
        _yind = ind_inRange_(_ytks, y0, y1)
        glD.update(dict(
            xlocs = [i for i in _xtks[_xind] if i%delta[0] == 0],
            ylocs = [i for i in _ytks[_yind] if i%delta[1] == 0],
            ))
    glD.update(kwargs)
    gl = ax.gridlines(**glD)


def da_map_(
        da,
        ax=None,
        ext=None,
        func='pcolormesh',
        sc=1,
        axK_={},
        pK_={},
        ):
    x, y = loa_(da)
    ax = plt.gca() if ax is None else ax
    if ext:
        ax.set_extent(ext, crs=ccrs.PlateCarree())
    axK_.setdefault('frame_on', False)
    ax.set(**axK_)
    _func = getattr(ax, func)
    o = _func(x, y, da.data*sc if isyx_(da) else da.data.T*sc,
              transform=ccrs.PlateCarree(),
              **pK_)
    return o


def da_hatch_(da, **kwargs):
    _K = dict(colors='none', zorder=5)
    _K.update(**kwargs)
    amsg = "'levels' and/or 'hatches' not specified!"
    assert 'levels' in _K and 'hatches' in _K, amsg
    return da_map_(da, func='contourf', **_K)


def das_map_(
        das,
        tis=None,
        ax=None,
        ext=None,
        func='pcolormesh',
        sc=1,
        axK_={},
        pK_={},
        txtK_={},
        out=False,
        ):
    ax = plt.gca() if ax is None else ax
    tis = ('',)*len(das) if tis is None else tis
    if ext:
        ax.set_extent(ext, crs=ccrs.PlateCarree())
    axK_.setdefault('frame_on', False)
    ax.set(**axK_)
    _func = getattr(ax, func)
    if out:
        o = []
    for i, (da, ti, c) in enumerate(zip(das, tis, 'kwbrmyg')):
        x, y = loa_(da)
        o_ = _func(x, y, da.data*sc if isyx_(da) else da.data.T*sc,
                   transform=ccrs.PlateCarree(),
                   **pK_)
        if ti:
            x0 = np.min(x)
            y1 = np.max(y)
            ax.text(x0, y1, ti, color=c, va='bottom', ha='left', **txtK_)
        if out:
            o.append(o_)
        if i>0:
            xx = [x[0, 0], x[0, -1], x[-1, -1], x[-1, 0], x[0, 0]]
            yy = [y[0, 0], y[0, -1], y[-1, -1], y[-1, 0], y[0, 0]]
            ax.plot(xx, yy, color=c, lw=.5)
    if out:
        return o


def uv_map_(
        uda,
        vda,
        ax=None,
        ext=None,
        func='quiver',
        sc_u=1,
        sc_v=1,
        axK_={},
        pK_={},
        ):
    ax = plt.gca() if ax is None else ax
    if ext:
        ax.set_extent(ext, crs=ccrs.PlateCarree())
    axK_.setdefault('frame_on', False)
    ax.set(**axK_)
    _func = getattr(ax, func)
    x, y = loa_(uda)
    udata = uda.data*sc_u if isyx_(uda) else uda.data.T*sc_u
    vdata = vda.data*sc_v if isyx_(vda) else vda.data.T*sc_v
    o = _func(x, y, udata, vdata,
              transform=ccrs.PlateCarree(),
              **pK_)
    return o


def frame_lw_(ax, lw):
    if isinstance(ax, mpl.axes.Axes):
        for i in ax.spines:
            ax.spines[i].set_linewidth(lw)
    elif isIter_(ax):
        for iax in ax:
            frame_lw_(iax, lw)


def spine_c_(ax, c, which='all'):
    if which == 'all':
        which = 'tblr'
    if isinstance(ax, mpl.axes.Axes):
        for i in ax.spines:
            if i[0] in which:
                ax.spines[i].set_color(c)
    elif isIter_(ax):
        for iax in ax:
            spine_c_(iax, c, which=which)


def fg_ax_(
        ax,
        name='ne_shaded',
        resolution='high',
        extent=None,
        catch={},
        zorder=99,
        alpha=.15,
        **kwargs,
        ):
    import json                                                                # import required packages
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from matplotlib.image import imread
    from cartopy.mpl.geoaxes import GeoAxes

    msg = "Axes should be cartopy.mpl.geoaxes.GeoAxes"                         # check axes
    assert isinstance(ax, GeoAxes), msg

    bgdir = os.getenv(                                                         # get file
            'CARTOPY_USER_BACKGROUNDS',
            '/home/clin/Documents/data/cartopy/data/raster/natural_earth',
            )
    json_file = os.path.join(bgdir, 'images.json')
    with open(json_file, 'r') as js_obj:
        fgD = json.load(js_obj)
    try:
        fname = fgD[name][resolution]
    except KeyError:
        raise ValueError(
            f'Image {name!r} and resolution {resolution!r} are not '
            f'present in the user background image metadata in directory '
            f'{bgdir!r}')
    if isinstance(catch, dict):
        if fname in catch:
            img = catch[fname]
        else:
            img = imread(os.path.join(bgdir, fname))
            upd_(catch, **{fname: img})
    else:
        img = imread(os.path.join(bgdir, fname))
    if img.ndim == 2:
        img = robust_bc2_(img, img.shape + (3,), axes=tuple(range(img.ndim)))
    if fgD[name]['__projection__'] == 'PlateCarree':
        source_proj = ccrs.PlateCarree()
    else:
        raise NotImplementedError('Background image projection undefined')

    if extent is None:                                                         # set extent
        extent = ax.get_extent()

    imKA_ = dict(alpha=alpha, zorder=zorder,
                 origin='upper',
                 transform=source_proj,
                 )
    if tuple(extent) == (-180, 180, -90, 90):                                  # global
        return ax.imshow(img, extent=extent, **imKA_)
    else:                                                                      # regional
        lod, lad = 180 / img.shape[1], 90 / img.shape[0]
        lo = np.arange(img.shape[1]) * lod * 2 + lod - 180
        la = 90 - np.arange(img.shape[0]) * lad * 2 - lad
        ind0 = ind_inRange_(la, *extent[2:])
        if extent[0] < 180 and extent[1] > 180:
            ind10 = ind_inRange_(lo, extent[0], 180)
            ind11 = ind_inRange_(lo, 180, extent[1], r_=360)
            img_ = np.concatenate(
                    (extract_(img, 1, ind10, 0, ind0),
                     extract_(img, 1, ind11, 0, ind0)),
                    axix=1,
                    )
            extent_ = (lo[ind10][0] - lod,
                       lo[ind11][-1] + lod + 360,
                       la[ind0][-1] - lad,
                       la[ind0][0] + lad)
        else:
            ind1 = ind_inRange_(lo, *extent[:2])
            img_ = extract_(img, 1, ind1, 0, ind0)
            extent_ = (lo[ind1][0] - lod,
                       lo[ind1][-1] + lod,
                       la[ind0][-1] - lad,
                       la[ind0][0] + lad)
        return ax.imshow(img_, extent=extent_, **imKA_)


def f2cmap(fn):
    rgb = np.load(fn)
    N, n34 = rgb.shape
    if rgb.max() > 1:
        rgb = rgb / 256
    if n34 == 3:
        rgb = np.hstack((rgb, np.ones((N, 1))))
    name = os.path.basename(os.path.splitext(fn)[0])
    return mpl.colors.LinearSegmentedColormap.from_list(name, rgb, N=N)


class _CM:
    def __init__(self, p):
        self.Path = p
        fnL = schF_keys_(p, ext='.npy')
        for fn in fnL:
            o = f2cmap(fn)
            setattr(self, o.name, o)
            setattr(self, f'{o.name}_r', o.reversed())


_here_ = os.path.dirname(__file__)
mycm = _CM(os.path.join(_here_, 'cmap'))

def cmnm_lsc_(
        cmb=mpl.cm.Spectral_r,
        cm0=.1,
        cm1=.9,
        cmn=None,
        nmb=mpl.colors.BoundaryNorm,
        nmv=None,
        nmk={},
        ):
    cmn = f'cmap{np.random.randint(0, 100):03}' if cmn is None else cmn
    o0 = mpl.colors.LinearSegmentedColormap.from_list(
            cmn, cmb(np.linspace(cm0, cm1, 256)),
            )
    if nmb == mpl.colors.BoundaryNorm:
        nma = (nmv, 256)
        nmk_ = dict(extend='both')
        nmk_.update(**nmk)
    elif nmb == mpl.colors.Normalize:
        nma = ()
        nmk_ = dict(vmin=nmv[0], vmax=nmv[1])
        nmk_.update(**nmk)
    else:
        nma = (nmv,)
        nmk_ = dict(extend='both')
        nmk_.update(**nmk)
    o1 = nmb(*nma, **nmk_)
    return (o0, o1)
