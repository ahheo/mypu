from cf_units import Unit as cUnit


__all__ = [
        'cUnit',
        'TSC_',
        'u2u_',
        ]


def u2u_(value, old, new, **kwargs):
    uold = old if isinstance(old, cUnit) else cUnit(old)
    unew = new if isinstance(new, cUnit) else cUnit(new)
    if not uold.is_convertible(unew):
        emsg = f"{uold!r} is not convertible to {unew!r}"
        raise ValueError(emsg)
    return uold.convert(value, unew, **kwargs)


TSC_ = lambda x: cUnit(f"days since {x}-1-1", calendar='standard')
