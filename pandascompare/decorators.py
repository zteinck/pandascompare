import pandas as pd
from functools import wraps


def ignore_nan(func):
    ''' wrapper function that ignores np.nan arguments '''

    @wraps(func)
    def wrapper(arg, *args, **kwargs):
        if pd.notnull(arg):
            return func(arg, *args, **kwargs)
        else:
            return arg

    return wrapper


def inplace_wrapper(func):
    ''' wrapper that adds inplace functionality to any function '''

    @wraps(func)
    def wrapper(df, *args, **kwargs):
        inplace = kwargs.get('inplace', False)

        if inplace:
            func(df, *args, **kwargs)
        else:
            return func(df.copy(deep=True), *args, **kwargs)

    return wrapper