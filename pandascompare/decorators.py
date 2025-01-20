import pandas as pd



def ignore_nan(func):
    ''' wrapper function that ignores np.nan arguments '''

    def wrapper(arg, *args, **kwargs):
        if pd.notnull(arg):
            return func(arg, *args, **kwargs)
        else:
            return arg

    return wrapper


def inplace_wrapper(func):
    ''' wrapper that adds inplace functionality to any function '''

    def wrapper(df, *args, **kwargs):
        inplace = kwargs.get('inplace', False)
        if not inplace: df = df.copy(deep=True)
        df = func(df, *args, **kwargs)
        return df

    return wrapper