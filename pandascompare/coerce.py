from functools import wraps
import pandas as pd
import numpy as np
from oddments.utils import validate_value


def _validate_array_like(data):
    ''' raises an error if data is not array-like '''
    validate_value(
        value=data,
        attr='data',
        types=(list, tuple, np.ndarray)
        )


def _get_data_dimensions(data):
    '''
    Description
    ------------
    Returns the number of dimensions of the input data. np.ndim returns:
        • 0 for single values (e.g. 'abc', 7, None, {'a': 9, 'b': [1, 2]}),
        • 1 for flat (un-nested) array-like data (e.g. [], [1, 2, 3])
        • 2 for once nested array-like data (e.g. [ [1, 3], [2, 4] ]

    However, it will raise a value error on irregularly shaped array-like data
    such as jagged lists (e.g. [[1, 2], [0]]) that are otherwise still DataFrame
    compatible. This function handles such edge cases by attempting to return the
    maximum number of dimensions among the constituent elements of the input data.

    Parameters
    ------------
    data : *
        input data

    Returns
    ------------
    ndim : int
        Maximum number of dimensions of the input data.
    '''
    try:
        return np.ndim(data)
    except ValueError:
        _validate_array_like(data)
        return max(np.ndim([x]) for x in data)


def _apply_default_name(obj, default_name):
    '''
    Description
    ------------
    Applies default name(s) to pandas object if necessary.

    Parameters
    ------------
    obj : pd.DataFrame | pd.Series
        pandas object to which the default name will be applied.
    default_name : str | None
        Serves as the default Series name attribute or DataFrame column
        name(s) when the former is None or the latter is the default
        range of ascending integers. The name(s) will have a numeric
        suffix (e.g., '_0', '_1', ...) appended.
        if None, pandas' default name(s) are left intact.

    Returns
    ------------
    obj : pd.Series | pd.DataFrame
        renamed object
    '''
    validate_value(
        value=default_name,
        attr='default_name',
        types=str,
        none_ok=True
        )

    if default_name is None:
        return obj

    # Unnamed Series
    if obj.ndim == 1 and obj.name is None:
        return obj.rename(default_name + '_0')

    # DataFrame with default integer column names
    if obj.ndim == 2 and all(
        isinstance(name, int) and name == index
        for index, name in enumerate(obj.columns)
        ):
        obj.columns = [
            f'{default_name}_{x}'
            for x in range(len(obj.columns))
            ]

    return obj


def _coercion_wrapper(func):

    @wraps(func)
    def wrapper(data, return_ndim=False, default_name='unnamed'):
        '''
        Description
        ------------
        Coerces data into a pandas DataFrame or Series.

        Parameters
        ------------
        data : *
            Data to be coerced to a Series or DataFrame.
            If 'data' is already of the desired type, a deep copy is
            returned.
        return_ndim : bool
            If True, the number of dimensions of the input data is returned.
        default_name : str | None
            see _apply_default_name() documentation

        Returns
        ------------
        out : pd.Series | pd.DataFrame
            Input data represented as a pandas object.
        ndim : int
            Only returned if 'return_ndim' is True.
        '''
        ndim = _get_data_dimensions(data)

        if ndim > 2:
            raise ValueError(
                f"Expected ndim to be ≤ 2, got: {ndim}. "
                f"Coercion failed for 'data': {data}."
                )

        out = func(data, _ndim=ndim).copy(deep=True)
        out = _apply_default_name(out, default_name)
        return (out, ndim) if return_ndim else out

    return wrapper


@_coercion_wrapper
def coerce_series(data, _ndim):
    ''' coerce data to pd.Series '''

    if isinstance(data, pd.Series):
        return data

    elif isinstance(data, pd.DataFrame):
        n_cols = len(data.columns)
        if n_cols != 1:
            raise ValueError(
                'Only DataFrames with exactly 1 column may be '
                f'converted to Series, got {n_cols} columns.'
                )
        return data.iloc[:, 0]

    elif isinstance(data, dict):
        n_keys = len(data)
        if n_keys == 0: # empty dictionary
            return coerce_series([])
        if n_keys != 1:
            raise ValueError(
                'Dictionary argument cannot have more '
                f'than one key, got {n_keys} keys.'
                )
        key, value = next(iter(data.items()))
        return coerce_series(value).rename(key)

    if _ndim == 0: # single value
        return coerce_series([data])

    elif _ndim == 1: # one-dimensional
        _validate_array_like(data)
        return pd.Series(data)

    raise ValueError(
        f"Expected ndim to be ≤ 1, got: {_ndim}. "
        f"Series coercion failed for 'data': {data}."
        )


@_coercion_wrapper
def coerce_dataframe(data, _ndim):
    ''' coerce data to pd.DataFrame '''

    if isinstance(data, pd.DataFrame):
        return data

    elif isinstance(data, pd.Series):
        return data.to_frame()

    elif isinstance(data, dict) and len(data) > 0:
        objs = [coerce_series({k: v}) for k, v in data.items()]
        return pd.concat(objs=objs, axis=1, join='outer')

    if _ndim <= 1: # single value or one-dimensional
        return coerce_series(data).to_frame()

    return pd.DataFrame(data)


def _coercion_handler(func, coerce, *args, **kwargs):
    '''
    Description
    ------------
    Function that enables coercion decorators to accommodate both standalone
    functions and instance methods.

    Note: does not work with static methods.

    Parameters
    ------------
    func : func
        decorated function
    coerce : func
        coercion function
    args : tuple
        func arguments
    kwargs : dict
        func keyword arguments

    Returns
    ------------
    out : pd.DataFrame | pd.Series
        coercion function output
    '''
    if not args:
        # user only passed kwargs
        kind = coerce.__name__.split('_')[-1]
        key = {'dataframe': 'df', 'series': 's'}[kind]
        args = (kwargs.pop(key),)

    if len(func.__qualname__.split('.')) > 1:
        # func is an instance method
        self, arg = args[0], coerce(args[1])
        return func(self, arg, *args[2:], **kwargs)
    else:
        # func is a standalone function
        arg = coerce(args[0])
        return func(arg, *args[1:], **kwargs)


def with_series_coercion(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        return _coercion_handler(func, coerce_series, *args, **kwargs)

    return wrapper


def with_dataframe_coercion(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        return _coercion_handler(func, coerce_dataframe, *args, **kwargs)

    return wrapper