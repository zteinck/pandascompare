import pandas as pd
import numpy as np
from pathpilot import purge_whitespace
from iterlab import to_iter



def get_index_names(df):
    ''' returns list of pandas DataFrame index name(s) '''
    return list(filter(None, list(df.index.names)))


def columns_apply(df, func, inplace=False):
    '''
    Description
    ------------
    applies a uniform function to all column names in a dataframe

    Parameters
    ------------
    df : pd.DataFrame
        dataframe to adjust column names
    func : func | str
        function to apply to each column name. If string, it will be interpreted
        as an attribute function of the column datatype (e.g. 'lower', 'upper',
        'title', 'int', etc.).

    Returns
    ------------
    out : pd.DataFrame | None
        renamed dataframe is returned if inplace=False otherwise None
    '''
    renames = {
        k: func(k) if callable(func) else getattr(k, func)()
        for k in df.columns
        }
    return df.rename(columns=renames, inplace=inplace)


def merge_dfs(a, b, **kwargs):
    ''' returns merged dataframe with index still intact '''
    kwargs.setdefault('how', 'left')
    a_index, b_index = map(get_index_names, (a, b))

    if 'suffixes' not in kwargs:
        s = pd.Series(
            a_index + a.columns.tolist() + \
            b_index + b.columns.tolist()
            ).value_counts()

        join_on = kwargs.get('on')
        if join_on is not None:
            s.drop(to_iter(join_on), inplace=True)

        dupes = s[ s > 1 ]
        if not dupes.empty:
            raise ValueError(
                "Duplicate column names detected and 'suffixes' keyword "
                f"argument not passed:\n\n{dupes[:10].to_frame()}\n"
                )

    out = (a.reset_index() if a_index else a).merge(
          (b.reset_index() if b_index else b), **kwargs)

    return out.set_index(a_index) if a_index else out


def merge_left_only(a, b, **kwargs):
    ''' returns 'a' dataframe filtered for records that are not in 'b' dataframe '''
    kwargs['indicator'] = True
    df = merge_dfs(a, b, **kwargs)
    df = df[ (df['_merge'] == 'left_only') ].drop('_merge', axis=1)
    return df


def join_left_only(a, b, **kwargs):
    ''' returns 'a' dataframe filtered for records that are not in 'b' dataframe '''
    a_index, b_index = map(get_index_names, (a, b))
    if not all([a_index, b_index]) or a_index != b_index:
        raise NotImplementedError(
            "DataFrame arguments must have the same named index: "
            f"'a' index = {a_index}; 'b' index = {b_index}"
            )
    return merge_left_only(a, b, on=a_index, **kwargs)


def drop_duplicates(df, **kwargs):
    ''' returns unique dataframe; pandas' drop_duplicates method does not account
        for different indices only columns '''
    index = get_index_names(df)
    if index:
        return df.reset_index().drop_duplicates(**kwargs).set_index(index)
    else:
        return df.drop_duplicates(**kwargs)


def column_name_is_datelike(name):
    name = name.lower()
    out = any(x in name for x in ('date','time')) or name[-2:] == 'dt'
    return out


def verify_no_duplicates(df, attr='columns', label=None):
    ''' verify there are no duplicates for a given attribute (e.g. columns or index) '''
    if isinstance(df, pd.Series) and attr == 'columns': return
    obj = getattr(df, attr)
    if not obj.has_duplicates: return
    s = obj.value_counts(dropna=False)
    dupes = s[ s > 1 ][:10].to_frame()
    msg = ['Duplicates detected in']
    if label is not None: msg.append(f"'{label}'")
    msg.extend([df.__class__.__name__, f"{attr}:\n\n{dupes}\n"])
    raise ValueError(' '.join(msg))


def infer_data_types(obj):
    ''' the appropriate data types are inferred for all columns in the dataframe '''

    @purge_whitespace
    def preprocess(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.copy(deep=True)
        elif isinstance(obj, pd.Series):
            return obj.to_frame()
        else:
            raise TypeError(f"'obj' argument of type '{type(obj)}' is not supported.")

    df = preprocess(obj)

    for k in df.columns:
        if column_name_is_datelike(k):
            try:
                df[k] = pd.to_datetime(df[k])
            except Exception as e:
                print(k, '-->', e)
        else:
            try:
                df[k] = pd.to_numeric(df[k])
            except:
                pass

    return df


def coerce_series(arg):
    '''
    Description
    ------------
    converts one-dimensional argument to pandas Series

    Parameters
    ------------
    arg : pd.DataFrame | pd.Series | list | tuple | np.ndarray
        One-dimensional data to be converted to a Series.
        If arg is already a Series, a copy is returned.

    Returns
    ------------
    s : pd.Series
        arg represented as a Series
    '''
    if isinstance(arg, pd.DataFrame):
        columns = arg.columns.tolist()
        if len(columns) != 1:
            raise ValueError(
                "cannot convert DataFrame with more than "
                f"one column to a Series: {columns}"
                )
        return arg[columns[0]]

    if isinstance(arg, pd.Series):
        s = arg.copy(deep=True)
        if s.name is None:
            s.name = 'untitled_0'
        return s

    elif isinstance(arg, (list, tuple, np.ndarray)):
        ndim = np.ndim(arg)
        if ndim == 1: # one-dimensional
            return pd.Series(arg, name='untitled_0')
        else:
            raise ValueError(f"'arg' ndim should be 1 not {ndim}.")

    else:
        raise TypeError(f"'arg' type '{type(arg)}' not supported.")


def coerce_dataframe(arg, return_ndim=False):
    '''
    Description
    ------------
    converts argument to pandas DataFrame

    Parameters
    ------------
    arg : pd.DataFrame | pd.Series | list | tuple | np.ndarray | *
        data to be converted to a DataFrame.
        If arg is already a DataFrame, a copy is returned.
    return_ndim : bool
        if True, arg's number of dimensions is returned
                 as well.

    Returns
    ------------
    df : pd.DataFrame
        arg represented as a DataFrame
    ndim : int
        only returned if return_ndim is True
    '''
    if isinstance(arg, pd.DataFrame):
        df, ndim = arg.copy(deep=True), 2

    elif isinstance(arg, pd.Series):
        df, ndim = arg.to_frame(), 1

    else:
        ndim = np.ndim(arg)
        if ndim == 0: # single value
            arg = [arg]
        if ndim <= 1: # one-dimensional
            df = coerce_series(arg).rename('untitled_0').to_frame()
        elif ndim == 2: # two-dimensional
            columns = [f'untitled_{x}' for x in range(len(arg[0]))]
            df = pd.DataFrame(arg, columns=columns)
        else:
            raise ValueError(f"'arg' ndim should be 0 ≤ ndim ≤ 2, not: {ndim}.")

    return (df, ndim) if return_ndim else df


def dropna_edges(s, drop_inf=False, ignore_gaps=False):
    '''
    Description
    ------------
    Drops leading and trailing NaN values in a Series.
    Only Series with indexes sorted in ascending order
    are supported.

    Parameters
    ------------
    drop_inf : bool
        if True, infinite values replaced with NaN
    ignore_gaps : bool
        if True, NaN values exist between valid values
                 are permitted.
        if False, if any NaN values exist between valid
                  values, an exception is raised.

    Returns
    ------------
    out : pd.Series
       Series with leading and trailing NaN values dropped.
    '''
    s = coerce_series(s)

    if not s.index.is_monotonic_increasing:
        raise NotImplementedError(
            "'s' index must be sorted in ascending order"
            )

    if drop_inf:
        s.replace(
            to_replace=[np.inf, -np.inf],
            value=np.nan,
            inplace=True
            )

    first_index = s.first_valid_index()
    last_index = s.last_valid_index()

    out = s.loc[ first_index : last_index ]

    if ignore_gaps or out.notnull().all():
        return out

    msg = ["'s' argument cannot contain NaN"]
    if drop_inf: msg.append('or Inf(+/-)')
    z = s.loc[ out[ out.isnull() ].index ]
    msg.append(f"between valid values: \n\n{z}\n")
    raise ValueError(' '.join(msg))