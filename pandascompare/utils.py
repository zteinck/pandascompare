import pandas as pd
from pathpilot import purge_whitespace
from iterlab import to_iter



def get_index_names(df):
    ''' returns list of pandas DataFrame index name(s) '''
    return list(filter(None, list(df.index.names)))


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
        function to apply to each column name. If string, it will be interpreted as an attribute function of the column datatype
        (e.g. 'lower', 'upper', 'title', 'int', etc.

    Returns
    ------------
    out : pd.DataFrame | None
        renamed dataframe is returned if inplace=False otherwise None
    '''
    return df.rename(columns={k: func(k) if callable(func) else getattr(k, func)() for k in df.columns}, inplace=inplace)


def merge_dfs(a, b, **kwargs):
    ''' returns merged dataframe with index still intact '''
    if 'how' not in kwargs: kwargs['how'] = 'left'
    a_index, b_index = map(get_index_names, (a, b))

    if 'suffixes' not in kwargs:
        s = pd.Series(
            a_index + a.columns.tolist() + \
            b_index + b.columns.tolist()
            ).value_counts()

        if 'on' in kwargs and kwargs['on'] is not None:
            s.drop(to_iter(kwargs['on']), inplace=True)

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
    index = get_index_names(a)
    if not index: raise NotImplementedError
    b.index.names = index
    return merge_left_only(a, b, on=index, **kwargs)


def drop_duplicates(df, **kwargs):
    ''' returns unique dataframe; pandas' drop_duplicates method does not account for different indices only columns '''
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