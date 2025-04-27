import pandas as pd
from iterlab import to_iter

from .utils import get_index_names


def merge_dfs(a, b, **kwargs):
    '''
    Description
    ------------
    Alternative to pandas' native pd.merge that offers some quality-of-life
    improvements. These include preserving indices, checking for duplicate
    column names when the 'suffixes' argument is not passed, and defaulting
    the 'how' argument to 'left' instead of 'inner'.

    Parameters
    ------------
    kwargs : dict
        keyword arguments passed to the native method.

    Returns
    ------------
    df : pd.DataFrame
        DataFrame representing 'a' and 'b' objects merged.
    '''
    kwargs.setdefault('how', 'left')
    a_index, b_index = list(map(get_index_names, (a, b)))

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
    ''' returns a DataFrame containing records from 'a' that are not present in 'b' '''
    kwargs['indicator'] = True
    df = merge_dfs(a, b, **kwargs)
    df = df[ (df['_merge'] == 'left_only') ].drop('_merge', axis=1)
    return df


def join_left_only(a, b, **kwargs):
    ''' same as 'merge_left_only', but performs the join on indices '''
    a_index, b_index = list(map(get_index_names, (a, b)))
    if not all([a_index, b_index]) or a_index != b_index:
        raise NotImplementedError(
            "DataFrame arguments must have the same named index: "
            f"'a' index = {a_index}; 'b' index = {b_index}"
            )
    return merge_left_only(a, b, on=a_index, **kwargs)