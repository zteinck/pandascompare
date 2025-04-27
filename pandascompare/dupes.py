import pandas as pd
from oddments.utils import validate_value

from .utils import get_index_names


def drop_duplicates(df, **kwargs):
    '''
    Description
    ------------
    Alternative to pandas' native DataFrame.drop_duplicates that does
    not ignore the index.

    Parameters
    ------------
    kwargs : dict
        keyword arguments passed to the native method.

    Returns
    ------------
    df : pd.DataFrame
        DataFrame with duplicate rows removed.
    '''
    ok_kwargs = ['subset','keep']
    bad_kwargs = [k for k in kwargs if k not in ok_kwargs]

    if bad_kwargs:
        raise NotImplementedError(
            f'Unsupported kwargs: {bad_kwargs}'
            )

    index = get_index_names(df)
    if index: df = df.reset_index()
    df = df.drop_duplicates(**kwargs)
    if index: df = df.set_index(index)
    return df


def verify_no_duplicates(df, attr, label=None, dropna=None):
    '''
    Description
    ------------
    Raises an error if duplicates are found in the specified DataFrame
    attribute.

    Parameters
    ------------
    df : pd.DataFrame | pd.Series
        pandas object to inspect for duplicates
    attr : str
        DataFrame attribute to inspect for duplicates:
            • 'columns' ➜ column names
            • 'index' ➜ index values
            • 'values' ➜ column values
    label : str | None
        • attr = 'values' ➜ Name of the column to inspect for duplicates; may
          be None if only one column is present.
        • attr in ('columns','index') ➜ Optional supplemental text to include
          in the error message for additional context.
    dropna : bool | None
        • True ➜ NaN values are ignored when identifying duplicates.
        • False ➜ NaN values are considered when identifying duplicates.
        • None ➜ defaults to True when attr='values', otherwise False.

    Returns
    ------------
    None
    '''
    validate_value(
        value=attr,
        attr='attr',
        types=str,
        whitelist=['columns','index','values']
        )

    if isinstance(df, pd.Series):
        if attr == 'columns': return
        df = df.to_frame()
    else:
        validate_value(value=df, types=pd.DataFrame)

    is_values = attr == 'values'

    if is_values and label is None:
        if len(df.columns) == 1:
            label = df.columns[0]
        else:
            raise ValueError(
                "A column name must be specified using the 'label' "
                "argument when attr='values' and multiple columns exist."
                )

    msg = ['Duplicates detected in']

    if is_values:
        verify_no_duplicates(df, attr='columns')
        obj = df[label]
        msg.append('column')
    else:
        obj = getattr(df, attr)
        # if not obj.has_duplicates: return

    if dropna is None:
        dropna = is_values

    s = obj.value_counts(dropna=dropna)

    dupes = s[ s > 1 ].head(10).to_frame()
    if dupes.empty: return

    if label is not None: msg.append(f"'{label}'")
    msg.append('column names' if attr == 'columns' else attr)
    raise ValueError(' '.join(msg) + f':\n\n{dupes}\n')