from .coerce import (
    coerce_series,
    coerce_dataframe,
    with_series_coercion,
    with_dataframe_coercion
    )

from .combine import (
    merge_dfs,
    merge_left_only,
    join_left_only
    )

from .compare import PandasCompare

from .decorators import (
    ignore_nan,
    inplace_wrapper
    )

from .dropna import dropna_edges

from .dupes import (
    drop_duplicates,
    verify_no_duplicates
    )

from .utils import *

__version__ = '0.4.0'
__author__ = 'Zachary Einck <zacharyeinck@gmail.com>'