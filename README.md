# pandascompare

<div>

[![Package version](https://img.shields.io/pypi/v/pandascompare?color=%2334D058&label=pypi)](https://pypi.org/project/pandascompare/)
[![License](https://img.shields.io/github/license/zteinck/pandascompare)](https://github.com/zteinck/pandascompare/blob/master/LICENSE)

</div>

`pandascompare` is a Python package designed to compare `DataFrame` objects, enabling you to quickly identify the differences between two datasets.

## Installation
```sh
pip install pandascompare
```

## Main Features
The `PandasCompare` class compares any two `DataFrame` objects along the following dimensions:
- `Rows` ➔ missing rows based on the join key(s) specified via the `join_on` argument.
- `Columns` ➔ name differences or missing columns.
- `Values` ➔ value mismatches in content or type.


## Example Usage
Please refer to the documentation within the code for more information.

### Imports
```python
from pandascompare import PandasCompare
```

### Create DataFrames
First, let's create two sample `DataFrame` objects to compare.
```python
import pandas as pd
import numpy as np

# February Data
left_df = pd.DataFrame({
    'id': [1, 2, 3],
    'date': [pd.to_datetime('2024-02-29')] * 3,
    'first_name': ['Alice', 'Mike', 'John'],
    'amount': [10.5, 5.3, 33.77],
    })

# January Data
right_df = pd.DataFrame({
    'id': [1, 2, 9],
    'date': [pd.to_datetime('2024-01-31')] * 3,
    'first_name': ['Alice', 'Michael', 'Zachary'],
    'last_name': ['Jones', 'Smith', 'Einck'],
    'amount': [11.1, np.nan, 14],
    })
```

### Compare DataFrames
Next, we will initialize a `PandasCompare` instance to perform the comparison. Please consult the in-code documentation for a comprehensive list of available arguments.
```python
pc = PandasCompare(
    left_data=left_df,
    right_data=right_df,
    left_label='feb',
    right_label='jan',
    join_on='id',
    left_ref=['first_name'],
    include_delta=True,
    verbose=True,
    )
```

### Export to Excel
Finally, let's export the compare report to an Excel file to view the results.
```python
pc.to_excel()
```