from pathfinder import File
from iterkit import natural_sort, to_iter

from collections import OrderedDict
import numpy as np
import pandas as pd

# pd.options.mode.chained_assignment = None



#+---------------------------------------------------------------------------+
# Freestanding functions
#+---------------------------------------------------------------------------+

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


@inplace_wrapper
def purge_whitespace(df):
    ''' trim leading and trailing whitespace and replace whitespace-only values with NaN '''
    for k in df.select_dtypes(include=['object']).columns:
        df[k] = df[k].str.strip().replace(r'^\s*$', np.nan, regex=True)
    return df


def column_name_is_datelike(name):
    name = name.lower()
    out = any(x in name for x in ('date','time')) or name[-2:] == 'dt'
    return out


def infer_data_types(df):
    ''' the appropriate data types are inferred for all columns in the dataframe '''
    df = purge_whitespace(columns_apply(df, 'strip'))

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



#+---------------------------------------------------------------------------+
# Classes
#+---------------------------------------------------------------------------+

class PandasCompare(object):
    '''
    Description
    --------------------
    compares two dataframes

    Class Attributes
    --------------------
    ...

    Instance Attributes
    --------------------
    left : PandasCompare.DataFrame
        Left dataframe to compare
    right : PandasCompare.DataFrame
        Right dataframe to compare
    refs : pd.DatFrame
        dataframe comprised of reference columns
    shared_columns : list
        Columns shared between the left and right dataframes. Columns included
        in the 'ignore' parameter are excluded from this list.
    reports : OrderedDict
        dictionary where key values are report names and values are dataframes
        representing the corresponding report.

    Note: remaining instance attributes are documented in the init parameter
          documentation below.
    '''

    def __init__(
        self,
        left,
        right,
        left_label='left',
        right_label='right',
        join_on=None,
        left_ref=[],
        right_ref=[],
        ignore=[],
        file_name=None,
        tolerance=0,
        matches_only=False,
        include_delta=False,
        infer_dtypes=False,
        ignore_whitespace=False,
        compare_strings_as_floats=False,
        label_template='{label}.{name}',
        include_diff_type=False,
        verbose=False,
        ):
        '''
        Parameters
        ----------
        left : pd.DataFrame
            Left dataframe to compare
        right : pd.DataFrame
            Right dataframe to compare
        left_label : str
            Suffix used to denote columns from left dataframe (e.g. '_a')
        right_label : str
            Suffix used to denote columns from right dataframe (e.g. '_b')
        join_on : str | list
            Name of column(s) on which to join left and right dataframes.
            Note that the index is reset on the 'left' and 'right' arguments
            so passing index name(s) is also supported.
        left_ref : str | list
            List of columns from the 'left' argument to include on each tab for
            reference.
        right_ref : str | list
            List of columns from the 'right' argument to include on each tab for
            reference.
        ignore : list
            List of columns that will be ignored during the comparison.
        file_name : str | FileBase
            Name of excel output file.
        tolerance : float
            Acceptable margin of error between two numeric values. Deltas in excess
             of this error tolerance threshold are categorized as differences. For example,
             if tolerance=0.01 then 1.01 vs 1.02 (0.01) would not count as a difference whereas
             1.01 vs 1.03 would (0.02).
        matches_only : bool
            If True, the report will only reflect differences between matching records. The report
            will not take into consideration row or column differences.
        include_delta : bool
            If True, a "delta" column is added to each report showing the difference observed between
            numeric or datelike values.
        infer_dtypes : bool
            If True, the appropriate data types are inferred for all columns in the 'left' and 'right'
            arugments.
        ignore_whitespace : bool
            If True, values with leading and trailing whitespace are stripped and values comprised of
            only whitespace are treated as np.nan for comparison purposes.
        compare_strings_as_floats : bool
            if True, strings are compared as floats if possible. If True, '0.25' and '0.250' would not count as left
            value difference for example. Conversely, you would want to set it to false when comparing values that
            are not meant to be interpreted as strings (e.g. leading zeroes such as '0505' vs '505')
        label_template : str
            string template that should contain placeholders for 'label' and 'name' (e.g. '{label}.{name}',
            '{name}_{label}', etc.) where the label will be populated with 'left_label' or 'right_label'
            as appropriate and the name corresponds with the column name.
        include_diff_type : bool
            if True, the column denoting the difference type 'value'/'type' is included in the reports
        verbose : bool
            if True, information will be printed
        '''
        # set self.DataFrame class atrributes
        self.DataFrame.join_on = join_on
        self.DataFrame.infer_dtypes = infer_dtypes
        self.DataFrame.label_template = label_template

        # set instance atrributes
        self.left = self.DataFrame(left, left_label, left_ref)
        self.right = self.DataFrame(right, right_label, right_ref)
        self.ignore = to_iter(ignore)
        self.file_name = file_name
        self.tolerance = tolerance
        self.matches_only = matches_only
        self.include_delta = include_delta
        self.ignore_whitespace = ignore_whitespace
        self.compare_strings_as_floats = compare_strings_as_floats
        self.include_diff_type = include_diff_type
        self.verbose = verbose
        self.shared_columns = natural_sort(
            self.left.df.columns.intersection(self.right.df.columns).difference(ignore).tolist())

        # build dataframe comprised of reference columns
        if self.left.ref_cols and self.right.ref_cols:
            df = self.left.refs.join(self.right.refs, how='inner')
            column_order = []
            for k in self.left.ref_cols:
                column_order.append(self.left.apply_label(k))
                if k in self.right.ref_cols:
                    column_order.append(self.right.apply_label(k))
            for k in self.right.ref_cols:
                if self.right.apply_label(k) not in column_order:
                    column_order.append(self.right.apply_label(k))
            df = df[column_order]
        elif self.left.ref_cols and not self.right.ref_cols:
            df = self.left.refs
        elif self.right.ref_cols and not self.left.ref_cols:
            df = self.right.refs
        else:
            df = None

        self.refs = df
        self.compare()



    #+---------------------------------------------------------------------------+
    # Classes
    #+---------------------------------------------------------------------------+

    class DataFrame(object):
        '''
        Description
        --------------------
        DataFrame object with some additional functionality that facilitates the compare

        Class Attributes
        --------------------
        join_on : str | list | tuple
            see documentation above
        label_template : str
            see documentation above
        infer_dtypes : bool | callable
            see documentation above

        Instance Attributes
        --------------------
        label : str
            see 'left_label' and 'right_label' documentation above
        ref_cols : str | list
            see 'left_ref' and 'right_ref' documentation above
        df : pd.DataFrame
            see 'left' and 'right' documentation above
        '''

        def __init__(self, obj, label, ref_cols):

            # set instance attributes
            self.label = label
            self.ref_cols = to_iter(ref_cols) if ref_cols else None

            # verify 'df' argument is of the correct type and then make a copy
            if isinstance(obj, pd.DataFrame):
                self.df = obj.copy(deep=True)
            elif isinstance(obj, pd.Series):
                self.df = obj.to_frame()
            else:
                raise TypeError(f"'{label}' dataframe argument of type '{type(obj)}' is not supported.")

            # if there is already an index then reset it to column(s)
            if get_index_names(self.df): self.df.reset_index(inplace=True)

            # verify there are no duplicate columns which can cause issues
            self.verify_no_duplicates('columns')

            if self.infer_dtypes: self.df = infer_data_types(self.df)

            # set index to the column(s) you intend to join on
            if self.join_on:
                self.df.set_index(self.join_on, inplace=True)
            else:
                self.join_on = self.df.index.names = ['index']

            # verify there are no duplicate index values which can cause issues
            self.verify_no_duplicates('index')



        #+---------------------------------------------------------------------------+
        # Instance Methods
        #+---------------------------------------------------------------------------+

        def verify_no_duplicates(self, attr):
            ''' verify there are no duplicates for a given attribute (e.g. columns or index) '''
            obj = getattr(self.df, attr)
            dupes = obj[ obj.duplicated() ].tolist()
            if dupes:
                raise ValueError(
                    "duplicates detected in '{0}' dataframe {1}:{2}"\
                    .format(self.label, attr, '\n\t• '.join([''] + [str(x) for x in dupes]))
                    )

        def apply_label(self, name):
            ''' add label to column name '''
            return self.label_template.format(**dict(label=self.label, name=name))


        def apply_labels(self, names=None):
            ''' run apply_label on all names in self.df '''
            return self.df[names or self.df.columns]\
                   .rename(columns={k: self.apply_label(k) for k in self.df.columns})



        #+---------------------------------------------------------------------------+
        # Properties
        #+---------------------------------------------------------------------------+

        @property
        def refs(self):
            ''' labeled self.df filtered for reference columns only '''
            return self.apply_labels(self.ref_cols)



    #+---------------------------------------------------------------------------+
    # Instance Methods
    #+---------------------------------------------------------------------------+

    def compare(self):

        def compare_values(left, right):

            @ignore_nan
            def ignore_whitespace(x):
                if not isinstance(x, str): return x
                out = x.strip()
                if out == '': out = np.nan
                return out

            if self.ignore_whitespace:
                left, right = map(ignore_whitespace, (left, right))

            if pd.isnull(left) and pd.isnull(right):
                return np.nan

            if left != right:
                if type(left) != type(right):
                    return 'type'
                try:
                    if isinstance(left, str) and not self.compare_strings_as_floats: raise
                    if abs(float(left) - float(right)) <= self.tolerance: return np.nan
                except:
                    pass
                return 'value'
            else:
                return np.nan


        def calculate_delta(name, left, right):
            try:
                return left - right
            except:
                if column_name_is_datelike(name):
                    try:
                        return pd.to_datetime(left) - pd.to_datetime(right)
                    except:
                        pass
            return np.nan



        self.reports = OrderedDict()
        equal_verbiage = f"DataFrames '{self.left.label}' and '{self.right.label}' are equal"

        if self.left.df.equals(self.right.df):
            if self.verbose: print(f'{equal_verbiage} - 1st pass')
            return

        if self.verbose: print(f"Comparing DataFrames '{self.left.label}' vs '{self.right.label}'")

        dfs = (self.left, self.right)
        sheet_name = '{0} {1} not in {2}'
        if not self.matches_only:
            for func in ('find_missing_cols','find_missing_rows'):
                for i in range(2):
                    df = getattr(self, func)(dfs[i].df, dfs[i - 1].df)
                    k = sheet_name.format(dfs[i].label, func.split('_')[-1], dfs[i - 1].label)
                    if not df.empty:
                        if self.verbose: print(f'\t• {k}')
                        self.reports[k] = df


        master = self.left.apply_labels(self.shared_columns).join(
                 self.right.apply_labels(self.shared_columns), how='inner')

        if not master.empty:

            for k in self.shared_columns:

                df = master[[self.left.apply_label(k), self.right.apply_label(k)]]
                left_k, right_k = df.columns.tolist()
                master.drop(df.columns, axis=1, inplace=True)

                k2 = self.DataFrame.label_template.format(**dict(label='compare', name=k))
                df[k2] = list(map(compare_values, df[left_k], df[right_k]))
                df.dropna(subset=k2, inplace=True)
                if not self.include_diff_type: df.drop(k2, axis=1, inplace=True)

                if not df.empty:
                    if self.refs is not None:
                        ref_cols = self.refs.columns.difference(df.columns).tolist()
                        if ref_cols: df = self.refs[ref_cols].join(df, how='inner')

                    if self.include_delta:
                        k3 = self.DataFrame.label_template.format(**dict(label='delta', name=k))
                        df[k3] = list(map(calculate_delta, [k] * len(df), df[left_k], df[right_k]))
                        if pd.isnull(df[k3]).all(): df.drop(k3, axis=1, inplace=True)

                    if self.verbose: print(f'\t• {k}')
                    self.reports[k] = df

        if not self.reports:
            if self.verbose: print(f'{equal_verbiage} - 2nd pass')


    def find_missing_cols(self, a, b):
        ''' find missing columns '''
        df = pd.DataFrame(
            [[k, str(a[k].dtype)] for k in a.columns if k not in b.columns],
            columns=['Missing Column', 'Data Type']
            )
        return df


    def find_missing_rows(self, a, b):
        ''' find missing rows '''
        df = merge_left_only(
            a,
            b.drop(b.columns, axis=1),
            on=self.DataFrame.join_on
            )
        return df


    def export_to_excel(self, **kwargs):
        ''' export reports to excel '''
        if not self.reports: return

        file = File(f"DataFrame Compare {self.left.label} vs {self.right.label}.xlsx")\
               .timestamp(fmt='%Y-%m-%d %I.%M %p', loc='suffix', encase=True)\
                if self.file_name is None else File(self.file_name)

        for sheet, df in self.reports.items():
            file.write_df(sheet=sheet, df=df, **kwargs)

        file.save()










if __name__ == '__main__':

    # example:

    left = pd.DataFrame({
        'UniqueID': [1,9,3],
        'first_name': ['alice','mike','john'],
        'strIntegers': ['1010.0', '2020.0', '3030.0'],
        'Integers': [1010.0, 2020.0, 3030.0],
        'Floats': [1010.05, 2020.05, 3030.05],
        'Percentage': [0.1,   0.2,   0.33],
        'Datetime': [f'2023-01-{str(x).zfill(2)} 00:01:04' for x in range(1,4)],
        'Date': pd.to_datetime([f'2023-01-{str(x).zfill(2)} 01:01:01' for x in range(1,4)]),
        }).tail(2)

    right = pd.DataFrame({
        'UniqueID': [1,2,3],
        'first_name': ['alice','viola','johnathan'],
        'strIntegers': ['900', '5000', '9000'],
        'Integers': [5, 3000, 3000],
        'Floats': [0.05, 2020.9, 3030.5],
        'Percentage': [0.15,   0.12,   0.7],
        'Datetime': [f'2024-01-{str(x).zfill(2)} 00:03:04' for x in range(1,4)],
        'Date': pd.to_datetime([f'2024-01-{str(x).zfill(2)} 01:04:01' for x in range(1,4)]),
        }).head(2)

    # print(left.dtypes)
    # left = infer_data_types(left)
    # print(left.dtypes)

    pc = PandasCompare(
        left=left,
        right=right,
        left_label='left',
        right_label='right',
        join_on=None,
        left_ref=[],
        right_ref=[],
        ignore=[],
        file_name=None,
        tolerance=0,
        matches_only=False,
        include_delta=True,
        infer_dtypes=False,
        ignore_whitespace=False,
        compare_strings_as_floats=False,
        label_template='{label}.{name}',
        include_diff_type=False,
        verbose=True,
        )

    pc.export_to_excel()