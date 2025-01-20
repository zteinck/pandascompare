from collections import OrderedDict
import numpy as np
import pandas as pd
from pathpilot import File, get_data_path
from iterlab import natural_sort, to_iter

from .decorators import ignore_nan

from .utils import (
    get_index_names,
    verify_no_duplicates,
    column_name_is_datelike,
    drop_duplicates,
    merge_left_only,
    infer_data_types,
    )

pd.options.mode.chained_assignment = None



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
    ref_def : set
        set of default reference columns to be included on each tab
    ref_map : dict
        dictionary where keys are column names and values are sets of
        reference column names to be included on those tabs specifically.
    shared_columns : list
        Columns shared between the left and right dataframes. Columns included
        in the 'ignore' parameter are excluded from this list.
    reports : OrderedDict
        dictionary where key values are report names and values are dataframes
        representing the corresponding report.

    Note: remaining instance attributes are documented in the init parameter
          documentation below.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

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
        allow_duplicates=False,
        include_data=False,
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
        left_ref : str | list | dict
            List of columns from the 'left' argument to include for reference.
            Column names represented as strings in the list are considered "default"
            reference columns and are included on every tab. Dictionaries represent a
            special case where the key denotes a specific tab, and the value is the list
            of reference columns to be included on that tab, in addition to the default
            reference columns.
        right_ref : str | list | dict
            List of columns from the 'right' argument to include on each tab for
            reference. See 'left_ref' documentation for more information.
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
        allow_duplicates : bool
            if True, duplicates in the index are allowed
        include_data : bool
            if True, the entire 'left' and 'right' dataframes are included in the report for reference.
        verbose : bool
            if True, information will be printed
        '''
        # set self.DataFrame class atrributes
        self.DataFrame.join_on = join_on
        self.DataFrame.infer_dtypes = infer_dtypes
        self.DataFrame.label_template = label_template
        self.DataFrame.allow_duplicates = allow_duplicates

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
        self.allow_duplicates = allow_duplicates
        self.include_data = include_data
        self.verbose = verbose

        self.shared_columns = natural_sort(
            self.left.df.columns\
            .intersection(self.right.df.columns)\
            .difference(ignore).tolist()
            )

        self.refs = self._build_refs()
        self.ref_map = self._build_ref_map()
        self.ref_def = self.left.ref_def.union(self.right.ref_def)

        self.compare()


    #╭-------------------------------------------------------------------------╮
    #| Classes                                                                 |
    #╰-------------------------------------------------------------------------╯

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
        allow_duplicates : bool
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

        #╭-----------------------------------╮
        #| Initialize Instance               |
        #╰-----------------------------------╯

        def __init__(self, obj, label, ref_cols):

            # set instance attributes
            self.label = label

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
            verify_no_duplicates(df=self.df, label=self.label, attr='columns')

            # drop duplicate rows
            self.df.drop_duplicates(inplace=True)

            # infer data types
            if self.infer_dtypes: self.df = infer_data_types(self.df)

            # set index to the column(s) you intend to join on
            if self.join_on:
                self.df.set_index(self.join_on, inplace=True)
            else:
                self.join_on = self.df.index.names = ['index']

            # verify there are no duplicate index values which can cause issues
            if not self.allow_duplicates:
                verify_no_duplicates(df=self.df, label=self.label, attr='index')

            self.parse_ref_cols(ref_cols)


        #╭-----------------------------------╮
        #| Properties                        |
        #╰-----------------------------------╯

        @property
        def refs(self):
            ''' labeled self.df filtered for reference columns only '''
            return self.apply_labels(self.ref_cols)


        #╭-----------------------------------╮
        #| Instance Methods                  |
        #╰-----------------------------------╯

        def apply_label(self, name):
            ''' add label to column name '''
            return self.label_template.format(**dict(label=self.label, name=name))


        def apply_labels(self, names=None):
            ''' run apply_label on all names in self.df '''
            return self.df[names or self.df.columns]\
                   .rename(columns={k: self.apply_label(k) for k in self.df.columns})


        def parse_ref_cols(self, columns):
            ''' derive ref_cols, ref_def, and ref_map instance attributes '''

            ref_cols = set()
            ref_def = set()
            ref_map = {}

            for x in to_iter(columns):
                if isinstance(x, str):
                    ref_cols.add(x)
                    ref_def.add(x)

                elif isinstance(x, dict):
                    for k, v in x.items():
                        if k not in ref_map:
                            ref_map[k] = set()
                        v = to_iter(v)
                        ref_cols.update(v)
                        ref_map[k].update(v)
                else:
                    raise TypeError(f"'ref_cols' element of type '{type(x)}' is not supported.")

            apply_label = lambda x: set(map(self.apply_label, x))

            for k in ref_map:
                ref_map[k].update(ref_def)
                ref_map[k] = apply_label(ref_map[k])

            missing_cols = ref_cols - set(self.df.columns)

            if missing_cols:
                raise ValueError(f"'{self.label}' dataframe is missing passed reference columns: {missing_cols}")

            self.ref_cols = list(ref_cols)
            self.ref_def = apply_label(ref_def)
            self.ref_map = ref_map


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

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

        summary = pd.DataFrame(
            columns=['Differences','Matches','Total','Match Rate %'],
            dtype='float'
            )
        summary.index.name = 'Comparison'
        self.reports['Compare Summary'] = summary

        dfs = (self.left, self.right)

        if self.include_data:
            for obj in dfs:
                self.reports[f'{obj.label} data'] = obj.df

        sheet_name = '{0} {1} not in {2}'
        if not self.matches_only:
            for axis in ('cols','rows'):
                for i in range(2):
                    df = getattr(self, f'find_missing_{axis}')(dfs[i].df, dfs[i - 1].df)
                    k = sheet_name.format(dfs[i].label, axis, dfs[i - 1].label)
                    if not df.empty:
                        if self.verbose: print(f'\t• {k}')
                        self.reports[k] = df
                        summary.loc[k, ['Differences','Total']] = len(df), \
                        len(dfs[i].df if axis == 'rows' else dfs[i].df.columns)

        master = self.left.apply_labels(self.shared_columns).join(
                 self.right.apply_labels(self.shared_columns), how='inner')

        if not master.empty:

            for k in self.shared_columns:

                df = master[[self.left.apply_label(k), self.right.apply_label(k)]]
                if self.allow_duplicates: df = drop_duplicates(df)
                left_k, right_k = df.columns.tolist()
                master.drop(df.columns, axis=1, inplace=True)

                k2 = self.DataFrame.label_template.format(**dict(label='compare', name=k))
                df[k2] = list(map(compare_values, df[left_k], df[right_k]))
                df.dropna(subset=[k2], inplace=True)
                if not self.include_diff_type: df.drop(k2, axis=1, inplace=True)

                if not df.empty:
                    if self.refs is not None:
                        ref_cols = [
                            x for x in self.refs.columns
                            if x in self.ref_map.get(k, self.ref_def)
                            and x not in df.columns
                            ]

                        if ref_cols:
                            refs = self.refs[ref_cols]
                            if self.allow_duplicates: refs = drop_duplicates(refs)
                            df = refs.join(df, how='inner')

                    if self.include_delta:
                        k3 = self.DataFrame.label_template.format(**dict(label='delta', name=k))
                        df[k3] = list(map(calculate_delta, [k] * len(df), df[left_k], df[right_k]))
                        if pd.isnull(df[k3]).all(): df.drop(k3, axis=1, inplace=True)

                    if self.verbose: print(f'\t• {k}')
                    self.reports[k] = df
                    summary.loc[k, ['Differences','Total']] = len(df), len(master)

        if summary.empty:
            self.reports.clear()
            if self.verbose: print(f'{equal_verbiage} - 2nd pass')
        else:
            summary['Matches'] = summary['Total'] - summary['Differences']
            summary['Match Rate %'] = summary['Matches'] / summary['Total']


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


    def _build_refs(self):
        ''' build dataframe comprised of reference columns '''
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

        return df


    def _build_ref_map(self):
        ''' combine left and right ref_map '''
        if self.refs is None: return {}
        out = self.left.ref_map.copy()
        for k, v in self.right.ref_map.items():
            if k in out:
                out[k].update(v)
            else:
                out[k] = v
        return out


    def export_to_excel(self, **kwargs):
        ''' export reports to excel '''
        if not self.reports: return

        file = get_data_path().join(self.__class__.__name__, read_only=False)\
               .join(f"Compare {self.left.label} vs {self.right.label}.xlsx")\
               .timestamp(fmt='%Y-%m-%d %I.%M.%S %p', loc='suffix', encase=True)\
               if self.file_name is None else File(self.file_name)

        for sheet, df in self.reports.items():
            file.write_df(sheet=sheet, df=df, **kwargs)

        file.save()