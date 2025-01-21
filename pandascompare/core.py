import warnings
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
    join_left_only,
    infer_data_types,
    )

pd.options.mode.chained_assignment = None



class PandasCompare(object):
    '''
    Description
    --------------------
    Compares two pandas DataFrame or Series objects across the following dimensions:
        • Rows ➔ Identifies discrepancies based on the join key(s) specified using
                  the 'join_on' argument.
        • Columns ➔ Detects differences in column names and identifies missing columns.
        • Values ➔ Highlights differences in data, including mismatches in value or type.

    Class Attributes
    --------------------
    ...

    Instance Attributes
    --------------------
    left : DataWrapper
        Left dataset to compare
    right : DataWrapper
        Right dataset to compare
    ref_df : pd.DataFrame
        DataFrame comprised of all extra columns the user wants to include
        in the comparison reports for reference.
    ref_def : set
        Set of default reference columns to be included on each report.
    ref_map : dict
        Dictionary where keys are column names and values are sets of
        reference column names to be included on those reports specifically.
    shared_columns : list
        List of column names shared between the left and right DataFrames.
        Column names included in the 'ignore' parameter are excluded from
        this list.
    ignored_columns : set
        See 'ignore' parameter documentation.
    reports : dict | None
        Dictionary where the keys are report names, and the values are
        pd.DataFrame objects representing the corresponding reports.

    Note: remaining instance attributes are documented in the __init__
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
        left_ref=None,
        right_ref=None,
        ignore=None,
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
        ------------
        left : pd.DataFrame | pd.Series
            "left" side dataset to compare
        right : pd.DataFrame | pd.Series
            "right" side dataset to compare
        left_label : str
            Suffix used to denote columns from left DataFrame (e.g. '_a')
        right_label : str
            Suffix used to denote columns from right DataFrame (e.g. '_b')
        join_on : str | list
            Name of column(s) on which to join left and right DataFrames.
            Note that the index is reset on the 'left' and 'right' arguments
            so passing index name(s) is also supported.
        left_ref : str | list | dict
            List of columns from the 'left' argument to include for reference.
            Column names represented as strings in the list are considered
            "default" reference columns and are included on every tab. Dictionaries
            represent a special case where the key denotes a specific tab, and the
            value is the list of reference columns to be included on that tab, in
            addition to the default reference columns.
        right_ref : str | list | dict
            List of columns from the 'right' argument to include on each tab for
            reference. See 'left_ref' documentation for more information.
        ignore : list
            List of column names that will be ignored during the comparison.
        tolerance : float
            Acceptable margin of error between two numeric values. Deltas in excess
            of this error tolerance threshold are categorized as differences. For
            example, if tolerance=0.01 then 1.01 vs 1.02 (0.01) would not count as
            a difference whereas 1.01 vs 1.03 would (0.02).
        matches_only : bool
            If True, the report will only reflect differences between matching records.
            The report will not take into consideration row or column differences.
        include_delta : bool
            If True, a "delta" column is added to each report showing the difference
            observed between numeric or date-like values.
        infer_dtypes : bool
            If True, the appropriate data types are inferred for all columns in the
            'left' and 'right' DataFrames.
        ignore_whitespace : bool
            If True, values with leading and trailing whitespace are stripped and
            values comprised of only whitespace are treated as np.nan for comparison
            purposes.
        compare_strings_as_floats : bool
            if True, strings are compared as floats if possible. If True, '0.25' and
            '0.250' would not count as left value difference, for example. Conversely,
            you would want to set it to False when comparing values that are not meant
            to be interpreted as strings (e.g. leading zeroes such as '0505' vs '505').
        label_template : str
            string template that should contain placeholders for 'label' and 'name'
            (e.g. '{label}.{name}', '{name}_{label}', etc.) where the label will be
            populated with 'left_label' or 'right_label' as appropriate and the name
            corresponds with the column name.
        include_diff_type : bool
            if True, the column denoting the difference type 'value'/'type' is
            included in the reports.
        allow_duplicates : bool
            if True, duplicates in the join key(s) are allowed (see 'join_on' arg).
        include_data : bool
            if True, the entire 'left' and 'right' DataFrames being compared are
            included in the report for reference.
        verbose : bool
            if True, information will be printed.
        '''
        self.file_name = file_name

        if file_name is not None:
            warnings.warn(
                "'file_name' argument is deprecated and will be removed in a future release. "
                "Please use the 'file' argument in export_to_excel() instead.",
                DeprecationWarning,
                stacklevel=2
                )

        self.join_on = join_on
        self.infer_dtypes = infer_dtypes
        self.label_template = label_template
        self.allow_duplicates = allow_duplicates

        self.left = self.DataWrapper(self, 'left', left, left_label, left_ref)
        self.right = self.DataWrapper(self, 'right', right, right_label, right_ref)

        self.tolerance = tolerance
        self.matches_only = matches_only
        self.include_delta = include_delta
        self.ignore_whitespace = ignore_whitespace
        self.compare_strings_as_floats = compare_strings_as_floats
        self.include_diff_type = include_diff_type
        self.include_data = include_data
        self.verbose = verbose

        self.ignored_columns = set() if ignore is None else set(to_iter(ignore))
        self.shared_columns = self._find_shared_cols()

        self.ref_df = self._combine_ref_dfs()
        self.ref_map = self._combine_ref_maps()
        self.ref_def = self.left.ref_def.union(self.right.ref_def)

        self.reports = self._compile_reports()


    #╭-------------------------------------------------------------------------╮
    #| Classes                                                                 |
    #╰-------------------------------------------------------------------------╯

    class DataWrapper(object):
        '''
        Description
        --------------------
        Wraps left and right DataFrames.

        Class Attributes
        --------------------
        ...

        Instance Attributes
        --------------------
        parent : PandasCompare
            object to which the data belongs
        side : str
            indicates which "side" the data belongs to (e.g. 'left' or 'right')
        label : str
            label for column names
        df : pd.DataFrame
            dataset to compare
        ref_cols : list
            ordered list of all reference column names
        ref_def : set
            see parent class documentation
        ref_map : dict
            see parent class documentation
        ref_df : pd.DataFrame
            subset of self.df that only includes labeled reference columns
        '''

        #╭-----------------------------------╮
        #| Initialize Instance               |
        #╰-----------------------------------╯

        def __init__(self, parent, side, data, label, ref_cols):
            self.parent = parent
            self.side = side

            # set data label attribute
            if not isinstance(label, str):
                raise TypeError(f"'{self.side}_label' must be a string, "
                                f"not: {type(label).__name__}.")
            self.label = label

            # verify 'data' argument is of the correct type and then make a copy
            if isinstance(data, pd.DataFrame):
                self.df = data.copy(deep=True)
            elif isinstance(data, pd.Series):
                self.df = data.to_frame()
            else:
                raise TypeError(f"'{self.side}' must be a pandas DataFrame, "
                                f"or Series, not {type(data).__name__}.")

            # if there is already an index then reset it to column(s)
            index_names = get_index_names(self.df)
            default_index_name = ['index']

            for name in (index_names or default_index_name):
                if name in self.df.columns:
                    raise ValueError(f"'{self.side}' index name '{name}' "
                                     "conflicts with an existing column name. "
                                     "Please rename the column or index.")

            if index_names:
                self.df.reset_index(inplace=True)
            else:
                self.df.index.names = default_index_name

            # verify there are no duplicate columns which can cause issues
            verify_no_duplicates(df=self.df, label=self.label, attr='columns')

            # drop duplicate rows
            self.df.drop_duplicates(inplace=True)

            # infer data types
            if self.parent.infer_dtypes:
                self.df = infer_data_types(self.df)

            # set index to the column(s) you intend to join on
            if self.parent.join_on is not None:
                self.df.set_index(self.parent.join_on, inplace=True)

            # verify there are no duplicate index values which can cause issues
            if not self.parent.allow_duplicates:
                verify_no_duplicates(df=self.df, label=self.label, attr='index')

            # decompose ref_cols argument
            self.ref_cols, self.ref_def, self.ref_map = self.parse_ref_cols(ref_cols)

            # create pd.DataFrame with labeled reference columns
            self.ref_df = self.get_labeled_subset(self.ref_cols)


        #╭-----------------------------------╮
        #| Instance Methods                  |
        #╰-----------------------------------╯

        def add_label(self, column_name):
            ''' add label to single column name '''
            return self.parent.label_template\
                .format(**dict(label=self.label, name=column_name))


        def get_labeled_subset(self, column_names=None):
            ''' returns subset of self.df with labeled column names '''
            if column_names is None: column_names = self.df.columns.tolist()
            renames = {k: self.add_label(k) for k in column_names}
            df = self.df[column_names].rename(columns=renames)
            return df


        def parse_ref_cols(self, ref_cols):
            ''' derive ref_cols, ref_def, and ref_map instance attributes '''
            ref_cols = [] if ref_cols is None else to_iter(ref_cols)
            arg_name = f"'{self.side}_ref'"
            ref_all, ref_map = [], {}

            for x in ref_cols:
                if isinstance(x, str):
                    ref_all.append(x)

                elif isinstance(x, dict):
                    for k in x:
                        v = to_iter(x[k])
                        invalid_types = list(set([type(e).__name__ for e in v
                                                  if not isinstance(e, str)]))
                        if invalid_types:
                            raise TypeError(f"{arg_name} dictionary values must be strings"
                                            f" or lists of strings, not: {invalid_types}")
                        else:
                            ref_map.setdefault(k, []).extend(v)

                else:
                    raise TypeError(f"{arg_name} may only include strings or "
                                    f"dictionaries, not: {type(x).__name__}")

            add_labels = lambda x: set(map(self.add_label, x))
            ref_def = set(ref_all)

            for k, v in ref_map.items():
                ref_all.extend(v)
                ref_map[k] = add_labels(set(v).union(ref_def))

            # drop duplicates but retain order
            ref_all = list(dict.fromkeys(ref_all))

            missing_ref_cols = [x for x in ref_all if x not in self.df.columns]

            if missing_ref_cols:
                raise KeyError(f"{arg_name} includes column names not found in "
                               f"'{self.side}': {missing_ref_cols}")

            return ref_all, add_labels(ref_def), ref_map


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def _compile_reports(self):
        '''
        Description
        ------------
        Compares left and right DataFrames and stores results in 'results'
        dictionary.

        Parameters
        ------------
        ...

        Returns
        ------------
        results : dict
            see self.reports documentation
        '''

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


        reports = {}
        equal_verbiage = f"DataFrames '{self.left.label}' and '{self.right.label}' are equal"

        if self.left.df.equals(self.right.df):
            if self.verbose: print(f'{equal_verbiage} - 1st pass')
            return reports

        if self.verbose:
            print(f"Comparing DataFrames '{self.left.label}' vs '{self.right.label}'")

        log = pd.DataFrame(
            columns=['Dimension','Differences','Matches','Total','Match Rate %'],
            dtype='float'
            )
        log.index.name = 'Comparison'
        reports['Compare Summary'] = log

        dfs = (self.left, self.right)

        if self.include_data:
            for obj in dfs:
                reports[f'{obj.label} data'] = obj.df

        sheet_name = '{0} {1} not in {2}'
        if not self.matches_only:
            for axis in ('cols','rows'):
                for i in range(2):
                    df = getattr(self, f'_find_missing_{axis}')(dfs[i].df, dfs[i - 1].df)
                    k = sheet_name.format(dfs[i].label, axis, dfs[i - 1].label)
                    if not df.empty:
                        if self.verbose: print(f'\t• {k}')
                        reports[k] = df
                        log.loc[k, ['Dimension','Differences','Total']] = \
                            'columns' if axis == 'cols' else axis, len(df), \
                            len(dfs[i].df if axis == 'rows' else dfs[i].df.columns)

        master = self.left.get_labeled_subset(self.shared_columns).join(
                 self.right.get_labeled_subset(self.shared_columns), how='inner')

        if not master.empty:

            for k in self.shared_columns:

                df = master[[self.left.add_label(k), self.right.add_label(k)]]
                if self.allow_duplicates: df = drop_duplicates(df)
                left_k, right_k = df.columns.tolist()
                master.drop(df.columns, axis=1, inplace=True)

                k2 = self.label_template.format(**dict(label='compare', name=k))
                df[k2] = list(map(compare_values, df[left_k], df[right_k]))
                df.dropna(subset=[k2], inplace=True)
                if not self.include_diff_type: df.drop(k2, axis=1, inplace=True)

                if not df.empty:
                    if self.ref_df is not None:
                        ref_cols = [
                            x for x in self.ref_df.columns
                            if x in self.ref_map.get(k, self.ref_def)
                            and x not in df.columns
                            ]

                        if ref_cols:
                            refs = self.ref_df[ref_cols]
                            if self.allow_duplicates: refs = drop_duplicates(refs)
                            df = refs.join(df, how='inner')

                    if self.include_delta:
                        k3 = self.label_template.format(**dict(label='delta', name=k))
                        df[k3] = list(map(calculate_delta, [k] * len(df), df[left_k], df[right_k]))
                        if pd.isnull(df[k3]).all(): df.drop(k3, axis=1, inplace=True)

                    if self.verbose: print(f'\t• {k}')

                    reports[k] = df
                    log.loc[k, ['Dimension','Differences','Total']] = \
                        'values', len(df), len(master)

        if log.empty:
            reports.clear()
            if self.verbose: print(f'{equal_verbiage} - 2nd pass')
        else:
            log['Matches'] = log['Total'] - log['Differences']
            log['Match Rate %'] = log['Matches'] / log['Total']

        return reports


    def _find_missing_cols(self, a, b):
        ''' find missing columns '''
        b_columns = set(b.columns).union(self.ignored_columns)
        data = [[k, str(a[k].dtype)] for k in a.columns if k not in b_columns]
        df = pd.DataFrame(data, columns=['Missing Column', 'Data Type'])
        return df


    def _find_missing_rows(self, a, b):
        ''' find missing rows '''
        return join_left_only(a, b.drop(b.columns, axis=1))


    def _find_shared_cols(self):
        ''' find shared columns '''
        out = self.left.df.columns\
            .intersection(self.right.df.columns)\
            .difference(self.ignored_columns)\
            .tolist()
        return natural_sort(out)


    def _combine_ref_dfs(self):
        ''' build pd.DataFrame comprised of reference columns '''
        if self.left.ref_cols and self.right.ref_cols:
            df = self.left.ref_df.join(self.right.ref_df, how='inner')
            column_order = []

            for k in self.left.ref_cols:
                column_order.append(self.left.add_label(k))
                if k in self.right.ref_cols:
                    column_order.append(self.right.add_label(k))

            for k in self.right.ref_cols:
                if self.right.add_label(k) not in column_order:
                    column_order.append(self.right.add_label(k))

            return df[column_order]

        elif self.left.ref_cols and not self.right.ref_cols:
            return self.left.ref_df

        elif self.right.ref_cols and not self.left.ref_cols:
            return self.right.ref_df


    def _combine_ref_maps(self):
        ''' combine left and right ref_map '''
        if self.ref_df is None: return {}
        out = self.left.ref_map.copy()
        for k, v in self.right.ref_map.items():
            out[k] = out[k].union(v) if k in out else v
        return out


    def export_to_excel(self, file=None, empty_ok=False, **kwargs):
        '''
        Description
        ------------
        Exports self.reports to Excel file.

        Parameters
        ------------
        file : None | str | ExcelFile
            Excel file to which reports will be exported
                • None ➜ default file name is assigned and the file is saved
                          in the data path (see get_data_path documentation).
                • str ➜ treated as a file path and passed to pathpilot.File
                • ExcelFile ➜ pathpilot.ExcelFile object or a derivative
        empty_ok : bool
            determines behavior when self.reports is empty (i.e. the compare
            found no differences).
                • True ➜ Excel file is created
                • False ➜ Excel file is not created
        kwargs : dict
            keyword arguments passed to write_df

        Returns
        ------------
        None
        '''
        empty_msg = 'No differences found'
        if not self.reports and not empty_ok:
            if self.verbose:
                print(f'Skipping export to Excel - {empty_msg.lower()}.')
            return

        if file is None: file = self.file_name

        if file is None:
            file_name = f"Compare {self.left.label} vs {self.right.label}.xlsx"
            folder = get_data_path().join(self.__class__.__name__, read_only=False)
            file = folder.join(file_name).timestamp(
                fmt='%Y-%m-%d %I.%M.%S %p',
                loc='suffix',
                encase=True
                )
        else:
            file = File(file)
            if file.ext != 'xlsx':
                raise ValueError(f"file extension must be 'xlsx', not: '{file.ext}'")

        if self.reports:
            # kwargs.setdefault('autofilter', True)
            file.save(self.reports, **kwargs)
        else:
            file.name += ' (empty)'
            file.save({'Compare Summary': pd.DataFrame({'Note': [empty_msg + '.']})})