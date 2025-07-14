from pathpilot import file_factory
import oddments as odd
import pandas as pd
import numpy as np

from ._data_wrapper import DataWrapper

pd.options.mode.chained_assignment = None


class PandasCompare(object):
    '''
    Description
    --------------------
    Compares two pandas DataFrame or Series objects across the following
    dimensions:
        • Rows ➔ missing rows based on the join key(s).
        • Columns ➔ name differences or missing columns.
        • Values ➔ value mismatches in content or type.

    Class Attributes
    --------------------
    summary_name : str
        Name of compare summary tab.

    Instance Attributes
    --------------------
    left : DataWrapper
        Left dataset to compare.
    right : DataWrapper
        Right dataset to compare.
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
    ignore_columns : set
        See 'ignore_columns' parameter documentation.
    summary : pd.DataFrame
        DataFrame containing comparison summary statistics.
    _reports : dict
        Dictionary where the keys are report names, and the values are
        pd.DataFrame objects representing the corresponding reports.

    Note: remaining instance attributes are documented in the __init__
          documentation below.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Class Attributes                                                        |
    #╰-------------------------------------------------------------------------╯

    summary_name = 'Compare Summary'


    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(
        self,
        left_data,
        right_data,
        left_label='left',
        right_label='right',
        delta_label='delta',
        label_template='{label}.{name}',
        join_on=None,
        left_ref=None,
        right_ref=None,
        dimensions=None,
        ignore_columns=None,
        include_data=False,
        include_diff_type=False,
        include_delta=False,
        tolerance=0,
        infer_dtypes=False,
        ignore_whitespace=False,
        try_numeric=False,
        allow_duplicates=False,
        verbose=False,
        ):
        '''
        Parameters
        ------------
        left_data : any
            "left" side dataset to compare.
        right_data : any
            "right" side dataset to compare.
        left_label : str
            Label used to denote columns from the left DataFrame
            (e.g. 'after').
        right_label : str
            Label used to denote columns from the right DataFrame
            (e.g. 'before').
        delta_label : str
            If include_delta is True, this label is used to denote
            delta columns (e.g. 'Δ').
        label_template : str
            String template that should contain placeholders for 'label' and
            'name' (e.g. '{label}.{name}', '{name}_{label}', etc.) where the
            label will be populated with 'left_label' or 'right_label' as
            appropriate and the name corresponds with the column name.
        join_on : str | list
            Name of column(s) on which to join left and right DataFrames.
            Note that the index is reset on the 'left' and 'right' arguments
            so passing index name(s) is also supported.
        left_ref : str | list | dict
            List of columns from the 'left' argument to include for reference.
            Column names represented as strings in the list are considered
            "default" reference columns and are included on every tab.
            Dictionaries represent a special case where the key denotes a
            specific tab, and the value is the list of reference columns to be
            included on that tab, in addition to the default reference columns.
        right_ref : str | list | dict
            List of columns from the 'right' argument to include on each tab
            for reference. See 'left_ref' documentation for more information.
        dimensions : set
            Specifies the dimensions along which the data will be compared.
            Valid options include 'rows', 'columns', and 'values'.
            If None, all dimensions are compared by default.
        ignore_columns : list
            List of column names to ignore during the comparison.
        include_data : bool
            if True, the entire 'left' and 'right' DataFrames being compared
            are included in the report for reference.
        include_diff_type : bool
            if True, the column denoting the difference type 'value'/'type'
            is included in the reports.
        include_delta : bool
            If True, a "delta" column is added to each report showing the
            difference observed between numeric or date-like values.
        tolerance : float
            Acceptable margin of error between two numeric values. Deltas in
            excess of this error tolerance threshold are categorized as
            differences. For example, if tolerance=0.01 then 1.01 vs 1.02
            (0.01) would not count as a difference whereas 1.01 vs 1.03 would
            (0.02).
        infer_dtypes : bool
            If True, the appropriate data types are inferred for all columns
            in the 'left' and 'right' DataFrames.
        ignore_whitespace : bool
            If True, values with leading and trailing whitespace are stripped
            and values comprised of only whitespace are treated as np.nan for
            comparison purposes.
        try_numeric : bool
            if True, values are compared as floats if possible. For example,
            '0.25' and '0.250' would not count as a value difference.
        allow_duplicates : bool
            if True, duplicates in the join key(s) are allowed.
        verbose : bool
            if True, relevant information is printed during the compare.
        '''

        self.join_on = join_on
        self.infer_dtypes = infer_dtypes
        self.label_template = label_template
        self.allow_duplicates = allow_duplicates

        self.left = DataWrapper(
            parent=self,
            side='left',
            data=left_data,
            label=left_label,
            ref_cols=left_ref,
            )

        self.right = DataWrapper(
            parent=self,
            side='right',
            data=right_data,
            label=right_label,
            ref_cols=right_ref,
            )

        self.delta_label = delta_label
        self.tolerance = tolerance
        self.dimensions = self._resolve_dims(dimensions)
        self.include_delta = include_delta
        self.ignore_whitespace = ignore_whitespace
        self.try_numeric = try_numeric
        self.include_diff_type = include_diff_type
        self.include_data = include_data
        self.verbose = verbose

        self.ignore_columns = set()
        if ignore_columns is not None:
            self.ignore_columns.update(
                odd.to_iter(ignore_columns)
                )

        self.shared_columns = self._find_shared_cols()

        self.ref_df = self._combine_ref_dfs()
        self.ref_map = self._combine_ref_maps()
        self.ref_def = self.left.ref_def.union(self.right.ref_def)

        self.summary = self._get_summary_frame()
        self._reports = {}
        self._compile_reports()


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def reports(self):
        ''' reports with summary prepended '''
        return {self.summary_name: self.summary} | self._reports


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def _get_summary_frame(self):
        ''' returns summary template '''
        df = pd.DataFrame(
            columns=[
                'Dimension',
                'Differences',
                'Matches',
                'Total',
                'Match Rate %'
                ],
            )
        df.index.name = 'Comparison'
        return df


    def _update_summary(self, label, dim, diffs, total):
        ''' updates summary '''
        columns = ['Dimension','Differences','Total']
        self.summary.loc[label, columns] = dim, diffs, total


    def _compute_summary_metrics(self):
        ''' computes final summary metrics '''
        df = self.summary
        df['Matches'] = df['Total'] - df['Differences']
        df['Match Rate %'] = df['Matches'] / df['Total']


    def _compile_reports(self):
        ''' compares left and right datasets and compiles results in reports
            dictionary '''

        label_verbiage = f'DataFrames {0!r} and {1!r}'\
            .format(self.left.label, self.right.label)

        equal_verbiage = f'{label_verbiage} are equal.'

        if self.left.df.equals(self.right.df):
            if self.verbose:
                print(f'{equal_verbiage} - 1st pass')
            return

        if self.verbose:
            print(f'Comparing {label_verbiage}')

        objs = [obj for obj in self._iter_objs()]

        if self.include_data:
            for obj in objs:
                self._reports[f'{obj.label} data'] = obj.df

        sheet_name = '{0} {1} not in {2}'

        # compare columns and rows
        for axis in ('cols','rows'):
            dim = 'columns' if axis == 'cols' else axis

            if dim not in self.dimensions:
                continue

            attr = f'_find_missing_{axis}'
            axis_index = 0 if axis == 'rows' else 1

            for i in (0, 1):
                df = getattr(self, attr)(
                    objs[i].df,
                    objs[i - 1].df
                    )

                k = sheet_name.format(
                    objs[i].label,
                    axis,
                    objs[i - 1].label
                    )

                if not df.empty:
                    self._reports[k] = df

                    self._update_summary(
                        label=k,
                        dim=dim,
                        diffs=len(df),
                        total=objs[i].df.shape[axis_index]
                        )

                    if self.verbose:
                        print(f'\t• {k}')

        # compare values
        if 'values' in self.dimensions:
            self._find_value_differences()

        # compute summary metrics
        if self.summary.empty:
            if self.verbose:
                print(f'{equal_verbiage} - 2nd pass')
        else:
            self._compute_summary_metrics()


    def _iter_objs(self):
        ''' iterate over data wrapper objects from left to right '''
        for obj in (self.left, self.right):
            yield obj


    def _find_missing_cols(self, a, b):
        ''' find missing columns '''
        b_columns = set(b.columns).union(self.ignore_columns)
        data = [
            [k, str(a[k].dtype)]
            for k in a.columns
            if k not in b_columns
            ]
        columns = ['Missing Column','Data Type']
        df = pd.DataFrame(data=data, columns=columns)
        return df


    def _find_missing_rows(self, a, b):
        ''' find missing rows '''
        return odd.join_left_only(a, b.drop(columns=b.columns))


    def _find_shared_cols(self):
        ''' find shared columns '''
        out = self.left.df.columns\
            .intersection(self.right.df.columns)\
            .difference(self.ignore_columns)\
            .tolist()
        return odd.natural_sort(out)


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


    def _find_value_differences(self):
        ''' find value differences '''

        columns = self.shared_columns[:]

        if not columns:
            return

        df = pd.concat(
            objs=[
                obj.get_labeled_subset(columns)
                for obj in self._iter_objs()
                ],
            axis=1,
            join='inner'
            )

        if df.empty:
            return

        for column in columns:
            labeled = [
                obj.add_label(column)
                for obj in self._iter_objs()
                ]
            self._compare_column_values(
                column=column,
                df=df[labeled]
                )


    def _compare_column_values(self, column, df):
        ''' compares column (series) values '''

        left_col, right_col = df.columns.tolist()
        total = len(df)

        if self.allow_duplicates:
            df = odd.drop_duplicates(df)

        diff_col = self.label_template.format(
            **dict(label='compare', name=column)
            )

        df[diff_col] = list(map(
            self._compare_values,
            df[left_col],
            df[right_col]
            ))

        df.dropna(subset=[diff_col], inplace=True)

        if not self.include_diff_type:
            df.drop(columns=diff_col, inplace=True)

        if df.empty: return

        if self.ref_df is not None:
            ref_cols = [
                x for x in self.ref_df.columns
                if x in self.ref_map.get(column, self.ref_def)
                and x not in df.columns
                ]

            if ref_cols:
                refs = self.ref_df[ref_cols]
                if self.allow_duplicates:
                    refs = odd.drop_duplicates(refs)
                df = refs.join(df, how='inner')

        if self.include_delta:
            delta_col = self.label_template.format(
                **dict(label=self.delta_label, name=column)
                )

            df[delta_col] = list(map(
                self._calculate_delta,
                [column] * len(df),
                df[left_col],
                df[right_col]
                ))

            if df[delta_col].isnull().all():
                df.drop(columns=delta_col, inplace=True)

        self._reports[column] = df

        self._update_summary(
            label=column,
            dim='values',
            diffs=len(df),
            total=total
            )

        if self.verbose:
            print(f'\t• {column}')


    def _compare_values(self, left, right):
        ''' compares two values '''

        @odd.ignore_nan
        def ignore_whitespace(x):
            if isinstance(x, str):
                return x.strip() or np.nan
            return x

        if self.ignore_whitespace:
            left, right = [
                ignore_whitespace(x)
                for x in (left, right)
                ]

        if self.try_numeric:
            left, right = [
                pd.to_numeric(x, errors='ignore')
                for x in (left, right)
                ]

        if pd.isnull(left) and pd.isnull(right):
            return np.nan

        if left == right:
            return np.nan

        if type(left) != type(right):
            return 'type'

        if not self.try_numeric and isinstance(left, str):
            return 'value'

        try:
            delta = abs(float(left) - float(right))
            if delta <= self.tolerance:
                return np.nan
        except (TypeError, ValueError):
            pass

        return 'value'


    def to_excel(self, file=None, empty_ok=False, **kwargs):
        '''
        Description
        ------------
        Exports compare reports to an Excel file.

        Parameters
        ------------
        file : None | str | ExcelFile
            File to which reports will be exported
                • None ➜ a generic default file name is used
                • str ➜ desired file path
                • ExcelFile ➜ pathpilot.ExcelFile object (or a derivative)
        empty_ok : bool
            Determines behavior when there are no reports (i.e. the compare
            found no differences).
                • True ➜ Excel file is created
                • False ➜ Excel file is not created
        kwargs : dict
            keyword arguments passed to write_df

        Returns
        ------------
        None
        '''
        empty_msg = 'No differences found.'
        is_empty = self.summary.empty

        if is_empty and not empty_ok:
            if self.verbose:
                print(f'Skipping Excel export. {empty_msg}')
            return

        if file is None:
            file = f"Compare {self.left.label} vs {self.right.label}.xlsx"

        file = file_factory(file, read_only=False)

        if file.ext != 'xlsx':
            raise ValueError(
                "File extension must be 'xlsx', "
                f"not: '{file.ext}'"
                )

        if is_empty:
            df = pd.DataFrame({'Note': [empty_msg]})
            file.name += ' (empty)'
            file.save({self.summary_name: df})
        else:
            # kwargs.setdefault('autofilter', True)
            file.save(self.reports, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Static Methods                                                          |
    #╰-------------------------------------------------------------------------╯

    @staticmethod
    def _resolve_dims(dims):
        ''' validates dimensions and returns them as a set '''
        whitelist = ['rows','columns','values']

        if dims is None:
            return set(whitelist)

        if isinstance(dims, str):
            dims = [dims]
        elif isinstance(dims, (tuple, set)):
            dims = list(dims)

        odd.validate_value(
            value=dims,
            attr='dimensions',
            types=list,
            empty_ok=False
            )

        for value in dims:
            odd.validate_value(
                value=value,
                attr='dimension values',
                types=str,
                whitelist=whitelist
                )

        return set(dims)


    @staticmethod
    def _calculate_delta(name, left, right):
        ''' attempts to calculate the difference between two values '''
        try:
            return left - right
        except:
            if odd.column_name_is_datelike(name):
                try:
                    left, right = map(pd.to_datetime, (left, right))
                    return left - right
                except (TypeError, ValueError):
                    pass

        return np.nan