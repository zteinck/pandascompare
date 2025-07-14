import oddments as odd


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
        Object to which the data belongs.
    side : str
        Indicates the "side" to which the data belongs
        (e.g. 'left' or 'right')
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

    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, parent, side, data, label, ref_cols):
        self.parent = parent
        self.side = side

        # set data label attribute
        odd.validate_value(
            value=label,
            attr=f'{self.side}_label',
            types=str,
            empty_ok=False
            )

        self.label = label

        # coerce 'data' argument to DataFrame
        self.df = odd.coerce_dataframe(data)

        # if there is already an index then reset it to column(s)
        index_names = odd.get_index_names(self.df)
        default_index_name = ['index']

        for name in (index_names or default_index_name):
            if name in self.df.columns:
                raise ValueError(
                    f'{self.side!r} index name {name!r} conflicts with an '
                    'existing column name. Please rename the column or index.'
                    )

        if index_names:
            self.df.reset_index(inplace=True)
        else:
            self.df.index.names = default_index_name

        # verify there are no duplicate columns which can cause issues
        odd.verify_no_duplicates(
            df=self.df,
            label=self.label,
            attr='columns'
            )

        # drop duplicate rows
        self.df.drop_duplicates(inplace=True)

        # infer data types
        if self.parent.infer_dtypes:
            self.df = odd.infer_data_types(self.df)

        # set index to the column(s) you intend to join on
        if self.parent.join_on is not None:
            self.df.set_index(self.parent.join_on, inplace=True)

        # verify there are no duplicate index values which can cause issues
        if not self.parent.allow_duplicates:
            odd.verify_no_duplicates(
                df=self.df,
                label=self.label,
                attr='index'
                )

        # decompose ref_cols argument
        self.ref_cols, self.ref_def, self.ref_map = \
            self.parse_ref_cols(ref_cols)

        # create pd.DataFrame with labeled reference columns
        self.ref_df = self.get_labeled_subset(self.ref_cols)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def add_label(self, column):
        ''' add label to single column name '''
        return self.parent.label_template\
            .format(**dict(label=self.label, name=column))


    def get_labeled_subset(self, columns=None):
        ''' returns subset of self.df with labeled column names '''
        if columns is None:
            columns = self.df.columns.tolist()
        renames = {k: self.add_label(k) for k in columns}
        df = self.df[columns].rename(columns=renames)
        return df


    def parse_ref_cols(self, ref_cols):
        ''' derive ref_cols, ref_def, and ref_map instance attributes '''
        ref_cols = [] if ref_cols is None else odd.to_iter(ref_cols)
        arg_name = f"'{self.side}_ref'"
        ref_all, ref_map = [], {}

        for x in ref_cols:
            if isinstance(x, str):
                ref_all.append(x)

            elif isinstance(x, dict):
                for k in x:
                    v = odd.to_iter(x[k])
                    invalid_types = list(set([
                        type(e).__name__ for e in v
                        if not isinstance(e, str)
                        ]))
                    if invalid_types:
                        raise TypeError(
                            f"{arg_name} dictionary values must be strings"
                            f" or lists of strings, not: {invalid_types}"
                            )
                    else:
                        ref_map.setdefault(k, []).extend(v)

            else:
                raise TypeError(
                    f"{arg_name} may only include strings or "
                    f"dictionaries, not: {type(x).__name__}"
                    )

        add_labels = lambda x: set(map(self.add_label, x))
        ref_def = set(ref_all)

        for k, v in ref_map.items():
            ref_all.extend(v)
            ref_map[k] = add_labels(set(v).union(ref_def))

        # drop duplicates but retain order
        ref_all = list(dict.fromkeys(ref_all))

        missing_ref_cols = [
            x for x in ref_all
            if x not in self.df.columns
            ]

        if missing_ref_cols:
            raise KeyError(
                f"{arg_name} includes column names not found "
                f"in '{self.side}': {missing_ref_cols}"
                )

        return ref_all, add_labels(ref_def), ref_map