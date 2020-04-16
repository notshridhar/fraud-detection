import pandas as pd


# Encode the categorical variables by an integer
# Preserves information without loss
class LabelEncoder:
    def __init__(self, drop_original=False):
        self.lbls_ = {}
        self.drop_ = drop_original

    def fit(self, data, columns):
        # save unique number of every category
        self.lbls_ = {}

        # do for all columns
        for col in columns:
            keys = data[col].fillna("NaN").unique()
            vals = [i + 1 for i in range(len(keys))]
            self.lbls_[col] = dict(zip(keys, vals))

    def transform(self, data, columns):
        # return full merged dataframe

        # temp dataframe
        encoded = pd.DataFrame()

        # do for all columns
        for col in columns:
            # check if already encoded
            key = col + "_lbl"
            if key in data.columns:
                continue

            dct = self.lbls_[col]
            encoded[key] = data[col].fillna("NaN").apply(lambda x: dct.get(x, 0))

        # drop encoded columns
        if self.drop_:
            rema = [col for col in data.columns if col not in columns]
            data = data[rema]

        # return merged dataframe
        return data.merge(encoded, left_index=True, right_index=True, how="left")

    def fit_transform(self, data, columns):
        # perform both fit and transform

        self.fit(data, columns)
        return self.transform(data, columns)


# Encode categorical variables with their frequencies by group
# Useful for the model to learn about abundance
class FrequencyEncoder:
    def __init__(self):
        self.freq_ = {}

    def fit(self, data, columns):
        # save frequencies of every category
        self.freq_ = {}

        # do for all columns
        for col in columns:
            self.freq_[col] = data[col].fillna("NaN").value_counts().to_dict()

    def transform(self, data, columns):
        # returns full merged dataframe

        # temp dataframe
        encoded = pd.DataFrame()

        # do for all columns
        for col in columns:
            # check if already encoded
            key = str(col) + "_frq"
            if key in data.columns:
                continue

            dct = self.freq_[col]
            encoded[key] = data[col].fillna("NaN").apply(lambda x: dct.get(x, 0))

        # return merged dataframe
        return data.merge(encoded, left_index=True, right_index=True, how="left")

    def fit_transform(self, data, columns):
        # perform both fit and transform

        self.fit(data, columns)
        return self.transform(data, columns)


# Encode quantitative variables with their aggregates by group
# Useful for the model to learn properties
class AggregateEncoder:
    def __init__(self, aggr_type, fillna=None):
        self.aggr_ = {}
        self.agty_ = aggr_type
        self.fill_ = fillna

    def fit(self, data, columns, uids):
        # save aggregates of every col-uid pair
        self.aggr_ = {}

        # do for all columns
        for col in columns:
            for uid in uids:
                # dict == {"col_uid_aggr": {"cat1": 1.23, "cat2": 4.56}}
                key = str(col) + "_" + str(uid) + "_" + str(self.agty_)
                dct = data.groupby(uid)[col].agg(self.agty_).to_dict()
                self.aggr_[key] = dct

    def transform(self, data, columns, uids):
        # returns full merged dataframe

        # temp dataframe
        encoded = pd.DataFrame()

        # do for all columns
        for col in columns:
            for uid in uids:
                # check if already encoded
                key = str(col) + "_" + str(uid) + "_" + str(self.agty_)
                if key in data.columns:
                    continue

                dct = self.aggr_[key]
                encoded[key] = data[uid].apply(lambda x: dct.get(x, 0))

        # fillna if requested
        if self.fill_:
            encoded.fillna(self.fill_, inplace=True)

        # return merged dataframe
        return data.merge(encoded, left_index=True, right_index=True, how="left")

    def fit_transform(self, data, columns, uids):
        # perform both fit and transform

        self.fit(data, columns, uids)
        return self.transform(data, columns, uids)

