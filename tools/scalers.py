import pandas as pd

# Scale the quantitative variables between 0 and 1
# operates by grouping column by uid
class MinMaxScaler:
    def __init__(self, fillna=None):
        self.prop_ = {}
        self.fill_ = fillna

    def fit(self, data, columns):
        # save min and max of every column
        self.prop_ = {}

        for col in columns:
            # calculate aggregates and save
            key = str(col) + "_abs"
            stat = data[col].agg(["min", "max"]).to_dict()
            self.prop_[col] = stat

    def transform(self, data, columns):
        # transform the column to fit
        # return full merged dataframe

        # temp dataframe
        encoded = pd.DataFrame()

        for col in columns:
            # find if already transformed
            key = str(col) + "_abs"
            if key in columns:
                continue

            # pre-calculate
            minval = self.prop_[col]["min"]
            ranges = self.prop_[col]["max"] - minval

            # transform with condition
            if ranges:
                encoded[key] = data[col].apply(lambda x: (x - minval) / ranges)
            else:
                encoded[key] = data[col].apply(lambda x: (x - minval))

        # fill missing values with -1
        if self.fill_:
            encoded.fillna(self.fill_, inplace=True)

        # merge encoded dataframe with original
        return data.merge(encoded, left_index=True, right_index=True, how="left")

    def fit_transform(self, data, columns):
        # perform both fit and transform

        self.fit(data, columns)
        return self.transform(data, columns)


# Scale the quantitative variables such that std = 1
# operates by grouping column by uid
class StandardScaler:
    def __init__(self, fillna=None):
        self.prop_ = {}
        self.fill_ = fillna

    def fit(self, data, columns):
        # save mean and std of every col
        self.prop_ = {}

        for col in columns:
            # calculate aggregates and save
            key = str(col) + "_std"
            stat = data[col].agg(["mean", "std"]).to_dict()
            self.prop_[col] = stat

    def transform(self, data, columns):
        # transform the column to fit
        # return full merged dataframe

        # temp dataframe
        encoded = pd.DataFrame()

        for col in columns:
            # find if already transformed
            key = str(col) + "_std"
            if key in columns:
                continue

            # pre-calculate
            mean = self.prop_[col]["mean"]
            stdv = self.prop_[col]["std"]

            # transform with condition
            if stdv:
                encoded[key] = data[col].apply(lambda x: (x - mean) / stdv)
            else:
                encoded[key] = data[col].apply(lambda x: (x - mean))

        # fill missing values
        if self.fill_:
            encoded.fillna(self.fill_, inplace=True)

        # merge encoded dataframe with original
        return data.merge(encoded, left_index=True, right_index=True, how="left")

    def fit_transform(self, data, columns):
        # perform both fit and transform

        self.fit(data, columns)
        return self.transform(data, columns)

