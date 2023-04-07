import pandas as pd


class Base:
    def day_of_week(self, df, datetime):
        feature = pd.to_datetime(df[datetime])
        feature = feature.dt.dayofweek.reset_index(drop=True)
        feature = feature.astype("float")
        return feature

    def week_of_year(self, df, datetime):
        feature = pd.to_datetime(df[datetime])
        feature = feature.dt.isocalendar().week.reset_index(drop=True)
        feature = feature.astype("float")
        return feature

    # convert series to supervised learning
    def series_to_supervised(self, df, lags=1, forecasts=1, dropnan=True):
        n_vars = 1 if type(df) is list else df.shape[1]
        df = pd.DataFrame(df)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(lags, 0, -1):
            cols.append(df.shift(i))
            names += [(str(df.columns[j]) + '(t-%d)' % (i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, forecasts):
            cols.append(df.shift(-i))
            if i == 0:
                names += [str(df.columns[j]) + '(t)' for j in range(n_vars)]
            else:
                names += [(str(df.columns[j]) + '(t+%d)' % (i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def time_features(self, df, datetime, resolution):
        self.data = pd.DataFrame()

        for r in resolution:
            feature = getattr(self, r)
            column = feature(df, datetime)
            self.data[r] = column

    def lag_features(self, df, output, lags):
        df = df[[output]].copy()

        data = self.series_to_supervised(df, lags=lags, forecasts=0, dropnan=False)
        self.data = pd.concat([self.data, data], axis="columns")

    def forecast_features(self, df, output, forecasts):
        df = df[[output]].copy()

        data = self.series_to_supervised(df, lags=0, forecasts=forecasts, dropnan=False)
        self.data = pd.concat([self.data, data], axis="columns")
        self.output = data.columns

    def additional_features(self, df, datetime, output):
        df = df.copy()
        df = df.drop(columns=[datetime, output])
        for c in df.columns:
            df[c] = df[c].astype("float")

        self.data = pd.concat([self.data, df], axis="columns")
        self.data = self.data.dropna().reset_index(drop=True)
