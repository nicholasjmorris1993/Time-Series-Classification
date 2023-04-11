import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import sys
sys.path.append("/home/nick/Time-Series-Classification/src")
from base import Base


def nnet(df, datetime, output, lags, forecasts, resolution, test_frac, deep):
    model = NNet()
    model.time_features(df, datetime, resolution)
    df = model.encode_labels(df, output)
    model.lag_features(df, output, lags)
    model.forecast_features(df, output, forecasts)
    model.additional_features(df, datetime, output)
    model.scale(test_frac)
    model.train(deep)
    model.predict()

    return model


class NNet(Base):
    def encode_labels(self, df, output):
        labels = df[output].tolist()
        self.encoder = LabelEncoder()
        df[output] = self.encoder.fit_transform(labels)

        return df

    def scale(self, test_frac):
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))
        X = train.copy().drop(columns=self.output)

        # standardize the inputs to take on values between 0 and 1
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = self.data.copy().drop(columns=self.output)
        columns = X.columns
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns=columns)

        Y = self.data.copy()[self.output]
        self.data = pd.concat([Y, X], axis="columns")

    def train(self, deep):
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = dict()

        if deep:
            layer = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
        else:
            layer = [128, 128]

        for out in self.output:
            X = train.copy().drop(columns=self.output)
            Y = train.copy()[[out]]

            model = MLPClassifier(
                max_iter=1000,
                hidden_layer_sizes=layer,
                learning_rate_init=0.001,
                learning_rate="adaptive",
                batch_size=16,
                random_state=42,
            )
            model.fit(X, Y.to_numpy().ravel().astype("int"))

            self.model[out] = model

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))
        
        self.metric = dict()
        self.predictions = dict()

        for out in self.output:
            X = test.copy().drop(columns=self.output)
            Y = test.copy()[[out]]

            model = self.model[out]
            y_pred = model.predict(X)
            y_true = Y.to_numpy().ravel().astype("int")

            metric = accuracy_score(
                y_true=y_true, 
                y_pred=y_pred,
            )
            metric = f"Accuracy: {100 * round(metric, 4)}%"

            predictions = pd.DataFrame({
                "Actual": self.encoder.inverse_transform(y_true),
                "Predicted": self.encoder.inverse_transform(y_pred),
            })

            self.metric[out] = metric
            self.predictions[out] = predictions
