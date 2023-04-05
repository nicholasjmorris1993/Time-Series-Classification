import pandas as pd
from river import stream
from river import forest
from river import metrics
import sys
sys.path.append("/home/nick/Time-Series-Classification/src")
from base import Base


def streaming(df, datetime, output, lags, forecasts, resolution, test_frac):
    model = Streaming()
    model.time_features(df, datetime, resolution)
    model.lag_features(df, output, lags)
    model.forecast_features(df, output, forecasts)
    model.additional_features(df, datetime, output)
    model.train(test_frac)
    model.predict()

    return model


class Streaming(Base):
    def train(self, test_frac):
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = dict()

        for out in self.output:
            X = train.copy().drop(columns=self.output)
            Y = train.copy()[[out]]

            model = forest.ARFClassifier(n_models=10, seed=42)

            for x, y in stream.iter_pandas(X, Y):
                model = model.learn_one(x, y[out])

            self.model[out] = model

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))
        
        self.metric = dict()
        self.predictions = dict()

        for out in self.output:
            X = test.copy().drop(columns=self.output)
            Y = test.copy()[[out]]

            model = self.model[out]
            metric = metrics.Accuracy()
            predictions = pd.DataFrame()

            for x, y in stream.iter_pandas(X, Y):
                y_pred = model.predict_one(x)
                metric = metric.update(y[out], y_pred)
                model = model.learn_one(x, y[out])

                pred = pd.DataFrame({
                    "Actual": [y[out]],
                    "Predicted": [y_pred],
                })
                predictions = pd.concat([
                    predictions, 
                    pred,
                ], axis="index").reset_index(drop=True)

            self.model[out] = model
            self.metric[out] = metric
            self.predictions[out] = predictions
