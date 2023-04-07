import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("/home/nick/Time-Series-Classification/src")
from online import streaming


data = pd.read_csv("/home/nick/Time-Series-Classification/test/traffic.txt", sep="\t")

model = streaming(
    df=data, 
    datetime="Day", 
    output="Vehicles", 
    lags=7, 
    forecasts=7, 
    resolution=["day_of_week", "week_of_year"],
    test_frac=0.5,
)

forecast = 7
print(model.metric[model.output[forecast - 1]])

predictions = model.predictions[model.output[forecast - 1]].copy()

labels = np.unique(predictions.to_numpy())
cmatrix = confusion_matrix(
    y_true=predictions["Actual"],  # rows
    y_pred=predictions["Predicted"],  # columns
    labels=labels,
)
cmatrix = pd.DataFrame(cmatrix, columns=labels, index=labels)
print(cmatrix)
