import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


data = load_iris()
df = pd.DataFrame(
    data['data'],
    columns = ['SL', 'SW', 'PL', 'PW']
)
df['target'] = data['target']
df['sp'] = df['target'].apply(
    lambda i: data['target_names'][i]
)
# print(df.shape)
# print(df.isnull().sum())



# # # SPLITTING # # #

x_train, x_test, y_train, y_test = train_test_split(
    df[['SL', 'SW', 'PL', 'PW']], 
    df['sp'], 
    test_size = .05
)



# FITTING Model Random Forest

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators = 100
)

model.fit(x_train, y_train)
# print(round(model.score(x_train, y_train) * 100, 2), '%')
# print(round(model.score(x_test, y_test) * 100, 2), '%')

# print(x_test.iloc[0])
# print(y_test.iloc[0])
# print(model.predict([x_test.iloc[0]]))
# print(model.predict_proba([x_test.iloc[0]]))
pred = (model.predict([[ 2.0, 3.2, 1.0, 1.9 ]]))
prob = (model.predict_proba([[ 2.0, 3.2, 1.0, 1.9 ]]))
print(round(np.max(prob) * 100, 2), '%', pred)
# 53 % Setosa

import joblib
joblib.dump(model, 'modelRanFor.joblib')