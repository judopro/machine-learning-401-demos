import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

def encode_labels(df):
    le = preprocessing.LabelEncoder()
    for column in df.columns:
        if df[column].dtype == type(object):
            df[column] = le.fit_transform(df[column])
    return df


training_df = pd.read_csv('files/Regression_Training.csv')
test_df = pd.read_csv('files/Regression_Test.csv')

int_rate = training_df["int_rate"]
training_df = training_df.drop('int_rate', axis=1)
training_df = training_df.drop('grade', axis=1)
test_df = test_df.drop('grade', axis=1)

training_df = encode_labels(training_df)
test_df = encode_labels(test_df)

model = LinearRegression().fit(training_df, int_rate)

res = model.predict(test_df)
predicted_df = pd.Series(res)

r_square = model.score(training_df, int_rate)
print("Score: " + str(r_square))
predicted_df.to_csv("files/prediction_linear_rgr.csv", index=False)
