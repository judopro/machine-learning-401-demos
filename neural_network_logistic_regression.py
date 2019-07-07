import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

def encode_labels(df):
    le = preprocessing.LabelEncoder()
    for column in df.columns:
        if df[column].dtype == type(object):
            df[column] = le.fit_transform(df[column])
    return df


training_df = pd.read_csv('files/Classification_Training.csv')
test_df = pd.read_csv('files/Classification_Test.csv')

approved = training_df["approved"]
training_df = training_df.drop('approved', axis=1)

training_df = encode_labels(training_df)
test_df = encode_labels(test_df)

model = MLPClassifier(hidden_layer_sizes=(21, 21, 21), max_iter=10000)
model.fit(training_df, approved)

res = model.predict_proba(test_df)

predicted_df = pd.Series(res[:, 1])
predicted_df.to_csv("files/prediction_neural_network_logistic_rgr.csv", index=False)
