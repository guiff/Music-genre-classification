import pickle
import warnings
import os.path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


warnings.filterwarnings('ignore')


features = ['f' + str(i) for i in range(1, 471)] + ['genre'] # 470 features in total
data = pd.read_csv('features_train.csv', names=features).values

X = data[:, 0:470] # Array with 470 features for each song of the training data
Y = data[:, 470] # Array with genre name


# Feature selection with Recursive Feature Elimination
def rfe(nb_features):
    if os.path.isfile('fit_data.dmp'):
        with open('fit_data.dmp','rb') as f:
            fit_data = pickle.load(f) # Load the RFE model if already on disk
    else:
        rfe = RFE(estimator=LogisticRegression(), n_features_to_select=nb_features)  # Select top features
        fit_data = rfe.fit(X, Y)
        with open('fit_data.dmp', 'wb') as f:
            pickle.dump(fit_data, f) # Save the RFE model to the disk

    filter_condition = np.hstack((fit_data.ranking_ == 1, [False]))
    filtered_features = data[:, filter_condition].astype(float)
    return filtered_features, data[:, -1], filter_condition[:-1]


# Keep only the top 150 features to train the classifier
x_train, y_train, selection_filter = rfe(150)


# Define the classifier
estimators = [('svm', SVC(probability=True)), ('logReg', LogisticRegression()), ('lda', LinearDiscriminantAnalysis()), ('xgb', XGBClassifier())]
clf_voting = VotingClassifier(estimators=estimators, weights=[1, 2, 1, 1], voting='soft')
clf_voting.fit(x_train, y_train)


# Predict the test dataset
test_features = pd.read_csv('features_test.csv', header=None).values
x_test = test_features[:, selection_filter].astype(float)
predictions = clf_voting.predict(x_test)


# Save the predictions to a txt file
file = open('predictions.txt', 'w')
for i in range(len(predictions)):
    print(predictions[i], file=file)
file.close()
