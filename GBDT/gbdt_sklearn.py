import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    learning_rate=0.2
    n_estimators=55
    subsample=1
    min_samples_split=2
    min_samples_leaf=1
    max_depth=3
    # name of features
    featName = ['Number', 'Plasma', 'Diastolic', 'Triceps', '2-Hour', 'Body', 'Diabetes', 'Age', 'Class']
    path = 'xg.csv'
    # read data file
    data = pd.read_csv(path, sep=',', header=0, names=featName)
    # set random seed
    np.random.seed(123)
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values,
                                                        test_size=0.2, random_state=123)

    gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample
                                  , min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_depth=max_depth
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )
    gbdt.fit(X_train, y_train)
    y_hat = gbdt.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    print(acc)

    import pickle

    with open('gbdt.pickle', 'wb') as f:
        pickle.dump(gbdt,f)
    with open('gbdt.pickle', 'rb') as f:
        clf2 = pickle.load(f)

    pred = clf2.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(acc)
