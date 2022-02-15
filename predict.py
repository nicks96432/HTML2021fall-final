import numpy
import matplotlib.pyplot
import pandas
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.model_selection
import sklearn.preprocessing
import imblearn.ensemble
import xgboost.sklearn

# preprocessing: one hot encode data
train_data = pandas.read_csv("./train.csv")
test_data = pandas.read_csv("./test.csv")
location = pandas.read_csv("./data/location.csv")

city_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, dtype=int).fit(
    location["City"].dropna().to_numpy().reshape(-1, 1))
transformed = city_encoder.transform(
    train_data["City"].to_numpy().reshape(-1, 1))
train_data = pandas.concat([train_data, pandas.DataFrame(
    transformed, columns=city_encoder.get_feature_names_out())], axis=1).drop("City", axis=1)
transformed = city_encoder.transform(
    test_data["City"].to_numpy().reshape(-1, 1))
test_data = pandas.concat([test_data, pandas.DataFrame(
    transformed, columns=city_encoder.get_feature_names_out())], axis=1).drop("City", axis=1)

train_y = train_data.pop("Churn Category")
train_x = pandas.get_dummies(train_data)
test_x = pandas.get_dummies(test_data)

train_x = pandas.DataFrame(sklearn.preprocessing.StandardScaler(
).fit_transform(train_x), columns=train_x.keys())
test_x = pandas.DataFrame(sklearn.preprocessing.StandardScaler(
).fit_transform(test_x), columns=test_x.keys())


# use matplotlib to show validation result
n_estimator = list(range(5, 500, 5))
n_scores = []
for n in n_estimator:
    model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n)
    scores: numpy.ndarray = sklearn.model_selection.cross_val_score(
        model, train_x, train_y, cv=12, n_jobs=12)
    n_scores.append(scores.mean())
matplotlib.pyplot.plot(n_estimator, n_scores)
matplotlib.pyplot.xlabel(
    'Value of n_estimators for BalancedRandomForestClassifier')
matplotlib.pyplot.ylabel('Cross-Validated Accuracy')
matplotlib.pyplot.show()


# # models

# # public: 0.26941, private: 0.26404
# result = sklearn.linear_model.LogisticRegression(
#     solver="liblinear").fit(train_x, train_y).predict(test_x)

# # public: 0.27870, private: 0.30595
# model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=45)
# result = model.fit(train_x, train_y).predict(test_x)

# # public: 0.30741, private: 0.30677
# model = xgboost.sklearn.XGBClassifier(
#     max_depth=3, learning_rate=0.09, n_estimators=185,
#     use_label_encoder=False, objective='binary:logistic',
#     min_child_weight=5, subsample=0.8)
# result = model.fit(train_x, train_y).predict(test_x)

# # public: 0.27456, private: 0.33861
# model = imblearn.ensemble.BalancedRandomForestClassifier(
#     n_estimators=600, max_depth=4)
# result = model.fit(train_x, train_y).predict(test_x)

# #output the result
# output = pandas.read_csv("./data/sample_submission.csv")
# output["Churn Category"] = result
# output.to_csv("output.csv", index=False)
