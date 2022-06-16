import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from get_data import read_params
import argparse
import joblib
import json

def eval_metrics(actual, pred):
    rmse = metrics.mean_absolute_error(actual, pred)
    mae = metrics.mean_squared_error(actual, pred)
    accuracy = metrics.accuracy_score(actual, pred)
    return rmse, mae, accuracy

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)


    rf = RandomForestClassifier(
        n_estimators = n_estimators)
    rf.fit(train_x, train_y)

    predicted_qualities = rf.predict(test_x)
    
    (rmse, mae, accuracy) = eval_metrics(test_y, predicted_qualities)

    print("RandomForestClassifier (n_estimators=%f):" % (n_estimators))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  ACCURACY: %s" % accuracy)


    ###################### reports & scores ###############################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "accuracy": accuracy
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "n_estimators": n_estimators,
        }
        json.dump(params, f, indent=4)


##################### save model ################################


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(rf, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)