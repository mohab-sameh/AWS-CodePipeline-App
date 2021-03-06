from __future__ import print_function

import argparse
import os

import joblib
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    print("Input Files:")
    print(input_files)
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )
    #raw_data = [pd.read_csv(file, header=None, engine="c", low_memory=False) for file in input_files]
    raw_data = [pd.read_csv(file, engine="c", low_memory=False) for file in input_files]
    train_data = pd.concat(raw_data)

    print("Data Head:")
    print(train_data.head())
    train_y = train_data[['wage_eur']]
    #train_X= train_data.drop('wage_eur', axis = 1)
    train_X = train_data[train_data.columns[~train_data.columns.isin(['wage_eur'])]]

    print("Train DF columns")
    print(train_y.columns)
    print(train_X.columns)



    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor(n_estimators = 1000, random_state = 0)
    clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf