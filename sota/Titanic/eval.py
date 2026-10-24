import argparse
import importlib
import os

from pathlib import Path as p
from os.path import join as pj
import sys


import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
import numpy as np
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def create_save_dir(save_root):
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)    

    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            i = -1
            while exp_name[i].isdigit():
                i -= 1
            i += 1
            if i != 0:
                n.append(int(exp_name[i:]))
    
    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = f"{save_root}/exp{sorted(n)[-1] + 1}"
    
    return save_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model", help="model file")
    parser.add_argument('--save_dir', type=str, default="trained", help="path where the trained model will be saved")
    parser.add_argument('--random_seed', type=int, default=42, help="random seed to use for deterministic evaluation")
    parser.add_argument('--variant_dir', type=str, default='models', help="directory where models will be written by LLM-GE")
    return parser.parse_args()



if __name__ == '__main__':
    script_directory = p(__file__).parent.resolve()
    os.chdir(script_directory)

    args = get_args()

    # Set the random seeds for deterministic results
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # This is LLM Guided Code
    # Import the module dynamically
    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)



    # this will get the gene id value 
    try:
        gene_id = args.model.split('model_')[1] # Causing an error
    except:
        gene_id = 'seed'
    save_dir = f'{args.save_dir}/{gene_id}' 
    create_save_dir(save_dir)


    train_df = pd.read_csv("data/processed_train.csv", index_col=0)
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]


    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=10
    )

    fold_aucs = []
    fp_count, fn_count = 0, 0

    for train_idx, val_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        model = model_module.Model()
        model.fit(X_train, y_train)
        prob_predictions = model.predict_prob(X_val)
        predictions = model.predict(X_val)
        auc = roc_auc_score(y_val,prob_predictions)
        fold_aucs.append(auc)
        
        matrix = sklearn.metrics.confusion_matrix(y_val, predictions)
        fp_count += matrix[0,1]
        fn_count += matrix[1,0]

        print(
            f"Fold {len(fold_aucs)}: "
            f"train={len(train_idx)}, "
            f"val={len(val_idx)}, "
            f"AUC={auc:.4f}"
        )

    mean_roc_auc = sum(fold_aucs) / len(fold_aucs)

    """
    Count Total parameters in the Model
    model.parameters() returns an iterable of all the model's parameter
    """
    results_text = f"roc_auc,fp_count,fn_count\n{mean_roc_auc},{fp_count},{fn_count}"

    """
    This line formats the results into comma-separated string
    fp_count: Number of false positives.
    fn_count: Number of false negatives.
    """

    # Define Output Filename
    """
    Defines an Output Filename
    gene_id is a unique identifier for the experiment or model.
    """
    filename = os.path.abspath(f'results/{gene_id}_results.csv')
    
    dir_path = os.path.dirname(filename)

    # Create the directory, ignore error if it already exists
    os.makedirs(dir_path, exist_ok=True)

    """
    os.path.dirname(filename) extracts the directory path from the filename.
    os.makedirs(dir_path, exist_ok=True) creates the directory if it does not already exist. 
    The exist_ok=True parameter ensures that no error is raised if the directory already exists.
    """

    # Open the file in write mode and write the text
    with open(filename, 'w') as file:
        file.write(results_text)
    """
    This block opens the specified file in write mode ('w').
    If the file does not exist, it will be created.
    The results_text string is written to the file.
    """

    print(f"results have been written to {filename}")

    print('='*120);print('job done');print('='*120)

