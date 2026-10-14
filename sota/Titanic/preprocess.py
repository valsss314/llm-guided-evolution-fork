
import random
import numpy as np
import pandas as pd

if __name__ == "__main__":
    random.seed(100)
    np.random.seed(100)

    train_data = pd.read_csv("data/train.csv")

    train_data.drop(
        columns=["Name", "Ticket", "Cabin"],
        inplace=True
    )
    train_data.set_index(
        keys=["PassengerId"],
        drop=True,
        inplace=True
    )

    train_nan_map = {
        "Age": train_data["Age"].mean(),
        "Fare": train_data["Fare"].mean(),
        "Embarked": train_data["Embarked"].mode()[0]
    }

    train_data.fillna(
        value=train_nan_map,
        inplace=True
    )

    columns_map = {
        "Embarked": {"C": 0, "Q": 1, "S": 2},
        "Sex": {"male": 0, "female": 1}
    }

    pd.set_option("future.no_silent_downcasting", True)
    train_data.replace(columns_map, inplace=True)

    train_data.to_csv("data/processed_train.csv")

    print("Preprocessing complete!")
    print("Shape:", train_data.shape)
    print("Columns:", train_data.columns.tolist())
