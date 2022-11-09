# modelop.schema.0: input_schema.avsc
# modelop.slot.1: in-use

import pandas as pd
import pickle
import numpy as np
from typing import List

# modelop.init
def begin() -> None:

    global lasso_model, train_encoded_columns, categorical_columns
    lasso_model = pickle.load(open("lasso_model.pickle", "rb"))
    train_encoded_columns = pickle.load(open("train_encoded_columns.pickle", "rb"))
    categorical_columns = pickle.load(open("categorical_columns.pickle", "rb"))


# modelop.score
def action(data: List[dict]) -> List[dict]:

    # Input data is a list of records (dicts) - Turn it into dataframe
    data = pd.DataFrame(data)

    data_ID = data["Id"]  # Saving the Id column
    data.drop("Id", axis=1, inplace=True)

    data["MasVnrType"] = data["MasVnrType"].fillna("None")
    data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
    data["Electrical"] = data["Electrical"].fillna("SBrkr")
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    data["GarageYrBlt"] = data["GarageYrBlt"].fillna(data["YearBuilt"])

    data["MSZoning"] = data["MSZoning"].fillna("RL")
    data["Functional"] = data["Functional"].fillna("Typ")
    data["BsmtHalfBath"] = data["BsmtHalfBath"].fillna(0)
    data["BsmtFullBath"] = data["BsmtFullBath"].fillna(0)
    data["Utilities"] = data["Utilities"].fillna("AllPub")
    data["SaleType"] = data["SaleType"].fillna("WD")
    data["GarageArea"] = data["GarageArea"].fillna(0)
    data["GarageCars"] = data["GarageCars"].fillna(2)
    data["KitchenQual"] = data["KitchenQual"].fillna("TA")
    data["TotalBsmtSF"] = data["TotalBsmtSF"].fillna(0)
    data["BsmtUnfSF"] = data["BsmtUnfSF"].fillna(0)
    data["BsmtFinSF2"] = data["BsmtFinSF2"].fillna(0)
    data["BsmtFinSF1"] = data["BsmtFinSF1"].fillna(0)
    data["Exterior2nd"] = data["Exterior2nd"].fillna("VinylSd")
    data["Exterior1st"] = data["Exterior1st"].fillna("VinylSd")

    data["MSSubClass"] = pd.Categorical(data.MSSubClass)
    data["YrSold"] = pd.Categorical(data.YrSold)
    data["MoSold"] = pd.Categorical(data.MoSold)

    #  Computing total square-footage as a new feature
    data["TotalSF"] = data["TotalBsmtSF"] + data["firstFlrSF"] + data["secondFlrSF"]

    #  Computing total 'porch' square-footage as a new feature
    data["Total_porch_sf"] = (
        data["OpenPorchSF"]
        + data["threeSsnPorch"]
        + data["EnclosedPorch"]
        + data["ScreenPorch"]
        + data["WoodDeckSF"]
    )

    #  Computing total bathrooms as a new feature
    data["Total_Bathrooms"] = (
        data["FullBath"]
        + (0.5 * data["HalfBath"])
        + data["BsmtFullBath"]
        + (0.5 * data["BsmtHalfBath"])
    )

    # Engineering some features into Booleans
    f = lambda x: bool(1) if x > 0 else bool(0)

    data["has_pool"] = data["PoolArea"].apply(f)
    data["has_garage"] = data["GarageArea"].apply(f)
    data["has_bsmt"] = data["TotalBsmtSF"].apply(f)
    data["has_fireplace"] = data["Fireplaces"].apply(f)

    data = data.drop(["threeSsnPorch", "PoolArea", "LowQualFinSF"], axis=1)

    print("Shape of data before `get dummies`: ", data.shape, flush=True)

    encoded_features = pd.get_dummies(data, columns=categorical_columns)

    print("Shape of data after `get dummies`: ", encoded_features.shape, flush=True)

    # Matching dummy variables from training set to current dummy variables
    missing_cols = set(train_encoded_columns) - set(encoded_features.columns)

    print("Number of Missing Columns: ", len(missing_cols), flush=True)
    for c in missing_cols:
        encoded_features[c] = 0

    # Matching order of variables to those used in training
    encoded_features = encoded_features[train_encoded_columns]

    print("Shape of Encoded Features: ", encoded_features.shape, flush=True)

    # Model was trained on log(SalePrice)
    log_predictions = lasso_model.predict(encoded_features)

    adjusted_predictions = {}
    adjusted_predictions["ID"] = data_ID.astype(int)
    adjusted_predictions["Lasso"] = np.expm1(log_predictions)

    # Round output
    output = np.round(np.array(pd.DataFrame(adjusted_predictions)), 2).tolist()

    return output


# Test Script
if __name__ == "__main__":
    begin()
    data = pd.read_json("sample_input.json")
    output = action(data)
    assert output == [[1460.0, 120429.01], [1462.0, 123970.31]]
