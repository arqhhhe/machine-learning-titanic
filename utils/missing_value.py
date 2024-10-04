import pandas as pd
from sklearn.impute import SimpleImputer

from util.util import score_dataset

def getMissingProecessedDataSet(X_train, X_valid, y_train, y_valid):

    result = score_dataset(X_train, X_valid, y_train, y_valid)
    # print("MAE from Approach 1 (Leave as is):")
    MAE = result.MAE
    data_set = result.data_set

    col_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    reduced_X_train = X_train.drop(col_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(col_with_missing, axis=1)

    result = score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)
    # print("MAE from Approach 2 (Drop missing value column):")
    if result.MAE < MAE:
        MAE = result.MAE
        data_set = result.data_set

    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    for col in col_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()


    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns


    result = score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)
    if result.MAE < MAE:
        data_set = result.data_set

    return data_set