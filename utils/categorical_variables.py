import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from util.util import score_dataset


def get_categorical_variables_processed_dataset(X_train, X_valid, y_train, y_valid):
    # Get list of categorical variables
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)

    # Make copy to avoid changing original data
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()

    # Apply ordinal encoder to each column with categorical data
    ordinal_encoder = OrdinalEncoder()
    label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
    label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

    result = score_dataset(label_X_train, label_X_valid, y_train, y_valid)
    MAE = result.MAE
    data_set = result.data_set

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    # Ensure all columns have string type
    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_valid.columns = OH_X_valid.columns.astype(str)


    result = score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)
    if result.MAE < MAE:
        data_set = result.data_set

    return data_set