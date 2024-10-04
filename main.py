import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from utils.categorical_variables import get_categorical_variables_processed_dataset
from utils.missing_value import getMissingProecessedDataSet

# print(os.getcwd())
# print(os.path.abspath('./data_sets/train.csv'))

df = pd.read_csv('utils/data_sets/train.csv', index_col="PassengerId")
# print(df.head(10))

X = df.drop(columns='Survived', axis=1)
y = df['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=5)


# X_train = pd.get_dummies(X_train, drop_first=True)
# X_valid = pd.get_dummies(X_valid, drop_first=True)
#
# # Align columns of training and validation sets
# X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)


data_set = getMissingProecessedDataSet(X_train, X_valid, y_train, y_valid)
# Destructure the data_set
X_train, X_valid, y_train, y_valid = data_set.values()
data_set = get_categorical_variables_processed_dataset(X_train, X_valid, y_train, y_valid)