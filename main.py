import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils.pipelines import get_pipeline, print_prediction_metrics, getNumericleColumns, getCategoricalColumns

# from utils.categorical_variables import get_categorical_variables_processed_dataset
# from utils.missing_value import getMissingProecessedDataSet

# print(os.getcwd())
# print(os.path.abspath('./data_sets/train.csv'))

df = pd.read_csv('utils/data_sets/train.csv', index_col="PassengerId")

# print(df.head(10))

X = df.drop(columns='Survived', axis=1)
y = df['Survived']

X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = train_test_split(X, y, test_size=0.20, random_state=5)

result = get_pipeline(X_train_raw, X_valid_raw, y_train_raw, y_valid_raw)
pipeline = result['pipeline']
X_train, X_valid, y_train, y_valid = result['data_set'].values()
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, predictions)
print('MAE:', score)

print_prediction_metrics(y_valid, predictions)

test_df = pd.read_csv('utils/data_sets/test.csv', index_col="PassengerId")
numerical_cols = getNumericleColumns(test_df)
categorical_cols = getCategoricalColumns(test_df)
my_cols = numerical_cols + categorical_cols
X_test = test_df[my_cols].copy()

predictions = pipeline.predict(X_test)

print(predictions)

results = pd.DataFrame({
    'PassengerId': X_test.index,  # PassengerId from the index
    'Survived': predictions          # Predictions from the model
})

# Step 5: (Optional) Save the results to a CSV file
results.to_csv('./utils/data_sets/submission1.csv', index=False)

# Print or display the results
print(results)