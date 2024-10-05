from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


def getNumericleColumns(X_train):
    return [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

def getCategoricalColumns(X_train):
    return [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

def getModel():

    return XGBClassifier(
        max_depth=3,  # Maximum tree depth
        learning_rate=0.1,  # Learning rate
        n_estimators=100,  # Number of trees
        # use_label_encoder=False, # deprecated  # Avoid label encoding warning
        eval_metric='logloss'  # Metric used for evaluation
    )

def get_pipeline(X_train, X_valid, y_train, y_valid):
    # label_encoder = LabelEncoder()
    # y_train = label_encoder.fit_transform(y_train)
    # y_valid = label_encoder.transform(y_valid)

    # Select numerical columns
    numerical_cols = getNumericleColumns(X_train)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = getCategoricalColumns(X_train)

    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train[my_cols].copy()
    X_valid = X_valid[my_cols].copy()

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    model = getModel()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

    return {
        'pipeline': pipeline,
        'data_set': {
            'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}
        }

def print_prediction_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")