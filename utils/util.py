from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, max_depth=100, min_samples_split=2, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    MAE = mean_absolute_error(y_valid, predictions)
    return {
        'MAE': MAE,
        'data_set': {'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}
    }
