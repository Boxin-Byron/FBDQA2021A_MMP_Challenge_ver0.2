import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob
import os
from sklearn.model_selection import KFold

# Load data
data_files = []
for i in range(10):
    data_files.extend(glob.glob(f'data_processed/sym{i}_data*_am.csv'))
    data_files.extend(glob.glob(f'data_processed/sym{i}_data*_pm.csv'))

file_dir = './data_processed'
i = 0
dates = list(range(12))
df = pd.DataFrame()
for date in dates:
    if (date & 1):
        file_name = f"snapshot_sym{i}_date{date//2}_am.csv"
    else:
        file_name = f"snapshot_sym{i}_date{date//2}_pm.csv"
    df = df.append(pd.read_csv(os.path.join(file_dir,file_name)))
    
# Concatenate all sampled data
data = pd.concat(df, ignore_index=True)

# Prepare features and labels
X = data.drop(columns=['date','time', 'sym', 'label5', 'label10', 'label20', 'label40', 'label60'])
y = data['label5']

# Split data into training and testing sets
# Ensure data is sorted by date and time to maintain the time series order
data = data.sort_values(by=['date', 'time'])

# Split data into training and testing sets without shuffling to prevent data leakage
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

X_train = train_data.drop(columns=['date', 'time', 'sym', 'label5', 'label10', 'label20', 'label40', 'label60'])
y_train = train_data['label5']
X_test = test_data.drop(columns=['date', 'time', 'sym', 'label5', 'label10', 'label20', 'label40', 'label60'])
y_test = test_data['label5']

# Train XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature importance
importance = model.feature_importances_
importance_normalized = importance / importance.max()
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance_normalized
}).sort_values(by='importance', ascending=False)

print(feature_importance)

# Prepare features and labels
X = data.drop(columns=['date', 'time', 'sym', 'label5', 'label10', 'label20', 'label40', 'label60'])
y = data['label5']

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=False)

mse_list = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train XGBoost model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

    # Feature importance
    importance = model.feature_importances_
    importance_normalized = importance / importance.max()
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance_normalized
    }).sort_values(by='importance', ascending=False)

    print(feature_importance)

# Print average Mean Squared Error
print(f'Average Mean Squared Error: {sum(mse_list) / len(mse_list)}')