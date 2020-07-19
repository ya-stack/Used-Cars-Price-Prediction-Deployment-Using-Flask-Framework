import pandas as pd
import datetime
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import model_selection
from xgboost import XGBRegressor

train_data = pd.read_excel("Data_Train.xlsx")
test_data = pd.read_excel("Data_Test.xlsx")
train_data.head()

# Data Preprocessing
train_data["brand_name"] = train_data["Name"].apply(lambda x: str(x).split(" ")[0])
test_data["brand_name"] = test_data["Name"].apply(lambda x: str(x).split(" ")[0])
train_data.drop('Name', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)


def fill_na_with_mode(ds, brandname):
    fill_value = ds.loc[ds['brand_name'] == brandname]['Seats'].mode()[0]
    condit = ((ds['brand_name'] == brandname) & (ds['Seats'].isnull()))
    ds.loc[condit, 'Seats'] = ds.loc[condit, 'Seats'].fillna(fill_value)

car_brand = ['Maruti', 'Hyundai', 'BMW', 'Fiat', 'Land', 'Ford', 'Toyota', 'Honda', 'Skoda', 'Mahindra']
for c in car_brand:
    fill_na_with_mode(train_data, c)
    fill_na_with_mode(test_data, c)

train_data["Mileage"] = train_data["Mileage"].str.replace("km/kg", "")
train_data["Mileage"] = train_data["Mileage"].str.replace("kmpl", "")
train_data["Engine"] = train_data["Engine"].str.replace("CC", "")
train_data["Power"] = train_data["Power"].str.replace("bhp", "")

test_data["Mileage"] = test_data["Mileage"].str.replace("km/kg", "")
test_data["Mileage"] = test_data["Mileage"].str.replace("kmpl", "")
test_data["Engine"] = test_data["Engine"].str.replace("CC", "")
test_data["Power"] = test_data["Power"].str.replace("bhp", "")

train_data["Mileage"] = pd.to_numeric(train_data["Mileage"], errors='coerce')
train_data["Engine"] = pd.to_numeric(train_data["Engine"], errors='coerce')
train_data["Power"] = pd.to_numeric(train_data["Power"], errors='coerce')

test_data["Mileage"] = pd.to_numeric(train_data["Mileage"], errors='coerce')
test_data["Engine"] = pd.to_numeric(train_data["Engine"], errors='coerce')
test_data["Power"] = pd.to_numeric(train_data["Power"], errors='coerce')

train_data.drop(train_data[train_data["brand_name"] == 'Smart'].index, axis=0, inplace=True)

test_data.drop(train_data[train_data["brand_name"] == 'Hindustan'].index, axis=0, inplace=True)

def missing_values(df, brandname, colname):
    value = df[df['brand_name'] == brandname][colname].mode()[0]
    condition = ((df['brand_name'] == brandname) & (df[colname].isnull()))
    df.loc[condition, colname] = df.loc[condition, colname].fillna(value)


mileage_brands = train_data[train_data['Mileage'].isnull()]['brand_name'].unique()
power_brands = train_data[train_data['Power'].isnull()]['brand_name'].unique()
engine_brands = train_data[train_data['Engine'].isnull()]['brand_name'].unique()

for x in mileage_brands:
    missing_values(train_data, x, 'Mileage')
for y in power_brands:
    missing_values(train_data, y, 'Power')
for z in engine_brands:
    missing_values(train_data, z, 'Engine')

power_brands = test_data[test_data['Power'].isnull()]['brand_name'].unique()
engine_brands = test_data[test_data['Engine'].isnull()]['brand_name'].unique()

for x in power_brands:
    missing_values(test_data, x, 'Power')
for y in engine_brands:
    missing_values(test_data, y, 'Engine')
print(mileage_brands)
print(power_brands)
print(engine_brands)

zero_mileage = train_data[train_data['Mileage'] == 0.0]['brand_name'].unique()

for x in zero_mileage:
    value = train_data[train_data['brand_name'] == x]['Mileage'].mode()[0]
    cond = ((train_data['brand_name'] == x) & (train_data['Mileage'] == 0.0))
    train_data.loc[cond, 'Mileage'] = value

zero_mileage = test_data[test_data['Mileage'] == 0.0]['brand_name'].unique()

for x in zero_mileage:
    value = test_data[test_data['brand_name'] == x]['Mileage'].mode()[0]
    cond = ((test_data['brand_name'] == x) & (test_data['Mileage'] == 0.0))
    test_data.loc[cond, 'Mileage'] = value

m1 = train_data['Seats'] == 0.0
train_data.loc[m1, 'Seats'] = 5.0

now = datetime.datetime.now()
train_data["Year_upd"] = train_data["Year"].apply(lambda x: now.year - x)
test_data["Year_upd"] = test_data["Year"].apply(lambda x: now.year - x)

train_data.drop("Year", axis=1, inplace=True)
test_data.drop("Year", axis=1, inplace=True)

# More than 75% of the values are missing in New Price
train_data.drop("New_Price", axis=1, inplace=True)
test_data.drop("New_Price", axis=1, inplace=True)

# It is of no relevance in price prediction of used cars
train_data.drop("Location", axis=1, inplace=True)
test_data.drop("Location", axis=1, inplace=True)

# Removing skewness from target variable
train_data['Price'] = np.log1p(train_data['Price'])

from sklearn.preprocessing import OneHotEncoder
import pickle

# Fit and save an OneHotEncoder in train
columns_to_fit = ['Fuel_Type', 'Transmission','Owner_Type','brand_name']
enc = OneHotEncoder(sparse=False,handle_unknown='ignore').fit(train_data.loc[:, columns_to_fit])
pickle.dump(enc, open('encoder.pickle', 'wb'))
column_names = enc.get_feature_names(columns_to_fit)
encoded_variables = pd.DataFrame(enc.transform(train_data.loc[:, columns_to_fit]), columns=column_names)
train_data = train_data.drop(columns_to_fit, 1)
train_set = pd.concat([train_data, encoded_variables], axis=1)

## -- test dataset
columns_to_fit = ['Fuel_Type', 'Transmission','Owner_Type','brand_name']
column_names = enc.get_feature_names(columns_to_fit)
encoded_variables = pd.DataFrame(enc.transform(test_data.loc[:, columns_to_fit]), columns=column_names)
test_data = test_data.drop(columns_to_fit, 1)
test_set = pd.concat([test_data, encoded_variables], axis=1)
train_set = train_set.fillna(value=0)
test_set = test_set.fillna(value=0)

# Splitting train data set into independent variables and dependent variable
X_train = train_set.drop("Price", axis=1)
y_train = train_set["Price"]
X_test = test_set

# Making predictions on validation set
X_train1, X_valid, y_train1, y_valid = train_test_split(X_train,y_train,test_size=0.3,random_state=1)

# Model building
regressor = XGBRegressor()
regressor.fit(X_train1, y_train1)
print(regressor.score(X_train1, y_train1))
print(regressor.score(X_valid, y_valid))

# Cross validation score
print(np.mean((model_selection.cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2'))))

# Let's look at the features of train and test dataset
train_features = X_train.columns
test_features = X_test.columns
print(train_features)
print(test_features)
print(len(train_features))
print(len(test_features))

# Dumping model using pickle
X_train = X_train.values
y_train = y_train.values
X_test = test_set.values
regressor.fit(X_train, y_train)
pickle.dump(regressor, open('model.pkl', 'wb'))
