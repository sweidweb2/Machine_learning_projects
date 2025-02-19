# Read in the car sales data
import pandas as pd
import numpy as np

car_sales = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales-extended-missing-data.csv")

print(car_sales.head())
print()
print(car_sales.info())
print(car_sales.isna().sum())
print(car_sales.dtypes)
car_sales.dropna(subset=["Price"], inplace=True)

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

categorical_features = ["Make", "Colour"]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

door_feature = ["Doors"]

door_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=4))])

numeric_features = ["Odometer (KM)"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="Median"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("door", door_transformer, door_feature),
        ("num", numeric_transformer,numeric_features)])

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

regression_models = {"Ridge": Ridge(),
                     "SVR_linear": SVR(kernel="linear"),
                     "SVR_rbf": SVR(kernel="rbf"),
                     "RandomForestRegressor": RandomForestRegressor()}


regression_results = {}

car_sales_X = car_sales.drop("Price",axis=1)

car_sales_y = car_sales["Price"]

car_X_train, car_X_test, car_y_train, car_y_test = train_test_split(car_sales_X,
                                                                    car_sales_y,
                                                                    test_size=20,
                                                                    random_state=42)


print(car_X_train, car_X_test, car_y_train, car_y_test)

for model_name, model in regression_models.items():
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                     ("model", model)])

    print(f"Fitting {model_name}...")
    model_pipeline.fit(car_X_train, car_y_train)

    print(f"Scoring {model_name}...")
    regression_results[model_name] = model_pipeline.score(car_X_test,
                                                          car_y_test)


print(regression_results)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

ridge_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                 ("model", Ridge())])

ridge_pipeline.fit(car_X_train, car_y_train)

car_y_preds = ridge_pipeline.predict(car_X_test)

print(car_y_preds[:50],car_y_test[:50])

mse = mean_squared_error(car_y_test, car_y_preds)
print(mse)

mae =  mean_absolute_error(car_y_test, car_y_preds)
print(mae)

r2 = r2_score(car_y_test, car_y_preds)
print(r2)



