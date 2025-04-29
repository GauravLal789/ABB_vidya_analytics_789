# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:34:32 2025

@author: 199309
"""
## Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression  
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
#%%
# Load the data
train_data = pd.read_csv('D:\\OneDrive - Tata Steel Limited\\Desktop\\NEW_WORK\\ABB_py\\train_v9rqX0R.csv')
test_data  = pd.read_csv('D:\\OneDrive - Tata Steel Limited\\Desktop\\NEW_WORK\\ABB_py\\test_AbJTz2l.csv')
#%%
#Replacing Item_Fat_Content and making the data set more homogeneeous
train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace({
    'Low Fat': 'LF',
    'low fat': 'LF',
    'LF': 'LF',
    'Regular': 'RG',
    'reg': 'RG'
})

test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace({
    'Low Fat': 'LF',
    'low fat': 'LF',
    'LF': 'LF',
    'Regular': 'RG',
    'reg': 'RG'
})
#%%
#Outlet_Size is missing, as it is categorical feature of outlet so not considering 
train_data =train_data.drop(['Outlet_Size'],axis =1)
test_data  =test_data.drop(['Outlet_Size'],axis =1)
#get the rows/items whose weight are missing and repalcing them with mean weight on group by Item_Fat_Content and Item_Type
nan_rows = train_data[train_data['Item_Weight'].isna()]
for i in range(0,nan_rows.shape[0]):
    nan_rows_get =pd.DataFrame(nan_rows.iloc[i,:]).T
    Item_Identifier = nan_rows_get.loc[:,'Item_Identifier'].values[0]
    Item_Fat_Content =nan_rows_get.loc[:,'Item_Fat_Content'].values[0]
    Item_Type =nan_rows_get.loc[:,'Item_Type'].values[0]
    Outlet_Location_Type =nan_rows_get.loc[:,'Outlet_Location_Type'].values[0]
    Outlet_Type =nan_rows_get.loc[:,'Outlet_Type'].values[0]
    get_short_data =train_data[(train_data['Item_Fat_Content'] == Item_Fat_Content) & (train_data['Item_Type'] == Item_Type) ]
    train_data['Item_Weight'][train_data['Item_Identifier']==Item_Identifier]= get_short_data['Item_Weight'].fillna(
        get_short_data['Item_Weight'].mean()
    )
#%%
nan_rows_test = test_data[test_data['Item_Weight'].isna()]
for i in range(0,nan_rows_test.shape[0]):
    nan_rows_get_test =pd.DataFrame(nan_rows_test.iloc[i,:]).T
    Item_Identifier_test = nan_rows_get_test.loc[:,'Item_Identifier'].values[0]
    Item_Fat_Content_test =nan_rows_get_test.loc[:,'Item_Fat_Content'].values[0]
    Item_Type_test =nan_rows_get_test.loc[:,'Item_Type'].values[0]
    Outlet_Location_Type_test =nan_rows_get_test.loc[:,'Outlet_Location_Type'].values[0]
    Outlet_Type_test =nan_rows_get_test.loc[:,'Outlet_Type'].values[0]
    get_short_data_test =test_data[(test_data['Item_Fat_Content'] == Item_Fat_Content_test) & (test_data['Item_Type'] == Item_Type_test) ]
    test_data['Item_Weight'][test_data['Item_Identifier']==Item_Identifier_test]= get_short_data_test['Item_Weight'].fillna(
        get_short_data_test['Item_Weight'].mean()
    )
#%%
#converting Outlet_Establishment_Year into Outlet_Age
train_data['Outlet_Age']= 2013-train_data['Outlet_Establishment_Year']  
train_data =train_data.drop(['Outlet_Establishment_Year'],axis =1) 
test_data ['Outlet_Age']= 2013-test_data['Outlet_Establishment_Year']  
test_data  =test_data.drop(['Outlet_Establishment_Year'],axis =1)  
#%%
cat_columns = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type',
               'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type']
num_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP',
               'Outlet_Age']
## Column Transformer converting category columns to one hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_columns),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_columns)
    ])
#%%
'''lasso = Lasso(alpha=0.01, random_state=42)  # Adjust alpha as needed
feature_selector = SelectFromModel(estimator=lasso)
 #Define the XGBoost Regressor model
xgb = XGBRegressor(random_state=42)
 #Create the Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
  ('feature_selection', feature_selector),  # Add feature selection
  ('regressor', xgb)])'''
#%%
#Feature Selection with Lasso (L1 regularization)
lasso = Lasso(alpha=0.01, random_state=42)  # Adjust alpha as needed
feature_selector = SelectFromModel(estimator=lasso)
#Define the Random Forest Regressor model
rf = RandomForestRegressor(random_state=42)
#Create the Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('feature_selection', feature_selector),
                        # Add feature selection
                        ('regressor', rf)])
#%%
# Split the data into training and testing sets
X = train_data.drop('Item_Outlet_Sales', axis=1)  # Features
X.isnull().sum()
y = train_data['Item_Outlet_Sales']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
#%%
#K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust n_splits
#Perform Cross-Validation and Store Scores
cv_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
print("Cross-validation R^2 scores:", cv_scores)
print("Mean cross-validation R^2 score:", np.mean(cv_scores))
#%%
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"R^2 Score: {accuracy}")
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
#%%
y_pred_test = pipeline.predict(test_data)
#%%
y_pred_test=pd.DataFrame(y_pred_test,columns =['Predicted_Item_Outlet_Sales'])
output= pd.concat([test_data,y_pred_test],axis =1)
output.to_excel(r'D:\OneDrive - Tata Steel Limited\Desktop\NEW_WORK\ABB_py\RF_output_test.xlsx', index=False)
