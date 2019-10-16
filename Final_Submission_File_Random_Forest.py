#Income-Prediction-ML 
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor

# Importing the datasets and removing the independent columns
# Importing Train dataset
train_dataset_file_name = 'tcd ml 2019-20 income prediction training (with labels).csv'
dataset_x = pd.read_csv(train_dataset_file_name)
#remove negative values
dataset_x = dataset_x[dataset_x['Income in EUR']>0]
dataset_x = dataset_x.drop(['Instance','Wears Glasses','Hair Color','Body Height [cm]'], axis = 1)
# Importing Test dataset
test_dataset_file_name='tcd ml 2019-20 income prediction test (without labels).csv'
test_dataset = pd.read_csv(test_dataset_file_name)
test_dataset_x = test_dataset.drop(['Instance','Wears Glasses','Hair Color','Body Height [cm]'],axis = 1)

# Treating missing values
class CategoricalDataFrameImputer(TransformerMixin):
    def _init_(self):
        """Impute missing values.Columns of dtype object are imputed with the most frequent value 
        in column.Columns of other types are imputed with mean of column."""
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
   
dataset_x = CategoricalDataFrameImputer().fit_transform(dataset_x)
test_dataset_x = test_dataset_x.iloc[:, :-1]
test_dataset_x = CategoricalDataFrameImputer().fit_transform(test_dataset_x)

#Target Encoding
def target_encoding(train_X, train_y, target_encoding_col, smoothness):
    train_target = train_X.copy()
    original_map = dict()    # stores mapping between original and encoded values
    global_map = dict() # stores global average of each variable
   
    for col in target_encoding_col:
        prior = train_X[train_y].mean()
        m = train_X.groupby(col).size()
        mu = train_X.groupby(col)[train_y].mean()
        mu_smooth = (m * mu + smoothness * prior) / (m + smoothness)
       
        train_target.loc[:, col] = train_target[col].map(mu_smooth)        
        original_map[col] = mu_smooth
        global_map[col] = prior        
    return train_target, original_map, global_map

target_encoding_var = ['Gender','Country','Profession','University Degree']
dataset, original_map, global_map = target_encoding(dataset_x,'Income in EUR', target_encoding_var, 20)

#Update the mean values to test data
for c in target_encoding_var:
    test_dataset_x.loc[:, c] = test_dataset_x[c].map(original_map[c])
    
X_train = dataset.iloc[:, :-1]
y_train = dataset.iloc[:, -1]
X_test = test_dataset_x

# Fitting RandomForestRegressor to the dataset
regressor = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=30)
regressor.fit(X_train, y_train)
X_test = CategoricalDataFrameImputer().fit_transform(X_test)
y_pred=regressor.predict(X_test)

# Exporting values to excel
outputIncome = pd.DataFrame(test_dataset['Instance'])
outputIncome['Income'] = pd.DataFrame(y_pred)
outputFileName = 'tcd ml 2019-20 income prediction submission file.csv'
outputIncome.to_csv(outputFileName)

#Accuracy testing
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split( X_train , y_train, test_size=0.25, random_state = 7)
regressor = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=30)
regressor.fit(X_train1, y_train1)
y_pred1 = regressor.predict(X_test1)
RMSE=np.sqrt(np.sum(((y_pred1 - y_test1)**2))/len(y_test1))
print("RMSE Score:", RMSE)


