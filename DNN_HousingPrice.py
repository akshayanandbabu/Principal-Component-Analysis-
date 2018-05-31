# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:44:06 2018

@author: Akshay Anandbabu
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from itertools import islice
import math
from sklearn import metrics
from sklearn.cross_validation import train_test_split

train = pd.read_csv("C:/Users/User/Downloads/ProjTrain.csv")
#For Simplicity consider the data consists only of numeric predictors
# (Perform encoding for categoric variables)
numeric_data_params = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train= train.select_dtypes(include=numeric_data_params)

test = pd.read_csv("C:/Users/User/Downloads/ProjTest.csv")
numeric_data_params = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
test= test.select_dtypes(include=numeric_data_params)

target = 'SalePrice'
predictors = [i for i in train.keys().tolist() if i!=target] # feature extraction
predictors = predictors[1:37]  # removing ID
train.describe()

train_obs = train.shape[0]
train_input_size = train.shape[1]-1  # to store the number of predictors


data =train[predictors].get_values()  #getting the feature matrix also removing id(irrelevant to the model)
data_test = test[predictors].get_values()
data_normalized=tf.contrib.keras.utils.normalize(
    data,
    axis=-1,
    order=2
)
label = train[target].get_values()   #output vector

train_new = pd.DataFrame(data_normalized,columns = predictors)
train_new['SalePrice'] = train['SalePrice'] #adding the SalesPrice to our Normalized df
train_new = train_new.fillna(0)
train_new=train_new.sample(frac=1)
cut = np.random.rand(len(train_new)) < 0.8

x_train = train_new[cut]

x_test = train_new[~cut]


# We repeat the same procedure for the test set
data_normalized_test=tf.contrib.keras.utils.normalize(
    data_test,
    axis=-1,
    order=2
)
test_new = pd.DataFrame(data_normalized_test,columns = predictors)
test_new = test_new.fillna(0)
test_new['SalePrice']=test['SalePrice']

# Deep Neural Network Parameters

learning_rate = 0.001
train_epochs = 1000
batch_size = 100
activation_func =  tf.nn.leaky_relu
features = list(train_new.columns)
features.remove('SalePrice')
feature_cols =[tf.contrib.layers.real_valued_column(j) for j in features]


#DNN Model

regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[1024,512,256,128,64],
                                          activation_fn=activation_func,
                                          model_dir = "akshay_DNNModel")

print(regressor.model_dir)


def input_func(df,prediction= False):
    
    if prediction == False:
        # get features and label as tensor dictionary
        features_as_tensors = {k:tf.constant(df[k].values) for k in features}
        label_as_tensor = tf.constant(df[target].values)
        return features_as_tensors, label_as_tensor
    
    if prediction == True:
       
        features_as_tensors = {k:tf.constant(df[k].values) for k in features}
        return features_as_tensors
        

#Roughly takes around 8-9mins on i5-980M 
       regressor.fit(input_fn=lambda: input_func(x_train),
              steps=train_epochs) 


#Predict values for test set (The test set does not have SalesPrice so I add the predicted column)
_eval = regressor.predict(input_fn = lambda: input_func(x_test))

predictions = list(islice(_eval, x_test.shape[0]))

test_new_after = x_test
test_new_after['predicted'] = predictions

accuracy_eval = tf.metrics.accuracy(labels=tf.argmax(x_test['SalePrice'], 1), 
                                    predictions=tf.argmax(predictions,1))
accuracy = list(islice(accuracy_eval, x_test.shape[0]))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
print(sess.run([acc, acc_op]))
print(sess.run([acc]))

ev = regressor.evaluate(input_fn = lambda: input_func(x_test))

# Compute rmsle.
root_mean_squared_error = math.sqrt(
      metrics.mean_squared_error(predictions,x_test['SalePrice']))

#creating a temp table to check actual and predicted values side by side
#temp = train_new
#temp['Predicted'] = predictions

print("RMSE : %0.4f" %root_mean_squared_error)















































#train = train.reindex(np.random.permutation(train.index))
#train["SalePrice"]/=1000

#train_subset = train.iloc[:,list(range(9)) + [-1]]
#train_subset_cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key = train_subset,vocabulary_list = ["MSZoning","Street","LotShape"])
#train_subset_num_col = tf.feature_column.numeric_column('MSSubClass','LotFrontage','LotArea')
#train_target = train_subset["SalePrice"]

#optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.001, name = "GradientDescent");

#Gradient clipping for exploding gradients
#optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)




