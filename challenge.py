# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:06:26 2018

@author: User
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import cross_decomposition   

#tried these models didnt get good results
from sklearn import svm
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


 
train = pd.read_csv("C:/Users/User/Downloads/class_train.csv",header= None)

test = pd.read_csv("C:/Users/User/Downloads/class_valid.csv",header = None)

#############Data Extraction##################

# creating 0-1 ,0-2, 0-3 dataframes for both train and test
train_0 = train[(train.iloc[:,-1] == 0)]
train_1 = train[(train.iloc[:,-1] == 1)]
train_2 = train[(train.iloc[:,-1] == 2)]
train_3 = train[(train.iloc[:,-1] == 3)]
train_0_1 =train_0.append(train_1)
train_0_2 =train_0.append(train_2)
train_0_3 =train_0.append(train_3)

train_0_1_normalized = pd.DataFrame(preprocessing.scale(train_0_1),
                                    columns = train_0_1.columns)
train_0_2_normalized = pd.DataFrame(preprocessing.scale(train_0_2),
                                    columns = train_0_1.columns)
train_0_3_normalized = pd.DataFrame(preprocessing.scale(train_0_3),
                                    columns = train_0_3.columns)
test_0 = test[(test.iloc[:,-1] == 0)]
test_1 = test[(test.iloc[:,-1] == 1)]
test_2 = test[(test.iloc[:,-1] == 2)]
test_3 = test[(test.iloc[:,-1] == 3)]
test_0_1 =test_0.append(test_1)
test_0_2 =test_0.append(test_2)
test_0_3 =test_0.append(test_3)


test_0_1_normalized = pd.DataFrame(preprocessing.scale(test_0_1),
                                    columns = test_0_1.columns)
test_0_2_normalized = pd.DataFrame(preprocessing.scale(test_0_2),
                                    columns = test_0_1.columns)
test_0_3_normalized = pd.DataFrame(preprocessing.scale(test_0_3),
                                    columns = test_0_3.columns)


##############################################################################
### Binary Classification for 0-1 with PCA and MLP with L-2 Regularization(lasso) #############
train_0_1_label = train_0_1.iloc[:,-1]
train_0_1_features = train_0_1_normalized.iloc[:,0:380]
pca_model = PCA(n_components =30)

pca_model.fit_transform(train_0_1_features)

plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
#we see we need 30 components
n_components = 30 

pca_0_1_df=pd.DataFrame(pca_model.transform(train_0_1_features), 
                    columns=['PCA%i' % i for i in range(n_components)], 
                    index=train_0_1_features.index)

pca_0_1_df['Label'] = train_0_1_label


pca_0_1_features = pca_0_1_df.iloc[:,0:30]
pca_0_1_label = pca_0_1_df['Label']


########Getting test_0_1 PCA Data################

test_0_1_label = test_0_1.iloc[:,-1]
test_0_1_features = test_0_1_normalized.iloc[:,0:380]


pca_model = PCA(n_components =30)

pca_model.fit_transform(test_0_1_features)

plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
#we see we need 30 components (same as train_0_1)
n_components = 30 

pca_0_1_df_test=pd.DataFrame(pca_model.transform(test_0_1_features), 
                    columns=['PCA%i' % i for i in range(n_components)], 
                    index=test_0_1_features.index)

pca_0_1_df_test['Label'] = test_0_1_label


pca_0_1_testfeatures = pca_0_1_df_test.iloc[:,0:30]
pca_0_1_testlabel = pca_0_1_df_test['Label']


###### Neural Network Model
                                      #Hyper-parameters
nn_model= MLPClassifier(hidden_layer_sizes=(30,30,30), 
                        activation='relu',  #activation function
                           solver='sgd',    # stochastic gradient descent optimization
                           alpha= 0.015,    #l2 penalty term
                           batch_size='auto',
                           learning_rate='adaptive', #updating learning rate 
                           learning_rate_init=0.0001, #starting learning rate
                           max_iter=3000,        #no of epochs to run neural net
                           verbose =True,     #true to see output of each iteration
              shuffle=True, random_state=None, tol=0.0001 #tolerance to stop if change is not >0.00001
               ) # moment vectors
#100,100,100 0.015,10

fit = nn_model.fit(pca_0_1_features,pca_0_1_label)


#Predictions and accuracy on training and test for 0-1 Classification

predictions_0_1_train = fit.predict(pca_0_1_features)
accuracy_0_1_train = metrics.accuracy_score(pca_0_1_label,predictions_0_1_train)


predictions_0_1_test = fit.predict(pca_0_1_testfeatures)
accuracy_0_1_test = metrics.accuracy_score(pca_0_1_testlabel,predictions_0_1_test)


##############################################################################
###Binary Classification for 0-2 with PCA and MLP with L-2 Regularization(lasso)#########################
train_0_2_label = train_0_2.iloc[:,-1]
train_0_2_features = train_0_2_normalized.iloc[:,0:380]
pca_model = PCA(n_components = 80)

pca_model.fit_transform(train_0_2_features)

plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
#we see we need 80 components
n_components = 80 

pca_0_2_df=pd.DataFrame(pca_model.transform(train_0_2_features), 
                    columns=['PCA%i' % i for i in range(n_components)], 
                    index=train_0_2_features.index)

pca_0_2_df['Label'] = train_0_2_label.values


pca_0_2_features = pca_0_2_df.iloc[:,0:80]
pca_0_2_label = pca_0_2_df['Label']


########Getting test_0_2 PCA Data################

test_0_2_label = test_0_2.iloc[:,-1]
test_0_2_features = test_0_2_normalized.iloc[:,0:380]


pca_model = PCA(n_components =80)

pca_model.fit_transform(test_0_2_features)

plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
#we see we need 30 components (same as train_0_2)
n_components = 80 

pca_0_2_df_test=pd.DataFrame(pca_model.transform(test_0_2_features), 
                    columns=['PCA%i' % i for i in range(n_components)], 
                    index=test_0_2_features.index)

pca_0_2_df_test['Label'] = test_0_2_label.values


pca_0_2_testfeatures = pca_0_2_df_test.iloc[:,0:80]
pca_0_2_testlabel = pca_0_2_df_test['Label']


## Neural Network Model
                                      #Hyper-parameters
nn_model_0_2= MLPClassifier(hidden_layer_sizes=(30,30,30), activation='relu', 
                           solver='sgd',alpha= 0.015, batch_size='auto',
                           learning_rate='adaptive', 
                           learning_rate_init=0.0001, max_iter=3000,
                           verbose =True,
              shuffle=True, random_state=None, tol=0.0001
               ) # moment vectors
#30,30,30 0.015,10

fit = nn_model.fit(pca_0_2_features,pca_0_2_label)


#Predictions and accuracy on training and test for 0-1 Classification

predictions_0_2_train = fit.predict(pca_0_2_features)
accuracy_0_2_train = metrics.accuracy_score(pca_0_2_label,predictions_0_2_train)


predictions_0_2_test = fit.predict(pca_0_2_testfeatures)
accuracy_0_2_test = metrics.accuracy_score(pca_0_2_testlabel,predictions_0_2_test)



##############################################################################
#Binary Classification for 0-3 with PCA and MLP with L-2 Regularization(lasso) #########################
train_0_3_label = train_0_3.iloc[:,-1]
train_0_3_features = train_0_3_normalized.iloc[:,0:380]
pca_model = PCA(n_components=35)  

pca_model.fit_transform(train_0_3_features)

plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
#we see we need 35 components 
n_components = 35

pca_0_3_df=pd.DataFrame(pca_model.transform(train_0_3_features), 
                    columns=['PCA%i' % i for i in range(n_components)], 
                    index=train_0_3_features.index)

pca_0_3_df['Label'] = train_0_3_label.values


pca_0_3_features = pca_0_3_df.iloc[:,0:35]
pca_0_3_label = pca_0_3_df['Label']


########Getting test_0_3 PCA Data################

test_0_3_label = test_0_3.iloc[:,-1]
test_0_3_features = test_0_3_normalized.iloc[:,0:380]


pca_model = PCA(n_components =35)

pca_model.fit_transform(test_0_3_features)

plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
#we see we need 35 components (same as train_0_3)
n_components = 35

pca_0_3_df_test=pd.DataFrame(pca_model.transform(test_0_3_features), 
                    columns=['PCA%i' % i for i in range(n_components)], 
                    index=test_0_3_features.index)

pca_0_3_df_test['Label'] = test_0_3_label.values


pca_0_3_testfeatures = pca_0_3_df_test.iloc[:,0:35]
pca_0_3_testlabel = pca_0_3_df_test['Label']


## Neural Network Model
                                      #Hyper-parameters
nn_model_0_3= MLPClassifier(hidden_layer_sizes=(30,30,30), activation='relu', 
                           solver='sgd',alpha= 0.01, batch_size='auto',
                           learning_rate='adaptive', 
                           learning_rate_init=0.001, max_iter=3000,
                           verbose =True,
              shuffle=True, random_state=None, tol=0.0001
               ) # moment vectors
#100,100,100 0.015,10

fit = nn_model.fit(pca_0_3_features,pca_0_3_label)


#Predictions and accuracy on training and test for 0-3 Classification

predictions_0_3_train = fit.predict(pca_0_3_features)
accuracy_0_3_train = metrics.accuracy_score(pca_0_3_label,predictions_0_3_train)


predictions_0_3_test = fit.predict(pca_0_3_testfeatures)
accuracy_0_3_test = metrics.accuracy_score(pca_0_3_testlabel,predictions_0_3_test)














































train_label =train_normalized.iloc[:,-381]
train_features = train_normalized.iloc[:,0:380]

#initial pca to find number of principal components
pca_model = PCA()
 
pca_model.fit_transform(train_features)
variance = pca_model.explained_variance_ratio_


a = pd.DataFrame(pca_model.components_)

#cumulative variance plot
plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# from the plot we see number of components required is 60 for capturing 90% of variance
pca_model = PCA(n_components =60)
temp= pd.DataFrame(pca_model.fit_transform(train_features))
variance = pca_model.explained_variance_ratio_
n_components = 60
pca_df=pd.DataFrame(pca_model.transform(train_features), 
                    columns=['PCA%i' % i for i in range(n_components)], 
                    index=train_features.index)

#we add our label(response variable) to the resulting pca data frame which we will use to create a model
pca_df['Label']=train.iloc[:,380]

## Neural Network Model
                                      #Hyper-parameters
nn_model=MLPClassifier(hidden_layer_sizes=(100,100), activation='relu', 
                           solver='sgd',alpha=0.015, batch_size=10,learning_rate='constant', 
                           learning_rate_init=0.0001, max_iter=1500,
                           verbose =True,
              shuffle=True, random_state=None, tol=0.0001
               ) # moment vectors
#100,100,100 0.015,10
pca_df_features = pca_df.iloc[:,0:60]
pca_df_label = pca_df['Label']


fit = nn_model.fit(pca_df_features,pca_df_label)

#Now we have to convert test data into PCA to run our neural network
#So we do the same steps as done on the train data
test_normalized = pd.DataFrame(preprocessing.scale(test),columns = test.columns) 

test_label =test_normalized.iloc[:,-381]
test_features = test_normalized.iloc[:,0:380]
pca_model_test = PCA(n_components = 60)

pca_test_df=pd.DataFrame(pca_model.transform(test_features), 
                    columns=['PCA%i' % i for i in range(n_components)], 
                    index=test_features.index)

pca_test_df['Label'] = test.iloc[:,380]
pca_test_features = pca_test_df.iloc[:,0:60]


#Predictions and accuracy on training

predictions_train = fit.predict(pca_df_features)
accuracy_train = metrics.accuracy_score(pca_df_label,predictions_train)


predictions_test = fit.predict(pca_test_features)
accuracy_test = metrics.accuracy_score(pca_test_df['Label'],predictions_test)






