library(tidyverse)
library(car)
library(dplyr)
library(readr)
library(readxl)
library(data.table)
library(mice)
library(Amelia)
library(gridExtra)
library(corrplot)
library(caret)
library(xgboost)
library(glmnet)
setwd("C:/Users/User/Downloads")
train<-read.csv("ProjTrain.csv",stringsAsFactors = F)
train<-setDT(train)
test<-read.csv("ProjTrain.csv",stringsAsFactors = F)
str(train)
#####################

missmap(train[-1], col=c('red', 'green'), y.cex=0.5, x.cex=0.8)  #Missingness map
sum(is.na(train)) / (nrow(train) *ncol(train))


# finding cat and num variabes
cat_var <- names(train)[which(sapply(train, is.character))]
cat_car <- c(cat_var, 'BedroomAbvGr', 'HalfBath', ' KitchenAbvGr','BsmtFullBath', 'BsmtHalfBath', 'MSSubClass')
numeric_var <- names(train)[which(sapply(train, is.numeric))]


train[,(cat_var) := lapply(.SD, as.factor), .SDcols = cat_var]  #converting catergorical vars to factors
train_cat <- train[,.SD, .SDcols = cat_var]
train_cont <- train[,.SD,.SDcols = numeric_var]  

#DISTRIBUTION PLOTS

bar_plot <- function(dataset, i) 
  {
  data <- data.frame(x=dataset[[i]])
  g <- ggplot(data=data, aes(x=factor(x))) + 
    stat_count() + 
    xlab(colnames(dataset)[i]) + 
    theme_light()  
  return (g)
}

plot_arrange <- function(dataset, fun, ii, ncol=3) {
  gg<- list()
  for (i in ii) {
    t <- fun(dataset=dataset, i=i)
    gg <- c(gg, list(t))
  }
  do.call("grid.arrange", c(gg, ncol=ncol))
}


density_plot <- function(dataset, i)
  {
  data <- data.frame(x=dataset[[i]], SalePrice = dataset$SalePrice)
  g <- ggplot(data) + 
    geom_line(aes(x = x), stat = 'density', size = 1,alpha = 1.0) +
    xlab(paste0((colnames(dataset)[i])))
  return(g)
}

################################

#Distribution of Categorical vars
plot_arrange(train_cat, fun = bar_plot, ii = 1:4, ncol = 2)
plot_arrange(train_cat, fun = bar_plot, ii = 5:8, ncol = 2)
plot_arrange(train_cat, fun = bar_plot, ii = 10:14, ncol = 2)
plot_arrange(train_cat, fun = bar_plot, ii = 15:19, ncol = 2)

#Density plots of continuous variables
plot_arrange(train_cont, fun = density_plot, ii = 2:6, ncol = 2)
plot_arrange(train_cont, fun = density_plot, ii = 7:12, ncol = 2)
plot_arrange(train_cont, fun = density_plot, ii = 13:17, ncol = 2)
plot_arrange(train_cont, fun = density_plot, ii = 18:23, ncol = 2)


corr <- cor(na.omit(train_cont[,-1, with = FALSE]))

# correlations
corr<- cor(na.omit(train_cont[,-1, with = FALSE]))
row_indic <- apply(corr, 1, function(x) sum(x > 0.3 | x < -0.3) > 1)

corr<- corr[row_indic ,row_indic ]
corrplot::corrplot(corr, method="square")

#Scatterplots for variables with high correlation:
train %>%
ggplot(aes(OverallQual,SalePrice))+
  geom_point()+
  geom_smooth(method = lm)


train %>%
  ggplot(aes(YearBuilt,SalePrice))+
  geom_point()+
  geom_smooth(method = lm)

train %>%
  ggplot(aes(YearRemodAdd,SalePrice))+
  geom_point()+
  geom_smooth(method = lm)

train %>%
  ggplot(aes(TotalBsmtSF,SalePrice))+
  geom_point()+
  geom_smooth(method = lm)

train %>%
  ggplot(aes(`1stFlrSF`,SalePrice))+
  geom_point()+
  geom_smooth(method = lm)


train %>%
  ggplot(aes(GrLivArea,SalePrice))+
  geom_point()+
  geom_smooth(method = lm)

##################################Feature Selection and transformation################################################

#We are combining the training and test sets,removing the SalePrice which is what we are predicting. 
#We remove Id as it has nothing to do with house pricing
df.all <- rbind(within(train, rm('Id','SalePrice','PoolQC','MiscFeature','Alley')), within(test, rm('Id','SalePrice','PoolQC','MiscFeature','Alley')))
dim(df.all)

# We see that poolQC, MiscFeatures, Alley have maximum  missing values so we remove them from our model.
# And we personally believe that the quality of the pool and alley does not drive major price changes

num_feat<- names(which(sapply(df.all, is.numeric)))
cat_feat <- names(which(sapply(df.all, is.character)))

df.numeric <- df.all[num_feat]

#Function that maps a categoric value to its corresponding numeric value and returns that column to the data frame
map.fcn <- function(cols, map.list, df){
  for (col in cols){
    df[col] <- as.numeric(map.list[df.all[,col]])
  }
  return(df)
}
#finding all qual columns to convert catergorical into numeric vars
Qual.cols <- c('ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual')
Qual.list <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)

df.numeric <- map.fcn(Qual.cols, Qual.list, df.numeric)

#Converting Basement cols
bsmt.list <- c('None' = 0, 'No' = 1, 'Mn' = 2, 'Av' = 3, 'Gd' = 4)
df.numeric = map.fcn(c('BsmtExposure'), bsmt.list, df.numeric)

#BsmtFinType1 and BsmtFinType2
bsmt.fin.list <- c('None' = 0, 'Unf' = 1, 'LwQ' = 2,'Rec'= 3, 'BLQ' = 4, 'ALQ' = 5, 'GLQ' = 6)
df.numeric <- map.fcn(c('BsmtFinType1','BsmtFinType2'), bsmt.fin.list, df.numeric)

#Home Functionality rating
functional.list <- c('None' = 0, 'Sal' = 1, 'Sev' = 2, 'Maj2' = 3, 'Maj1' = 4, 'Mod' = 5, 'Min2' = 6, 'Min1' = 7, 'Typ'= 8)
df.numeric['Functional'] <- as.numeric(functional.list[df.all$Functional])

#Garage Finish
garage.fin.list <- c('None' = 0,'Unf' = 1, 'RFn' = 1, 'Fin' = 2)
df.numeric['GarageFinish'] <- as.numeric(garage.fin.list[df.all$GarageFinish])

#Fence
fence.list <- c('None' = 0, 'MnWw' = 1, 'GdWo' = 1, 'MnPrv' = 2, 'GdPrv' = 4)
df.numeric['Fence'] <- as.numeric(fence.list[df.all$Fence])


MSdwell.list <- c('20' = 1, '30'= 0, '40' = 0, '45' = 0,'50' = 0, '60' = 1, '70' = 0, '75' = 0, '80' = 0, '85' = 0, '90' = 0, '120' = 1, '150' = 0, '160' = 0, '180' = 0, '190' = 0)
df.numeric['NewerDwelling'] <- as.numeric(MSdwell.list[as.character(df.all$MSSubClass)])

#We now find correlations between the variables and sales price to see 
#which features have the highest correlation with sales price

corr.df <- cbind(df.numeric[1:1460,], train['SalePrice']) #we check on the training dataset
correlation<-cor(corr.df)

corr.Max <- as.matrix(sort(correlation[,'SalePrice'], decreasing = TRUE))

corr.idx <- names(which(apply(corr.Max, 1, function(x) (x > 0.5 | x < -0.5)))) # correlation greater than 0.5 in either direction
corrplot(as.matrix(correlation[corr.idx,corr.idx]), type = 'upper', method='color', addCoef.col = 'black', tl.cex = .7,cl.cex = .7, number.cex=.7)

#We see that the 12 variables impact the sales prices the most
#They are OverallQual, GrLivArea,ExternalQual,KitchenQual,
#GarageCars,GarageArea,TotalBsmtSF, X1stFlrSF,FullBath,
#TotRmsAbvGrd,YearBuilt,YearRemodAdd

##########################################Normalizing and creating Dummy Variables########################################
scaler <- preProcess(df.numeric)
df.numeric <- predict(scaler, df.numeric)

dummy <- dummyVars(" ~ .",data=df.all[,cat_feat])
df.categoric <- data.frame(predict(dummy,newdata=df.all[,cat_feat]))

#Combining numeric and cat variables
df.total<-cbind(df.numeric, df.categoric)

qqnorm(train$SalePrice)
qqline(train$SalePrice)

#log transform of Sales Price to remove skewness of salesprice
log_train<-log(train$SalePrice+1)


qqnorm(log_train)
qqline(log_train)

######################################## MODEL BUILDING #############################################

#We train a decision tree model using XGB. We decided XGBoost would be best for our model 
#as it consists of a large number of predictors so dividing them into weak learners and training each weak learner
# to perform better would be our best option. This also gave us a good practical experience in using boosted trees

xgb_train <- df.total[1:1460,]

xgb_test <- df.total[1461:nrow(df.total),]

densetrain <- xgb.DMatrix(as.matrix(xgb_train), label=log_train)  #dense matrix since most of the values are non-zeroes
densetest <- xgb.DMatrix(as.matrix(xgb_test))

control <- trainControl(method = "repeatedcv", repeats = 1,number = 4, 
                        allowParallel=T)    #repeated cv for k-fold cross validation

grid <- expand.grid(nrounds = 750,
                        eta = c(0.01,0.005,0.001),
                        max_depth = c(4,6,8),
                        colsample_bytree=c(0,1,10),
                        min_child_weight = 2,
                        subsample=c(0,0.2,0.4,0.6),
                        gamma=0.01)
set.seed(84)

xgb.tune <- train(as.matrix(xgb_train),
                        log_train,
                        method="xgbTree",
                        trControl=control,
                        tuneGrid=grid,
                        verbose=T,
                        metric="RMSE",
                        nthread =4)
                 
xgb.paramaters <- list( booster = 'gbtree',
                   objective = 'reg:linear',
                   colsample_bytree=1,
                   eta=0.005,
                   max_depth=4,
                   min_child_weight=3,
                   alpha=0.3,
                   lambda=0.4,
                   gamma=0.01, # less overfit
                   subsample=0.6,
                   seed=5,
                   silent=TRUE)
                 
xgb.cv(xgb.paramaters, densetrain, nrounds = 10000, nfold = 4, early_stopping_rounds = 500)

bst<-xgb.train(densetrain,params = xgb.paramaters, nrounds = 10000,
                early_stopping_rounds = 300, 
                watchlist = list(train=densetrain))
                 
#### Have to Perform Regularization to improve scores