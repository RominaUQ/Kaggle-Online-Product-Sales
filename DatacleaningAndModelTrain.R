setwd("C:/Users/Nicholas/Desktop/Kaggle")
train<-read.csv("TrainingDataset.csv", header=T, na.strings="NaN")
library(caTools)

#ID=seq(1, 751)
#train=cbind(train, ID)

train[train=="NAN"]=train$"NA"

lapply(train, median)


#######convertig the data file into long format## each monthly sales represents a raw and removing all NAs #####
library(reshape)
train2<-reshape(train, varying =1:12, direction="long",timevar="MONTH", v.names="Sales")
train3<-train2[!is.na(train2$Sales), ]; str(train3)
str(train3)

############################### manipulating categorical varibales
table(sapply(train3,class))
cat.flag <- grepl("Cat",names(train3))
summary(cat.flag)

categoricals<- train3[,cat.flag]
str(categoricals)
table(sapply(categoricals,function(x) sum(is.na(x))))
categoricals <- data.frame(lapply(categoricals,factor))
str(categoricals)
summary(categoricals)

##############################remove categoricals with one level####

categorical.levels <- sapply(categoricals,function(x) length(levels(x)))
categorical.levels
table(categorical.levels)

#############################finding and removing categoricals with zero variance
categoricals <- categoricals[ ,categorical.levels > 1]

cat.levels <- sapply(categoricals,function(x) length(levels(x)))
table(cat.levels,useNA='always')

lapply(categoricals, table)
sum(is.na(categoricals))

#####################################manipulating Continues variables####

Quant.flag=grepl("Quan", names(train3))
str(Quant.flag)
table(Quant.flag)
Quant=train3[ ,Quant.flag]
Quant=data.frame(lapply(Quant, as.numeric))
str(Quant)
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.na))

summary(Quant)

###########################Replacing all NAs with Median for quant variables####################
f=function(x){
  x[is.na(x)] =median(x, na.rm=TRUE) 
  x
}

Quant=data.frame(apply(Quant,2,f))
summary(Quant)

##########################finding and removing variables with zero variance#######
lapply(Quant, median)
sum(is.na(Quant))

Quant=Quant[ , -(27:31)]

#################################### converting Campiagne days to days/month/year####
Dates= train3[ , c("Date_1", "Date_2")]
sum(is.na(Dates))
f=function(x){
  x[is.na(x)] =median(x, na.rm=TRUE) 
  x #display the column
}

Dates=data.frame(apply(Dates,2,f))
summary(Dates)


moduluDate<-function (data,x1,x2) {
  x1=Dates[ ,1]
  x2=Dates[ ,2]
  x1=as.matrix(x1)
  x2=as.matrix(x2)
  Numyear_1=x1%/%365
  Numyear_2=x2%/%365
  rem_1=x1%%365
  rem_2=x2%%365
  Nummonth_1=rem_1%/%12
  Nummonth_2=rem_2%/%12
  Numdays_1=rem_1%%12
  Numdays_2=rem_2%%12
  Dates_all=data.frame(Numyear_1,Nummonth_1,Numdays_1,Numyear_2,Nummonth_2,Numdays_2 )
  return (Dates_all) 
}
Dates_All=moduluDate(Dates, Dates$Date_1, Dates$Date_2)

#################################Merge all the data together
AllData=cbind(train3$Sales,train3$MONTH, Dates_All, Quant, categoricals)
AllData = rename(AllData, c("train3$Sales"="Sales", c("train3$MONTH"="Month")))


###########near zero variance variables######################
#categoricals2= subset(categoricals, select= -c(Cat_496, Cat_409, Cat_410, Cat_411, Cat_257, Cat_267, Cat_280, Cat_282, Cat_285, Cat_310, Cat_315, Cat_319, Cat_354, Cat_387, Cat_409, Cat_410, Cat_411, Cat_422, Cat_424, Cat_434, Cat_465, Cat_466, Cat_470, 
                                             #  Cat_495, Cat_496, Cat_501, Cat_513,Cat_18, Cat_19, Cat_29,Cat_44, Cat_59, Cat_67,Cat_68, Cat_70, Cat_81, Cat_82, 
                                             #  Cat_118,Cat_120, Cat_130, Cat_140, Cat_174, Cat_193, Cat_246, Cat_247,Cat_273 ))


###########  zero variance variables######################
categoricals= subset(categoricals, select= -c(Cat_496, Cat_409, Cat_410,Cat_411))

AllData2=cbind(train3$Sales,train3$MONTH, Dates_All, Quant, categoricals)
AllData2 = rename(AllData2, c("train3$Sales"="Sales", c("train3$MONTH"="Month")))
AllData2$MonthstoSales_1=((AllData2$Numyear_1*12)+(AllData2$Nummonth_1))- AllData2$Month
AllData2$MonthstoSales_2=((AllData2$Numyear_2*12)+(AllData2$Nummonth_2))- AllData2$Month

##Creat Additional predictors of when the campaignes were launched####
#AllData$MonthstoSales_1=((AllData$Numyear_1*12)+(AllData$Nummonth_1))- AllData$Month
#AllData$MonthstoSales_2=((AllData$Numyear_2*12)+(AllData$Nummonth_2))- AllData$Month
#summary(AllData$MonthstoSales_1)

str(AllData2)
sum(is.na(AllData2))
AllData2$Sales=as.numeric(AllData2$Sales)
lapply(AllData2, table)

####################Partition data to train and Test###################

#intrain<-createDataPartition(y=AllData2$Month,p=0.8,list=FALSE)
#training<-AllData2[intrain,]
#testing<-AllData2[-intrain,]


#########################One Hot Encoding#########################

str(AllData)

sparsematrix2<- sparse.model.matrix(~. -1, data=AllData2)


head(sparsematrix2)
dim(sparsematrix2)
names(sparsematrix2)


#############################XGBM#############nonvalidated#####################
dataX2=sparsematrix2[,-1]
y2 = sparsematrix2[,1]

set.seed(1)
xgb2 <- xgboost(data = data.matrix(dataX2), label=y2,
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               c = "rmse",
               objective = "reg:linear")
xgb2
summary(xgb2)

model <- xgb.dump(xgb2, with.stats = T)
model[1:10] 

# Get the feature real names
names2 <- dimnames(data.matrix(dataX2[,-1]))[[2]]


# Compute feature importance matrix
importance_matrix2 <- xgb.importance(names2, model=xgb2)
head(importance_matrix2, 50)


# graph
xgb.plot.importance(importance_matrix2[1:20,])


predictgbm=predict(xgb2, data.matrix(dataX2[,-1]))

RSME=sqrt(mean((AllDatatest2$Sales-predictgbm)^2))
RSME

error=RMSE(predictgbm,AllDatatest2$Sales)
Rsq=R2(predictgbm,AllDatatest2$Sales)


cor=cor(predictgbm,y2)
R2=cor^2
R2
#######GBM TRAINING#####################
library(caret)
library(gbm)
library(doSNOW)
library(foreach)
library(parallel)

ptm<-proc.time()

## first Round estimation

ctrl1=trainControl(method="cv", number=5)
grid1=expand.grid(.n.trees=c(5000), .interaction.depth=c(1,2,3), .shrinkage=c(0.01,0.0025), .n.minobsinnode=c(1,1,1))

set.seed(1)
model2=train(Sales~.,data=AllData2, method="gbm", preProcess=c("center", "scale", "YeoJohnson"), weights=NULL, maximize=FALSE, metric="RMSE", na.action=na.pass, verbose=TRUE, trControl=ctrl1, tuneGrid=grid1)
model2
head(summary (model2), 200)

#############IMProved Train Controls###########GBM##############

ctrl5=trainControl(method="cv", number=7)
grid3=expand.grid(.n.trees=c(600,1500), .interaction.depth=c(1,2,3), .shrinkage=c(0.01,0.0025), .n.minobsinnode=c(1,1,1))

set.seed(5)
model4=train(Sales~.,data=AllData2, method="gbm", preProcess=c("center", "scale", "YeoJohnson"), weights=NULL, maximize=FALSE, metric="RMSE", na.action=na.pass, verbose=TRUE, trControl=ctrl5, tuneGrid=grid3)
model4
head(summary (model4), 100)

##############----Basic Training using XGBoost in caret Library-######################

grid3=expand.grid(nrounds = c(1000,2000),
  eta = c(0.01, 0.001, 0.1),
  max_depth = c(1,2,3,4),gamma=0, colsample_bytree = 1,min_child_weight = 1 )

xgb_trcontrol= trainControl(method = "cv",
  number = 10,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                       
  allowParallel = TRUE)



########################## train the model for each parameter combination in the grid, using CV to evaluate
set.seed(2)
xgb_train_3 = train( x=data.matrix(dataX2),y=y2,trControl = xgb_trcontrol,tuneGrid = grid3, method = "xgbTree")
xgb_train_3
summary(xgb_train_3)
head(varImp(xgb_train_3), 50)

#########################################################model4##########################
set.seed(3)
xgb_train_4 = train( x=data.matrix(dataX2),y=y2,trControl = xgb_trcontrol,tuneGrid = grid3, method = "xgbTree")
xgb_train_4
summary(xgb_train_4)
head(varImp(xgb_train_4), 50)

actual=c(y2)
predicted=c(predict_xgb_train4)
rmsle(actual, predicted)

###############################################Model5########################

y2= training$Sales

set.seed(4)
xgb_train_5 = train( x=data.matrix(training[ ,-1]),y=y2,trControl = xgb_trcontrol,tuneGrid = grid3, method = "xgbTree")
xgb_train_5
summary(xgb_train_5)
head(varImp(xgb_train_5), 100)

################################################

y2= training$Sales
set.seed(6)
xgb_train_6 = train(x=dataX2,y=y2,trControl = xgb_trcontrol,tuneGrid = grid3, method = "xgbTree")
xgb_train_6
summary(xgb_train_6)
imp<-head(varImp(xgb_train_5), 100)
m1=xgb_train_6$bestTune
plot(m1)

################################

data.matrix(AllDatatest2)
predict_xgb_train5=predict(xgb_train_5,newdata=data.matrix(AllDatatest2[ ,-1]))
error5=RMSE(predict_xgb_train5,ytest)
error5

rsquare5=R2(predict_xgb_train5,ytest)
rsquare5

#####################   ON TEST DATA ###############################NOT USEFUL#################

test2<-reshape(test, varying =1:12, direction="long",timevar="MONTH", v.names="Sales")
test3<-test2[!is.na(test2$Sales), ]; str(test3)
summary(test3)
##### manipulating categorical varibales
table(sapply(test3,class))
cat.flagtest <- grepl("Cat",names(test3))
summary(cat.flagtest)

categoricalstest<- test3[ ,cat.flagtest]

str(categoricalstest)
table(sapply(categoricalstest,function(x) sum(is.na(x))))
categoricalstest <- data.frame(lapply(categoricalstest,factor))
str(categoricalstest)
summary(categoricalstest)

########remove categoricalstest with one level####

categorical.levelstest =categorical.levels
  
categorical.levelstest<- sapply(categoricalstest,function(x) length(levels(x)))
categorical.levelstest
table(categorical.levelstest)

##finding and removing categoricalstest with zero variance
categoricalstest <- categoricalstest[ ,categorical.levels>  1]

cat.levelstest <- sapply(categoricalstest,function(x) length(levels(x)))
table(cat.levelstest,useNA='always')

lapply(categoricalstest, table)
sum(is.na(categoricalstest))

########manipulating Continues variables####

quanttest.flag=grepl("Quan", names(test3))
str(quanttest.flag)
table(quanttest.flag)
quanttest=test3[ ,quanttest.flag]
quanttest=data.frame(lapply(quanttest, as.numeric))
str(quanttest)
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.na))

summary(quanttest)

#####Replacing all NAs with Median for quanttest variables####################
f=function(x){
  x[is.na(x)] =median(x, na.rm=TRUE) 
  x #display the column
}

quanttest=data.frame(apply(quanttest,2,f))
summary(quanttest)

######finding and removing variables with zero variance#######
lapply(quanttest, median)
sum(is.na(quanttest))

quanttest=quanttest[ , -(27:31)]

###### converting Campiagne days to days/month/year####
Datestest= test3[ , c("Date_1", "Date_2")]
sum(is.na(Datestest))
f=function(x){
  x[is.na(x)] =median(x, na.rm=TRUE) 
  x #display the column
}

Datestest=data.frame(apply(Datestest,2,f))
summary(Datestest)


moduluDate<-function (data,x1,x2) {
  x1=Datestest[ ,1]
  x2=Datestest[ ,2]
  x1=as.matrix(x1)
  x2=as.matrix(x2)
  Numyear_1=x1%/%365
  Numyear_2=x2%/%365
  rem_1=x1%%365
  rem_2=x2%%365
  Nummonth_1=rem_1%/%12
  Nummonth_2=rem_2%/%12
  Numdays_1=rem_1%%12
  Numdays_2=rem_2%%12
  Datestest_all=data.frame(Numyear_1,Nummonth_1,Numdays_1,Numyear_2,Nummonth_2,Numdays_2 )
  return (Datestest_all) 
}
Datestest_All=moduluDate(Datestest, Datestest$Date_1, Datestest$Date_2)

######Merge all the data together
AllDatatest=cbind(test3$Sales,test3$MONTH, Datestest_All, quanttest, categoricalstest)
AllDatatest = rename(AllDatatest, c("test3$Sales"="Sales","test3$MONTH"="Month"))

################remove zero or near zero variance vairables
categoricalstest2= subset(categoricalstest, select= -c(Cat_496, Cat_409, Cat_410, Cat_411, Cat_257, Cat_267, Cat_280, Cat_282, Cat_285, Cat_310, Cat_315, Cat_319, Cat_354, Cat_387, Cat_409, Cat_410, Cat_411, Cat_422, Cat_424, Cat_434, Cat_465, Cat_466, Cat_470, 
                                                       Cat_495, Cat_496, Cat_501, Cat_513,Cat_18, Cat_19, Cat_29,Cat_44, Cat_59, Cat_67,Cat_68, Cat_70, Cat_81, Cat_82, 
                                                       Cat_118,Cat_120, Cat_130, Cat_140, Cat_174, Cat_193, Cat_246, Cat_247,Cat_273 ))

AllDatatest2=cbind(test3$Sales,test3$MONTH, Datestest_All, quanttest, categoricalstest2)
AllDatatest2 = rename(AllDatatest2, c("test3$Sales"="Sales", c("test3$MONTH"="Month")))
AllDatatest2$MonthstoSales_1=((AllDatatest2$Numyear_1*12)+(AllDatatest2$Nummonth_1))- AllDatatest2$Month
AllDatatest2$MonthstoSales_2=((AllDatatest2$Numyear_2*12)+(AllDatatest2$Nummonth_2))- AllDatatest2$Month

##Creat Additional predictors of when the campaignes were launched####

str(AllDatatest2)
sum(is.na(AllDatatest2))
AllDatatest2$Sales=as.numeric(AllDatatest2$Sales)


head(lapply(AllDatatest2, table),230)

sparsematrixtest<- sparse.model.matrix(~. -1, data=AllDatatest2)


head(sparsematrixtest)
dim(sparsematrixtest)
names(sparsematrixtest)

dataX2test=sparsematrixtest[,-1]
ytest=AllDatatest2$Sales
