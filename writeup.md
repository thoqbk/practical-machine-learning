# Practical machine learning assignment

## Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

## Data
The training data for this project are available here:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

## Load and process data
First, we load training data and replace all blank or “NA” or “NULL” by NA
```R
set.seed(2016)
rawTraining <- read.csv("pml-training.csv", na.strings = c("", "NA", "NULL"))
```

Check the dimension of raw training data:
```R
dim(rawTraining)
[1] 19622   160
```
The rawTraining data frame has ~200K rows and 160 columns but some columns contain alot of NA value. 

Therefore we remove all columns existing a NA value plus un-relevant columns like "X", "user_name".

```R
naTraining <- apply(rawTraining, 2, function(x){sum(is.na(x))})

columns.woNa <- names(naTraining[naTraining == 0])
columns.woUnrelevant <- columns.woNa[!(columns.woNa %in% c("X","user_name", "raw_timestamp_part_1",
                                                   "raw_timestamp_part_2", "cvtd_timestamp", 
                                                   "new_window", "num_window"))]
validTraining <- rawTraining[, columns.woUnrelevant]
```

Check the dimension again. Now there are only 53 columns
```R
dim(validTraining)
[1] 19622    53
```

Check correlation of columns to remove highly correlated columns which have correlation value > 0.9
```R
columnsCor <- cor(validTraining[sapply(validTraining, is.numeric)])
columnsCor[!lower.tri(columnsCor)] <- 0
validTraining.withCorrelation <- cbind(validTraining[,!apply(columnsCor,2, function(x) any(x > 0.9))],classe=validTraining$classe)
```

View correlation graph:
```R
require(lattice)
levelplot(columnsCor, scales=list(x=list(rot=90,cex=.7),y=list(cex=.7)))
```

![correlation graph](https://raw.githubusercontent.com/thoqbk/practical-machine-learning/master/correlation-plot.png)

Check dimension of valid training data after preprocessing
```R
dim(validTraining.withCorrelation)
[1] 19622    50
```

## Slice data for training and cross validation
We use 70% data for training and 30% for testing
```R
library(caret)
inTrain <- createDataPartition(y=validTraining.withCorrelation$classe, p = 0.7, list = FALSE)
training <- validTraining.withCorrelation[inTrain,]
testing <- validTraining.withCorrelation[-inTrain,]
```
## Model 1: use regression tree implementation in Caret package
```R
modFitRpart <- train(classe ~ ., method = "rpart", data = training)
answers <- predict(modFitRpart, newdata = testing)
sum(answers == testing[["classe"]]) / nrow(testing)
[1] 0.5813084
```
The 0.58 is a bad prediction result. Check the final model
```
   1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
     2) pitch_forearm< -26.45 1239   57 A (0.95 0.046 0 0 0) *
     3) pitch_forearm>=-26.45 12498 9774 A (0.22 0.21 0.19 0.18 0.2)  
       6) accel_belt_z>=-187.5 11778 9058 A (0.23 0.22 0.2 0.19 0.15)  
        12) yaw_belt>=169.5 546   56 A (0.9 0.044 0 0.048 0.011) *
        13) yaw_belt< 169.5 11232 8656 B (0.2 0.23 0.21 0.2 0.16)  
          26) magnet_dumbbell_z< -93.5 1309  552 A (0.58 0.29 0.046 0.057 0.032) *
          27) magnet_dumbbell_z>=-93.5 9923 7587 C (0.15 0.22 0.24 0.22 0.18)  
            54) pitch_belt< -42.95 591  104 B (0.022 0.82 0.095 0.024 0.036) *
            55) pitch_belt>=-42.95 9332 7052 C (0.16 0.18 0.24 0.23 0.19)  
             110) magnet_dumbbell_y< 291.5 4113 2470 C (0.19 0.11 0.4 0.16 0.15) *
             111) magnet_dumbbell_y>=291.5 5219 3738 D (0.13 0.24 0.12 0.28 0.22)  
               222) accel_dumbbell_y>=-42.5 4708 3237 D (0.15 0.26 0.062 0.31 0.22)  
                 444) magnet_dumbbell_x>=-264.5 1011  459 B (0.035 0.55 0.035 0.038 0.35) *
                 445) magnet_dumbbell_x< -264.5 3697 2264 D (0.18 0.19 0.069 0.39 0.18)  
                   890) accel_forearm_x>=-88.5 1721 1291 E (0.22 0.25 0.12 0.16 0.25)  
                    1780) roll_forearm< 119.5 843  497 A (0.41 0.23 0.049 0.18 0.13) *
                    1781) roll_forearm>=119.5 878  557 E (0.047 0.26 0.19 0.13 0.37) *
                   891) accel_forearm_x< -88.5 1976  810 D (0.14 0.13 0.022 0.59 0.12) *
               223) accel_dumbbell_y< -42.5 511  164 C (0 0.051 0.68 0.02 0.25) *
       7) accel_belt_z< -187.5 720    5 E (0.0056 0.0014 0 0 0.99) *
```

## Model 2: use Tree package
```R
library(tree)
modFitTree <- tree(classe ~ ., data = training)
answers <- predict(modFitTree, newdata = testing, type = "class")
sum(answers == testing[["classe"]]) / nrow(testing)
[1] 0.6762957
```
The 0.67 is better than the result of model 1 but it is still not a good prediction. 

## Model 3: use Random Forest
```R
library(randomForest)
modFitRf <- randomForest(classe ~ ., data=training)
answers <- predict(modFitRf, newdata = testing, type = "class")
sum(answers == testing[["classe"]]) / nrow(testing)
[1] 0.9942226
```
The 0.994 is highly accurate prediction. Check the model:

```
 randomForest(formula = classe ~ ., data = training) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 7

        OOB estimate of  error rate: 0.58%
Confusion matrix:
     A    B    C    D    E  class.error
A 3904    0    0    1    1 0.0005120328
B   15 2637    6    0    0 0.0079006772
C    0   13 2381    2    0 0.0062604341
D    0    0   28 2222    2 0.0133214920
E    0    0    3    8 2514 0.0043564356
```
There are 500 trees in this model and it usually uses 7 predictors to build up a tree. The error rate is 0.58%

## Use the random forest model for pml-testing.csv data
```R
rawTesting <- read.csv("pml-testing.csv")
answers <- predict(modFitRf, newdata = rawTesting, type="class")
answers
```

Result:
```
1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
Levels: A B C D E
```

## About me
Tho Q Luong, email: thoqbk@gmail.com, homepage: [thoqbk.github.io](http://thoqbk.github.io/)
