set.seed(2016)

rawTraining <- read.csv("pml-training.csv", na.strings = c("", "NA", "NULL"))
dim(rawTraining)

naTraining <- apply(rawTraining, 2, function(x){sum(is.na(x))})

columns.woNa <- names(naTraining[naTraining == 0])
columns.woUnrelevant <- columns.woNa[!(columns.woNa %in% c("X","user_name", "raw_timestamp_part_1",
                                                   "raw_timestamp_part_2", "cvtd_timestamp", 
                                                   "new_window", "num_window"))]
validTraining <- rawTraining[, columns.woUnrelevant]

dim(validTraining)

# Correlation
columnsCor <- cor(validTraining[sapply(validTraining, is.numeric)])
columnsCor[!lower.tri(columnsCor)] <- 0
validTraining.withCorrelation <- cbind(validTraining[,!apply(columnsCor,2, function(x) any(x > 0.9))],classe=validTraining$classe)

# Draw correlation graph
require(lattice)
levelplot(columnsCor, scales=list(x=list(rot=90,cex=.7),y=list(cex=.7)))

dim(validTraining.withCorrelation)

# Model 1: use regression tree implementation in Caret package
library(caret)
inTrain <- createDataPartition(y=validTraining.withCorrelation$classe, p = 0.7, list = FALSE)
training <- validTraining.withCorrelation[inTrain,]
testing <- validTraining.withCorrelation[-inTrain,]

modFitRpart <- train(classe ~ ., method = "rpart", data = training)

answers <- predict(modFitRpart, newdata = testing)
sum(answers == testing[["classe"]]) / nrow(testing)

# Model 2: use Tree package
library(tree)
modFitTree <- tree(classe ~ ., data = training)
answers <- predict(modFitTree, newdata = testing, type = "class")
sum(answers == testing[["classe"]]) / nrow(testing)

# Model 3: use Random Forest
library(randomForest)
modFitRf <- randomForest(classe ~ ., data=training)
answers <- predict(modFitRf, newdata = testing, type = "class")
sum(answers == testing[["classe"]]) / nrow(testing)

# Use the random forest model for pml-testing.csv data
rawTesting <- read.csv("pml-testing.csv")
answers <- predict(modFitRf, newdata = rawTesting, type="class")
answers

