## ----message=FALSE----------------------------------------------------------------------------------
load("Breast_cancer.RData")
library(dplyr)
head(Breast_cancer_data[,c(1:6)])
data <- Breast_cancer_data
dim(data)
data <- data %>% mutate(diagnosis_0_1 = ifelse(diagnosis == "M", 1, 0))
var <- c(1,2,5,6,9,10,11,12,15,16,25,31)
data <- data[,-c(1,2)]
data <- data[,var]
colnames(data)
library(ROCR)
library(caret)
set.seed(101)
training <- createDataPartition(data$diagnosis_0_1, p=0.8, list=FALSE)
train <- data[ training, ] #--> training data 
test <- data[ -training, ] #--> test data
trainfull <- train[,-c(1,2)]
uni.model <- glm(diagnosis_0_1~smoothness_worst,family=binomial,data=trainfull)
full.model <- glm(diagnosis_0_1~.,family=binomial,data=trainfull)
summary(full.model)$coef


## ---------------------------------------------------------------------------------------------------
new.patients <- test
new.patients
p.full = predict(full.model, new.patients, type="response")
p.full[1:10]


## ----eval=TRUE--------------------------------------------------------------------------------------
my.classifier <- function(model,patient,threshold=0.5)  ifelse(predict(model, patient, type="response")>threshold,1,0)
my.classifier(full.model,new.patients[1:10,])


## ---------------------------------------------------------------------------------------------------
pred.full <- my.classifier(full.model,test)
res.full <- confusionMatrix(as.factor(pred.full),as.factor(test$diagnosis_0_1),positive="1") 
table(pred.full,test$diagnosis_0_1)
#res.full$table
res.full$overall[1]
res.full$byClass[c(1,2,5,6)]


## ---------------------------------------------------------------------------------------------------
pred.uni <- my.classifier(uni.model,test)
res.uni <- confusionMatrix(as.factor(pred.uni),as.factor(test$diagnosis_0_1),positive="1") 
res.uni$table
res.uni$overall[1]
res.uni$byClass[c(1,2,5,6)]


## ---------------------------------------------------------------------------------------------------
res.uni$byClass[7]


## ---------------------------------------------------------------------------------------------------
res.full$byClass[7]


## ----eval=TRUE--------------------------------------------------------------------------------------

pr_full = prediction(pred.full, test$diagnosis_0_1)
perf_full = performance(pr_full, measure = "tpr", x.measure = "fpr")

pr_uni = prediction(pred.uni, test$diagnosis_0_1)
perf_uni = performance(pr_uni, measure = "tpr", x.measure = "fpr")


auc1 = performance(pr_full, measure = "auc")
auc1 = auc1@y.values[[1]]
auc2 = performance(pr_uni, measure = "auc")
auc2 = auc2@y.values[[1]]
print(paste("AUC full test", signif(auc1,digits=4)))
print(paste("AUC uni test", signif(auc2,digits=4)))


library(ROCit)
r1=rocit(predict(uni.model, test), test$diagnosis_0_1, method = "bin")
r3=rocit(predict(full.model, test), test$diagnosis_0_1, method = "bin")



plot(r1$TPR~r1$FPR, type = "l", xlab = "False Positive Rate", lwd = 2,
     ylab = "Sensitivity", col= "gold4",cex.lab=2,cex.axis = 2)
grid()
lines(r3$TPR~r3$FPR, lwd = 2, col = "orange")
segments(0,0,1,1, col = "2", lwd = 2)
segments(0,0,0,1, col = "darkgreen", lwd = 2)
segments(1,1,0,1, col = "darkgreen", lwd = 2)

legend("bottomright", c("Perfectly Separable", 
                        "Univariate", "Full model", "Chance Line"), cex=2,
       lwd = 2, col = c("darkgreen", "gold4",
                        "orange", "red"), bty = "n")



## ----message=FALSE----------------------------------------------------------------------------------
library(MASS)
library(tidyverse)
data("Boston", package = "MASS")
ind <- which(Boston$medv==50)
Boston.s <- Boston[-ind,]
train.data  <- Boston.s

model1 <- lm(medv~rm,data=train.data)


p1 <- ggplot(train.data, aes(rm, medv)) +
  geom_point(size=2) + stat_smooth(method = "lm", col = "red") +
  xlab('Average number of rooms per dwelling (rm)') + ylab('House prices in $1000 (medv)')+
  theme(legend.position = 'bottom')  + theme_bw()
p1 + theme(axis.text = element_text(size = 10))+ theme(axis.title = element_text(size = 10))   


## ---------------------------------------------------------------------------------------------------
model.lm <- lm(medv~lstat,data=train.data)
model.quad <- lm(medv~lstat+I(lstat^2),data=train.data)

p2 <- ggplot(train.data, aes(lstat, medv) ) +
  geom_point(size=2) +
  stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))+
  stat_smooth(method = lm, formula = y ~ x,col='red')+
  xlab('Lower status of the population in % (lstat)') + ylab('House prices in $1000 (medv)')+
  theme(legend.position = 'bottom')  + theme_bw()
p2 + theme(axis.text = element_text(size = 10))+ theme(axis.title = element_text(size = 10))   


## ----tidy=TRUE,tidy.opts = list(blank = FALSE, width.cutoff = 60),cache=TRUE------------------------
file_training_set_image <- "train-images.idx3-ubyte"
file_training_set_label <- "train-labels.idx1-ubyte"
file_test_set_image <- "t10k-images.idx3-ubyte"
file_test_set_label <- "t10k-labels.idx1-ubyte"

extract_images <- function(file, nbimages = NULL) {
  if (is.null(nbimages)) { # We extract all images
    nbimages <- as.numeric(paste("0x", paste(readBin(file, "raw", n = 8)[5:8], collapse = ""), sep = ""))
  }
 nbrows <- as.numeric(paste("0x", paste(readBin(file, "raw", n = 12)[9:12], collapse = ""), sep = ""))
 nbcols <- as.numeric(paste("0x", paste(readBin(file, "raw", n = 16)[13:16], collapse = ""), sep = ""))
 raw <- readBin(file, "raw", n = nbimages * nbrows * nbcols + 16)[-(1:16)]
return(array(as.numeric(paste("0x", raw, sep="")), dim = c(nbcols, nbrows, nbimages)))
}

extract_labels <- function(file) {
 nbitem <- as.numeric(paste("0x", paste(readBin(file, "raw", n = 8)[5:8], collapse = ""), sep = ""))
 raw <- readBin(file, "raw", n = nbitem + 8)[-(1:8)]
return(as.numeric(paste("0x", raw, sep="")))
}

images_training_set <- extract_images(file_training_set_image, 60000)
images_test_set <- extract_images(file_test_set_image, 10000)
labels_training_set <- extract_labels(file_training_set_label)
labels_test_set <- extract_labels(file_test_set_label)

labels_training_set[1:10]
# par(ask = TRUE)
par(mfrow=c(2,2))
for (i in 1:4) image(as.matrix(rev(as.data.frame(images_training_set[,,i]))), col = gray((255:0)/256))



## ---------------------------------------------------------------------------------------------------
vectorized_result <- function(j) {
  e <- as.matrix(rep(0, 10))
  e[j + 1] <- 1
  return(e)
}

Xtrain <- matrix(0,nrow=60000,ncol=784)
Ytrain <- matrix(0,nrow=60000,ncol=10)
for (i in 1:60000) {
  Xtrain[i,] <- as.vector(images_training_set[,,i]) / 256
  Ytrain[i,] <- t(vectorized_result(labels_training_set[i]))
}
Ytrain[which(Ytrain==0)] <- -1

Xtest <- matrix(0,nrow=10000,ncol=784)
Ytest <- matrix(0,nrow=10000,ncol=10)
for (i in 1:10000) {
  Xtest[i,] <- as.vector(images_test_set[,,i]) / 256
  Ytest[i,] <- t(vectorized_result(labels_test_set[i]))
}


## ----cache=TRUE-------------------------------------------------------------------------------------
library(MASS)
mat <- ginv(Xtrain)


## ---------------------------------------------------------------------------------------------------
model <- matrix(0,nrow=784,ncol=10)
for(i in 1:10){
model[,i] <- mat%*%matrix(Ytrain[,i],ncol=1)  
}


## ---------------------------------------------------------------------------------------------------

pred.model <- function(x,Xtest=Xtest){sum(x*Xtest)}

linear.class <- function(model,Xtest){
  res <- apply(model,MARGIN=2,FUN=pred.model,Xtest=Xtest)
  pred <- which.max(res)-1
  return(pred)
}


## ---------------------------------------------------------------------------------------------------
result <- 0
res <- rep(0,10000)
for (i in 1:10000){
Xtest1 <- (as.vector(images_test_set[,,i]) / 256)
resi <- linear.class(model,Xtest=Xtest1)
res[i] <- resi
if(abs(resi-labels_test_set[i])<0.5) result <- result +1
}
accuracy <- result/10000
accuracy


## ---------------------------------------------------------------------------------------------------
library(caret)
caret::confusionMatrix(factor(res),factor(labels_test_set))$table


## ----echo=FALSE-------------------------------------------------------------------------------------
set.seed(11)
x <- seq(0,1,length=10)
y <- sin(2*pi*x)-cos(2*pi*x)+rnorm(10,0,0.2)
mydata <- data.frame(y=y,x=x)
save(mydata,file="data-poly.Rdata")
library(ggplot2)
ggplot(mydata, aes(x, y)) + 
  geom_point() 


## ---------------------------------------------------------------------------------------------------
load("data-poly.Rdata")
y <- mydata$y
x <- mydata$x
model0 <- glm(y~1)
model1 <- glm(y~poly(x,1))
model2 <- glm(y~poly(x,2))
model3 <- glm(y~poly(x,3))
model4 <- glm(y~poly(x,4))
model5 <- glm(y~poly(x,5))
model6 <- glm(y~poly(x,6))
model7 <- glm(y~poly(x,7))
model8 <- glm(y~poly(x,8))
model9 <- glm(y~poly(x,9))

# Much better
model.list <- vector("list", length = 10)
model.list[[1]] <- glm(y~1)
for (i in 1:9){
  formula <- as.formula(paste("y ~ poly(x,",i,")",sep=""))
  model.list[[i+1]] <- glm(formula) 
}



my.formula <- y ~ 1
p0 <- ggplot(mydata, aes(x, y)) + 
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, 
              formula = my.formula, 
              colour = "red")+
  labs(title=element_text("Polynomial fit with k=0"))+
  theme(legend.position = "none") + theme_bw()


my.formula <- y ~ poly(x, 1, raw = TRUE)

p1 <- ggplot(mydata, aes(x, y)) + 
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, 
              formula = my.formula, 
              colour = "red")+
  labs(title=element_text("Polynomial fit with k=1"))+
  theme(legend.position = "none") + theme_bw()

my.formula <- y ~ poly(x, 3, raw = TRUE)

p3 <- ggplot(mydata, aes(x, y)) + 
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, 
              formula = my.formula, 
              colour = "red")+
  labs(title=element_text("Polynomial fit with k=3"))+
  theme(legend.position = "none")+ theme_bw()


my.formula <- y ~ poly(x, 9, raw = TRUE)


p9 <- ggplot(mydata,aes(x, y=y)) + 
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, 
              formula = my.formula, 
              colour = "red")+
  labs(title=element_text("Polynomial fit with k=9"))+
  theme(legend.position = "none") + theme_bw()

library(ggpubr)
ggarrange(p0,p1,p3,p9,ncol = 2, nrow = 2)


## ---------------------------------------------------------------------------------------------------
RMSE <- function(model,dataXY){
  pred <- predict.glm(model,newdata=data.frame(x=dataXY$X))
  RMSE <- mean((pred-dataXY$Y)**2)
  return(RMSE)
}

data.train <- data.frame(X=x,Y=y)

RMSE.result <- matrix(NA,nrow=10,ncol=2)
for(i in 1:10){
  model <- model.list[[i]]
  RMSE.result[i,] <- c(i-1,RMSE(model,data.train))
}

data.result <- as.data.frame(RMSE.result)
colnames(data.result) <- c("M","Train")
mydata <- data.frame(Model=data.result$M,Performance=data.result$Train,Group=rep("Train",each=10),color=rep("red",each=10))



linetype = c('solid', 'solid')
LegendTitle = ""
p <- ggplot(mydata, aes(Model,Performance, group = Group,color=color))+
  geom_line(aes(group = Group,color=color))+#,"red")) +
  scale_color_identity(labels = c(blue = "blue",red = "red"))+
  #scale_linetype_manual(name = LegendTitle, values = linetype) +
  scale_x_continuous(name="Model (k)",breaks=0:9,labels=0:9)+
  scale_y_continuous(name="Mean Square Error")+
  theme_bw() +
  annotate(geom="text", x=7.5, y=1, label="E_train ",color="red")+
  annotate("segment", x = 6.6, xend = 7, y = 1, yend = 1,
         colour = "red")
p



## ---------------------------------------------------------------------------------------------------
set.seed(1123)
testx <- seq(0,1,length=10000)
testy <- sin(2*pi*testx)-cos(2*pi*testx)+rnorm(10000,0,0.2)
data.test <- data.frame(X=testx,Y=testy)


RMSE.result <- matrix(NA,nrow=10,ncol=3)
for(i in 1:10){
  model <- model.list[[i]]
  RMSE.result[i,] <- c(i-1,RMSE(model,data.train),RMSE(model,data.test))
}


colnames(RMSE.result) <- c("M","Train","Test")
data.result <- as.data.frame(RMSE.result)
colnames(RMSE.result) <- c("M","Train","Test")
mydata <- data.frame(Model=rep(data.result$M,2),Performance=c(data.result$Train,data.result$Test),Group=rep(c("Train","Production"),each=10),color=rep(c("red","black"),each=10))

linetype = c('solid', 'solid')
LegendTitle = ""
p <- ggplot(mydata, aes(Model,Performance, group = Group,color=color))+
  geom_line(aes(group = Group,color=color))+#,"red")) +
  scale_color_identity(labels = c(blue = "blue",red = "red"))+
  scale_x_continuous(name="Model (k)",breaks=0:9,labels=0:9)+
  scale_y_continuous(name="Mean Square Error")+
  theme_bw() +
  annotate(geom="text", x=7.5, y=1, label="E_train", parse=TRUE
             ,color="red")+
  annotate("segment", x = 6.6, xend = 7, y = 1, yend = 1,
         colour = "red")+
  annotate(geom="text", x=7.6, y=0.95, label="E_unseen", parse=TRUE
           ,color="black")+
annotate("segment", x = 6.6, xend = 7, y = 0.95, yend = 0.95,
         colour = "black")
p



## ----cache=TRUE,message=FALSE-----------------------------------------------------------------------
load("Breast_cancer.RData")
library(glmnet)
data <- Breast_cancer_data
data <- data %>% mutate(diagnosis_0_1 = ifelse(diagnosis == "M", 1, 0))
library(caret)
set.seed(101)
training <- createDataPartition(data$diagnosis_0_1, p=0.8, list=FALSE)
train <- data[ training, ]
test <- data[ -training, ]
trainfull <- train[,-c(1,2)]
testfull <- data[ -training, -c(1,2)]
model.lasso <- glmnet(trainfull[,-31],trainfull[,31],family="binomial",alpha=1)
model.ridge <- glmnet(trainfull[,-31],trainfull[,31],family="binomial",alpha=0)
par(mfrow=c(2,2))
plot(model.ridge,xvar="lambda")
plot(model.lasso,xvar="lambda")
plot(model.ridge,xvar="dev")
plot(model.lasso,xvar="dev")


## ---------------------------------------------------------------------------------------------------
ind.lambda.r <- which.min(model.ridge$dev.ratio<0.8)
#model.ridge$beta[,ind.lambda.r]
ind.lambda.l <- which.min(model.lasso$dev.ratio<0.8)
#model.lasso$beta[,ind.lambda.l]
res <- cbind(model.ridge$beta[,ind.lambda.r],model.lasso$beta[,ind.lambda.l])
colnames(res) <- c("ridge","lasso")
res


## ---------------------------------------------------------------------------------------------------
## Example Lasso and ridge regression 
## Breast cancer 
### A practice on Breast Cancer Data
model.lasso <- cv.glmnet(as.matrix(trainfull[,-31]),trainfull[,31],family="binomial",alpha=1,type.measure = "class")
model.ridge <- cv.glmnet(as.matrix(trainfull[,-31]),trainfull[,31],family="binomial",alpha=0,type.measure = "class")
plot(model.lasso)


## ---------------------------------------------------------------------------------------------------
res.l <- predict(model.lasso,s = "lambda.min",newx=as.matrix(testfull[,-31]),type="response")
classifier.l <- ifelse(res.l>0.5,"1","0")
table(classifier.l,testfull[,31])
result.lasso <- caret::confusionMatrix(factor(classifier.l),factor(testfull[,31]),positive="1")
result.lasso$table
result.lasso$overall[1]
result.lasso$byClass[c(1,2,5,6)]


## ----cache=TRUE-------------------------------------------------------------------------------------
library(glmnet)
Ytrain[which(Ytrain==-1)] <- 0
fit=cv.glmnet(Xtrain[1:1000,],Ytrain[1:1000,],family = "multinomial",type.measure = "class")
plot(fit)


## ----message=NA-------------------------------------------------------------------------------------
library(IMIFA)
pred=predict(fit,Xtest,s=fit$lambda.min,type="class")-1
par(mfrow=c(2,3))
for(i in 1:6){
  show_digit(Xtest[i,])
  title(sprintf("prediction = %s",pred[i]))
}
mean(labels_test_set==pred)


## ----cache=TRUE-------------------------------------------------------------------------------------
library(nnet)
model <- multinom(Ytrain[1:20000,] ~., family = "multinomial", MaxNWts =10000000, maxit=50,data=as.data.frame(Xtrain[1:20000,]));


## ---------------------------------------------------------------------------------------------------
results <- predict(model, newdata=Xtest, type='probs')
prediction <- max.col(results)
prediction <- prediction - 1
cl <- mean(prediction != labels_test_set)
print(paste('Accuracy', 1 - cl))


## ---------------------------------------------------------------------------------------------------
load("versatile-boundaries.RData")
library(ggplot2)
ggplot(data) + geom_point(aes(x = x,y = y,color = as.character(label)), size = 1) +
  theme_bw(base_size = 15) +
  xlim(-2.5, 1.5) +
  ylim(-2.5, 2.5) +
  coord_fixed(ratio = 0.8) +
  theme(axis.ticks=element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text=element_blank(),
        axis.title=element_blank(),
        legend.position = "none")


## ---------------------------------------------------------------------------------------------------
train_data <- data
trainX <- train_data[, c(1, 2)]
trainY <- train_data[, 3]
trainY <- ifelse(trainY == 1, 0, 1)
data.glm <- data.frame(Y=trainY,X1=trainX[,1],X2=trainX[,2])
model.logistic <- glm(Y~X1+X2,data=data.glm,family=binomial)
## As a reminder we check the shape of our classifier
step <- 0.01
x_min <- min(trainX[, 1]) - 0.2
x_max <- max(trainX[, 1]) + 0.2
y_min <- min(trainX[, 2]) - 0.2
y_max <- max(trainX[, 2]) + 0.2
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = step), seq(y_min,
                                                                y_max, by = step)))

data.grid <- data.frame(X1=grid[,1],X2=grid[,2])

Z <- predict(model.logistic,newdata=data.grid,type="response")
Z <- ifelse(Z <0.5, 1, 2)


g1 <- ggplot() +
  geom_tile(aes(x = grid[, 1], y = grid[, 2], fill = as.character(Z)), alpha = 0.3, show.legend = F)+
  geom_point(data = train_data, aes(x = x, y = y, color = as.character(trainY)),size = 1)+
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  labs(x=expression(x[1]),cex=2) +labs(y=expression(x[2]))+
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
       panel.grid.minor = element_blank(), legend.position = "none",axis.title=element_text(size=18,face="italic"))
  
g1


## ---------------------------------------------------------------------------------------------------
model.logistic.nl2 <- glm(Y~polym(X1, X2, degree=2, raw=TRUE),data=data.glm,family=binomial)


Z <- predict(model.logistic.nl2,newdata=data.grid,type="response")
Z <- ifelse(Z <0.5, 1, 2)

g2 <- ggplot() +
  geom_tile(aes(x = grid[, 1], y = grid[, 2], fill = as.character(Z)), alpha = 0.3, show.legend = F)+
  geom_point(data = train_data, aes(x = x, y = y, color = as.character(trainY)),size = 1)+
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")+
  theme(axis.title=element_text(size=18,face="italic"))

g2+ labs(x=expression(x[1]),cex=2) +labs(y=expression(x[2]),cex=2)


## ---------------------------------------------------------------------------------------------------
model.logistic.nl3 <- glm(Y~polym(X1, X2, degree=4, raw=TRUE),data=data.glm,family=binomial)
model.logistic.nl4 <- glm(Y~polym(X1, X2, degree=8, raw=TRUE),data=data.glm,family=binomial)

Z <- predict(model.logistic.nl3,newdata=data.grid,type="response")
Z <- ifelse(Z <0.5, 1, 2)
par(mfrow=c(1,2))
g2 <- ggplot() +
  geom_tile(aes(x = grid[, 1], y = grid[, 2], fill = as.character(Z)), alpha = 0.3, show.legend = F)+
  geom_point(data = train_data, aes(x = x, y = y, color = as.character(trainY)),size = 1)+
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(), legend.position = "none")+
  theme(axis.title=element_text(size=18,face="italic"))

g2+ labs(x=expression(x[1]),cex=2) +labs(y=expression(x[2]),cex=2)


Z <- predict(model.logistic.nl4,newdata=data.grid,type="response")
Z <- ifelse(Z <0.5, 1, 2)
g3 <- ggplot() +
  geom_tile(aes(x = grid[, 1], y = grid[, 2], fill = as.character(Z)), alpha = 0.3, show.legend = F)+
  geom_point(data = train_data, aes(x = x, y = y, color = as.character(trainY)),size = 1)+
  theme_bw(base_size = 15) + coord_fixed(ratio = 0.8) +
  labs(x=expression(x[1]),cex=2) +labs(y=expression(x[2]))+
  theme(axis.ticks = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), legend.position = "none",axis.title=element_text(size=18,face="italic"))


g3+ labs(x=expression(x[1]),cex=2) +labs(y=expression(x[2]),cex=2)


## ----cache=TRUE,message=NA--------------------------------------------------------------------------
clr1 <- c(rgb(1,0,0,1),rgb(0,1,0,1),rgb(0,0,1,1),rgb(0.5,0.5,0.5,1))
clr2 <- c(rgb(1,0,0,.1),rgb(0,1,0,.1),rgb(0,0,1,.1),rgb(0.5,0.5,0.5,.5))
x <- c(.4,.55,.65,.9,.1,.35,.5,.15,.2,.85,0.2,0.7,0.3,0.5,0.6,0.55,0.4)
y <- c(.85,.95,.8,.87,.5,.55,.5,.2,.1,.3,0.7,0.5,0.3,0.7,0.6,0.65,0.7)
z <- c(1,2,2,2,1,0,0,1,0,0,1,2,1,3,3,3,3)
df <- data.frame(x,y,z)
plot(x,y,pch=19,cex=2,col=clr1[z+1],ylab=expression(x[2]),xlab=expression(x[1]),cex.lab=1.4)
#write.csv(df,file="synthetic_data_4_class.csv",row.names = FALSE)
library(nnet)
model.mult <- multinom(z~x+y,data=df)

pred_mult <- function(x,y){
  res <- predict(model.mult,
                 newdata=data.frame(x=x,y=y),type="probs")
  apply(res,MARGIN=1,which.max)
}
x_grid<-seq(0,1,length=601)
y_grid<-seq(0,1,length=601)
z_grid <- outer(x_grid,y_grid,FUN=pred_mult)


image(x_grid,y_grid,z_grid,col=clr2,ylab=expression(x[2]),xlab=expression(x[1]),cex.lab=1.4)

points(x,y,pch=19,cex=2,col=clr1[z+1])
legend("topleft", inset=0.02, legend=c("class 1", "class 2","class 3","class 4"),
       col=clr1[c(1,3,2,4)], cex=0.9, pch=c(19,19))

model.mult <- multinom(z~polym(x,y, degree=2, raw=TRUE),data=df)


x_grid<-seq(0,1,length=601)
y_grid<-seq(0,1,length=601)
z_grid <- outer(x_grid,y_grid,FUN=pred_mult)

image(x_grid,y_grid,z_grid,col=clr2,ylab=expression(x[2]),xlab=expression(x[1]),cex.lab=1.4)

points(x,y,pch=19,cex=2,col=clr1[z+1])
legend("topleft", inset=0.02, legend=c("class 1", "class 2","class 3","class 4"),
       col=clr1[c(1,3,2,4)], cex=0.9, pch=c(19,19))


model.mult <- multinom(z~polym(x,y, degree=8, raw=TRUE),data=df)


x_grid<-seq(0,1,length=601)
y_grid<-seq(0,1,length=601)
z_grid <- outer(x_grid,y_grid,FUN=pred_mult)


image(x_grid,y_grid,z_grid,col=clr2,ylab=expression(x[2]),xlab=expression(x[1]),cex.lab=1.4)

points(x,y,pch=19,cex=2,col=clr1[z+1])
legend("topleft", inset=0.02, legend=c("class 1", "class 2","class 3","class 4"),
       col=clr1[c(1,3,2,4)], cex=0.9, pch=c(19,19))

