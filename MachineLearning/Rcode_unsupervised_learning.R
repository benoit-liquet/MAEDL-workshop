## ----fig.width=5,eval=FALSE-----------------------------------------------------------------------------
library(animation)
set.seed(101)
library(mvtnorm)
x = rbind(rmvnorm(40, mean=c(0,1),sigma = 0.05*diag(2)),rmvnorm(40, mean=c(0.5,0),sigma = 0.05*diag(2)),rmvnorm(40, mean=c(1,1),sigma = 0.05*diag(2)))
par(mfrow=c(3,2))
colnames(x) = c("x1", "x2")
kmeans.ani(x, centers = matrix(c(0.5,1,0.5,0,1,1),byrow=T,ncol=2))


## ----eval=TRUE,cache=TRUE-------------------------------------------------------------------------------
library(ggplot2)
library(jpeg)
img <- readJPEG("Yoni-ben-pool-seg.jpg")

# Obtain the dimension
imgDm <- dim(img)

# Assign RGB channels to data frame
imgRGB <- data.frame(
  x = rep(1:imgDm[2], each = imgDm[1]),
  y = rep(imgDm[1]:1, imgDm[2]),
  R = as.vector(img[,,1]),
  G = as.vector(img[,,2]),
  B = as.vector(img[,,3])
)

par(mfrow=c(3,1))

# Plot the original image
p1 <- ggplot(data = imgRGB, aes(x = x, y = y)) + 
  geom_point(colour = rgb(imgRGB[c("R", "G", "B")])) +
  labs(title = "Original Image") +
  xlab("x") +
  ylab("y") 
p1


kClusters <- 2
kMeans <- kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters)
kColours <- rgb(kMeans$centers[kMeans$cluster,])


p2 <- ggplot(data = imgRGB, aes(x = x, y = y)) + 
  geom_point(colour = kColours) +
  labs(title = paste("k-Means Clustering of", kClusters, "Colours")) +
  xlab("x") +
  ylab("y") 
p2

kClusters <- 6
kMeans <- kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters)
kColours <- rgb(kMeans$centers[kMeans$cluster,])


p6 <- ggplot(data = imgRGB, aes(x = x, y = y)) + 
  geom_point(colour = kColours) +
  labs(title = paste("k-Means Clustering of", kClusters, "Colours")) +
  xlab("x") +
  ylab("y") 
p6


## -------------------------------------------------------------------------------------------------------
load("Breast_cancer.RData")
head(Breast_cancer_data[,c(1:6)])
data <- Breast_cancer_data
dim(data)


## ----fig.width=3----------------------------------------------------------------------------------------
library(FactoMineR)
library(ggplot2)
library(dplyr)

pca <- PCA(data[,-c(1,2)],ncp=2,graph=FALSE)
dat <- data.frame(data,pc1=pca$ind$coord[,1],pc2=pca$ind$coord[,2],diagnosis=as.factor(data[,2]))

#dat <- dat %>% filter(pc1<7 & pc2<10) 

p1 <- ggplot(data = dat, aes(x = pc1, y = pc2))+
  geom_hline(yintercept = 0, lty = 2) +
  geom_vline(xintercept = 0, lty = 2) +
  geom_point(alpha = 0.8,size=2.5) + theme_bw()
 
p1 + theme(axis.text = element_text(size = 20))+ theme(axis.title = element_text(size = 20))   


## -------------------------------------------------------------------------------------------------------
pca$eig[1:2,]


## ----fig.width=5----------------------------------------------------------------------------------------
p2 <- ggplot(data = dat, aes(x = pc1, y = pc2, color = diagnosis))+
  geom_hline(yintercept = 0, lty = 2) +
  geom_vline(xintercept = 0, lty = 2) +
  geom_point(alpha = 0.8,size=2.5) + theme_bw()+
  theme(legend.position=c(0.15,0.85),legend.title=element_blank())
p3 <- p2 +  scale_color_discrete( labels = c("benign", "malignant"))

p3 + theme(legend.text=element_text(size=20),axis.text = element_text(size = 20))+ theme(axis.title = element_text(size = 20))


## ----fig.width=8,fig.height=10,eval=FALSE---------------------------------------------------------------
if (!"jpeg" %in% installed.packages()) install.packages("jpeg")
# Read image file into an array with three channels (Red-Green-Blue, RGB)
myImage <- jpeg::readJPEG("CODE_WORKSHOP/pool_graysacle.jpg")

r <- myImage[, , 1] 
# Performs full SVD 
myImage.r.svd <- svd(r)# ; lmyImage.g.svd <- svd(g) ; myImage.b.svd <- svd(b)
rgb.svds <- list(myImage.r.svd)#



plot.image <- function(pic, main = "") {
  h <- dim(pic)[1] ; w <- dim(pic)[2]
  plot(x = c(0, h), y = c(0, w), type = "n", xlab = "", ylab = "", main = main)
  rasterImage(pic, 0, 0, h, w)
}


compress.image <- function(rgb.svds, nb.comp) {
  # nb.comp (number of components) should be less than min(dim(img[,,1])), 
  # i.e., 170 here
  svd.lower.dim <- lapply(rgb.svds, function(i) list(d = i$d[1:nb.comp], 
                                                     u = i$u[, 1:nb.comp], 
                                                     v = i$v[, 1:nb.comp]))
  img <- sapply(svd.lower.dim, function(i) {
    img.compressed <- i$u %*% diag(i$d) %*% t(i$v)
  }, simplify = 'array')
  img[img < 0] <- 0
  img[img > 1] <- 1
  return(list(img = img, svd.reduced = svd.lower.dim))
}



par(mfrow = c(2, 2))
plot.image(r, "Original image")

p <- 10 ; plot.image(compress.image(rgb.svds, p)$img[,,1], 
                     paste("SVD with", p, "components"))

p <- 30 ; plot.image(compress.image(rgb.svds, p)$img[,,1], 
                     paste("SVD with", p, "components"))


p <- 50 ; plot.image(compress.image(rgb.svds, p)$img[,,1], 
                     paste("SVD with", p, "components"))



## ----eval=TRUE,cache=TRUE,message=FALSE-----------------------------------------------------------------
# To install the R package downlaod the tar.gz file at https://cran.r-project.org/src/contrib/Archive/ruta/
library(ruta)
library(rARPACK)
library(ggplot2)
###############
### Function plot 
###############
plot_digit <- function(digit, ...) {
  image(keras::array_reshape(digit, c(28, 28), "F")[, 28:1], xaxt = "n", yaxt = "n", col=gray(1:256 / 256), ...)
}

plot_sample <- function(digits_test, model1,model2,model3, sample) {
  sample_size <- length(sample)
  layout(
    matrix(c(1:sample_size, (sample_size + 1):(4 * sample_size)), byrow = F, nrow = 4)
  )
  
  
  for (i in sample) {
    par(mar = c(0,0,0,0) + 1)
    plot_digit(digits_test[i, ])
    plot_digit(model1[i, ])
    plot_digit(model2[i, ])
    plot_digit(model3[i, ])
  }
}


#######################
#### Load MNIST DATA
#######################

mnist = keras::dataset_mnist()

# Normalization to the [0, 1] interval
x_train <- keras::array_reshape(
  mnist$train$x, c(dim(mnist$train$x)[1], 784)
)
x_train <- x_train / 255.0
x_test <- keras::array_reshape(
  mnist$test$x, c(dim(mnist$test$x)[1], 784)
)
x_test <- x_test / 255.0

if(T){
network <- input() + dense(30, "tanh") + output("sigmoid")
network1 <- input() + dense(50, "tanh") +dense(10, "linear")+dense(50, "tanh") +output("sigmoid")
}

### model simple
network.simple <- autoencoder(network)#, loss = "binary_crossentropy")
model = train(network.simple, x_train, epochs = 10)
decoded.simple <- reconstruct(model, x_test)


### model deep
my_ae2 <- autoencoder(network1)#, loss = "binary_crossentropy")
model2 = train(my_ae2, x_train, epochs = 10)
decoded2 <- reconstruct(model2, x_test)

#### Linear interpolation between two digits
digit_A = x_train[which(mnist$train$y==3)[1],]#MNIST digit with 3 (This is the first digit in the train set that has 3)
digit_B = x_train[which(mnist$train$y==3)[10],]#another MNIST digit with 3 (This is the 10[th] digit in the train set that has 3)
latent_A = encode(model2,matrix(digit_A,nrow=1))
latent_B = encode(model2,matrix(digit_B,nrow=1))
lambda = 0.5
latent_interpolation = lambda*latent_A + (1-lambda)*latent_B
rought_interpolation = lambda*digit_A + (1-lambda)*digit_B

output_interpolation = decode(model2,latent_interpolation)


par(mar = c(0,0,0,0) + 1,mfrow=c(1,3))
plot_digit(digit_A)
plot_digit(as.vector(output_interpolation))
plot_digit(digit_B)


par(mar = c(0,0,0,0) + 1,mfrow=c(1,3))
plot_digit(digit_A)
plot_digit(as.vector(rought_interpolation))
plot_digit(digit_B)


