## KOMD
## @author: Michele Donini
## @email: mdonini@math.unipd.it

## KOMD is a kernel machine for ranking and classification.
## Paper: "A Kernel Method for the Optimization of the Margin Distribution"
## by F. Aiolli, G. Da San Martino, and A. Sperduti.


## REQUIREMENTS: kernlab package
## install.packages("kernlab")
library(kernlab)

## Toy task
data(spam)
m <- 500
set <- sample(1:dim(spam)[1],m)
x <- scale(as.matrix(spam[,-58]))[set,]
y <- as.integer(spam[set,58])
y[y==2] <- -1

## train and test: 1 fold of a 10-Fold cross-validation
id_train <- sample(1:length(y),9 * length(y)/10)
id_test <- setdiff(1:length(y),id_train)

## Settings
lambda <- 0.1
xtr <- x[id_train,]
xte <- x[id_test,]
ytr <- y[id_train]
yte <- y[id_test]
rbf <- rbfdot(sigma = 0.1)
K <- kernelPol(rbf,x,,y)
Ktr <- K[id_train,id_train]
Kte <- K[id_test,id_train]


komd_train  <- function(K,y,lambda) {
  
  ## if m := cardinality of the training set then
  ## K : kernel matrix {m x m}
  ## y : labels {m}
  ## lambda : hyperparameter in the real interval [0,1]
  
  m = length(y)
  Y <- diag(y)
  H <- 2 * ((1-lambda) * Y %*% K %*% Y + diag(nrow=m))
  c <- matrix(rep(0.0,m))
  A <- t(matrix(c(y==1, y==-1)+ 0.0))
  dim(A) <- c(dim(A)[2]/2,2)
  A <- t(A)
  b <- matrix(rep(1.0,2))
  l <- matrix(rep(0.0,m))
  u <- matrix(rep(1.0,m))
  r <- matrix(rep(0.0,2))
  
  ##ipop solves the quadratic programming problem :
  ##  min(c' ??? x + 1/2 ??? x' ??? H ??? x)
  ##     subject to:
  ##           b <= A ??? x <= b + r
  ##           l <= x <= u
  ##ipop(c, H, A, b, l, u, r, sigf = 7, maxiter = 40, margin = 0.05, bound = 10, verb = 0)
  
  komd_clf <- ipop(c,H,A,b,l,u,r)
  ##gamma <- primal(komd_clf) ## KOMD solution
  ##dual_obj <- dual(komd_clf) ## dual objective function
  print(how(komd_clf))
  return(komd_clf)
}

komd_rank   <- function(K,y,gamma) {
  
  ## if m := cardinality of the training set; n:= cardinality of the test set then
  ## K : kernel matrix {m x n}
  ## y : labels of the train examples {m}
  ## gamma : komd primal model solution
  
  Y <- diag(y)
  ## ranking
  rank <- (gamma %*% Y) %*% t(K) 
  return(rank)
}

komd_classification <- function(Kte,Ktr,y,gamma) {
  
  ## if m := cardinality of the training set; n:= cardinality of the test set then
  ## K : kernel matrix {m x n}
  ## y : labels of the train examples {m}
  ## gamma : komd primal model solution
  
  Y <- diag(y)
  ## ranking
  rank <- komd_rank(Kte,y,gamma)
  bias <- 0.5 * t(gamma) %*% (Ktr %*% (Y %*% gamma))
  classification <- sign(rank-bias[1])
  return(classification)
}

## train the model
komd_clf <- komd_train(Ktr,ytr,lambda)
gamma <- primal(komd_clf)
## ranking
rank <- komd_rank(Kte,ytr,gamma)
## classification
classification <- komd_classification(Kte,Ktr,ytr,gamma)

## accuracy
accuracy <- sum(classification == yte) / length(yte)
print(c("Accuracy:",accuracy))

## PLOT
##RED negative examples, BLUE positive examples 
plot(rank[1,],col=(yte+3)) 
## threshold as a black solid line 
abline(h = bias)
