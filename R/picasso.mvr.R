#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# picasso.mvr(): The user interface for mvr()                                       #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 1st, 2014                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

picasso.mvr <- function(X, 
                        Y, 
                        lambda = NULL,
                        nlambda = NULL,
                        lambda.min.ratio = NULL,
                        method="l12",
                        alg = "cyclic",
                        res.sd = FALSE,
                        prec = 1e-4,
                        max.ite = 1e4,
                        verbose = TRUE)
{
  n = nrow(X)
  d = ncol(X)
  m = ncol(Y)
  if(verbose)
    cat("Sparse multivariate regression. \n")
  if(n==0 || d==0) {
    cat("No data input.\n")
    return(NULL)
  }
  maxdf = max(n,d)
  design.sd = TRUE
  if(design.sd){
    xm=matrix(rep(colMeans(X),n),nrow=n,ncol=d,byrow=T)
    x1=X-xm
    sdx=sqrt(diag(t(x1)%*%x1)/(n-1))
    Cxinv=diag(1/sdx)
    xx=x1%*%Cxinv
    ym=matrix(rep(colMeans(Y),n),nrow=n,ncol=m,byrow=T)
    yy=Y-ym
  }else{
    xx = X
    yy = Y
  }
  
  if(!is.null(lambda)) nlambda = length(lambda)
  if(is.null(lambda)){
    if(is.null(nlambda))
      nlambda = 5
    if(is.null(lambda.min.ratio)){
      lambda.min.ratio = 0.25
    }
    lambda.max = max(abs(crossprod(xx,yy/n/m)))
    cat("lambda.max=",lambda.max,"\n")
    lambda.min = lambda.min.ratio*lambda.max
    lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda))
    rm(lambda.max,lambda.min,lambda.min.ratio)
    gc()
  }
  begt=Sys.time()
  if(method=="l12") {
    if (alg=="cyclic")
      out = mvr.cyclic(yy, xx, lambda, nlambda, n, d, m, max.ite, prec, verbose)
    if (alg=="greedy")
      out = mvr.greedy(yy, xx, lambda, nlambda, n, d, m, max.ite, prec, verbose)
    if (alg=="prox")
      out = mvr.prox(yy, xx, lambda, nlambda, n, d, m, max.ite, prec, verbose)
    if (alg=="stoc")
      out = mvr.stoc(yy, xx, lambda, nlambda, n, d, m, max.ite, prec, verbose)
  }

  runt=Sys.time()-begt
  
  df=rep(0,nlambda)
  for(i in 1:nlambda)
    df[i] = sum(out$beta[[i]]!=0)/m
  
  est = list()
  beta1 = vector("list", nlambda)
  intcpt = vector("list", nlambda)
  if(design.sd){
    for(k in 1:nlambda){
      tmp.beta = out$beta[[k]]
      beta1[[k]]=Cxinv%*%tmp.beta
      intcpt[[k]]=ym[1,]-xm[1,]%*%beta1[[k]]+out$intcpt[[k]]
    }
  }else{
    for(k in 1:nlambda){
      beta1[[k]]=out$beta[[k]]
      intcpt[[k]] = out$intcpt[[k]]
    }
  }
  
  est$size.act = out$size.act
  est$beta = beta1
  est$intercept = intcpt
  est$Y = Y
  est$X = X
  est$lambda = lambda
  est$nlambda = nlambda
  est$gamma = gamma
  est$df = df
  est$method = method
  est$alg = alg
  est$ite =out$ite
  est$verbose = verbose
  est$runtime = runt
  class(est) = "mvr"
  return(est)
}

print.mvr <- function(x, ...)
{  
  cat("\n MVR options summary: \n")
  cat(x$nlambda, " lambdas used:\n")
  print(signif(x$lambda,digits=3))
  cat("Average df:",min(x$sparsity),"----->",max(x$sparsity),"\n")
  if(units.difftime(x$runtime)=="secs") unit="secs"
  if(units.difftime(x$runtime)=="mins") unit="mins"
  if(units.difftime(x$runtime)=="hours") unit="hours"
  cat("Runtime:",x$runtime," ",unit,"\n")
}
