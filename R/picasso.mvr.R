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
                        method="l1",
                        alg = "cyclic",
                        gamma = 3,
                        design.sd = TRUE,
                        max.act.in = 3, 
                        truncation = 0, 
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
  if(method!="l1" && method!="mcp" && method!="scad"){
    cat(" Wrong \"method\" input. \n \"method\" should be one of \"l1\", \"mcp\" and \"scad\".\n", 
        method,"does not exist. \n")
    return(NULL)
  }
  if(alg!="cyclic" && alg!="greedy" && alg!="prox" && alg!="stoc"){
    cat(" Wrong \"alg\" input. \n \"alg\" should be one of \"cyclic\", \"greedy\", \"prox\" and \"stoc\".\n", 
        alg,"does not exist. \n")
    return(NULL)
  }
  maxdf = max(n,d)
  if(design.sd){
    xm=matrix(rep(colMeans(X),n),nrow=n,ncol=d,byrow=T)
    x1=X-xm
    xinvc.vec = 1/sqrt(colSums(x1^2)/(n-1))
    xinvc=diag(xinvc.vec)
    xx=x1%*%xinvc
    ym=matrix(rep(colMeans(Y),n),nrow=n,ncol=m,byrow=T)
    yy=Y-ym
  }else{
    xinvc.vec = rep(1,d)
    xx = X
    yy = Y
  }
  
  est = list()
  
  S = colSums(xx^2)
  Uinv.vec = 1/sqrt(S/(n)) #??
  Uinv = diag(Uinv.vec)
  xx1 = xx%*%Uinv
  est$uinv = Uinv.vec
  
  if(!is.null(lambda)) nlambda = length(lambda)
  if(is.null(lambda)){
    if(is.null(nlambda))
      nlambda = 5
    if(is.null(lambda.min.ratio)){
      lambda.min.ratio = 0.25
    }
    lambda.max = sqrt(max(rowSums((t(xx1)%*%yy)^2)))/n
    lambda.min = lambda.min.ratio*lambda.max
    lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda))
    rm(lambda.max,lambda.min,lambda.min.ratio)
    gc()
  }else{
    lambda = lambda*max(Uinv.vec)
  }
  begt=Sys.time()
  if(method=="l1") {
    method.flag = 1
    if (alg=="cyclic")
      out = mvr.cyclic.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag, max.act.in, truncation)
    if (alg=="greedy")
      out = mvr.greedy.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag)
    if (alg=="prox")
      out = mvr.prox.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag)
    if (alg=="stoc")
      out = mvr.stoc.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag, max.act.in, truncation)
  }
  if(method=="mcp") {
    method.flag = 2
    if (gamma<=1) {
      cat("gamma > 1 is required for MCP. Set to default value 3. \n")
      gamma = 3
    }
    if (alg=="cyclic")
      out = mvr.cyclic.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag, max.act.in, truncation)
    if (alg=="greedy")
      out = mvr.greedy.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag)
    if (alg=="prox")
      out = mvr.prox.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag)
    if (alg=="stoc")
      out = mvr.stoc.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag, max.act.in, truncation)
  }
  if(method=="scad") {
    method.flag = 3
    if (gamma<=2) {
      cat("gamma > 2 is required for SCAD. Set to default value 3. \n")
      gamma = 3
    }
    if (alg=="cyclic")
      out = mvr.cyclic.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag, max.act.in, truncation)
    if (alg=="greedy")
      out = mvr.greedy.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag)
    if (alg=="prox")
      out = mvr.prox.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag)
    if (alg=="stoc")
      out = mvr.stoc.orth(yy, xx1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, xinvc.vec, Uinv.vec, Uinv, method.flag, max.act.in, truncation)
  }

  runt=Sys.time()-begt
  
  df=rep(0,nlambda)
  for(i in 1:nlambda)
    df[i] = sum(out$beta[[i]]!=0)/m
  
  beta1 = vector("list", nlambda)
  intcpt = vector("list", nlambda)
  if(design.sd){
    for(k in 1:nlambda){
      tmp.beta = out$beta[[k]]
      beta1[[k]]=xinvc%*%tmp.beta
      intcpt[[k]]=ym[1,]-xm[1,]%*%beta1[[k]]+out$intcpt[[k]]
    }
  }else{
    for(k in 1:nlambda){
      beta1[[k]]=out$beta[[k]]
      intcpt[[k]] = out$intcpt[[k]]
    }
  }
  
  est$xinvc = xinvc.vec
  est$runt = out$runt
  est$obj = out$obj
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
  cat("Average df:",min(x$df),"----->",max(x$df),"\n")
  if(units.difftime(x$runtime)=="secs") unit="secs"
  if(units.difftime(x$runtime)=="mins") unit="mins"
  if(units.difftime(x$runtime)=="hours") unit="hours"
  cat("Runtime:",x$runtime," ",unit,"\n")
}
