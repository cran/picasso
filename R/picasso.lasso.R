#----------------------------------------------------------------------------------#
# Package: picasso                                                                 #
# picasso.lasso(): The user interface for lasso()                                  #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Jul 5th, 2014                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

picasso.lasso <- function(X, 
                          Y, 
                          lambda = NULL,
                          nlambda = NULL,
                          lambda.min.ratio = NULL,
                          method="l1",
                          alg = "cyclic",
                          gamma = 3,
                          design.sd = TRUE,
                          res.sd = FALSE,
                          gr = NULL,
                          gr.d = NULL,
                          gr.size = NULL,
                          max.act.in = 3, 
                          truncation = 0, 
                          prec = 1e-4,
                          max.ite = 1e4,
                          verbose = TRUE)
{
  n = nrow(X)
  d = ncol(X)
  if(verbose)
    cat("Sparse linear regression. \n")
  if(n==0 || d==0) {
    cat("No data input.\n")
    return(NULL)
  }
  if(method!="l1" && method!="mcp" && method!="scad" && method!="group" && method!="group.mcp" && method!="group.scad"){
    cat(" Wrong \"method\" input. \n \"method\" should be one of \"l1\", \"mcp\", \"scad\", \"group\", \"group.mcp\" and \"group.scad\".\n", 
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
    xinvc.vec=1/sqrt(colSums(x1^2)/(n-1))
    xx=x1%*%diag(xinvc.vec)
    ym=mean(Y)
    y1=Y-ym
    if(res.sd == TRUE){
      sdy=sqrt(sum(y1^2)/(n-1))
      yy=y1/sdy
    }else{
      sdy = 1
      yy = y1
    }
  }else{
    xinvc.vec = rep(1,d)
    sdy = 1
    xx = X
    yy = Y
  }
  
  if(method=="l1"||method=="mcp"||method=="scad") {
    if(!is.null(lambda)) nlambda = length(lambda)
    if(is.null(lambda)){
      if(is.null(nlambda))
        nlambda = 5
      if(is.null(lambda.min.ratio)){
        lambda.min.ratio = 0.25
      }
      lambda.max = max(abs(crossprod(xx,yy/n)))
      lambda.min = lambda.min.ratio*lambda.max
      lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda))
      rm(lambda.max,lambda.min,lambda.min.ratio)
      gc()
    }
  }
  begt=Sys.time()
  if(method=="l1") {
    method.flag = 1
    if (alg=="cyclic")
      out = lasso.cyclic(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag, max.act.in, truncation)
    if (alg=="greedy")
      out = lasso.greedy(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag)
    if (alg=="prox")
      out = lasso.prox(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag)
    if (alg=="stoc")
      out = lasso.stoc(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag, max.act.in, truncation)
  }
  if(method=="scad") {
    method.flag = 3
    if (gamma<=2) {
      cat("gamma > 2 is required for SCAD. Set to default value 3. \n")
      gamma = 3
    }
    if (alg=="cyclic")
      out = lasso.cyclic(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag, max.act.in, truncation)
    if (alg=="greedy")
      out = lasso.greedy(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag)
    if (alg=="prox")
      out = lasso.prox(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag)
    if (alg=="stoc")
      out = lasso.stoc(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag, max.act.in, truncation)
  }
  if(method=="mcp") {
    method.flag = 2
    if (gamma<=1) {
      cat("gamma > 1 is required for MCP. Set to default value 3. \n")
      gamma = 3
    }
    if (alg=="cyclic")
      out = lasso.cyclic(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag, max.act.in, truncation)
    if (alg=="greedy")
      out = lasso.greedy(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag)
    if (alg=="prox")
      out = lasso.prox(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag)
    if (alg=="stoc")
      out = lasso.stoc(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, method.flag, max.act.in, truncation)
  }
  if(method=="group"||method=="group.mcp"||method=="group.scad") {
    if (is.null(gr)) {
      gr = list()
      if(is.null(gr.d)){
        if(is.null(gr.size)){
          gr.d = 2
          gr.n = ceiling(d/gr.d)
          gr.size = rep(gr.d,gr.n)
          if(sum(gr.size)>d) gr.size[gr.n] = d - sum(gr.size[1:(gr.n-1)])
        }else{
          if(sum(gr.size)!=d) {
            cat('Group size error... sum(gr.size) !=',d,'\n')
            return(NULL)
          }
          gr.n = length(gr.size)
        }
      }else{
        if(gr.d>d){
          cat('Dimension of per group error... gr.d >',d,'\n')
          return(NULL)
        }
        if(!is.null(gr.size))
          cat('Group decided by gr.d \n')
        gr.n = ceiling(d/gr.d)
        gr.size = rep(gr.d,gr.n)
        if(sum(gr.size)>d) gr.size[gr.n] = d - sum(gr.size[1:(gr.n-1)])
      }
      idx = 1
      for(i in 1:gr.n){
        gr[[i]] = c(idx:(idx+gr.size[i]-1))
        idx = idx + gr.size[i]
      }
    }else{
      if(is.null(gr.n)){
        gr.n = length(gr)
        if(is.null(gr.size)){
          gr.size = rep(0,gr.n)
          for(i in 1:gr.n){
            gr.size[i] = length(gr[[i]])
            if(max(gr[[i]])>d) {
              max.idx = which(gr[[i]]==max(gr[[i]]))
              cat('Group index error... gr[[',i,']][',max.idx,'] >',d,'\n')
              return(NULL)
            }
          }
        }else{
          if(sum(gr.size)!=d) {
            cat('Group size error... sum(gr.size) !=',d,'\n')
            return(NULL)
          }
        }
      }else{
        if(gr.n != length(gr)){
          gr.n = length(gr)
          cat('Group size error... gr.n !=',gr.n,'. Set gr.n=length(gr) \n')
        }
        if(is.null(gr.size)){
          gr.size = rep(1,gr.n)
          idx = c(1:d)
          for(i in 1:gr.n){
            gr.size[i] = length(gr[[i]])
            if(max(gr[[i]])>d) {
              max.idx = which(gr[[i]]==max(gr[[i]]))
              cat('Group index error... gr[[',i,']][',max.idx,'] >',d,'\n')
              return(NULL)
            }
            idx[gr[[i]]] = 0
          }
          if(sum(idx)>0)
            cat('Index ', which(idx==1),' not in the group \n')
        }else{
          if(sum(gr.size)!=d) {
            cat('Group size error... sum(gr.size) !=',d,'\n')
            return(NULL)
          }
        }
      }
    }
    xx1 = xx
    Uinv.list = vector("list", gr.n)
    for(i in 1:gr.n){
      #Uinv.list[[i]] = chol2inv(chol(chol(crossprod(X[,gr[[i]]])/n)))
      Uinv.list[[i]] = solve(chol(crossprod(xx[,gr[[i]]])/n))
      xx1[,gr[[i]]] = xx[,gr[[i]]]%*%Uinv.list[[i]]
    }
    if(!is.null(lambda)) nlambda = length(lambda)
    if(is.null(lambda)){
      if(is.null(nlambda))
        nlambda = 5
      if(is.null(lambda.min.ratio)){
        lambda.min.ratio = 0.25
      }
      lambda.max = max(abs(crossprod(xx,yy/n)))
      lambda.min = lambda.min.ratio*lambda.max
      lambda = exp(seq(log(lambda.max), log(lambda.min), length = nlambda))
      rm(lambda.max,lambda.min,lambda.min.ratio)
      gc()
    }
    if (method=="group"){
      method.flag = 1 # glasso
      if (alg=="cyclic") 
        out = group.cyclic.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag, max.act.in, truncation)
      if (alg=="greedy") 
        out = group.greedy.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag)
      if (alg=="prox") 
        out = group.prox.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag)
      if (alg=="stoc") 
        out = group.stoc.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag, max.act.in, truncation)
    }
    if (method=="group.mcp"){
      method.flag = 2
      if (alg=="cyclic") 
        out = group.cyclic.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag, max.act.in, truncation)
      if (alg=="greedy") 
        out = group.greedy.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag)
      if (alg=="prox") 
        out = group.prox.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag)
      if (alg=="stoc") 
        out = group.stoc.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag, max.act.in, truncation)
    }
    if (method=="group.scad"){
      method.flag = 3
      if (alg=="cyclic") 
        out = group.cyclic.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag, max.act.in, truncation)
      if (alg=="greedy") 
        out = group.greedy.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag)
      if (alg=="prox") 
        out = group.prox.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag)
      if (alg=="stoc") 
        out = group.stoc.orth(yy, xx1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, max.ite, prec, verbose, Uinv.list, method.flag, max.act.in, truncation)
    }
  }
  runt=Sys.time()-begt
  
  df=rep(0,nlambda)
  for(i in 1:nlambda)
    df[i] = sum(out$beta[[i]]!=0)
  
  est = list()
  intcpt=matrix(0,nrow=1,ncol=nlambda)
  beta1=matrix(0,nrow=d,ncol=nlambda)
  if(design.sd){
    for(k in 1:nlambda){
      tmp.beta = out$beta[[k]]
      beta1[,k]=xinvc.vec*tmp.beta*sdy
      intcpt[k] = ym-as.numeric(xm[1,]%*%beta1[,k])+out$intcpt[k]*sdy
    }
  }else{
    for(k in 1:nlambda){
      beta1[,k]=out$beta[[k]]
      intcpt[k] = out$intcpt[k]
    }
  }
  
  est$obj = out$obj
  est$runt = out$runt
  est$gr.d = gr.d
  est$gr.size = gr.size
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
  class(est) = "lasso"
  return(est)
}

print.lasso <- function(x, ...)
{  
  cat("\n Lasso options summary: \n")
  cat(x$nlambda, " lambdas used:\n")
  print(signif(x$lambda,digits=3))
  cat("Method =", x$method, "\n")
  cat("Alg =", x$alg, "\n")
  cat("Degree of freedom:",min(x$df),"----->",max(x$df),"\n")
  if(units.difftime(x$runtime)=="secs") unit="secs"
  if(units.difftime(x$runtime)=="mins") unit="mins"
  if(units.difftime(x$runtime)=="hours") unit="hours"
  cat("Runtime:",x$runtime," ",unit,"\n")
}

plot.lasso <- function(x, ...)
{
  matplot(x$lambda, t(x$beta), type="l", main="Regularization Path",
          xlab="Regularization Parameter", ylab="Coefficient")
}

coef.lasso <- function(object, lambda.idx = c(1:3), beta.idx = c(1:3), ...)
{
  lambda.n = length(lambda.idx)
  beta.n = length(beta.idx)
  cat("\n Values of estimated coefficients: \n")
  cat(" index     ")
  for(i in 1:lambda.n){
    cat("",formatC(lambda.idx[i],digits=5,width=10),"")
  }
  cat("\n")
  cat(" lambda    ")
  for(i in 1:lambda.n){
    cat("",formatC(object$lambda[lambda.idx[i]],digits=4,width=10),"")
  }
  cat("\n")
  cat(" intercept ")
  for(i in 1:lambda.n){
    cat("",formatC(object$intercept[i],digits=4,width=10),"")
  }
  cat("\n")
  for(i in 1:beta.n){
    cat(" beta",formatC(beta.idx[i],digits=5,width=-5))
    for(j in 1:lambda.n){
      cat("",formatC(object$beta[beta.idx[i],lambda.idx[j]],digits=4,width=10),"")
    }
    cat("\n")
  }
}

predict.lasso <- function(object, newdata, lambda.idx = c(1:3), Y.pred.idx = c(1:5), ...)
{
  pred.n = nrow(newdata)
  lambda.n = length(lambda.idx)
  Y.pred.n = length(Y.pred.idx)
  intcpt = matrix(rep(object$intercept[,lambda.idx],pred.n),nrow=pred.n,
                  ncol=lambda.n,byrow=T)
  Y.pred = newdata%*%object$beta[,lambda.idx] + intcpt
  cat("\n Values of predicted responses: \n")
  cat("   index   ")
  for(i in 1:lambda.n){
    cat("",formatC(lambda.idx[i],digits=5,width=10),"")
  }
  cat("\n")
  cat("   lambda  ")
  for(i in 1:lambda.n){
    cat("",formatC(object$lambda[lambda.idx[i]],digits=4,width=10),"")
  }
  cat("\n")
  for(i in 1:Y.pred.n){
    cat("    Y",formatC(Y.pred.idx[i],digits=5,width=-5))
    for(j in 1:lambda.n){
      cat("",formatC(Y.pred[Y.pred.idx[i],j],digits=4,width=10),"")
    }
    cat("\n")
  }
  return(Y.pred)
}