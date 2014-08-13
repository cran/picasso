#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# picasso.logit(): The user interface for lasso()                                   #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Jul 26th, 2014                                                             #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

picasso.logit <- function(X, 
                          Y, 
                          lambda = NULL,
                          nlambda = NULL,
                          lambda.min.ratio = NULL,
                          method="l1",
                          alg = "cyclic",
                          gamma = 3,
                          gr = NULL,
                          gr.n = NULL,
                          gr.size = NULL,
                          prec = 1e-4,
                          max.ite = 1e4,
                          verbose = TRUE)
{
  n = nrow(X)
  d = ncol(X)
  if(verbose)
    cat("Sparse logistic regression. \n")
  if(n==0 || d==0) {
    cat("No data input.\n")
    return(NULL)
  }
  design.sd = TRUE
  if(design.sd==TRUE){
    maxdf = max(n,d)
    xm=matrix(rep(colMeans(X),n),nrow=n,ncol=d,byrow=T)
    x1=X-xm
    sdxinv=1/sqrt(colSums(x1^2)/(n-1))
    xx=x1*matrix(rep(sdxinv,n),nrow=n,ncol=d,byrow=T)
  }else{
    xx = X
  }
  yy = Y
  
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
  begt=Sys.time()
  if(method=="l1") {
    if (alg=="cyclic")
      out = logit.l1.cyclic(yy, xx, lambda, nlambda, n, d, max.ite, prec, verbose)
    if (alg=="greedy")
      out = logit.l1.greedy(yy, xx, lambda, nlambda, n, d, max.ite, prec, verbose)
    if (alg=="prox")
      out = logit.l1.prox(yy, xx, lambda, nlambda, n, d, max.ite, prec, verbose)
    if (alg=="stoc")
      out = logit.l1.stoc(yy, xx, lambda, nlambda, n, d, max.ite, prec, verbose)
  }
  if(method=="scad") {
    if (gamma<=2) {
      cat("\"gamma\">2 is required for SCAD. Set to default value 3. \n")
      gamma = 3
    }
    if (alg=="cyclic")
      out = logit.scad.cyclic(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose)
    if (alg=="greedy")
      out = logit.scad.greedy(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose)
    if (alg=="prox")
      out = logit.scad.prox(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose)
    if (alg=="stoc")
      out = logit.scad.stoc(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose)
  }
  if(method=="mcp") {
    if (gamma<=1) {
      cat("\"gamma\">1 is required for MCP. Set to default value 3. \n")
      gamma = 3
    }
    if (alg=="cyclic")
      out = logit.mcp.cyclic(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose)
    if (alg=="greedy")
      out = logit.mcp.greedy(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose)
    if (alg=="prox")
      out = logit.mcp.prox(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose)
    if (alg=="stoc")
      out = logit.mcp.stoc(yy, xx, lambda, nlambda, gamma, n, d, max.ite, prec, verbose)
  }
  if(method=="glasso") {
    if (is.null(gr)) {
      gr = list()
      if(is.null(gr.n)){
        if(is.null(gr.size)){
          igr.size = 2
          gr.n = ceiling(d/igr.size)
          gr.size = rep(igr.size,gr.n)
          if(sum(gr.size)>d) gr.size[gr.n] = gr.size[gr.n] - (sum(gr.size)-d)
        }else{
          if(sum(gr.size)!=d) {
            cat('Group size error... sum(gr.size) !=',d,'\n')
            return(NULL)
          }
          gr.n = length(gr.size)
        }
      }else{
        if(gr.n>d){
          cat('Group size error... gr.n >',d,'\n')
          return(NULL)
        }
        if(is.null(gr.size)){
          igr.size1 = ceiling(d/gr.n)
          igr.size2 = igr.size1-1
          gr.n1 = d-gr.n*igr.size2
          gr.n2 = gr.n-gr.n1
          gr.size = c(rep(igr.size1,gr.n1),rep(igr.size2,gr.n2))
        }else{
          if(sum(gr.size)!=d) {
            cat('Group size error... sum(gr.size) !=',d,'\n')
            return(NULL)
          }
          if(length(gr.size)!=gr.n) {
            cat('Group size does not match... length(gr.size)!=gr.n \n')
            return(NULL)
          }
        }
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
        if(gr.n>d){
          cat('Group size error... gr.n >',d,'\n')
          return(NULL)
        }
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
      }
    }
    if (alg=="cyclic") 
      out = logit.gr.cyclic(yy, xx, gr, gr.n, gr.size, lambda, nlambda, n, d, max.ite, prec, verbose)
    if (alg=="greedy") 
      out = logit.gr.greedy(yy, xx, gr, gr.n, gr.size, lambda, nlambda, n, d, max.ite, prec, verbose)
    if (alg=="prox") 
      out = logit.gr.prox(yy, xx, gr, gr.n, gr.size, lambda, nlambda, n, d, max.ite, prec, verbose)
    if (alg=="stoc") 
      out = logit.gr.stoc(yy, xx, gr, gr.n, gr.size, lambda, nlambda, n, d, max.ite, prec, verbose)
  }
  runt=Sys.time()-begt
  
  df=rep(0,nlambda)
  for(i in 1:nlambda)
    df[i] = sum(out$beta[[i]]!=0)
  
  est = list()
  intcpt=matrix(0,nrow=1,ncol=nlambda)
  beta1=matrix(0,nrow=d,ncol=nlambda)
  
  if(design.sd==TRUE){
    for(k in 1:nlambda){
      tmp.beta = out$beta[[k]]
      beta1[,k]=sdxinv*tmp.beta
      intcpt[k] = -as.numeric(xm[1,]%*%beta1[,k])+out$intcpt[k]
    }
  }else{
    for(k in 1:nlambda){
      beta1[,k]=out$beta[[k]]
      intcpt[k] = out$intcpt[k]
    }
  }
  
  est$gr = gr
  est$gr.n = gr.n
  est$gr.size = gr.size
  est$beta = beta1
  res = X%*%beta1+matrix(rep(intcpt,n),nrow=n,byrow=TRUE)
  est$p = exp(res)/(1+exp(res))
  est$intercept = intcpt
  est$Y = Y
  est$X = X
  est$lambda = lambda
  est$nlambda = nlambda
  est$df = df
  est$method = method
  est$alg = alg
  est$ite =out$ite
  est$verbose = verbose
  est$runtime = runt
  class(est) = "logit"
  return(est)
}

print.logit <- function(x, ...)
{  
  cat("\n Logit options summary: \n")
  cat(x$nlambda, " lambdas used:\n")
  print(signif(x$lambda,digits=3))
  cat("Method =", x$method, "\n")
  cat("Alg =", x$alg, "\n")
  cat("Sparsity level:",min(x$sparsity),"----->",max(x$sparsity),"\n")
  if(units.difftime(x$runtime)=="secs") unit="secs"
  if(units.difftime(x$runtime)=="mins") unit="mins"
  if(units.difftime(x$runtime)=="hours") unit="hours"
  cat("Runtime:",x$runtime," ",unit,"\n")
}

plot.logit <- function(x, ...)
{
  matplot(x$lambda, t(x$beta), type="l", main="Regularization Path",
          xlab="Regularization Parameter", ylab="Coefficient")
}

coef.logit <- function(object, lambda.idx = c(1:3), beta.idx = c(1:3), ...)
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

predict.logit <- function(object, newdata, lambda.idx = c(1:3), p.pred.idx = c(1:5), ...)
{
  pred.n = nrow(newdata)
  lambda.n = length(lambda.idx)
  p.pred.n = length(p.pred.idx)
  intcpt = matrix(rep(object$intercept[,lambda.idx],pred.n),nrow=pred.n,
                  ncol=lambda.n,byrow=T)
  res = newdata%*%object$beta[,lambda.idx] + intcpt
  p.pred = exp(res)/(1+exp(res))
  cat("\n Values of predicted Bernoulli parameter: \n")
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
  for(i in 1:p.pred.n){
    cat("    Y",formatC(p.pred.idx[i],digits=5,width=-5))
    for(j in 1:lambda.n){
      cat("",formatC(p.pred[p.pred.idx[i],j],digits=4,width=10),"")
    }
    cat("\n")
  }
  return(p.pred)
}
