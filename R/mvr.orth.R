#----------------------------------------------------------------------------------#
# Package: picasso                                                                 #
# mvr.orth(): Multivariate regression                                              #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 22nd, 2014                                                             #
# Version: 0.2.0                                                                   #
#----------------------------------------------------------------------------------#

mvr.cyclic.orth <- function(Y, X1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, 
                            xinvc.vec, Uinv.vec, Uinv, method.flag, max.act.in, truncation)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("Multivariate regression with L1 regularization via cyclic active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Multivariate regression with MCP regularization via cyclic active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Multivariate regression with SCAD regularization via cyclic active set identification and coordinate descent\n")
  }
  beta = array(0,dim=c(nlambda,d,m))
  beta.intcpt = matrix(0,m,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  obj = matrix(0,1,nlambda)
  runt = matrix(0,1,nlambda)
  str=.C("picasso_mvr_cyclic_orth", as.double(Y), as.double(X1), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(m), as.integer(ite.lamb), as.integer(ite.cyc), as.integer(size.act),
         as.double(obj), as.double(runt), as.double(xinvc.vec), as.double(lambda), 
         as.integer(nlambda), as.double(gamma), as.integer(max.ite), as.double(prec), 
         as.double(Uinv.vec), as.integer(method.flag), as.integer(max.act.in), 
         as.double(truncation), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  beta.intcpt = vector("list", nlambda)
  dm = d*m
  for(i in 1:nlambda){
    beta.i = matrix(unlist(str[3])[((i-1)*dm+1):(i*dm)],nrow=d,ncol=m,byrow = FALSE)
    beta.i = Uinv%*%beta.i
    beta.list[[i]] = beta.i
    beta.intcpt[[i]] = matrix(unlist(str[4])[((i-1)*m+1):(i*m)],nrow=1,ncol=m,byrow=TRUE)
  }
  ite.lamb = unlist(str[8])
  ite.cyc = unlist(str[9])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[10])
  obj = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[12]),ncol=nlambda,byrow = FALSE)
  
  return(list(beta = beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act,
              obj = obj, runt = runt))
}

mvr.greedy.orth <- function(Y, X1, lambda, nlambda, gamma, n, d, m, max.ite, prec, 
                                verbose, xinvc.vec, Uinv.vec, Uinv,method.flag)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("Multivariate regression with L1 regularization via greedy active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Multivariate regression with MCP regularization via greedy active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Multivariate regression with SCAD regularization via greedy active set identification and coordinate descent\n")
  }
  beta = array(0,dim=c(nlambda,d,m))
  beta.intcpt = matrix(0,m,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  obj = matrix(0,1,nlambda)
  runt = matrix(0,1,nlambda)
  str=.C("picasso_mvr_greedy_orth", as.double(Y), as.double(X1), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(m), as.integer(ite.lamb), as.integer(ite.cyc), 
         as.integer(size.act), as.double(obj), as.double(runt), as.double(xinvc.vec), 
         as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), 
         as.double(Uinv.vec), as.integer(method.flag), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  beta.intcpt = vector("list", nlambda)
  dm = d*m
  for(i in 1:nlambda){
    beta.i = matrix(unlist(str[3])[((i-1)*dm+1):(i*dm)],nrow=d,ncol=m,byrow = FALSE)
    beta.i = Uinv%*%beta.i
    beta.list[[i]] = beta.i
    beta.intcpt[[i]] = matrix(unlist(str[4])[((i-1)*m+1):(i*m)],nrow=1,ncol=m,byrow=TRUE)
  }
  ite.lamb = unlist(str[8])
  ite.cyc = unlist(str[9])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[10])
  obj = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[12]),ncol=nlambda,byrow = FALSE)
  
  return(list(beta = beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act,
              obj = obj, runt = runt))
}

mvr.prox.orth <- function(Y, X1, lambda, nlambda, gamma, n, d, m, max.ite, prec, 
                              verbose, xinvc.vec, Uinv.vec, Uinv, method.flag)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("Multivariate regression with L1 regularization via proximal gradient active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Multivariate regression with MCP regularization via proximal gradient active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Multivariate regression with SCAD regularization via proximal gradient active set identification and coordinate descent\n")
  }
  #Sigma = crossprod(X1)
  #L = max(colSums(abs(Sigma)))
  #L = eigen(Sigma, only.values=TRUE)$values[1]
  #S = Sigma[c(0:(d-1))*(d+1)+1]
  L = d*n
  S = colSums(X1^2)
  beta = array(0,dim=c(nlambda,d,m))
  beta.intcpt = matrix(0,m,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  obj = matrix(0,1,nlambda)
  runt = matrix(0,1,nlambda)
  str=.C("picasso_mvr_prox_orth", as.double(Y), as.double(X1), as.double(S), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(m), as.integer(ite.lamb), as.integer(ite.cyc), 
         as.integer(size.act), as.double(obj), as.double(runt), as.double(xinvc.vec), 
         as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), as.double(L), 
         as.double(Uinv.vec), as.integer(method.flag), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  beta.intcpt = vector("list", nlambda)
  dm = d*m
  for(i in 1:nlambda){
    beta.i = matrix(unlist(str[4])[((i-1)*dm+1):(i*dm)],nrow=d,ncol=m,byrow = FALSE)
    beta.i = Uinv%*%beta.i
    beta.list[[i]] = beta.i
    beta.intcpt[[i]] = matrix(unlist(str[5])[((i-1)*m+1):(i*m)],nrow=1,ncol=m,byrow=TRUE)
  }
  ite.lamb = unlist(str[9])
  ite.cyc = unlist(str[10])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[11])
  obj = matrix(unlist(str[12]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[13]),ncol=nlambda,byrow = FALSE)
  
  return(list(beta = beta.list, intcpt = beta.intcpt, ite = ite, size.act = size.act,
              obj = obj, runt = runt))
}

mvr.stoc.orth <- function(Y, X1, lambda, nlambda, gamma, n, d, m, max.ite, prec, verbose, 
                          xinvc.vec, Uinv.vec, Uinv, method.flag, max.act.in, truncation)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("Multivariate regression with L1 regularization via stochastic active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Multivariate regression with MCP regularization via stochastic active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Multivariate regression with SCAD regularization via stochastic active set identification and coordinate descent\n")
  }
  beta = array(0,dim=c(nlambda,d,m))
  beta.intcpt = matrix(0,m,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  obj = matrix(0,1,nlambda)
  runt = matrix(0,1,nlambda)
  str=.C("picasso_mvr_stoc_orth", as.double(Y), as.double(X1), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(m), as.integer(ite.lamb), as.integer(ite.cyc), as.integer(size.act),
         as.double(obj), as.double(runt), as.double(xinvc.vec), as.double(lambda), 
         as.integer(nlambda), as.double(gamma), as.integer(max.ite), as.double(prec), 
         as.double(Uinv.vec), as.integer(method.flag), as.integer(max.act.in), 
         as.double(truncation), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  beta.intcpt = vector("list", nlambda)
  dm = d*m
  for(i in 1:nlambda){
    beta.i = matrix(unlist(str[3])[((i-1)*dm+1):(i*dm)],nrow=d,ncol=m,byrow = FALSE)
    beta.i = Uinv%*%beta.i
    beta.list[[i]] = beta.i
    beta.intcpt[[i]] = matrix(unlist(str[4])[((i-1)*m+1):(i*m)],nrow=1,ncol=m,byrow=TRUE)
  }
  ite.lamb = unlist(str[8])
  ite.cyc = unlist(str[9])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[10])
  obj = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[12]),ncol=nlambda,byrow = FALSE)
  
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act,
              obj = obj, runt = runt))
}
