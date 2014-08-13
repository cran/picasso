#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# mvr.prox(): Multivariate regression with proximal actic set identification       #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 4th, 2014                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

mvr.prox <- function(Y, X, lambda, nlambda, n, d, m, max.ite, prec, verbose)
{
  if(verbose==TRUE)
    cat("L12 regularization via proximal gradient actic set identification and coordinate descent.\n")
  Sigma = crossprod(X)
  L = eigen(Sigma, only.values=TRUE)$values[1]
  S = Sigma[c(0:(d-1))*(d+1)+1]
  beta = array(0,dim=c(nlambda,d,m))
  beta.intcpt = matrix(0,m,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  str=.C("picasso_mvr_prox", as.double(Y), as.double(X), as.double(S), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(m), as.integer(ite.lamb), as.integer(ite.cyc), 
         as.integer(size.act), as.double(lambda), as.integer(nlambda), 
         as.integer(max.ite), as.double(prec), as.double(L), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  beta.intcpt = vector("list", nlambda)
  dm = d*m
  for(i in 1:nlambda){
    beta.i = matrix(unlist(str[4])[((i-1)*dm+1):(i*dm)],nrow=d,ncol=m,byrow = FALSE)
    beta.list[[i]] = beta.i
    beta.intcpt[[i]] = matrix(unlist(str[5])[((i-1)*m+1):(i*m)],nrow=1,ncol=m,byrow=TRUE)
  }
  ite.lamb = unlist(str[9])
  ite.cyc = unlist(str[10])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[11])
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act))
}