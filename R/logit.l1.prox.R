#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# logit.l1.prox(): Proximal gradient actic set identification                      #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 6th, 2014                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

logit.l1.prox <- function(Y, X, lambda, nlambda, n, d, max.ite, prec,verbose)
{
  if(verbose==TRUE)
    cat("L1 regularization via proximal gradient actic set identification and coordinate descent\n")
  L = eigen(crossprod(X)/n, only.values=TRUE)$values[1]
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  str=.C("picasso_logit_l1_prox", as.double(Y), as.double(X), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(size.act), 
         as.double(lambda), as.integer(nlambda), as.integer(max.ite), 
         as.double(prec), as.double(L), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[5])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  size.act = unlist(str[9])
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act))
}
