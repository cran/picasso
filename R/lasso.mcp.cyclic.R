#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# lasso.mcp.cyclic(): Cyclic actic set identification                              #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Jul 27th, 2014                                                             #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

lasso.mcp.cyclic <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,verbose)
{
  if(verbose==TRUE)
    cat("Lasso MCP via Cyclic Coordinate Descent\n")
  S = colSums(X^2)
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  str=.C("picasso_lasso_mcp_cyclic", as.double(Y), as.double(X), as.double(S), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(size.act),
         as.double(lambda), as.integer(nlambda), as.double(gamma), 
         as.integer(max.ite), as.double(prec), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[4])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[5])
  ite.int = unlist(str[8])
  ite = list()
  ite[[1]] = ite.int
  size.act = unlist(str[9])
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = size.act))
}
