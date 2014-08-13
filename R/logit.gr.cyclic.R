#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# logit.gr.cyclic(): Cyclic actic set identification                               #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 1st, 2014                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

logit.gr.cyclic <- function(Y, X, gr, gr.n, gr.size, lambda, nlambda, n, d, max.ite, prec,verbose)
{
  if(verbose==TRUE)
    cat("Group sparse logistic regression via cyclic actic set identification \n")
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  gr.size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  gr1 = rep(0, gr.n)
  for(i in 1:gr.n){
    gr1[i] = gr[[i]][1]-1
  }
  str=.C("picasso_logit_gr_cyclic", as.double(Y), as.double(X), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(gr.size.act),as.integer(gr1),
         as.integer(gr.n), as.integer(gr.size), as.double(lambda), 
         as.integer(nlambda), as.integer(max.ite), as.double(prec), 
         PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.int = unlist(str[7])
  ite = list()
  ite[[1]] = ite.int
  gr.size.act = unlist(str[8])
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = gr.size.act))
}
