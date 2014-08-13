#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# logit.gr.greedy(): Greedy actic set identification                               #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 5th, 2014                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

logit.gr.greedy <- function(Y, X, gr, gr.n, gr.size, lambda, nlambda, n, d, max.ite, prec,verbose)
{
  if(verbose==TRUE)
    cat("Group sparse logistic regression via greedy actic set identification \n")
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  gr.size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  gr1 = rep(0, gr.n)
  for(i in 1:gr.n){
    gr1[i] = gr[[i]][1]-1
  }
  str=.C("picasso_logit_gr_greedy", as.double(Y), as.double(X), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(gr.size.act),
         as.integer(gr1), as.integer(gr.n), as.integer(gr.size), 
         as.double(lambda), as.integer(nlambda), as.integer(max.ite), 
         as.double(prec), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  gr.size.act = unlist(str[9])
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = gr.size.act))
}
