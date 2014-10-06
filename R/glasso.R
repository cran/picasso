#----------------------------------------------------------------------------------#
# Package: picasso                                                                 #
# glasso(): Group regularization                                                   #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 22nd, 2014                                                             #
# Version: 0.2.0                                                                   #
#----------------------------------------------------------------------------------#

group.cyclic.orth <- function(Y, X1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, 
                              max.ite, prec, verbose, Uinv.list, method.flag, max.act.gr.in, truncation)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("Gropu L1 regularization via cyclic active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Group MCP regularization via cyclic active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("Group SCAD regularization via cyclic active set identification and coordinate descent\n")
  }
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  gr.size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = matrix(0,1,nlambda)
  obj = matrix(0,1,nlambda)
  gr1 = rep(0, gr.n)
  for(i in 1:gr.n){
    gr1[i] = gr[[i]][1]-1
  }
  str=.C("picasso_group_cyclic_orth", as.double(Y), as.double(X1), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(gr.size.act),
         as.double(obj), as.double(runt), as.integer(gr1),as.integer(gr.n),
         as.integer(gr.size), as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), 
         as.integer(method.flag), as.integer(max.act.gr.in), as.double(truncation), 
         PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    for(j in 1:gr.n){
      beta.i[gr[[j]]] = Uinv.list[[j]]%*%beta.i[gr[[j]]]
    }
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  gr.size.act = unlist(str[9])
  obj = matrix(unlist(str[10]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = gr.size.act,
         obj = obj, runt = runt))
}

group.greedy.orth <- function(Y, X1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, 
                              max.ite, prec, verbose, Uinv.list, method.flag)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("Gropu L1 regularization via greedy active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Group MCP regularization via greedy active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("Group SCAD regularization via greedy active set identification and coordinate descent\n")
  }
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  gr.size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = matrix(0,1,nlambda)
  obj = matrix(0,1,nlambda)
  gr1 = rep(0, gr.n)
  for(i in 1:gr.n){
    gr1[i] = gr[[i]][1]-1
  }
  str=.C("picasso_group_greedy_orth", as.double(Y), as.double(X1), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(gr.size.act),
         as.double(obj), as.double(runt), as.integer(gr1),as.integer(gr.n),
         as.integer(gr.size), as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), 
         as.integer(method.flag), PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    for(j in 1:gr.n){
      beta.i[gr[[j]]] = Uinv.list[[j]]%*%beta.i[gr[[j]]]
    }
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  gr.size.act = unlist(str[9])
  obj = matrix(unlist(str[10]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = gr.size.act,
         obj = obj, runt = runt))
}

group.prox.orth <- function(Y, X1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, 
                            max.ite, prec, verbose, Uinv.list, method.flag)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("Gropu L1 regularization via proximal gradient active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Group MCP regularization via proximal gradient active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("Group SCAD regularization via proximal gradient active set identification and coordinate descent\n")
  }
  #Sigma = crossprod(X1)
  #L = max(colSums(abs(Sigma)))
  #L = eigen(Sigma, only.values=TRUE)$values[1]
  L = d*n
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  gr.size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = matrix(0,1,nlambda)
  obj = matrix(0,1,nlambda)
  gr1 = rep(0, gr.n)
  for(i in 1:gr.n){
    gr1[i] = gr[[i]][1]-1
  }
  str=.C("picasso_group_prox_orth", as.double(Y), as.double(X1), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(gr.size.act),
         as.double(obj), as.double(runt), as.integer(gr1),as.integer(gr.n),
         as.integer(gr.size), as.double(lambda), as.integer(nlambda), as.double(gamma), 
         as.integer(max.ite), as.double(prec), as.double(L), as.integer(method.flag), 
         PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    for(j in 1:gr.n){
      beta.i[gr[[j]]] = Uinv.list[[j]]%*%beta.i[gr[[j]]]
    }
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  gr.size.act = unlist(str[9])
  obj = matrix(unlist(str[10]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = gr.size.act,
         obj = obj, runt = runt))
}

group.stoc.orth <- function(Y, X1, gr, gr.n, gr.size, lambda, nlambda, gamma, n, d, 
                            max.ite, prec, verbose, Uinv.list, method.flag, max.act.gr.in, truncation)
{
  if(verbose==TRUE){
    if(method.flag==1)
      cat("Gropu L1 regularization via stochastic active set identification and coordinate descent\n")
    if(method.flag==2)
      cat("Group MCP regularization via stochastic active set identification and coordinate descent\n")
    if(method.flag==3)
      cat("Group SCAD regularization via stochastic active set identification and coordinate descent\n")
  }
  beta = matrix(0,nrow=d,ncol=nlambda)
  beta.intcpt = rep(0,nlambda)
  gr.size.act = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.cyc = rep(0,nlambda)
  runt = matrix(0,1,nlambda)
  obj = matrix(0,1,nlambda)
  gr1 = rep(0, gr.n)
  for(i in 1:gr.n){
    gr1[i] = gr[[i]][1]-1
  }
  str=.C("picasso_group_stoc_orth", as.double(Y), as.double(X1), 
         as.double(beta), as.double(beta.intcpt), as.integer(n), as.integer(d), 
         as.integer(ite.lamb), as.integer(ite.cyc), as.integer(gr.size.act),
         as.double(obj), as.double(runt), as.integer(gr1),as.integer(gr.n),
         as.integer(gr.size), as.double(lambda), as.integer(nlambda), 
         as.double(gamma), as.integer(max.ite), as.double(prec), 
         as.integer(method.flag), as.integer(max.act.gr.in), as.double(truncation), 
         PACKAGE="picasso")
  beta.list = vector("list", nlambda)
  for(i in 1:nlambda){
    beta.i = unlist(str[3])[((i-1)*d+1):(i*d)]
    for(j in 1:gr.n){
      beta.i[gr[[j]]] = Uinv.list[[j]]%*%beta.i[gr[[j]]]
    }
    beta.list[[i]] = beta.i
  }
  beta.intcpt = unlist(str[4])
  ite.lamb = unlist(str[7])
  ite.cyc = unlist(str[8])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  gr.size.act = unlist(str[9])
  obj = matrix(unlist(str[10]),ncol=nlambda,byrow = FALSE)
  runt = matrix(unlist(str[11]),ncol=nlambda,byrow = FALSE)
  return(list(beta=beta.list, intcpt = beta.intcpt, ite=ite, size.act = gr.size.act,
         obj = obj, runt = runt))
}
