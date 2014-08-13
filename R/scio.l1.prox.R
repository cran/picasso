#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# scio.l1.prox(): Sparse Column Inverse Operator                                   #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 7th, 2014                                                              #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

scio.l1.prox <- function(S, lambda, nlambda, d, maxdf, prec, max.ite, verbose){
  
  if(verbose==TRUE)
    cat("L1 regularization via proximal gradient actic set identification and coordinate descent \n")
  d_sq = d^2
  L = eigen(S, only.values=TRUE)$values[1]
  nlambda = length(lambda)
  icov = array(0,dim=c(d,d,nlambda))
  ite.lamb = rep(0,d*nlambda)
  ite.cyc = rep(0,d*nlambda)
  obj = array(0,dim=c(max.ite,nlambda))
  runt = array(0,dim=c(max.ite,nlambda))
  x = array(0,dim=c(d,maxdf,nlambda))
  col_cnz = rep(0,d+1)
  row_idx = rep(0,d*maxdf*nlambda)
  begt=Sys.time()
  str=.C("picasso_scio_l1_prox", as.double(S), as.double(icov), 
         as.integer(d), as.integer(ite.lamb), as.integer(ite.cyc), 
         as.double(lambda), as.integer(nlambda), as.integer(max.ite), 
         as.double(prec), as.double(x), as.integer(col_cnz), 
         as.integer(row_idx), as.double(L), PACKAGE="picasso")
  runt1=Sys.time()-begt
  obj = 0
  icov_list = vector("list", nlambda)
  icov_list1 = vector("list", nlambda)
  for(i in 1:nlambda){
    icov_i = matrix(unlist(str[2])[((i-1)*d_sq+1):(i*d_sq)], byrow = FALSE, ncol = d)
    icov_list1[[i]] = icov_i
    icov_list[[i]] = icov_i*(abs(icov_i)<=abs(t(icov_i)))+t(icov_i)*(abs(t(icov_i))<abs(icov_i))
    obj[i] = sum(abs(icov_i))
  }
  ite.lamb = unlist(str[4])
  ite.cyc = unlist(str[5])
  ite = list()
  ite[[1]] = ite.lamb
  ite[[2]] = ite.cyc
  x = unlist(str[10])
  col_cnz = unlist(str[11])
  row_idx = unlist(str[12])
  return(list(icov=icov_list, icov1=icov_list1, ite=ite, obj=obj, runt=runt1,
              x=x, col_cnz=col_cnz, row_idx=row_idx))
}
