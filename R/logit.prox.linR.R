#----------------------------------------------------------------------------------#
# Package: picasso                                                                  #
# logit.prox.lin(): Proximal Gradient actic set identification                     #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Jul 24th, 2014                                                             #
# Version: 0.1.0                                                                   #
#----------------------------------------------------------------------------------#

logit.prox.linR <- function(Y, X, lambda, nlambda, n, d, max.ite, prec,verbose)
{
  if(verbose==TRUE)
    cat("Sparse Logistic Regression via Cyclic Coordinate Descent\n")
  d1 = d+1
  d2 = d-1
  alp = 0.8
  beta0 = matrix(0,nrow=d,ncol=nlambda)
  beta1 = beta0
  beta.intcpt = rep(0,nlambda)
  ite.lamb = rep(0,nlambda)
  ite.init = rep(0,nlambda)
  ite.int = rep(0,nlambda)
  max.ite1 = max.ite
  max.ite2 = max.ite
  max.ite3 = max.ite
  prec1 = prec
  prec2 = prec
  size.a = 0
  set.act = NULL
  set.ina = c(1:d)
  beta.intcpt[1] = mean(Y)
  y.hat = Y-beta.intcpt[1]
  p = rep(0,n)
  w = 1/4
  S = crossprod(X)/n
  L0 = eigen(S,only.values=TRUE)$values[1]*w
  z = rep(0,n)
  for(i in 1:nlambda){
    ilambda = lambda[i]
    if(i>1){
      beta1[set.act,i] = beta1[set.act,i-1]
    }
    L = L0
    if(size.a>0){
      beta.intcpt[i] = beta.intcpt[i-1]-sum(p-Y)/n/w # update beta_0
    }
    obj.base = sum(log(1+exp(beta.intcpt[i]+X[,set.act]%*%as.matrix(beta1[set.act,i],ncol=1)))-
                Y*(beta.intcpt[i]+X[,set.act]%*%as.matrix(beta1[set.act,i],ncol=1)))/n
    for(t in 1:n){
      p[t] = 1/(1+exp(-beta.intcpt[i]-X[t,]%*%beta1[,i]))
    }
    g = crossprod(X,p-Y)/n
    ite1 = 0
    beta.tmp = beta1[,i]
    track = 1
    while(track==1 && L>1e-1){
      beta.tild = beta1[,i] - g/L # get estimation
      ilamb = ilambda/L
      beta.tmp = matrix(0,nrow=d,ncol=1)
      for(j in 1:d){
        if(beta.tild[j]>ilamb){
          beta.tmp[j] = beta.tild[j]-ilamb
        }else{
          if(beta.tild[j]<(-ilamb)){
            beta.tmp[j] = beta.tild[j]+ilamb
          }else{
            beta.tmp[j] = 0
          }
        }
      }
      Q0 = obj.base+sum(g*(beta.tmp-beta1[,i]))+L*sum((beta.tmp-beta1[,i])^2)/2
      F0 = sum(log(1+exp(beta.intcpt[i]+X%*%beta.tmp))-
                 Y*(beta.intcpt[i]+X%*%beta.tmp))/n
      if(Q0>F0){
        L = L*alp
      }else{
        L = L/alp
        track = 0
      }
      ite1 = ite1+1
    }
    ite.init[i] = ite1
    beta.tild = beta1[,i] - g/L # get estimation
    ilamb = ilambda/L
    for(j in 1:d){
      if(beta.tild[j]>ilamb){
        beta1[j,i] = beta.tild[j]-ilamb
      }else{
        if(beta.tild[j]<(-ilamb)){
          beta1[j,i] = beta.tild[j]+ilamb
        }else{
          beta1[j,i] = 0
        }
      }
    }
    set.act = NULL
    size.a = 0
    for(j in 1:d){
      if(beta1[j,i]!=0){
        size.a = size.a+1
        set.act[size.a] = j
      }
    }
    
    dif = 1
    ite2 = 0
    while(dif>prec1 && ite2<max.ite1){
      beta.intcpt[i] = beta.intcpt[i]-sum(p-Y)/n/w # update beta_0
      for(t in 1:n){
        p[t] = 1/(1+exp(-beta.intcpt[i]-X[t,]%*%beta1[,i]))
      }
      g = crossprod(X,p-Y)/n
      if(L>L0){
        beta.tild = beta1[,i] - g/L # get estimation
        ilamb = ilambda/L
        for(j in 1:d){
          if(beta.tild[j]>ilamb){
            beta1[j,i] = beta.tild[j]-ilamb
          }else{
            if(beta.tild[j]<(-ilamb)){
              beta1[j,i] = beta.tild[j]+ilamb
            }else{
              beta1[j,i] = 0
            }
          }
        }
      }else{
        track = 1
        obj.base = sum(log(1+exp(beta.intcpt[i]+X[,set.act]%*%as.matrix(beta1[set.act,i],ncol=1)))-
                         Y*(beta.intcpt[i]+X[,set.act]%*%as.matrix(beta1[set.act,i],ncol=1)))/n
        ite2 = 0
        beta.tmp = beta1[,i]
        ite3 = 0
        while(track == 1 && L<1e3){
          beta.tild = beta1[,i] - g/L # get estimation
          ilamb = ilambda/L
          beta.tmp = matrix(0,nrow=d,ncol=1)
          for(j in 1:d){
            if(beta.tild[j]>ilamb){
              beta.tmp[j] = beta.tild[j]-ilamb
            }else{
              if(beta.tild[j]<(-ilamb)){
                beta.tmp[j] = beta.tild[j]+ilamb
              }else{
                beta.tmp[j] = 0
              }
            }
          }
          Q0 = obj.base+sum(g*(beta.tmp-beta1[,i]))+L*sum((beta.tmp-beta1[,i])^2)/2
          F0 = sum(log(1+exp(beta.intcpt[i]+X%*%beta.tmp))-
                     Y*(beta.intcpt[i]+X%*%beta.tmp))/n
          if(Q0>F0){
            track = 0
          }else{
            L = L/alp
          }
          ite3 = ite3+1
        }
        ite.int[i] = ite.int[i]+ite3
        beta.tild = beta1[,i] - g/L # get estimation
        ilamb = ilambda/L
        for(j in 1:d){
          if(beta.tild[j]>ilamb){
            beta1[j,i] = beta.tild[j]-ilamb
          }else{
            if(beta.tild[j]<(-ilamb)){
              beta1[j,i] = beta.tild[j]+ilamb
            }else{
              beta1[j,i] = 0
            }
          }
        }
        #cat("ite2=",ite2,",ite3=",ite3,",dif=",dif,",loss=",F0,",l1norm=",sum(abs(beta1[,i])),"\n")
      }
      ite2 = ite2+1
      set.act = NULL
      size.a = 0
      for(j in 1:d){
        if(beta1[j,i]!=0){
          size.a = size.a+1
          set.act[size.a] = j
        }
      }
      dif = norm(beta1[,i]-beta0[,i],"2") # stopping criterion
      beta0[,i] = beta1[,i]
    }
    ite.lamb[i] = ite2
    obj = sum(log(1+exp(beta.intcpt[i]+X[,set.act]%*%as.matrix(beta1[set.act,i],ncol=1)))-
                Y*(beta.intcpt[i]+X[,set.act]%*%as.matrix(beta1[set.act,i],ncol=1)))/n
    if(size.a>0)
      obj = obj+ilambda*sum(abs(beta1[set.act,i]))
    cat("ilambda=",i,", ite1=",ite.init[i],", ite2=",ite.lamb[i],", ite3=",ite.int[i],
        ",df=",sum(beta1[,i]!=0),",obj=",obj,"\n")
  }
}
