#----------------------------------------------------------------------------------#
# Package: picasso                                                                 #
# picasso(): The user interface for picasso()                                      #
# Author: Xingguo Li                                                               #
# Email: <xingguo.leo@gmail.com>                                                   #
# Date: Aug 17th, 2014                                                             #
# Version: 0.2.0                                                                   #
#----------------------------------------------------------------------------------#

picasso <- function(X, 
                    Y, 
                    lambda = NULL,
                    nlambda = NULL,
                    lambda.min.ratio = NULL,
                    family = "gaussian",
                    method = "l1",
                    alg = "cyclic",
                    gamma = 3,
                    sym = "or",
                    standardize = FALSE,
                    perturb = TRUE,
                    design.sd = TRUE,
                    res.sd = FALSE,
                    gr = NULL,
                    gr.d = NULL,
                    gr.size = NULL,
                    max.act.in = 3,
                    truncation = 0.01, 
                    prec = 1e-4,
                    max.ite = 1e4,
                    verbose = TRUE)
{
  if(family=="gaussian"){
    p = ncol(Y)
    if(p==1){
      out = picasso.lasso(X = X, Y = Y, lambda = lambda, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio,
                          method = method, alg = alg, gamma = gamma, design.sd = design.sd, res.sd = res.sd, gr = gr, 
                          gr.d = gr.d,gr.size = gr.size, max.act.in = max.act.in, truncation = truncation, prec = prec, 
                          max.ite = max.ite, verbose = verbose)
    }
    if(p>1){
      out = picasso.mvr(X = X, Y = Y, lambda = lambda, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio,
                        method = method, alg = alg, gamma = gamma, design.sd = design.sd, max.act.in = max.act.in, 
                        truncation = truncation, prec = prec, max.ite = max.ite, verbose = verbose)
    }
  }
  if(family=="logit"){
    out = picasso.logit(X = X, Y = Y, lambda = lambda, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio,
                        method = method, alg = alg, gamma = gamma, design.sd = design.sd, gr = gr, gr.d = gr.d,
                        gr.size = gr.size, max.act.in = max.act.in, truncation = truncation, prec = prec, 
                        max.ite = max.ite, verbose = verbose)
  }
  if(family=="npn"){
    out = picasso.scio(X = X, lambda = lambda, nlambda = nlambda, lambda.min.ratio = lambda.min.ratio,
                       method = method, alg = alg, gamma = gamma, sym = sym, truncation = truncation, prec = prec, 
                       max.ite = max.ite, standardize = standardize, perturb = perturb, verbose = verbose)
  }
  out$family = family
  return(out)
}
