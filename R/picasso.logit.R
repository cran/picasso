picasso.logit <- function(X, 
                          Y, 
                          lambda = NULL,
                          nlambda = NULL,
                          lambda.min.ratio = NULL,
                          method="l1",
                          gamma = 3,
                          standardize = TRUE,
                          intercept = TRUE,
                          prec = 1e-4,
                          max.ite = 1e4,
                          verbose = FALSE)
{
  dims = .picasso_validate_design(X)
  n = dims$n
  d = dims$d
  Y = as.factor(Y)
  if (length(levels(Y)) != 2){
    stop(sprintf(
      "Response vector must contain exactly 2 levels; found %d.",
      length(levels(Y))
    ))
  }
  Yb = rep(0, n)
  Yb[which(Y == levels(Y)[2])] = 1

  begt = Sys.time()

  if (verbose)
    cat("Sparse logistic regression. \n")

  design = .picasso_prepare_design(X, standardize)
  xx = design$xx
  xm = design$xm
  xinvc.vec = design$xinvc.vec

  yy = Yb
  
  lambda.max = max(abs(crossprod(xx,yy/n)))
  lambda.info = .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda = lambda.info$lambda
  nlambda = lambda.info$nlambda

  method.info = .picasso_method_flag(method, gamma)
  method.flag = method.info$flag
  gamma = method.info$gamma
  
  out = logit_solver(yy, xx, lambda, nlambda, gamma, 
              n, d, max.ite, prec, intercept, verbose, 
              method.flag)
  
  df = vapply(out$beta, function(beta.k) sum(beta.k != 0), FUN.VALUE = integer(1))
  
  est = list()
  beta.raw = do.call(cbind, out$beta)
  scaled = .picasso_rescale_solution(beta.raw, out$intcpt, standardize, xinvc.vec, xm)

  runt = Sys.time()-begt
  est$runt = out$runt
  est$beta = Matrix(scaled$beta)
  res = X %*% scaled$beta + matrix(rep(scaled$intercept, n), nrow = n, byrow = TRUE)
  est$p = exp(res)/(1+exp(res))
  est$intercept = scaled$intercept
  est$lambda = lambda
  est$nlambda = nlambda
  est$df = df
  est$method = method
  est$alg = "actnewton"
 
  est$ite =out$ite
  est$verbose = verbose
  est$runtime = runt
  class(est) = "logit"
  return(est)
}

print.logit <- function(x, ...)
{  
  .picasso_print_summary(x, " Logit options summary: ", method_label = "Method", show_alg = TRUE)
}

plot.logit <- function(x, ...)
{
  .picasso_plot_path(x)
}

coef.logit <- function(object, lambda.idx = c(1:3), beta.idx = c(1:3), ...)
{
  .picasso_extract_coef(object, lambda.idx, beta.idx)
}

predict.logit <- function(object, newdata, lambda.idx = c(1:3), p.pred.idx = c(1:5), ...)
{
  .picasso_predict(
    object,
    newdata,
    lambda.idx,
    p.pred.idx,
    default_response_idx = c(1:5),
    transform = function(z) exp(z) / (1 + exp(z))
  )
}
