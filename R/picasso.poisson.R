picasso.poisson <- function(X, 
                          Y, 
                          lambda = NULL,
                          nlambda = NULL,
                          lambda.min.ratio = NULL,
                          method = "l1",
                          gamma = 3,
                          standardize = TRUE,
                          intercept = FALSE,
                          prec = 1e-4,
                          max.ite = 1e4,
                          verbose = FALSE)
{
  dims = .picasso_validate_design(X)
  n = dims$n
  d = dims$d
  if (!isTRUE(all(Y == floor(Y))) || !isTRUE(all(Y >= 0))) 
    stop("The response must only contain non-negative integer values for poisson regression")

  if (sum(Y) <= 0)
    stop("The response vector is an all-zero vector. The problem is ill-conditioned.")

  begt = Sys.time()
  if (verbose)
    cat("Sparse poisson regression. \n")

  design = .picasso_prepare_design(X, standardize)
  xx = design$xx
  xm = design$xm
  xinvc.vec = design$xinvc.vec

  yy = Y
  avr_y = mean(yy)
  
  lambda.max = max(abs(crossprod(xx,(yy-avr_y)/n)))
  lambda.info = .picasso_lambda_path(lambda, nlambda, lambda.min.ratio, lambda.max)
  lambda = lambda.info$lambda
  nlambda = lambda.info$nlambda

  method.info = .picasso_method_flag(method, gamma)
  method.flag = method.info$flag
  gamma = method.info$gamma
  
  out = poisson_solver(yy, xx, lambda, nlambda, gamma, 
              n, d, max.ite, prec, intercept = intercept, verbose, 
              method.flag)
  
  df = vapply(out$beta, function(beta.k) sum(beta.k != 0), FUN.VALUE = integer(1))
  
  est = list()
  beta.raw = do.call(cbind, out$beta)
  scaled = .picasso_rescale_solution(beta.raw, out$intcpt, standardize, xinvc.vec, xm)
  runt = Sys.time()-begt
  est$runt = out$runt
  est$beta = Matrix(scaled$beta)
  est$intercept = scaled$intercept
  est$lambda = lambda
  est$nlambda = nlambda
  est$df = df
  est$method = method
 
  est$ite =out$ite
  est$verbose = verbose
  est$runtime = runt
  class(est) = "poisson"
  return(est)
}

print.poisson <- function(x, ...)
{  
  .picasso_print_summary(x, " Poisson regression options summary: ", method_label = "Regularization")
}

plot.poisson <- function(x, ...)
{
  .picasso_plot_path(x)
}

coef.poisson <- function(object, lambda.idx = c(1:3), beta.idx = c(1:3), ...)
{
  .picasso_extract_coef(object, lambda.idx, beta.idx)
}

predict.poisson <- function(object, newdata, lambda.idx = c(1:3), p.pred.idx = c(1:5), ...)
{
  .picasso_predict(
    object,
    newdata,
    lambda.idx,
    p.pred.idx,
    default_response_idx = c(1:5),
    transform = exp
  )
}
