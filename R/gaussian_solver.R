gaussian_solver <- function(Y, X, lambda, nlambda, gamma, n, d, max.ite, prec,
                    verbose, intercept, method.flag, type.gaussian)
{
  if (verbose){
    if (method.flag == 1)
      cat("L1 regularization via active set identification and coordinate descent\n")
    if (method.flag == 2)
      cat("MCP regularization via active set identification and coordinate descent\n")
    if (method.flag == 3)
      cat("SCAD regularization via active set identification and coordinate descent\n")
  }

  # Kept for API compatibility with historical covariance mode.
  # The current backend uses the same implementation for both values.
  solver_name <- if (type.gaussian == "covariance") "picasso_gaussian_cov" else "picasso_gaussian_naive"
  str <- .C(
    solver_name,
    as.double(Y), as.double(X),
    as.integer(n), as.integer(d),
    as.double(lambda), as.integer(nlambda),
    as.double(gamma), as.integer(max.ite), as.double(prec),
    as.integer(method.flag), as.integer(intercept),
    as.double(rep(0, d * nlambda)), as.double(rep(0, nlambda)),
    as.integer(rep(0, nlambda)), as.integer(rep(0, d * nlambda)),
    as.double(rep(0, nlambda)),
    PACKAGE = "picasso"
  )

  runt <- matrix(unlist(str[16]), ncol = nlambda, byrow = FALSE)

  return(list(
    beta = unlist(str[12]),
    intcpt = unlist(str[13]),
    beta.idx = unlist(str[15]),
    ite = unlist(str[14]),
    runt = runt,
    err = 0  # TODO: add detailed error propagation from C++ layer.
  ))
}
