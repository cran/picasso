\name{picasso}
\alias{picasso}

\title{
Pathwise Calibrated Sparse Shooting Algorithm
}

\description{
The function "picasso" implements The user interface.
}

\usage{
picasso(X, Y, lambda = NULL, nlambda = NULL, lambda.min.ratio = NULL,
        family = "gaussian", method="l1", alg = "cyclic", gamma = 3, 
        sym = "or", standardize = FALSE, perturb = TRUE, design.sd = TRUE,
        res.sd = FALSE, gr = NULL, gr.d = NULL, gr.size = NULL, max.act.in = 3,
        truncation = 0.01, prec = 1e-4, max.ite = 1e4, verbose = TRUE)
}

\arguments{
  \item{X}{For sparse linear regression and sparse logistic regression, \code{X} is an \eqn{n} by \eqn{d} design matrix. For sparse multivariate regression, \code{X} is an \eqn{n} by \eqn{d} design matrix. For sparse inverse column operation, there are 2 options when \code{family = "npn"}: (1) \code{data} is an \code{n} by \code{d} data matrix (2) a \code{d} by \code{d} sample covariance matrix. The program automatically identifies the input matrix by checking the symmetry. (\code{n} is the sample size and \code{d} is the dimension). }
  \item{Y}{For sparse linear regression and sparse logistic regression, \code{Y} is an \eqn{n} dimensional response vector. For sparse multivariate regression, \code{Y} is an \eqn{n} by \eqn{m} response matrix. For sparse inverse column operation, no input for \code{Y} is needed.}
  \item{lambda}{A sequence of decresing positive value to control the regularization. Typical usage is to leave the input \code{lambda = NULL} and have the program compute its own \code{lambda} sequence based on \code{nlambda} and \code{lambda.min.ratio}. Users can also specify a sequence to override this. Default value is from \eqn{lambda.max} to \code{lambda.min.ratio*lambda.max}. For Lq regression, the default value of \eqn{lambda.max} is \eqn{\pi\sqrt{\log(d)/n}}. The default value of \eqn{lambda.max} is the minimum regularization parameter which yields an all-zero estiamtes.}
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 5.}
  \item{lambda.min.ratio}{The smallest value for \code{lambda}, as a fraction of the uppperbound (\code{MAX}) of the regularization parameter. The program can automatically generate \code{lambda} as a sequence of length = \code{nlambda} starting from \code{MAX} to \code{lambda.min.ratio*MAX} in log scale. The default value is \code{0.25}.}
  \item{family}{Options for model. Sparse linear regression and sparse multivariate regression is applied if \code{family = "gaussian"}, sparse logistic regression is applied if \code{family = "logit"} and sparse column inverse operation is applied if \code{family = "npn"}. The default value is \code{"gaussian"}.}
  \item{method}{Options for regularization. Lasso is applied if \code{method = "l1"}, MCP is applied if \code{method = "mcp"} and SCAD Lasso is applied if \code{method = "scad"}. For sparse linear regression and sparse logistic regression, group lasso is applied if \code{method = "group"}, group MCP is applied if \code{method = "group.mcp"} and group SCAD is applied if \code{method = "group.scad"}. The default value is \code{"l1"}.}
  \item{alg}{Options for active set identification. Cyclic search is applied if \code{alg = "cyclic"}, greedy search is applied if \code{alg = "greedy"}, proximal gradient based search is applied if \code{alg = "prox"} and stochastic search is applied if \code{alg = "stoc"}. The default value is \code{"cyclic"}.}
  \item{gamma}{Parameter for MCP and SCAD. The default value is 3.}
  \item{sym}{Symmetrization of output graphs. If \code{sym = "and"}, the edge between node \code{i} and node \code{j} is selected ONLY when both node \code{i} and node \code{j} are selected as neighbors for each other. If \code{sym = "or"}, the edge is selected when either node \code{i} or node \code{j} is selected as the neighbor for each other. The default value is \code{"or"}.}
  \item{standardize}{Variables are standardized to have mean zero and unit standard deviation if \code{standardize = TRUE}. The default value is \code{FALSE}.}
  \item{perturb}{The diagonal of \code{Sigma} is added by a positive value to guarantee that \code{Sigma} is positive definite if \code{perturb = TRUE}. User can specify a numeric value for \code{perturbe}. The default value is \code{perturb = TRUE}.}
  \item{design.sd}{Flag of whether the design matrix is standardized. The default value is \code{TRUE}.}
  \item{res.sd}{Flag of whether the response varialbles are standardized. The default value is \code{FALSE}.}
  \item{gr}{A list which defines the grouping of the variables. The default setting is grouping every two variables if \code{method = "glasso"}.}
  \item{gr.d}{The number of variables in each group. The default value is \code{gr.d = "2"}.}
  \item{gr.size}{A vector of the number of variables in each group. The default value is a vector of 2 of size d/2.}
  \item{max.act.in}{The maximum number of active variables to add into the active set when \code{alg = "cyclic"}. The default value is 3.}
  \item{truncation}{The value of \eqn{(1+truncation)*\lambda} for active set update when \code{alg = "cyclic"}. The default value is 0.}
  \item{prec}{Stopping criterion. The default value is 1e-4.}
  \item{max.ite}{The iteration limit. The default value is 1e4.}
  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{TRUE}.}
  }
\details{
For sparse linear regression,

\deqn{
    \min_{\beta} {\frac{1}{2n}}|| Y - X \beta ||_2^2 + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP, SCAD, group \eqn{\ell_1}, group MCP or group SCAD.

For sparse multivariate regression,

\deqn{
    \min_{\beta} {\frac{1}{2nm}}|| Y - X \beta ||_F^2 + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} is \eqn{\ell_{12}} norm of matrix, group MCP or group SCAD.
    
For sparse logistic regression,

\deqn{
    \min_{\beta} {\frac{1}{n}}\sum_{i=1}^n (\log(1+e^{x_i^T \beta}) - y_i x_i^T \beta) + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP, SCAD, group \eqn{\ell_1}, group MCP or group SCAD.
    
For sparse column inverse operation,
\deqn{
    \min_{\beta} {\frac{1}{2}} \beta^T S \beta - e^T \beta + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP or SCAD.

}

\value{
An object with S3 classes \code{"lasso"}, \code{"logit"}, \code{"mvr"} and \code{"scio"} corresponding to  sparse linear regression, sparse logistic regression, sparse multivariate regression and sparse column inverse operator respectively is returned: 
  \item{beta}{
  A matrix of regression estimates whose columns correspond to regularization parameters for sparse linear regression and sparse logistic regression. A list of matrices of regression estimation corresponding to regularization parameters for sparse multivariate regression and sparse column inverse operator.
  }
  \item{intercept}{The value of intercepts corresponding to regularization parameters for sparse linear regression, sparse multivariate regression and sparse logistic regression. }
  \item{Y}{The value of \code{Y} used in the program. }
  \item{X}{The value of \code{X} used in the program. }
  \item{lambda}{
  The sequence of regularization parameters \code{lambda} used in the program.
  }
  \item{nlambda}{
  The number of values used in \code{lambda}.
  }
  \item{family}{
  The \code{family} from the input. 
  }
  \item{method}{
  The \code{method} from the input. 
  }
  \item{alg}{
  The \code{alg} from the input. 
  }
  \item{sym}{
  The \code{sym} from the input.
  }
  \item{path}{
  A list of \code{d} by \code{d} adjacency matrices of estimated graphs as a graph path corresponding to \code{lambda}.
  }
  \item{sparsity}{
  The sparsity levels of the graph path for sparse inverse column operator.
  }
  \item{standardize}{The \code{standardize} from the input.}
  \item{perturb}{The \code{perturb} from the input.}
  \item{df}{
  The degree of freecom (number of nonzero coefficients) along the solution path for sparse linear regression, sparse multivariate regression and sparse logistic regression. 
  }
  \item{ite}{
  A list of vectors where the i-th entries of ite[[1]] and ite[[2]] correspond to the outer iteration and inner iteration of i-th regularization parameter respectively.
  }
  \item{verbose}{The \code{verbose} from the input.}
}

\author{
Xingguo Li, Tuo Zhao and Han Liu \cr
Maintainer: Xingguo Li <xingguo.leo@gmail.com>
}

\references{
1. J. Friedman, T. Hastie and H. Hofling and R. Tibshirani. Pathwise coordinate optimization. \emph{The Annals of Applied Statistics}, 2007.\cr
2. C.H. Zhang. Nearly unbiased variable selection under minimax concave penalty. \emph{Annals of Statistics}, 2010.\cr
3. J. Fan and R. Li. Variable selection via nonconcave penalized likelihood and its oracle
properties. \emph{Journal of the American Statistical Association}, 2001.\cr
4. M. Yuan and Y. Lin. Model selection and estimation in the gaussian graphical model. \emph{Biometrika}, 2007.\cr
5. J. Huang, P. Breheny and S. Ma. A Selective Review of Group Selection in High-Dimensional Models. \emph{Statistical Science}, 2012.
}

\seealso{
\code{\link{picasso-package}}.
}

\examples{
################################################################
## Sparse linear regression
## Generate the design matrix and regression coefficient vector
n = 100
d = 400
X = matrix(rnorm(n*d), n, d)
beta = c(3,2,0,1.5,rep(0,d-4))

## Generate response using Gaussian noise, and fit sparse linear models
noise = rnorm(n)
Y = X\%*\%beta + noise
out.l1.cyclic = picasso(X, Y, nlambda=10)
out.l1.greedy = picasso(X, Y, nlambda=10, alg="greedy")
out.mcp.cyclic = picasso(X, Y, nlambda=10, method="mcp")
out.gr.prox = picasso(X, Y, nlambda=10, method="group", alg="prox")

## Visualize the solution path
plot(out.l1.cyclic)
plot(out.l1.greedy)
plot(out.mcp.cyclic)
plot(out.gr.prox)


################################################################
## Sparse logistic regression
## Generate the design matrix and regression coefficient vector
n = 100
d = 400
X = matrix(rnorm(n*d), n, d)
beta = c(3,2,0,1.5,rep(0,d-4))

## Generate response and fit sparse logistic models
p = exp(X\%*\%beta)/(1+exp(X\%*\%beta))
Y = rbinom(n,rep(1,n),p)
out.l1.cyclic = picasso(X, Y, nlambda=10, family="logit")
out.l1.greedy = picasso(X, Y, nlambda=10, family="logit", alg="greedy")
out.mcp.cyclic = picasso(X, Y, nlambda=10, family="logit", method="mcp")
out.gr.prox = picasso(X, Y, nlambda=10, family="logit", method="group", alg="prox")

## Visualize the solution path
plot(out.l1.cyclic)
plot(out.l1.greedy)
plot(out.mcp.cyclic)
plot(out.gr.prox)

## Estimate of Bernoulli parameters
p.l1 = out.l1.cyclic$p


################################################################
## Sparse multivariate regression
## Generate the design matrix and regression coefficient vector
n = 100
d = 200
p = 10
X = matrix(rnorm(n*d), n, d)
beta = matrix(0,d,p)
beta[c(1,2,4),] = c(3,2,1.5)

## Generate response using Gaussian noise, and fit sparse linear models
noise = matrix(rnorm(n*p),nrow=n)
Y = X\%*\%beta + noise
out.cyclic = picasso(X, Y, nlambda=10)


################################################################
## Sparse column inverse operator
## generating data
n = 100
d = 200
D = scio.generator(n=n,d=d,graph="band",g=1)
plot(D)

## sparse precision matrix estimation
out1 = picasso(D$data, nlambda=10, family="npn")
plot(out1)
scio.plot(out1$path[[4]])

}