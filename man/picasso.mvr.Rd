\name{picasso.mvr}
\alias{picasso.mvr}

\title{
Sparse Multivariate Regression
}

\description{
The function "picasso.mvr" implements sparse multivariate regression.
}

\usage{
picasso.mvr(X, Y, lambda = NULL, nlambda = NULL, lambda.min.ratio = NULL,
            method="l12", alg = "cyclic", res.sd = FALSE, prec = 1e-4, 
            max.ite = 1e4, verbose = TRUE)
}

\arguments{
  \item{Y}{The \eqn{n} by \eqn{m} response matrix.}
  \item{X}{The \eqn{n} by \eqn{d} design matrix.}
  \item{lambda}{A sequence of decresing positive value to control the regularization. Typical usage is to leave the input \code{lambda = NULL} and have the program compute its own \code{lambda} sequence based on \code{nlambda} and \code{lambda.min.ratio}. Users can also specify a sequence to override this. Default value is from \eqn{lambda.max} to \code{lambda.min.ratio*lambda.max}. For Lq regression, the default value of \eqn{lambda.max} is \eqn{\pi\sqrt{\log(d)/n}}. For Dantzig selector, the default value of \eqn{lambda.max} is the minimum regularization parameter, which yields an all-zero estiamtes.}
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 5.}
  \item{lambda.min.ratio}{The smallest value for \code{lambda}, as a fraction of the uppperbound (\code{MAX}) of the regularization parameter. The program can automatically generate \code{lambda} as a sequence of length = \code{nlambda} starting from \code{MAX} to \code{lambda.min.ratio*MAX} in log scale. The default value is \code{0.25} for Lq mvr and \code{0.5} for Dantzig selector.}
  \item{method}{Options for regularization. L1 is applied if \code{method = "l1"}, MCP is applied if \code{method = "scad"}, SCAD mvr is applied if \code{method = "scad"} and group mvr is applied if \code{method = "gmvr"}. The default value is \code{"l1"}.}
  \item{alg}{Options for active set identification. Cyclic search is applied if \code{method = "cyclic"}, greedy search is applied if \code{method = "greedy"}, proximal gradient based search is applied if \code{method = "prox"} and stochastic search is applied if \code{method = "stoc"}. The default value is \code{"cyclic"}.}
  \item{res.sd}{Flag of whether the response varialbles are standardized. The default value is \code{FALSE}.}
  \item{prec}{Stopping criterion. The default value is 1e-4.}
  \item{max.ite}{The iteration limit. The default value is 1e4.}
  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{TRUE}.}
  }
\details{
\deqn{
    \min_{\beta} {\frac{1}{2nm}}|| Y - X \beta ||_F^2 + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} is \eqn{\ell_12} norm of matrix.
}
\value{
An object with S3 class \code{"picasso.mvr"} is returned: 
  \item{beta}{
  A matrix of regression estimates whose columns correspond to regularization parameters.
  }
  \item{intercept}{The value of intercepts corresponding to regularization parameters. }
  \item{Y}{The value of \code{Y} used in the program. }
  \item{X}{The value of \code{X} used in the program. }
  \item{lambda}{
  The sequence of regularization parameters \code{lambda} used in the program.
  }
  \item{nlambda}{
  The number of values used in \code{lambda}.
  }
  \item{method}{
  The \code{method} from the input. 
  }
  \item{df}{
  The average degree of freecom (number of nonzero coefficients) along the solution path. 
  }
  \item{ite}{
  A list of vectors where the i-th entry corresponding to the i-th regularization parameter. 
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
4. M. Yuan and Y. Lin. Model selection and estimation in the gaussian graphical model. \emph{Biometrika
}, 2007
}

\seealso{
\code{\link{picasso-package}}.
}

\examples{
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
out.cyclic = picasso.mvr(X = X, Y = Y, nlambda = 10)
}