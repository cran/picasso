\name{picasso.scio}
\alias{picasso.scio}

\title{
Sparse Column Inverse Operator
}

\description{
The function "picasso.scio" implements sparse column inverse operator.
}

\usage{
picasso.scio(data, lambda = NULL, nlambda = NULL, lambda.min.ratio = NULL,
             method = "l1", alg = "cyclic", gamma = 3, sym = "or", prec = 1e-4,
             max.ite = 1e4, standardize = FALSE, perturb = TRUE, verbose = TRUE)
}

\arguments{
  \item{data}{There are 2 options for \code{"clime"}: (1) \code{data} is an \code{n} by \code{d} data matrix (2) a \code{d} by \code{d} sample covariance matrix. The program automatically identifies the input matrix by checking the symmetry. (\code{n} is the sample size and \code{d} is the dimension). For \code{"tiger"}, covariance input is not supported and \code{d}\eqn{ \ge }3 is required. For \code{"clime"}, \code{d}\eqn{ \ge }2 is required.}
  \item{lambda}{A sequence of decresing positive value to control the regularization. Typical usage is to leave the input \code{lambda = NULL} and have the program compute its own \code{lambda} sequence based on \code{nlambda} and \code{lambda.min.ratio}. Users can also specify a sequence to override this. Default value is from \eqn{lambda.max} to \code{lambda.min.ratio*lambda.max}. The default value of \eqn{lambda.max} is the minimum regularization parameter which yields an all-zero estiamtes.}
  \item{nlambda}{The number of values used in \code{lambda}. Default value is 5.}
  \item{lambda.min.ratio}{The smallest value for \code{lambda}, as a fraction of the uppperbound (\code{MAX}) of the regularization parameter. The program can automatically generate \code{lambda} as a sequence of length = \code{nlambda} starting from \code{MAX} to \code{lambda.min.ratio*MAX} in log scale. The default value is \code{0.25}.}
  \item{method}{Options for regularization. L1 is applied if \code{method = "l1"}, MCP is applied if \code{method = "scad"} and SCAD scio is applied if \code{method = "scad"}. The default value is \code{"l1"}.}
  \item{alg}{Options for active set identification. Cyclic search is applied if \code{method = "cyclic"}, greedy search is applied if \code{method = "greedy"}, proximal gradient based search is applied if \code{method = "prox"} and stochastic search is applied if \code{method = "stoc"}. The default value is \code{"cyclic"}.}
  \item{gamma}{Parameter for MCP and SCAD. The default value is 3.}
  \item{sym}{Symmetrization of output graphs. If \code{sym = "and"}, the edge between node \code{i} and node \code{j} is selected ONLY when both node \code{i} and node \code{j} are selected as neighbors for each other. If \code{sym = "or"}, the edge is selected when either node \code{i} or node \code{j} is selected as the neighbor for each other. The default value is \code{"or"}.}
  \item{prec}{Stopping criterion. The default value is 1e-4.}
  \item{max.ite}{The iteration limit. The default value is 1e4.}
  \item{standardize}{Variables are standardized to have mean zero and unit standard deviation if \code{standardize = TRUE}. The default value is \code{FALSE}.}
  \item{perturb}{The diagonal of \code{Sigma} is added by a positive value to guarantee that \code{Sigma} is positive definite if \code{perturb = TRUE}. User can specify a numeric value for \code{perturbe}. The default value is \code{perturb = TRUE}.}
  \item{verbose}{Tracing information is disabled if \code{verbose = FALSE}. The default value is \code{TRUE}.}
  }
\details{
\deqn{
    \min_{\beta} {\frac{1}{2}} \beta^T S \beta - e^T \beta + \lambda R(\beta),
    }\cr
    where \eqn{R(\beta)} can be \eqn{\ell_1} norm, MCP or SCAD.
}
\value{
An object with S3 class \code{"picasso.scio"} is returned: 
  \item{data}{
  The \code{n} by \code{d} data matrix or \code{d} by \code{d} sample covariance matrix from the input.
  }
  \item{cov.input}{
  An indicator of the sample covariance. 
  }
  \item{lambda}{
  The sequence of regularization parameters \code{lambda} used in the program.
  }
  \item{nlambda}{
  The number of values used in \code{lambda}.
  }
  \item{icov}{
  A list of \code{d} by \code{d} precision matrices corresponding to regularization parameters.
  }
  \item{sym}{
  The \code{sym} from the input.
  }
  \item{method}{
  The \code{method} from the input. 
  }
  \item{alg}{
  The \code{alg} from the input. 
  }
  \item{path}{
  A list of \code{d} by \code{d} adjacency matrices of estimated graphs as a graph path corresponding to \code{lambda}.
  }
  \item{sparsity}{
  The sparsity levels of the graph path.
  }
  \item{ite}{
  A list of vectors where the i-th entries of ite[[1]] and ite[[2]] correspond to the outer iteration and inner iteration of i-th regularization parameter respectively.
  }
  \item{standardize}{The \code{standardize} from the input.}
  \item{perturb}{The \code{perturb} from the input.}
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
4. W. Liu and X. Luo. High-dimensional sparse precision matrix estimation via sparse
column inverse operator. \emph{arXiv preprint arXiv:1203.3896}, 2012.
}

\seealso{
\code{\link{picasso-package}}.
}

\examples{
## generating data
n = 100
d = 200
D = scio.generator(n=n,d=d,graph="band",g=1)
plot(D)

## sparse precision matrix estimation with method "clime"
out1 = picasso.scio(D$data, nlambda = 10)
plot(out1)
scio.plot(out1$path[[4]])
}