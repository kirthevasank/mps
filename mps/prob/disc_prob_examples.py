"""
  Implements some example discriminative models.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-method
# pylint: disable=no-member

from copy import deepcopy
import numpy as np
try:
  from scipy.stats import multivariate_normal
except ImportError:
  from numpy.random import multivariate_normal
# Local
from ..exd.domains import EuclideanDomain
from .prob_distros import ParametricDiscriminativeModel, BayesianDiscriminativeModel
try:
  from ..prob.edward_prob_distros import EdwardBayesianDiscriminativeModel
except ImportError:
  EdwardBayesianDiscriminativeModel = object


class LogisticWithGaussianNoise(ParametricDiscriminativeModel):
  """ An example demonstrating a logistic with Gaussian Noise.
      Precisely, implements f(x) = a/(1 + exp(b(x-c))), and y = f(x) + e where
      e~N(0,eta2).
  """

  def __init__(self, a, b, c, eta2, x_domain=None, *args, **kwargs):
    """ Constructor. """
    theta = [a, b, c, eta2]
    if x_domain is None:
      x_domain = EuclideanDomain([[-np.inf, np.inf]])
    y_domain = EuclideanDomain([[-np.inf, np.inf]])
    super(LogisticWithGaussianNoise, self).__init__(x_domain, y_domain, theta,
                                                    None, *args, **kwargs)

  def _eval_fx(self, X):
    """ Evaluates f(x) = a/(1 + exp(b(x-c))). """
    a = self.theta[0]
    b = self.theta[1]
    c = self.theta[2]
    X_ = np.array(X).ravel()
    return a/(1 + np.exp(b * (X_ - c)))

  def _sample_y_giv_x_single(self, num_samples, x):
    """ Samples y given x. """
    mean_val = self._eval_fx(x)
    ret = mean_val + np.sqrt(self.theta[3]) * np.random.normal(size=(num_samples,))
    return ret

  def eval_y_mean_giv_x(self, X):
    """ Evaluate the mean. """
    return list(self._eval_fx(X))

  def eval_y_var_giv_x(self, X):
    """ Evaluate the variance. """
    return list(self.theta[3] * np.ones((len(X),)))

  def eval_y_covar_giv_x(self, X):
    """ Evaluate the covariance. """
    return np.diag(self.eval_y_var_giv_x(X))

  def __str__(self):
    """ Returns a string representation. """
    return ('LogisticWithGaussian: a=%0.3f, b=%.3f, c=%.3f, eta2=%0.3f'%(
            self.theta[0], self.theta[1], self.theta[2], self.theta[3]))


class BayesianLogisticWithGaussianNoise(EdwardBayesianDiscriminativeModel):
  """ An example similar to LogisticWithGaussianNoise but a, b, c, eta2 are
      drawn from a prior.
  """

  def __init__(self, x_domain, t_domain, prior_info, *args, **kwargs):
    """ Constructor. See EdwardBayesianDiscriminativeModel for details. """
    if x_domain is None:
      x_domain = EuclideanDomain([[-np.inf, np.inf]])
    if t_domain is None:
      t_domain = EuclideanDomain([[-np.inf, np.inf]] * 3 + [[0, np.inf]])
    y_domain = EuclideanDomain([[-np.inf, np.inf]])
    super(BayesianLogisticWithGaussianNoise, self).__init__(y_domain, x_domain, t_domain,
                                                            prior_info, *args, **kwargs)

  @classmethod
  def get_param_vector_from_dict(cls, param_dict):
    """ Returns the parameter vector from a dictionary representation. """
    return [param_dict['a'], param_dict['b'], param_dict['c'], param_dict['eta2']]

  @classmethod
  def get_param_dict_from_vector(cls, param_vector):
    """ Returns the parameter as a dictionaryfrom a vector representation. """
    return {'a':param_vector[0], 'b':param_vector[1], 'c':param_vector[2],
            'eta2':param_vector[3]}

  def get_edward_distribution_for_y_given_x_t(self, param_distros, X_data):
    """ Returns the discriminative distribution in Edward. """
    loc = self._compute_fx(X_data, param_distros['a'], param_distros['b'],
                           param_distros['c'])
    num_data = int(X_data.shape[0])
    scale = self.tf.sqrt(param_distros['eta2']) * self.tf.ones(num_data)
    return self.ed.models.Normal(loc=loc, scale=scale)

  def _compute_fx(self, X_data, a, b, c):
    """ Computes f(x). """
#     ret = a / (1 + self.tf.exp(b*(self.tf.reshape(X_data, [-1]) - c)))
    D = int(X_data.shape[1])
    ret = self.tf.divide(a, 1+self.tf.exp(b*(self.ed.dot(X_data, self.tf.ones(D))-c)))
    return ret

  def _draw_single_sample_from_posterior(self, posterior):
    """ Override the method to ensure that a, b, c are not negative. """
    sample = super(BayesianLogisticWithGaussianNoise,
                   self)._draw_single_sample_from_posterior(posterior)
    if sample['a'] > 0 and sample['b'] > 0 and sample['c'] > 0 and sample['eta2'] > 0:
      return sample
    else:
      return self._draw_single_sample_from_posterior(posterior)

  # Instantiate y_giv_x_t functions ----------------------------------------------------
  def _eval_fXT(self, X, T):
    """ Evaluates f(x) = a/(1 + exp(b(x-c))). Here X can be a matrix but t is a single
        vector specifying one value for the parameter.
    """
    if self.t_domain.is_a_member(T):
      a = T[0]
      b = T[1]
      c = T[2]
    else:
      a = T[:, 0]
      b = T[:, 1]
      c = T[:, 2]
    X_ = np.array(X).ravel()
    return a/(1 + np.exp(b * (X_ - c)))

  def _sample_y_giv_x_t_single(self, num_samples, x, t):
    """ Samples y given x. """
    mean_val = self._eval_fXT(x, t)
    ret = mean_val + np.sqrt(t[3]) * np.random.normal(size=(num_samples,))
    return ret

  def eval_y_mean_giv_x_t(self, X, T):
    """ Evaluate the mean. """
    return list(self._eval_fXT(X, T))

  def eval_y_var_giv_x_t(self, X, T):
    """ Evaluate the variance. """
    if self.t_domain.is_a_member(T):
      T = np.array([T] * len(X))
    return T[:, 3]

  def eval_y_covar_giv_x(self, X):
    """ Evaluate the covariance. """
    return np.diag(self.eval_y_var_giv_x(X))

  def __str__(self):
    """ Returns a string representation. """
    return 'BayesianLogisticWithGaussian: %s'%(self.prior_info)


# An interface for Gaussian Processes --------------------------------------------
class GPonGrid(BayesianDiscriminativeModel):
  """ A class for GPs on a grid.
    This is just an interface in our framework to the actual GP object in gp/ that
    is doing the work.
  """

  def __init__(self, x_domain, t_domain, grid, gp_obj, *args, **kwargs):
    """ Constructor. """
    y_domain = EuclideanDomain([[-np.inf, np.inf]])
    self.grid = grid
    self.gp_obj = gp_obj
    super(GPonGrid, self).__init__(y_domain, x_domain, t_domain, *args, **kwargs)

  def _sample_y_giv_x_t_multiple(self, num_samples, x, t):
    """ Here t is assumed to be a tuple where the second is a GP object. """
    grid_data = t
    gp_obj = deepcopy(self.gp_obj)
    aug_X = gp_obj.X + list(grid_data[0])
    aug_Y = gp_obj.Y + list(grid_data[1])
    gp_obj.set_data(aug_X, aug_Y)
    return gp_obj.draw_samples(num_samples, [x])

  def _sample_y_giv_x_t_single(self, num_samples, x, t):
    """ Here t is assumed to be a tuple where the second is a GP object. """
    return self._sample_y_giv_x_t_multiple(num_samples, x, t)[0]

  def _sample_t_giv_data_in_xy_form_single(self, post_gp):
    """ Samples from the posterior. """
    grid_samples = post_gp.draw_samples(1, self.grid)[0]
    return self.grid, grid_samples

  def _sample_t_giv_data_in_xy_form(self, num_samples, X, Y):
    """ Samples from the posterior. """
    post_gp = deepcopy(self.gp_obj)
    post_gp.set_data(X, Y)
    return [self._sample_t_giv_data_in_xy_form_single(post_gp) for _ in
            range(num_samples)]


# A linear model with Basis functions --------------------------------------------------
class LinearRBF(ParametricDiscriminativeModel):
  """ Implements y = f(x) + e, where
      f(x) = w^T phi(x), where our phi(x) kernel functions are Gaussians
      with different means and the same variance. The error terms are e~N(0,eta2).
  """
  def __init__(self, centers, rbf_var, weights, eta2, x_domain=None, *args, **kwargs):
    """ Constructor.
        centers: locations of centers of Gaussian RBF's
        rbf_var: variance of Gaussian RBF's
        weights: weights of the linear combination that determines f(x)
        eta2: noise parameter
    """
    self.centers = centers
    self.var = rbf_var
    self.eta2 = eta2
    theta = weights
    if x_domain is None:
      x_domain = EuclideanDomain([[-np.inf, np.inf], [-np.inf, np.inf]])
    y_domain = EuclideanDomain([[-np.inf, np.inf]])
    super(LinearRBF, self).__init__(x_domain, y_domain, theta,
                                    None, *args, **kwargs)

  def _eval_fx(self, X):
    """ Evaluates f(x) = w^T phi(x). """
    weights = self.theta
    centers = self.centers
    var = self.var
    gaussians = [multivariate_normal(mean=c, cov=var * np.eye(2)) for c in centers]
    densities = np.array([g.pdf(X) for g in gaussians]).T
    y = densities.dot(weights)
    return y

  def _sample_y_giv_x_single(self, num_samples, x):
    """ Samples y given x. """
    mean_val = self._eval_fx(x)
    eta2 = self.eta2
    ret = mean_val + np.sqrt(eta2) * np.random.normal(size=(num_samples,))
    return ret

  def eval_y_mean_giv_x(self, X):
    """ Evaluate the mean. """
    return list(self._eval_fx(X))

  def eval_y_var_giv_x(self, X):
    """ Evaluate the variance. """
    eta2 = self.eta2
    return list(eta2 * np.ones((len(X),)))

  def eval_y_covar_giv_x(self, X):
    """ Evaluate the covariance. """
    return np.diag(self.eval_y_var_giv_x(X))

  def __str__(self):
    """ Returns a string representation. """
    weights = self.theta
    eta2 = self.eta2
    min_weight = min(weights)
    max_weight = max(weights)
    str1 = 'LiearRBF: num_centers=%d, kernel_var=%.3f, '%(len(self.centers), self.var)
    str2 = 'eta2=%.3f, min_weight=%.3f, max_weight=%.3f'%(eta2, min_weight, max_weight)
    return str1 + str2


class BayesianLinearRBF(BayesianDiscriminativeModel):
  """ An example similar to LinearRBF but weights are drawn from a prior.
  """

  def __init__(self, centers, var, eta2, x_domain, t_domain, prior_info, *args, **kwargs):
    """ The constructor.
        centers should be a list of coordinates for the centers of the Gaussian RBFs.
        var should specify the variance of the Gaussian RBFs.
        prior_info should be of the form (mu_0, Lambda_0) containing the mean and
          precision matrix specifying a Gaussian distribution.
    """
    self.centers = np.array(centers)
    self.var = var
    self.dim = len(centers[0])
    self.eta2 = eta2
    self.prior = prior_info
    self.gaussians = [multivariate_normal(mean=c,
                                          cov=var * np.eye(self.dim)) for c in centers]
    if x_domain is None:
      x_domain = EuclideanDomain([[-np.inf, np.inf]] * self.dim)
    if t_domain is None:
      t_domain = EuclideanDomain([[-np.inf, np.inf]] * len(centers))
    y_domain = EuclideanDomain([[-np.inf, np.inf]])
    super(BayesianLinearRBF, self).__init__(y_domain, x_domain, t_domain,
                                            *args, **kwargs)

  def _sample_y_giv_x_t_single(self, num_samples, x, t):
    """ Draw a Y sample at the point x, for parameter t. """
    weights = t
    densities = np.array([g.pdf(x) for g in self.gaussians]).T
    y = densities.dot(weights)
    return y + np.sqrt(self.eta2) * np.random.normal(size=(num_samples,))

  def eval_y_mean_giv_x_t(self, X, T):
    """ Evalute the mean given x and t. """
    weights = T
    densities = np.array([g.pdf(X) for g in self.gaussians]).T
    if self.t_domain.is_a_member(T):
      y = densities.dot(weights)
    else:
      y = np.sum(densities * weights, axis=1)
    return y

  def eval_y_covar_giv_x_t(self, X, T):
    """ Evaluate the standard deviation given x and t. """
    return np.diag(self.eta2 * np.ones(len(X)))

  def sample_t_from_prior(self, num_samples, *args, **kwargs):
    """ Samples from the prior. """
    mu_0, Lambda_0 = self.prior
    prior = multivariate_normal(mean=mu_0, cov=self.eta2 * np.linalg.inv(Lambda_0))
    if num_samples == 1:
      return [prior.rvs(size=1)]
    return prior.rvs(size=num_samples)

  def _sample_t_giv_data_in_xy_form(self, num_samples, X, Y, *args, **kwargs):
    """ Samples from the posterior when given as two lists of X and Y.
        Given eta2, the posterior for our weights (t) is normal.
    """
    if len(X) == 0:
      return self.sample_t_from_prior(num_samples, *args, **kwargs)
    num_data = len(X)
    mu_0, Lambda_0 = self.prior
#     import pdb; pdb.set_trace()
    X = np.array(X)
    if len(X.shape) == 1:
      X = np.array([X])
#     else:
#       X = np.array(X)
#     X = np.array([g.pdf(X) for g in self.gaussians]).T
    X = np.array([g.pdf(X) for g in self.gaussians]).reshape((num_data, -1))
    t_mu = np.linalg.inv(X.T.dot(X) + Lambda_0).dot(Lambda_0.dot(mu_0) + X.T.dot(Y))
    t_var = self.eta2 * np.linalg.inv(X.T.dot(X) + Lambda_0)
    posterior = multivariate_normal(mean=t_mu, cov=t_var)
    if num_samples == 1:
      return [posterior.rvs(size=1)]
    return posterior.rvs(size=num_samples)

  def _sample_t_giv_data_in_tuple_form(self, num_samples, data, *args, **kwargs):
    """ Samples from the posterior when the data is given as a list of tuples. """
    X = [tup[0] for tup in data]
    Y = [tup[1] for tup in data]
    return self._sample_t_giv_data_in_xy_form(self, num_samples, X, Y, *args, **kwargs)

  def __str__(self):
    return "BayesianLinearRBF with num_centers=%d"%(len(self.centers))


