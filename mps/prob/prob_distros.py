"""
  Harness for handling discriminative, generative models along with Bayesian versions
  which include priors.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used
# pylint: disable=abstract-method

import numpy as np
# Local
from ..utils.reporters import get_reporter

disc_model_args = []
bayesian_disc_model_args = []

class DiscriminativeModel(object):
  """ Parent class for all discriminative models.
      Here, we are modeling Y|X under the model so Y is the observation given X.
  """

  def __init__(self, y_domain, x_domain, options=None, reporter=None):
    """ Constructor. """
    self.y_domain = y_domain
    self.x_domain = x_domain
    self.reporter = get_reporter(reporter)
    self.options = options

  def sample_y_giv_x(self, num_samples, X):
    """ Returns a list of num_samples objects where each element is an iterable with a
        sample for each element in X.
        If X is not a list, then returns a list of num_samples samples.
    """
    if self.x_domain.is_a_member(X):
      return self._sample_y_giv_x_single(num_samples, X)
    else:
      return self._sample_y_giv_x_multiple(num_samples, X)

  def _sample_y_giv_x_single(self, num_samples, x):
    """ Draw a Y sample at the point x. """
    raise NotImplementedError('Implement in a child class.')

  def _sample_y_giv_x_multiple(self, num_samples, X):
    """ Draw Y samples at each point X. The naive implementation repeatedly calls
        _sample_y_giv_x_single, but perhaps there are more efficient implementations for
        each child class. Can be overridden. """
    samples_per_point = [self._sample_y_giv_x_single(num_samples, x) for x in X]
    return map(list, zip(*samples_per_point))

  def eval_y_mean_giv_x(self, X):
    """ Evaluate the mean of Y at the X points. """
    raise NotImplementedError('Implement in a child class.')

  def eval_y_covar_giv_x(self, X):
    """ Evaluate the covariance of Y at the X points. """
    raise NotImplementedError('Implement in a child class.')

  def eval_y_var_giv_x(self, X):
    """ Evaluate the variance of Y at the X points. By default uses eval_y_covar_giv_x
        but can be overridden.
    """
    return np.diag(self.eval_y_covar_giv_x(X))

  def eval_y_giv_x(self, X, uncert_form='none'):
    """ Evaluate the mean and uncertainty of Y given X and T.
        Can be overridden by a child class.
    """
    return _return_mean_and_uncert_from_args(self.eval_y_mean_giv_x,
                                             self.eval_y_covar_giv_x,
                                             self.eval_y_var_giv_x,
                                             (X,), uncert_form)


class BayesianDiscriminativeModel(DiscriminativeModel):
  """ Parent class for all discriminative models. Here we are modelling P(Y|X,T)
      where T is a random variable.
  """

  def __init__(self, y_domain, x_domain, t_domain, options=None, reporter=None):
    """ Constructor. """
    self.t_domain = t_domain
    super(BayesianDiscriminativeModel, self).__init__(y_domain, x_domain, options,
                                                      reporter)

  # Y given X and t ----------------------------------------------------------------
  def sample_y_giv_x_t(self, num_samples, X, T):
    """ Returns a list of num_samples objects where each element is an iterable with a
        sample for each element in X and T.
        If X is not a list, then returns a list of num_samples samples.
    """
    if self.x_domain.is_a_member(X) and self.t_domain.is_a_member(T):
      return self._sample_y_giv_x_t_single(num_samples, X, T)
    else:
      return self._sample_y_giv_x_t_multiple(num_samples, X, T)

  def _sample_y_giv_x_t_single(self, num_samples, x, t):
    """ Draw a Y sample at the point x, for parameter t. """
    raise NotImplementedError('Implement in a child class. ')

  def _sample_y_giv_x_t_multiple(self, num_samples, X, T):
    """ Draw Y samples at each X and T. The naive implementation repeatedly calls
        _sample_y_giv_x_single, but perhaps there are more efficient implementations for
        each child class. Can be overridden. """
    samples_per_point = [self._sample_y_giv_x_t_single(num_samples, x, t) for
                         (x, t) in zip(X, T)]
    return map(list, zip(*samples_per_point))

  def eval_y_mean_giv_x_t(self, X, T):
    """ Evalute the mean given x and t. """
    raise NotImplementedError('Implement in a child class.')

  def eval_y_covar_giv_x_t(self, X, T):
    """ Evaluate the standard deviation given x and t. """
    raise NotImplementedError('Implement in a child class.')

  def eval_y_var_giv_x_t(self, X, T):
    """ Evaluate the variance of Y given X and T. By default uses
        eval_y_covar_giv_x_t but can be overridden.
    """
    return np.diag(self.eval_y_covar_giv_x_t(X, T))

  def eval_y_giv_x_t(self, X, T, uncert_form='none'):
    """ Evaluate the mean and uncertainty of Y given X and T. """
    return _return_mean_and_uncert_from_args(self.eval_y_mean_giv_x_t,
                                             self.eval_y_covar_giv_x_t,
                                             self.eval_y_var_giv_x_t,
                                             (X, T), uncert_form)

  # T given data ------------------------------------------------------------------------
  def sample_t_from_prior(self, num_samples, *args, **kwargs):
    """ Samples from the prior. """
    raise NotImplementedError('Implement in a child class.')

  def sample_t_giv_data(self, num_samples, data_or_X, Y=None, *args, **kwargs):
    """ Given observations, this draws from the posterior for theta.
        - If Y is None, then data_or_X is a list of n object where each element is
          tuple/list of length 2 corresponding to an action-observation (x, y) pair.
        - If Y is not None, then data_or_X is a list of n actions and Y is a list of
          n observations.
    """
    if Y is None:
      try:
        return self._sample_t_giv_data_in_tuple_form(num_samples, data_or_X,
                                                     *args, **kwargs)
      except NotImplementedError:
        X = [elem[0] for elem in data_or_X]
        Y = [elem[1] for elem in data_or_X]
        return self._sample_t_giv_data_in_xy_form(num_samples, X, Y, *args, **kwargs)
    else:
      try:
        return self._sample_t_giv_data_in_xy_form(num_samples, data_or_X, Y,
                                                  *args, **kwargs)
      except NotImplementedError:
        return self._sample_t_giv_data_in_tuple_form(num_samples, zip(data_or_X, Y),
                                                     *args, **kwargs)

  def _sample_t_giv_data_in_xy_form(self, num_samples, X, Y, *args, **kwargs):
    """ Samples from the posterior when given as two lists of X and Y. """
    raise NotImplementedError('Implement in a child class or ' +
                              '_sample_t_giv_data_in_tuple_form.')

  def _sample_t_giv_data_in_tuple_form(self, num_samples, data, *args, **kwargs):
    """ Samples from the posterior when given as a list of tuples. """
    raise NotImplementedError('Implement in a child class or ' +
                              '_sample_t_giv_data_in_xy_form.')


# Ancillary utilities we need above ===================================================
def _return_mean_and_uncert_from_args(eval_mean_func, eval_covar_func, eval_var_func,
                                      pass_args, uncert_form):
  """ A wrapper which evaluates the mean and uncertainty and returns. """
  ret_mean = eval_mean_func(*pass_args)
  if uncert_form == 'none':
    ret_uncert = None
  else:
    if uncert_form == 'covar':
      ret_uncert = eval_covar_func(*pass_args)
    elif uncert_form == 'var':
      ret_uncert = eval_var_func(*pass_args)
    elif uncert_form == 'std':
      ret_uncert = np.sqrt(eval_var_func(*pass_args))
    else:
      raise ValueError('Unknown option %s for uncert_form.'%(uncert_form))
  return ret_mean, ret_uncert

# Some specific abstract subclasses of the above ======================================
class ParametricDiscriminativeModel(DiscriminativeModel):
  """ A Discriminiative model with a parameter.
      Here, we are modeling Y|X,theta under the model so Y is the observation given X
      with parameter theta.
  """

  def __init__(self, x_domain, y_domain, theta, theta_domain=None, *args, **kwargs):
    """ Constructor. """
    self.theta = theta
    self.theta_domain = theta_domain
    super(ParametricDiscriminativeModel, self).__init__(x_domain, y_domain,
                                                        *args, **kwargs)

