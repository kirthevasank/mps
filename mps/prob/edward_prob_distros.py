"""
  Interface for Probability distributions defined in our library to Edward.
  -- kandasamy@cs.cmu.edu
"""
# N.B: Ideally, you won't need to import Edward anywhere outside of this file.

# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=abstract-class-not-used
# pylint: disable=abstract-class-little-used

import numpy as np
import edward as ed
import tensorflow as tf
# Local
from .prob_distros import BayesianDiscriminativeModel, bayesian_disc_model_args
from ..utils.option_handler import get_option_specs
from ..exd.exd_utils import load_options_and_reporter

edward_args_specific = [
  get_option_specs('edw_dflt_inference_method', False, 'vi',
    'Inference method for VI in edward.'),
  get_option_specs('edw_dflt_inference_num_iters', False, 5000,
    'Number of iterations for inference in edward.'),
  get_option_specs('edw_dflt_inference_num_restarts', False, 5,
    'Number of re-starts for Edward.'),
  ]

edward_args = bayesian_disc_model_args + edward_args_specific


# Some ancillary functions we will need for Edward -------------------------------------
def get_edward_prior_from_info(info):
  """ Returns a prior from the info. """
  if info.distro.lower() == 'normal_1d':
    return ed.models.Normal(loc=info.mu, scale=info.sigma)
  elif info.distro.lower() == 'inverse_gamma':
    return ed.models.InverseGamma(concentration=info.concentration, rate=info.rate)
  elif info.distro.lower() == 'const_1d':
    return ed.models.Deterministic(info.const)
  else:
    NotImplementedError('Not Implemented %s distribution yet!'%(info.distro))

def get_edward_vi_approximation(info):
  """ Returns the class of approximating distributions for VI. """
  # Now return
  if info.vi_distro.lower() == 'normal_1d':
    return ed.models.Normal(
      loc=tf.Variable(tf.constant(info.mu) + 0.01 * tf.random_normal([])),
      scale=tf.nn.softplus(tf.Variable(tf.constant(info.sigma)  +
                                       0.01 * tf.random_normal([]))))
  elif info.vi_distro.lower() == 'inverse_gamma':
    return ed.models.InverseGamma(
      concentration=tf.nn.softplus(tf.Variable(tf.constant(info.concentration) +
                                                0.01 * tf.random_normal([]))),
      rate=tf.nn.softplus(tf.Variable(tf.constant(info.rate) +
                                                0.01 * tf.random_normal([]))))
  elif info.vi_distro.lower() == 'const_1d':
    return ed.models.Deterministic(info.const)
  else:
    NotImplementedError('Not Implemented %s distribution for VI.'%(info.vi_distro))


class EdwardBayesianDiscriminativeModel(BayesianDiscriminativeModel):
  """ Bayesian Discriminative model with edward. """

  def __init__(self, y_domain, x_domain, t_domain, prior_info,
               options=None, reporter=None):
    """ Constructor.
      dflt_inference_method is a string which is either vi or mcmc.
      prior_info is a dictionary with the information on the prior for each variable.
      Some pointers:
        - self.tf is a pointer to tensorflow so that a child class can use self.tf instead
          of importing tensorflow.
        - self.ed is a pointer to edward so that a child class can use self.ed instead
          of importing edward.
    """
    options, reporter = load_options_and_reporter(edward_args, options, reporter)
    super(EdwardBayesianDiscriminativeModel, self).__init__(y_domain, x_domain, t_domain,
                                                            options, reporter)
    self.prior_info = prior_info
    self.prior_distros = {}
    self.tf = tf
    self.ed = ed
    self.reset()

  def reset(self):
    """ Reset priors. """
    self._set_param_distros()

  def _set_param_distros(self):
    """ Set the priors in the child. """
    for key, value in self.prior_info.iteritems():
      self.prior_distros[key] = get_edward_prior_from_info(value)

  @classmethod
  def get_param_vector_from_dict(cls, param_dict):
    """ Returns the parameter vector from a dictionary representation. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def get_param_dict_from_vectr(cls, param_vector):
    """ Returns the parameter as a dictionaryfrom a vector representation. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _run_edward_inference(cls, inference, inference_num_iters, posterior):
    """ Runs edward inference. """
    for _ in range(inference_num_iters-1):
      inference.update()
    info_dict = inference.update()
    inference.finalize()
    return posterior, info_dict['loss']

  def _get_new_inference_and_posterior_objects(self, X, Y, inference_method):
    """ Returns a new inference object. """
    param_distros = {param_name: get_edward_prior_from_info(info) for
                     (param_name, info) in self.prior_info.iteritems()}
    num_data = len(X)
    if num_data == 0:
      X = np.zeros(shape=(0, self.x_domain.dim))
      Y = np.zeros(shape=(0,))
    X = np.array(X)
    Y = np.array(Y)
    X_data = tf.placeholder(tf.float32, [num_data, self.x_domain.dim])
    Y_data = self.get_edward_distribution_for_y_given_x_t(param_distros, X_data)
    inference_data = {X_data:X, Y_data:Y}
    if inference_method == 'vi':
      posterior = {key: get_edward_vi_approximation(value) for
                   (key, value) in self.prior_info.iteritems()}
      inference_creator = ed.KLqp
    elif inference_method == 'hmc':
      raise NotImplementedError('Not implemented HMC yet!')
    else:
      raise ValueError('Unknown inference_type %s.'%(inference_method))
    # Dictionary mapping priors to posteriors
    prior_post_mappings = {param_distros[param_name]: posterior[param_name]
                           for param_name in self.prior_info.keys()}
    # Do the inference
    inference = inference_creator(prior_post_mappings, data=inference_data)
    inference.initialize(n_print=0)
    return inference, posterior

  def compute_posterior(self, X, Y, inference_num_restarts=None,
                        inference_num_iters=None, inference_method=None):
    """ X, Y are arrays/lists of data which has been observed. This computes the
        posterior for the parameters given X and Y.
    """
    # Determine inference parameters
    inference_num_restarts = inference_num_restarts if inference_num_restarts \
                             is not None else \
                             self.options.edw_dflt_inference_num_restarts
    inference_method = inference_method if inference_method is not None else \
                     self.options.edw_dflt_inference_method
    inference_num_iters = inference_num_iters if inference_num_iters is not None else \
                          self.options.edw_dflt_inference_num_iters
    # Create inference and posterior objects
    inferences_and_posteriors = [
      self._get_new_inference_and_posterior_objects(X, Y, inference_method)
      for _ in range(inference_num_restarts)]
    # Initialise and run
    tf.global_variables_initializer().run()
    posteriors_and_losses = [
      self._run_edward_inference(inference, inference_num_iters, posterior) for
      (inference, posterior) in inferences_and_posteriors]
    # Determine the best posterior
    best_posterior = None
    best_loss = np.inf
    for posterior, loss in posteriors_and_losses:
      if loss < best_loss:
        best_loss = loss
        best_posterior = posterior
    return best_posterior

  def get_edward_distribution_for_y_given_x_t(self, param_distros, X_data):
    """ Returns the discriminative distribution in Edward. """
    raise NotImplementedError('Implement in a child class.')

  def sample_t_from_prior(self, num_samples, *args, **kwargs):
    """ Samples from the prior. """
    prior_samplers = {param_name: self.prior_distros[param_name].sample() for
                                 param_name in self.prior_distros.keys()}
    return self._sample_t_wrap_up(prior_samplers, num_samples)

  def _sample_t_giv_data_in_xy_form(self, num_samples, X, Y, *args, **kwargs):
    """ Samples from the posterior. """
    # TODO: This does not work for MCMC methods since we need to do interleaving.
    posterior = self.compute_posterior(X, Y, *args, **kwargs)
    posterior_samplers = {param_name: posterior[param_name].sample() for param_name in
                          posterior.keys()}
    return self._sample_t_wrap_up(posterior_samplers, num_samples)

  def _sample_t_wrap_up(self, samplers, num_samples):
    """ Wraps up and returns samples from the samplers. """
    samples = [{param_name: samplers[param_name].eval()
                for param_name in samplers.keys()}
               for _ in range(num_samples)]
    samples = [self.get_param_vector_from_dict(elem) for elem in samples]
    return samples

  def _draw_single_sample_from_posterior(self, posterior):
    """ Returns a single sample in the form of a dictionary. """
    sample = {}
    for key in self.prior_info:
      sample[key] = posterior[key].sample().eval()
    return sample

