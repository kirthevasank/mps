"""
  Implements random policies.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-method

from argparse import Namespace
from copy import deepcopy
import numpy as np
# Local
from utils.oper_utils import maximise_with_method
from exd.exd_core import ExperimentDesigner
from utils.option_handler import get_option_specs

vared_args_specific = [
  get_option_specs('vared_acq_maximisation_method', False, 'rand',
    'Method to minimise penalty_next'),
  get_option_specs('vared_acq_maximisation_num_iters', False, -1,
    'Method to minimise penalty_next'),
  ]

vared_args = vared_args_specific


class VaRedExperimentDesigner(ExperimentDesigner):
  """ Implements Random querying for Experiment Design. """

# Overriding some policy methods from Blackbox Optimiser ------------------------
  def _policy_set_up(self):
    """ Sets up the policy. """
    pass

  def _get_policy_str(self):
    """ Describes the policy. """
    return 'vared'

  def _child_build_new_model(self):
    """ Builds a new model. """
    pass

  def _policy_update_history(self, qinfo):
    """ Updates to history from the policy. """
    pass

  def _get_policy_report_results_str(self):
    """ Reports updates current policy. """
    return ''

  def _policy_run_experiments_initialise(self):
    """ Initialisation for the policy. """
    pass

  def is_an_mf_policy(self):
    """ Returns True if a multi-fidelity policy. """
    return False

  def _get_num_iters_for_acq_max(self):
    """ Returns the number of iterations for penalty minimisation. """
    if self.options.vared_acq_maximisation_num_iters > 0:
      return self.options.vared_acq_maximisation_num_iters
    else:
      return min(1000, max(50,
                 20 * self.experiment_caller.domain.get_dim() * (1 + self.step_idx)**2))

  @classmethod
  def compute_exponentiated_variance(cls, x, post_gp):
    """ Computes the Exponentiated Variance of the GP. """
    p_mu, p_sigma = post_gp.eval([x], uncert_form='std')
    return (np.exp(p_sigma**2) - 1) * np.exp(2*p_mu + p_sigma**2)

  def _get_maximisation_obj_from_post_gp(self, post_gp):
    """ Returns the objective to be maximised from the posterior GP. """
    return lambda x: self.compute_exponentiated_variance(x, post_gp)

  def _get_maximisation_obj(self, past_X, past_Y):
    """ Returns the objective to be maximised. """
    post_gp = deepcopy(self.model.gp_obj)
    post_gp.set_data(past_X, past_Y)
    return self._get_maximisation_obj_from_post_gp(post_gp)

  def _determine_next_query(self):
    """ Determines the next query. """
    past_X, past_Y = self.get_past_data()
    max_obj = self._get_maximisation_obj(past_X, past_Y)
    _, next_experiment_point, _ = maximise_with_method(
                            self.options.vared_acq_maximisation_method,
                            max_obj, self.experiment_caller.domain,
                            self._get_num_iters_for_acq_max(), vectorised=False)
    return Namespace(point=next_experiment_point)

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of queries. """
    qinfos = [self._determine_next_query() for _ in range(batch_size)]
    return qinfos

