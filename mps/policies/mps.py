"""
  Implements posterior sampling for Bayesian experiment design.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=abstract-method
# pylint: disable=invalid-name
# pylint: disable=no-member

from argparse import Namespace
# Local
from ..exd.exd_core import ExperimentDesigner
try:
  from ..prob.edward_prob_distros import edward_args_specific
except ImportError:
  edward_args_specific = []
from ..prob.prob_distros import bayesian_disc_model_args
from ..utils.oper_utils import maximise_with_method
from ..utils.option_handler import get_option_specs

mps_args_specific = [
  get_option_specs('mps_num_lookahead_steps', False, 1,
    'Number of steps to look-ahead when determining a point.'),
  get_option_specs('mps_num_y_giv_x_t_samples', False, 50,
    'Number of Y|X,theta samples to approximate reward_next.'),
  get_option_specs('mps_reward_minimisation_method', False, 'rand',
    'Method to minimise reward_next'),
  get_option_specs('mps_reward_minimisation_num_iters', False, -1,
    'Method to minimise reward_next'),
  ]

go_args_specific = mps_args_specific

mps_args = mps_args_specific + bayesian_disc_model_args + edward_args_specific
go_args = go_args_specific + bayesian_disc_model_args + edward_args_specific


class MPSExperimentDesigner(ExperimentDesigner):
  """ Implements Posterior sampling for Experiment Design. """

  # Overriding some policy methods from Blackbox Optimiser ------------------------
  def _policy_set_up(self):
    """ Sets up the policy. """
    pass

  def _get_policy_str(self):
    """ Describes the policy. """
    return 'mps_%d'%(self.options.mps_num_lookahead_steps)

  def _child_build_new_model(self):
    """ Builds a new model. """
    # Ideally, you want a learn a prior from the data but we won't do this for now.
    pass

  def is_an_mf_policy(self):
    """ Returns True if a multi-fidelity policy. """
    return False

  def _policy_update_history(self, qinfo):
    """ Updates to history from the policy. """
    pass

  def _get_policy_report_results_str(self):
    """ Reports updates current policy. """
    return ''

  def _policy_run_experiments_initialise(self):
    """ Initialisation for the policy. """
    pass

  # Determine next query -----------------------------------------------------------
  def get_past_data(self):
    """ Returns the data in past evaluations. """
    X = self.prev_eval_points + self.history.query_points
    Y = self.prev_eval_vals + self.history.query_vals
    return X, Y

  def _compute_reward_next(self, x, theta, past_X, past_Y, sample_y_giv_x_t):
    """ Computes the expected reward if evaluated at x. """
    if self.reward_next is not None:
      return self.reward_next(x, theta, past_X, past_Y)
    else:
      # This function computes the reward by augmenting the past data with _x, _y
      def _reward_next_obs(_x, _y, _theta, _past_X, _past_Y):
        """ This function computes the reward by augmenting the past data with _x, _y """
        return self.reward(_theta, _past_X + [_x], _past_Y + [_y])
      # Now compute the average and return
      y_samples = sample_y_giv_x_t(self.options.mps_num_y_giv_x_t_samples, x, theta)
      reward_vals = [_reward_next_obs(x, y, theta, past_X, past_Y) for y in y_samples]
      return sum(reward_vals) / float(len(reward_vals))

  def _get_reward_next_obj(self, theta, past_X, past_Y):
    """ Returns an objective as a function of x that will compute x. """
    return lambda x: self._compute_reward_next(x, theta, past_X, past_Y,
                                                self.model.sample_y_giv_x_t)

  def _get_num_iters_for_reward_min(self):
    """ Returns the number of iterations for reward minimisation. """
    if self.options.mps_reward_minimisation_num_iters > 0:
      return self.options.mps_reward_minimisation_num_iters
    else:
      return min(1000, max(50,
                 20 * self.experiment_caller.domain.get_dim() * (1 + self.step_idx)**2))

  def _determine_next_query(self):
    """ Determines the next point for evaluation. """
    if self.options.mps_num_lookahead_steps > 1:
      raise NotImplementedError('Only implemented 1 look-ahead yet.')
    past_X, past_Y = self.get_past_data()
    theta = self.model.sample_t_giv_data(1, past_X, past_Y)[0]
    return self._determine_query_from_theta_and_data(theta, past_X, past_Y)

  def _determine_query_from_theta_and_data(self, theta, past_X, past_Y):
    """ Determines the query from a given value of theta. """
    reward_next_obj = self._get_reward_next_obj(theta, past_X, past_Y)
    _, next_experiment_point, _ = maximise_with_method(
                                    self.options.mps_reward_minimisation_method,
                                    reward_next_obj,
                                    self.experiment_caller.domain,
                                    self._get_num_iters_for_reward_min(),
                                    vectorised=False,
                                    )
    qinfo = Namespace(point=next_experiment_point)
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of queries. """
    return [self._determine_next_query() for _ in range(batch_size)]


# Myopic Optimal Policy which knows the true parameter ===============================
class MOExperimentDesigner(MPSExperimentDesigner):
  """ Implements Posterior sampling for Experiment Design. """

  def _determine_next_query(self):
    """ Determines the next query usint true_theta. """
    past_X, past_Y = self.get_past_data()
    return self._determine_query_from_theta_and_data(self.true_theta, past_X, past_Y)

  def _get_policy_str(self):
    """ Describes the policy. """
    return 'mo_%d'%(self.options.mps_num_lookahead_steps)

