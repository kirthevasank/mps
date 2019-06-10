"""
  A class for goal oriented experiment design problems.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=abstract-method
# pylint: disable=invalid-name
# pylint: disable=no-member

import numpy as np
# Local imports
from .exd_core import ExperimentDesigner, ed_core_args
from .exd_utils import load_options_and_reporter

goal_oriented_exd_args_specific = []
goal_oriented_exd_args = ed_core_args + goal_oriented_exd_args_specific


class GoalOrientedExperimentDesigner(ExperimentDesigner):
  """ A Guided experiment designer has a reward function. The goal is to choose data
      in such a way so as to minimise this reward.
  """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, experiment_caller, worker_manger, model, reward,
               true_reward=None, reward_next=None, options=None, reporter=None):
    """ Constructor.
      - description is a string describing the problem.
      - reward is a function which can take arguments of the form
        reward(theta, X, Y) or reward(theta, data). Here, data is a list of (x, y)
        tuples, and X, Y are equal length iterables of inputs and outputs.
      - If reward_next is not None, we will use reward to compute the look-ahead
        reward empirically.
      - True reward computes the true reward that we are trying to minimise.
    """
    self.reward = reward
    self.reward_next = reward_next
    self.true_reward = true_reward
    options, reporter = load_options_and_reporter(goal_oriented_exd_args, options,
                                                  reporter)
    super(GoalOrientedExperimentDesigner, self).__init__(experiment_caller, worker_manger,
                                                         model, options, reporter)

  def _problem_set_up(self):
    """ Set up for the problem. """
    self.curr_best_reward = -np.inf
    self.curr_reward = -np.inf
    # Set up history
    self.history.curr_best_reward = []
    self.history.curr_reward = []

  def _get_problem_str(self):
    """ Return a string describing the problem. Can be overridden by a child method. """
    return ''

  def _problem_update_history(self, _):
    """ Updates the history with some statistics relevant to the problem. """
    self._update_reward_values()
    self.history.curr_reward.append(self.curr_reward)
    self.history.curr_best_reward.append(self.curr_best_reward)

  def _update_reward_values(self):
    """ Updates the reward values. """
    # First update the reward value
    if self.true_reward is not None:
      past_X, past_Y = self.get_past_data()
      self.curr_reward = self.true_reward(past_X, past_Y)
    else:
      self.curr_reward = np.nan
    # Update the best reward
    if self.curr_reward > self.curr_best_reward:
      self.curr_best_reward = self.curr_reward

  def _get_problem_report_results_str(self):
    """ Returns a string reporting status on the problem. """
    return 'curr_rew: %0.3f, best_rew: %0.3f'%(self.curr_reward, self.curr_best_reward)

  def _problem_handle_prev_evals(self):
    """ Handles previous evaluations for the problem. Assumes that self.prev_eval_points
        and self.prev_eval_vals have been filled in already. """
    self._update_reward_values()

  def _problem_run_experiments_initialise(self):
    """ Initialisation for the problem. """
    pass

