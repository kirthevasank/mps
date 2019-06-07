"""
  A class for guided experiment design problems.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=abstract-method
# pylint: disable=invalid-name
# pylint: disable=no-member

import numpy as np
# Local imports
from exd.exd_core import ExperimentDesigner, ed_core_args
from exd.exd_utils import load_options_and_reporter

guided_exd_args_specific = []
guided_exd_args = ed_core_args + guided_exd_args_specific


class GuidedExperimentDesigner(ExperimentDesigner):
  """ A Guided experiment designer has a penalty function. The goal is to choose data
      in such a way so as to minimise this penalty.
  """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, experiment_caller, worker_manger, model, penalty,
               true_penalty=None, penalty_next=None, options=None, reporter=None):
    """ Constructor.
      - description is a string describing the problem.
      - penalty is a function which can take arguments of the form
        penalty(theta, X, Y) or penalty(theta, data). Here, data is a list of (x, y)
        tuples, and X, Y are equal length iterables of inputs and outputs.
      - If penalty_next is not None, we will use penalty to compute the look-ahead
        penalty empirically.
      - True penalty computes the true penalty that we are trying to minimise.
    """
    self.penalty = penalty
    self.penalty_next = penalty_next
    self.true_penalty = true_penalty
    options, reporter = load_options_and_reporter(guided_exd_args, options, reporter)
    super(GuidedExperimentDesigner, self).__init__(experiment_caller, worker_manger,
                                                   model, options, reporter)

  def _problem_set_up(self):
    """ Set up for the problem. """
    self.curr_best_penalty = np.inf
    self.curr_penalty = np.inf
    # Set up history
    self.history.curr_best_penalties = []
    self.history.curr_penalties = []

  def _get_problem_str(self):
    """ Return a string describing the problem. Can be overridden by a child method. """
    return 'guided'

  def _problem_update_history(self, _):
    """ Updates the history with some statistics relevant to the problem. """
    self._update_penalty_values()
    self.history.curr_penalties.append(self.curr_penalty)
    self.history.curr_best_penalties.append(self.curr_best_penalty)

  def _update_penalty_values(self):
    """ Updates the penalty values. """
    # First update the penalty value
    if self.true_penalty is not None:
      past_X, past_Y = self.get_past_data()
      self.curr_penalty = self.true_penalty(past_X, past_Y)
    else:
      self.curr_penalty = np.nan
    # Update the best penalty
    if self.curr_penalty < self.curr_best_penalty:
      self.curr_best_penalty = self.curr_penalty

  def _get_problem_report_results_str(self):
    """ Returns a string reporting status on the problem. """
    return 'curr_pen: %0.3f, best_pen: %0.3f'%(self.curr_penalty, self.curr_best_penalty)

  def _problem_handle_prev_evals(self):
    """ Handles previous evaluations for the problem. Assumes that self.prev_eval_points
        and self.prev_eval_vals have been filled in already. """
    self._update_penalty_values()

  def _problem_run_experiments_initialise(self):
    """ Initialisation for the problem. """
    pass

