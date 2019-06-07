"""
  Implements a class for initialising.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-method

from ..exd.guided_exd import GoalOrientedExperimentDesigner


class Initialiser(GoalOrientedExperimentDesigner):
  """ Implements Random querying for Experiment Design. """

  def __init__(self, experiment_caller, worker_manager):
    """ Constructor. """
    super(Initialiser, self).__init__(experiment_caller, worker_manager, None, None)

# Overriding some policy methods from Blackbox Optimiser ------------------------
  def _policy_set_up(self):
    """ Sets up the policy. """
    pass

  def _get_policy_str(self):
    """ Describes the policy. """
    return 'rand'

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

  def _determine_next_query(self):
    """ Determines the next query. """
    raise ValueError('No need to call this method in an initialiser. ')

  def _determine_next_batch_of_queries(self, _):
    """ Determines the next batch of queries. """
    raise ValueError('No need to call this method in an initialiser. ')

  def initialise(self):
    """ Initialise. """
    return self.run_experiments(0)

  def is_an_mf_policy(self):
    """ Returns True if it is an MF policy. """
    return False

