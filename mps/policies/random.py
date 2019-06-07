"""
  Implements random policies.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-method

from argparse import Namespace
import numpy as np
# Local
from ..exd.exd_core import ExperimentDesigner
from ..utils.general_utils import map_to_bounds

rand_args = []

class RandomExperimentDesigner(ExperimentDesigner):
  """ Implements Random querying for Experiment Design. """

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


class EuclideanRandomExperimentDesigner(RandomExperimentDesigner):
  """ Designs randomly in Euclidean spaces. """

  def is_an_mf_policy(self):
    """ Returns True if a multi-fidelity policy. """
    return False

  def _determine_next_query(self):
    """ Determines the next query. """
    qinfo = Namespace(point=map_to_bounds(np.random.random(self.domain.dim),
                                          self.domain.bounds))
    return qinfo

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of queries. """
    qinfos = [self._determine_next_query() for _ in range(batch_size)]
    return qinfos

