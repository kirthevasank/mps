"""
  Implementation of Active select.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=abstract-method
# pylint: disable=invalid-name
# pylint: disable=no-member


import numpy as np
from argparse import Namespace
# Local
from exd.exd_core import ExperimentDesigner
from utils.option_handler import get_option_specs
from utils.general_utils import map_to_bounds


actsel_args_specific = [
  get_option_specs('actsel_init_size', False, -1,
    'Method to minimise penalty_next'),
  ]

actsel_args = actsel_args_specific

class ActSelExperimentDesigner(ExperimentDesigner):
  """ Active select. """
  # pylint: disable=attribute-defined-outside-init

  # Overriding some policy methods from Blackbox Optimiser ------------------------
  def _policy_set_up(self):
    """ Sets up the policy. """
    pass

  def _get_policy_str(self):
    """ Describes the policy. """
    return 'actsel'

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

  def _get_init_threshold(self):
    """ Returns the initial threshold. """
    if self.options.actsel_init_size > 0:
      return self.options.actsel_init_size
    else:
      return 5

  def _make_pool(self):
    """ Make a pool to query from. """
    pool_size = self._get_init_threshold()
    cand_size = 2 * pool_size
    candidates = map_to_bounds(np.random.random((cand_size, self.domain.dim)),
                               self.domain.bounds)
    past_X, _ = self.get_past_data()
    dists = np.zeros((cand_size, len(past_X)))
    for i in range(cand_size):
      for j in range(len(past_X)):
        dists[i, j] = np.linalg.norm(candidates[i] - past_X[j])
    min_dists = dists.min(axis=1)
    sample_probs = min_dists/min_dists.sum()
    pool_idxs = list(np.random.choice(cand_size, pool_size, replace=False,
                                      p=sample_probs))
    self.pool = [candidates[i] for i in pool_idxs]

  def _determine_next_query(self):
    """ Determine the next query. """
    if self.step_idx < self._get_init_threshold():
      next_point = map_to_bounds(np.random.random(self.domain.dim), self.domain.bounds)
    else:
      if not hasattr(self, 'pool') or len(self.pool) == 0:
        self._make_pool()
      next_point = self.pool.pop(0)
    return Namespace(point=next_point)

  def _determine_next_batch_of_queries(self, batch_size):
    """ Determines the next batch of queries. """
    qinfos = [self._determine_next_query() for _ in range(batch_size)]
    return qinfos


