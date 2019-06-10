"""
  Utilities for the policy modules.
  -- kirthevasank
"""

from .mps import mps_args, mo_args
from .random import rand_args
from ..utils.option_handler import load_options
from ..exd.goal_oriented_exd import goal_oriented_exd_args

def load_options_for_policy(policy, policy_options='default', problem_options=None,
                            reporter=None):
  """ Load options for policy. """
  if problem_options is None:
    problem_options = []
  # determine policy options
  if policy_options is None or policy_options == 'default':
    if policy == 'mps':
      list_of_policy_options = mps_args
    elif policy == 'mo':
      list_of_policy_options = mo_args
    elif policy == 'rand':
      list_of_policy_options = rand_args
    else:
      raise ValueError('Unknown policy %s.'%(policy))
  else:
    list_of_policy_options = policy_options
  list_of_options = list_of_policy_options + problem_options + goal_oriented_exd_args
  return load_options(list_of_options, reporter=reporter)

