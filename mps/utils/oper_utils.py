"""
  A collection of wrappers for optimisng a function.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from argparse import Namespace
from datetime import datetime
import numpy as np
import os
from warnings import warn
try:
  import utils.direct_fortran.direct as direct_ft_wrap
except ImportError:
  direct_ft_wrap = None
from utils.general_utils import map_to_bounds

# Various utilities for global optimisation of *cheap* functions on Euclidean domains ====
# A wrapper for all methods
def maximise_with_method(method, obj, domain, max_evals, return_history=False,
                         *args, **kwargs):
  """ A wrapper which calls one of the functions below based on the method. """
  if method.lower().startswith('rand') and domain.get_type() == 'euclidean':
    max_val, max_pt, history = \
      random_maximise(obj, domain.bounds, max_evals, return_history, *args, **kwargs)
  elif method.lower().startswith('direct') and domain.get_type() == 'euclidean':
    max_val, max_pt, history = \
      direct_ft_maximise(obj, domain.bounds, max_evals, return_history, *args, **kwargs)
  else:
    raise ValueError('Unknown maximisation method %s on domain %s.'%(
                     method, domain.get_type()))
  return max_val, max_pt, history

def minimise_with_method(method, obj, domain, max_evals, *args, **kwargs):
  """ A wrapper which calls maximise_with_method on the negative of the function. """
  neg_obj = lambda x: -obj(x)
  max_val, min_pt, history = maximise_with_method(method, neg_obj, domain, max_evals,
                                                  *args, **kwargs)
  min_val = - max_val
  return min_val, min_pt, history


# Random sampling -------------------------------
def random_sample(obj, bounds, max_evals, vectorised=True):
  """ Optimises a function by randomly sampling and choosing its maximum. """
  dim = len(bounds)
  rand_pts = map_to_bounds(np.random.random((int(max_evals), dim)), bounds)
  if vectorised:
    obj_vals = obj(rand_pts)
  else:
    obj_vals = np.array([obj(x) for x in rand_pts])
#   import matplotlib.pyplot as plt
#   plt.plot(rand_pts.ravel(), obj_vals.ravel(), '.')
#   plt.show()
  return rand_pts, obj_vals

# Random maximisation
def random_maximise(obj, bounds, max_evals, return_history=False, vectorised=True):
  """ Optimises a function by randomly sampling and choosing its maximum. """
  rand_pts, obj_vals = random_sample(obj, bounds, max_evals, vectorised)
  max_idx = obj_vals.argmax()
  max_val = obj_vals[max_idx]
  max_pt = rand_pts[max_idx]
  if return_history:
    history = Namespace(query_vals=obj_vals, query_points=rand_pts)
  else:
    history = None
  return max_val, max_pt, history

# DIRECT -----------------------------------------
# Some constants
_MAX_DIRECT_FN_EVALS = 2.6e6 # otherwise the fortran software complains

def direct_ft_minimise(obj, bounds, max_evals,
                       return_history=False,
                       eps=1e-5,
                       max_iterations=None,
                       alg_method=0,
                       fglobal=-1e100,
                       fglper=0.01,
                       volper=-1.0,
                       sigmaper=-1.0,
                       log_file_name='',
                       results_file_name='',
                       vectorised=False,
                       alternative_if_direct_not_loaded=None,
                      ):
  """
    A wrapper for the fortran implementation. The four mandatory arguments are self
    explanatory. If return_history is True it also returns the history of evaluations.
    max_iterations is the maximum number of iterations of the direct algorithm.
    I am not sure what the remaining arguments are for.
  """
  # pylint: disable=too-many-locals
  # pylint: disable=too-many-arguments

  if direct_ft_wrap is None:
    report_str = 'Attempted to use direct, but fortran library could not be imported.'
    if alternative_if_direct_not_loaded is None:
      report_str += ' Alternative not specified. Raising exception.'
      print report_str
      raise Exception(report_str)
    elif alternative_if_direct_not_loaded.lower().startswith('rand'):
      report_str += 'Using random optimiser.'
      warn(report_str)
      return random_maximise(obj, bounds, max_evals, vectorised)
    else:
      report_str += 'Unknown option for alternative_if_direct_not_loaded: %s'%(
                     alternative_if_direct_not_loaded)
      raise ValueError(report_str)

  # Preliminaries.
  max_evals = min(_MAX_DIRECT_FN_EVALS, max_evals) # otherwise the fortran sw complains.
  max_iterations = max_evals if max_iterations is None else max_iterations
  bounds = np.array(bounds, dtype=np.float64)
  lower_bounds = bounds[:, 0]
  upper_bounds = bounds[:, 1]
  if len(lower_bounds) != len(upper_bounds):
    raise ValueError('The dimensionality of the lower and upper bounds should match.')

  # Create a wrapper to comply with the fortran requirements.
  def _objective_wrap(x, *_):
    """ A wrapper to comply with the fortran requirements. """
    return (obj(x), 0)

  # Some dummy data to comply with the fortran requirements.
  iidata = np.ones(0, dtype=np.int32)
  ddata = np.ones(0, dtype=np.float64)
  cdata = np.ones([0, 40], dtype=np.uint8)

  # Set up results_file_name if necessary
  if return_history:
    results_file_name = (results_file_name if results_file_name else
                         'direct_results_%s'%(datetime.now().strftime('%m%d-%H%M%S')))
  # Call the function.
  min_pt, min_val, _ = direct_ft_wrap.direct(_objective_wrap,
                                             eps,
                                             max_evals,
                                             max_iterations,
                                             lower_bounds,
                                             upper_bounds,
                                             alg_method,
                                             log_file_name,
                                             results_file_name,
                                             fglobal,
                                             fglper,
                                             volper,
                                             sigmaper,
                                             iidata,
                                             ddata,
                                             cdata
                                            )
  if return_history:
    history = get_direct_history(results_file_name)
    os.remove(results_file_name)
  else:
    history = None
  # return
  return min_val, min_pt, history


def direct_ft_maximise(obj, bounds, max_evals, *args, **kwargs):
  """
    A wrapper for maximising a function which calls direct_ft_minimise. See arguments
    under direct_ft_minimise for more details.
  """
  min_obj = lambda x: -obj(x)
  min_val, max_pt, history = direct_ft_minimise(min_obj, bounds, max_evals,
                                                *args, **kwargs)
  max_val = - min_val
  if history is not None:
    history.curr_opt_vals = -history.curr_opt_vals
    history.query_vals = -history.query_vals
  return max_val, max_pt, history


def get_direct_history(results_file_name):
  """ Reads the optimisation history from the direct results file. """
  curr_opt_vals = []
  query_vals = []
  query_points = []
  results_file_handle = open(results_file_name, 'r')
  for line in results_file_handle.readlines():
    words = line.strip().split()
    if words[0].isdigit():
      curr_opt_vals.append(float(words[1]))
      query_vals.append(float(words[2]))
      query_points.append([float(w) for w in words[3:]])
  return Namespace(curr_opt_vals=np.array(curr_opt_vals),
                   query_vals=np.array(query_vals),
                   query_points=np.array(query_points))


def get_history_from_direct_log(log_file_name):
  """ Returns the history from the direct log file. """
  saved_iterations = [0]
  saved_max_vals = [-np.inf]
  phase = 'boiler'
  log_file_handle = open(log_file_name, 'r')
  for line in log_file_handle.readlines():
    words = line.strip().split()
    if phase == 'boiler':
      if words[0] == 'Iteration':
        phase = 'results'
    elif phase == 'results':
      if len(words) == 3 and words[0].isdigit():
        saved_iterations.append(int(words[1]))
        saved_max_vals.append(-float(words[2]))
      else:
        phase = 'final'
    elif phase == 'final':
      if words[0] == 'Final':
        saved_max_vals.append(max(-float(words[-1]), saved_max_vals[-1]))
        # doing max as the fortran library rounds off the last result for some reason.
      if words[0] == 'Number':
        saved_iterations.append(int(words[-1]))
  # Now fill in the rest of the history.
  curr_opt_vals = np.zeros((saved_iterations[-1]), dtype=np.float64)
  for i in range(len(saved_iterations)-1):
    curr_opt_vals[saved_iterations[i]:saved_iterations[i+1]] = saved_max_vals[i]
  curr_opt_vals[-1] = saved_max_vals[-1]
  return Namespace(curr_opt_vals=curr_opt_vals)

# DIRECT end --------------------------------------------

# Utilities for sampling from combined domains ===================================
def random_sample_from_discrete_domain(dscr_vals, num_points=None):
  """ Samples from a discrete domain. """
  def _sample_single_point():
    """ Samples a single point. """
    return [np.random.choice(categ) for categ in dscr_vals]
  # Now draw num_points of them
  num_points_to_sample = 1 if num_points is None else num_points
  if len(dscr_vals) == 0:
    ret = [[]] * num_points_to_sample
  else:
    ret = [_sample_single_point() for _ in range(num_points_to_sample)]
  if num_points is None:
    return ret[0]
  else:
    return ret

def random_sample_cts_dscr(obj, cts_bounds, dscr_vals, max_evals, vectorised=True):
  """ Sample from a joint continuous and discrete space. """
  dim = len(cts_bounds)
  cts_rand_pts = map_to_bounds(np.random.random((int(max_evals), dim)), cts_bounds)
  dscr_rand_pts = random_sample_from_discrete_domain(dscr_vals, max_evals)
  if vectorised:
    obj_vals = obj(cts_rand_pts, dscr_rand_pts)
  else:
    obj_vals = np.array([obj(cx, dx) for (cx, dx) in zip(cts_rand_pts, dscr_rand_pts)])
  return cts_rand_pts, dscr_rand_pts, obj_vals

# Random maximisation
def random_maximise_cts_dscr(obj, cts_bounds, dscr_vals, max_evals,
                             return_history=False, vectorised=True):
  """ Optimises a function by randomly sampling and choosing its maximum. """
  cts_rand_pts, dscr_rand_pts, obj_vals = random_sample_cts_dscr(obj, cts_bounds,
                                            dscr_vals, max_evals, vectorised)
  max_idx = obj_vals.argmax()
  max_val = obj_vals[max_idx]
  max_cts_pt = cts_rand_pts[max_idx]
  max_dscr_pt = dscr_rand_pts[max_idx]
  if return_history:
    history = Namespace(query_vals=obj_vals, query_cts_points=cts_rand_pts,
                        query_dscr_points=dscr_rand_pts)
  else:
    history = None
  return max_val, max_cts_pt, max_dscr_pt, history

