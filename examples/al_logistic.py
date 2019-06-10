"""
  An active learning example on a 1D parametric problem.
  -- kandasamy@cs.cmu.edu

  To execute this example, you will need to install the Edward probabilistic
  programming language (http://edwardlib.org).
  See http://edwardlib.org/iclr2017 for instructions.
"""

# pylint: disable=invalid-name
# pylint: disable=no-name-in-module
# pylint: disable=abstract-method

from argparse import Namespace
import numpy as np
from scipy.optimize import minimize
# Local
from mps.exd.domains import EuclideanDomain
from mps.exd.goal_oriented_exd import GoalOrientedExperimentDesigner #guided_exd_args
from mps.exd.experiment_caller import EuclideanFunctionCaller
from mps.prob.disc_prob_examples import BayesianLogisticWithGaussianNoise
from mps.policies import mps
from mps.policies import random


def get_problem_params(options=None, reporter=None):
  """ Returns model parameters. """
  a = 2.1
  b = 7
  c = 6
  eta2 = 0.01
  true_theta = np.array([a, b, c, eta2])
  x_domain = EuclideanDomain([[0, 10]])
  prior_info = {
    'a': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=2.0, sigma=0.2),
    'b': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=6.0, sigma=2.0),
    'c': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=5.0, sigma=2.0),
    'eta2': Namespace(distro='const_1d', vi_distro='const_1d', const=eta2),
  }
  model = BayesianLogisticWithGaussianNoise(x_domain, None, prior_info,
                                            options=options, reporter=reporter)
  experiment_eval_func = lambda x: model.sample_y_giv_x_t(1, x, true_theta)[0]
  experiment_caller = EuclideanFunctionCaller(experiment_eval_func, x_domain,
                      'gauss_logistic')
  return true_theta, model, experiment_caller


def compute_lwn_least_squares_est(X, Y, min_successes=20):
  """ Tries optimizing using different starting points until a minimum number
      of successful points are reached.
      Model is specified a function model_pred(params, X) that returns the predicted
      Y for X given by a model specified by params.
      Function choose_init_pt chooses a random initial point.
  """
  # Model prediction ---------------------------------------------
  def _model_pred(params):
    """ Computes the sum-of-squares error for params. """
    a, b, c = params
    return a / (1 + np.exp(b * (X.ravel() - c)))
  # Sum of squared errors ----------------------------------------
  def _sse(params):
    """ Computes the sum-of-squares error for params. """
    y_preds = _model_pred(params)
    return np.sum(np.square(Y - y_preds))
  # Choose the init point ----------------------------------------
  def _choose_init_pt():
    """ Chooses an initial point. """
    return np.random.random((3,)) * np.array([4, 10, 1])
  locs = [] # locations of points we reach from optimization
  vals = [] # function values at locs
  iters_ran = 0
  max_iters = 10 * min_successes
  while len(locs) < min_successes:
    x0 = _choose_init_pt()
    optim_result = minimize(_sse, x0, method='L-BFGS-B')
    if optim_result.success:
      locs.append(optim_result.x)
      vals.append(optim_result.fun)
    iters_ran += 1
    if iters_ran == max_iters:
      break
  if len(vals) > 0:
    i_min = np.argmin(vals)
    best_abc = locs[i_min]
  else:
    best_abc = optim_result.x
  best_sse = _sse(best_abc)
  eta2 = best_sse / float(len(X))
  ret = list(best_abc) + [eta2]
  return np.array(ret)


class GaussLogisticProblem(GoalOrientedExperimentDesigner):
  """ Describe problem for Surfactant based active Learning. """

  def __init__(self, experiment_caller, worker_manager, model, true_theta,
               options=None, reporter=None, *args, **kwargs):
    """ Constructor. """
    self.true_theta = true_theta
    super(SurfactantProblem, self).__init__(experiment_caller, worker_manager,
      model, self._penalty, self._true_penalty, options=options, reporter=reporter,
      *args, **kwargs)

  def _penalty(self, theta, X, Y):
    """ The penalty function. """
    raw_X = self.experiment_caller.get_raw_domain_coords(X)
    est_theta = compute_lwn_least_squares_est(raw_X, Y)
    norm_err = (theta - est_theta) / (self.true_theta + 0.001)
    ret = np.linalg.norm(norm_err)**2
    return ret

  def _true_penalty(self, X, Y):
    """ The True penalty. """
    return self._penalty(self.true_theta, X, Y)


class GaussLogisticActiveLearnerMPS(GaussLogisticProblem, mps.MPSExperimentDesigner):
  """ Active Learning on the Bayesian Logistic Model with Posterior Sampling. """
  pass

class GaussLogisticActiveLearnerMO(GaussLogisticProblem,
                                   mps.MyopicOracleExperimentDesigner):
  """ Active Learning on the Bayesian Logistic Model with the Oracle policy. """
  pass

class GaussLogisticActiveLearnerRandom(GaussLogisticProblem,
                                       random.EuclideanRandomExperimentDesigner):
  """ Random Designer on GaussLogistic problem. """
  pass


def main():
  """ Main function. """
  budget = 40
  true_theta, model, experiment_caller = get_problem_params()
  worker_manager = SyntheticWorkerManager(1)

  # Random sampling
  print('\nRandom designer:')
  worker_manager.reset()
  rand_options = load_options_for_policy('rand')
  rand_designer = GaussLogisticActiveLearnerRandom(experiment_caller, worker_manager,
                    model, true_theta, options=rand_options)
  rand_designer.run_experiments(budget)

  # Random sampling
  print('\nOracle designer:')
  worker_manager.reset()
  mo_options = load_options_for_policy('mo')
  mo_designer = GaussLogisticActiveLearnerMO(experiment_caller, worker_manager,
                                             model, true_theta, options=mo_options)
  mo_designer.run_experiments(budget)

  # Posterior sampling
  print('\nMPS designer:')
  worker_manager.reset()
  mps_options = load_options_for_policy('mps')
  mps_designer = GaussLogisticActiveLearnerMPS(experiment_caller, worker_manager,
                                               model, true_theta, options=mps_options)
  mps_designer.run_experiments(budget)


if __name__ == '__main__':
  main()

