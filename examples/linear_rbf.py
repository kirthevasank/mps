"""
  Unit test for PSExperimentDesigner with LinearRBF.
  -- rrz@andrew.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-name-in-module
# pylint: disable=abstract-method

import numpy as np
try:
  from scipy.stats import multivariate_normal
except ImportError:
  from numpy.random import multivariate_normal
# Local
from mps.exd.domains import EuclideanDomain
from mps.exd.goal_oriented_exd import GoalOrientedExperimentDesigner # args
from mps.exd.worker_manager import SyntheticWorkerManager
from mps.exd.experiment_caller import EuclideanFunctionCaller
from mps.prob.disc_prob_examples import BayesianLinearRBF
from mps.policies import mps
from mps.policies import random


def get_problem_params(options=None, reporter=None):
  """ Returns problem parameters. Picks centers in [0, 1]^2. """
  n1 = 2 # number of centers in the bottom left quadrant = n1 ^ 2
  n2 = 1 # number of centers in each of the other 3 quadrants = n2 ^ 2
  grid_botleft = np.linspace(0, 0.5, n1 + 2)[1:-1]
  centers_botleft = np.array([[x, y] for y in grid_botleft for x in grid_botleft])
  grid_else = np.linspace(0, 0.5, n2 + 2)[1:-1]
  centers_else = np.array([[x, y] for y in grid_else for x in grid_else])
  centers_topright = centers_else + 0.5
  centers_topleft = centers_else + np.array([0, 0.5])
  centers_botright = centers_else + np.array([0.5, 0])
  centers = np.concatenate((centers_botleft, centers_topright, centers_topleft,
                            centers_botright), axis=0)
  def _weight_fn(pt):
    """ A toy function that determines the weights for each center. """
    x, y = pt
    return np.sin(4 * np.pi * (x**2 + y))
  true_theta = np.array([_weight_fn(c) for c in centers])
  x_domain = EuclideanDomain([[0.0, 1.0], [0.0, 1.0]])
  # prior mean and covariance
  prior_info = (np.zeros(len(centers)), np.eye(len(centers)))
  rbf_var = 0.1
  eta2 = 0.001
  model = BayesianLinearRBF(centers, rbf_var, eta2, x_domain, None, prior_info,
                            options=options, reporter=reporter)
  experiment_eval_func = lambda x: model.sample_y_giv_x_t(1, x, true_theta)[0]
  experiment_caller = EuclideanFunctionCaller(experiment_eval_func, x_domain,
                                              'linear_rbf')
  return true_theta, model, experiment_caller


def compute_least_squares_est(centers, cov, X, Y, regularize=True):
  """ Compute regularized least-squares estimate for theta given X, Y. """
  gaussians = [multivariate_normal(mean=c, cov=cov) for c in centers]
  densities = np.array([g.pdf(X) for g in gaussians]).T # n x d, where d is num centers
  if len(densities.shape) == 1:
    densities = np.reshape(densities, (1, len(densities)))
  # compute least-squares weights with or without regularization
  if regularize:
    if len(Y) == 1:
      reg_lambda = 0.1
    else:
      reg_lambda = np.std(Y) / (10 * len(X))
    densities_T_densities = densities.T.dot(densities)
    reg_term = reg_lambda * np.eye(len(densities_T_densities))
    return np.linalg.lstsq(densities_T_densities + reg_term, densities.T.dot(Y))[0]
  else:
    return np.linalg.lstsq(densities, Y)[0]


class LinearRBFProblem(GoalOrientedExperimentDesigner):
  """ Describes the problem for active learning. """

  def __init__(self, experiment_caller, worker_manager, model, true_theta,
               options=None, reporter=None, *args, **kwargs):
    """ Constructor. """
    self.true_theta = true_theta
    super(LinearRBFProblem, self).__init__(experiment_caller, worker_manager, model,
         self._penalty, self._true_penalty, options=options, reporter=reporter,
         *args, **kwargs)

  def _penalty(self, theta, X, Y):
    """ The penalty function. """
    if len(X) == 0:
      return np.inf
    raw_X = self.experiment_caller.get_raw_domain_coords(X)
    est_theta = compute_least_squares_est(self.model.centers, self.model.var, raw_X, Y)
    norm_err = (theta - est_theta) / (self.true_theta + 0.001)
    ret = np.linalg.norm(norm_err)**2
    return ret

  def _true_penalty(self, X, Y):
    """ The True penalty. """
    return self._penalty(self.true_theta, X, Y)


# The following classes inherit the problem and policy classes ==========================
class LinearRBFActiveLearnerMPS(LinearRBFProblem, mps.MPSExperimentDesigner):
  """ Active Learning on the LinearRBF Model with Posterior Sampling. """
  pass

class LinearRBFActiveLearnerGO(LinearRBFProblem, mps.GOExperimentDesigner):
  """ Active Learning on the LinearRBF Model with Posterior Sampling using the
      Oracle policy. """
  pass

class LinearRBFActiveLearnerRandom(LinearRBFProblem,
                                   random.EuclideanRandomExperimentDesigner):
  """ Random Designer on the LinearRBF problem. """
  pass


def main():
  """ Main function. """
  budget = 40
  true_theta, model, experiment_caller = get_problem_params()
  worker_manager = SyntheticWorkerManager(1)

  # Posterior sampling
  mps_designer = LinearRBFActiveLearnerMPS(experiment_caller, worker_manager,
                                           model, true_theta)
  mps_designer.run_experiments(budget)


if __name__ == '__main__':
  main()

