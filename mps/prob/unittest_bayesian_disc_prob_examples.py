"""
  Unit tests for the example Bayesian discriminiative models in disc_prob_examples.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

from argparse import Namespace
import numpy as np
from matplotlib import pyplot as plt
# Local
from . import disc_prob_examples as dpe
from ..utils.base_test_class import BaseTestClass, execute_tests

class BayesianDiscModelBaseTestCase(object):
  """ Implements tests for all child classes. """

  def setUp(self):
    """ Initial set up. """
    self.num_samples = 1000
    self.num_post_samples = 10
    self.model = None
    self.X = None
    self.T = None
    self.data = Namespace(X=None, Y=None, true_model=None)
    self._test_visualisation = False
    self._child_set_up()

  def test_mean_std(self):
    """ Test mean and std for y given x and t. """
    self.report('Testing sizes of returned mean and std for %s.'%(self.model))
    mean_vals, std_vals = self.model.eval_y_giv_x_t(self.X, self.T, uncert_form='std')
    assert len(mean_vals) == len(self.X)
    assert len(std_vals) == len(self.X)

  def test_discriminative_sampling(self):
    """ Test sampling. """
    self.report('Testing discriminative sampling for %s.'%(self.model))
    samples = self.model.sample_y_giv_x_t(self.num_samples, self.X, self.T)
    mean_vals, std_vals = self.model.eval_y_giv_x_t(self.X, self.T, uncert_form='std')
    sample_mean = np.array(samples).mean(axis=0)
    sample_std = np.array(samples).std(axis=0)
    tolerance = 5 * np.linalg.norm(std_vals) / np.sqrt(self.num_samples)
    err_mean = np.linalg.norm(sample_mean - np.array(mean_vals))
    err_std = np.linalg.norm(sample_std - np.array(std_vals))
    self.report(' err_mean=%0.4f, err_std=%0.4f, tol=%0.4f'%(
                err_mean, err_std, tolerance), 'test_result')
    assert err_mean <= tolerance
    assert err_std <= tolerance

  def test_posterior_sampling(self):
    """ Test posterior sampling. """
    self.report('Testing posterior sampling for %s.'%(self.model))
    t_samples = self.model.sample_t_giv_data(self.num_post_samples, self.data.X,
                                             self.data.Y, inference_num_iters=100)
    assert len(t_samples) == self.num_post_samples
    for t_sample in t_samples[:10]:
      assert self.model.t_domain.is_a_member(t_sample)

  def test_visualisation(self):
    """ Testing visualisation. """
    if self._test_visualisation:
      self.report('Testing visualisation for %s.'%(self.model))
      self.visualise()


class BayesianLogisticWithGaussianNoiseTestCase(BayesianDiscModelBaseTestCase,
                                                BaseTestClass):
  """ Unit test for dpe.BayesianLogisticWithGaussianNoise. """

  def _child_set_up(self):
    """ Set up. """
    prior_info = {
      # Prior option 1 ----------
      'a': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=1.0, sigma=0.1),
      'b': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=4.0, sigma=1.0),
      'c': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=5.0, sigma=1.0),
      'eta2': Namespace(distro='inverse_gamma', vi_distro='inverse_gamma',
                        concentration=10000.0, rate=1.0),
#       # Prior option 2: more concentrated around the truth ------------
#       'a': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=1.0, sigma=0.01),
#       'b': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=3.0, sigma=0.01),
#       'c': Namespace(distro='normal_1d', vi_distro='normal_1d', mu=4.4, sigma=0.2),
#       'eta2': Namespace(distro='const_1d', vi_distro='const_1d', const=0.0001),
      }
    self.model = dpe.BayesianLogisticWithGaussianNoise(None, None, prior_info)
    num_points = 13
    self.X = 10 * np.random.random((num_points,))
    self.T = 4 * np.random.random((num_points, 4))
    self.T[:, 3] = 0.0001 * self.T[:, 3]
    # Create Data
    self.data.true_model = dpe.LogisticWithGaussianNoise(1, 3, 4.5, 0.0001)
    num_tr_data = 4
    self.data.X = np.reshape(np.linspace(0, 10, num_tr_data + 2), (num_tr_data + 2, 1))
    self.data.X = self.data.X[1:-1]
    self.data.Y = np.array(self.data.true_model.sample_y_giv_x(1, self.data.X)[0])
    self._test_visualisation = True

  def visualise(self):
    """ Visualise the model. """
    t_post_samples = self.model.sample_t_giv_data(self.num_post_samples, self.data.X,
                                             self.data.Y, inference_num_iters=3000)
    t_prior_samples = self.model.sample_t_from_prior(self.num_post_samples)
    print t_post_samples
    grid = np.linspace(0, 10, 100)
    grid_arr = np.reshape(grid, (len(grid), 1))
    # First figure, the data, true function and the samples
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title('posterior')
    ax.plot(self.data.X.ravel(), self.data.Y, 'kx', alpha=1.0, label='data')
    ax.plot(grid, self.data.true_model.eval_y_mean_giv_x(grid_arr), 'k', alpha=1,
            label='true')
    prior_sample_curves = np.array(
                          [self._get_sample_curves_on_grid(grid_arr, sample)
                          for sample in t_prior_samples])
    post_sample_curves = np.array(
                          [self._get_sample_curves_on_grid(grid_arr, sample)
                          for sample in t_post_samples])
    ax.plot(grid, prior_sample_curves[0].T, 'g', lw=2, alpha=0.3, label='prior_draws')
    ax.plot(grid, prior_sample_curves[1:].T, 'g', lw=2, alpha=0.3)
    ax.plot(grid, post_sample_curves[0].T, 'r', lw=2, alpha=0.3, label='posterior_draws')
    ax.plot(grid, post_sample_curves[1:].T, 'r', lw=2, alpha=0.3)
    ax.legend()
    # Second figure, histograms for each parameter.
    plt.show()

  @classmethod
  def _get_sample_curves_on_grid(cls, grid_arr, sample):
    """ Returns the grid values of the sample. """
    sample_model = dpe.LogisticWithGaussianNoise(sample[0], sample[1],
                     sample[2], sample[3])
    return sample_model.eval_y_mean_giv_x(grid_arr)


if __name__ == '__main__':
  execute_tests(5453)

