"""
  Unit tests for the example discriminiative models in disc_prob_examples.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

import numpy as np
from matplotlib import pyplot as plt
# Local
from . import disc_prob_examples as dpe
from ..utils.base_test_class import BaseTestClass, execute_tests

class DiscModelBaseTestCase(object):
  """ Implements tests for all child classes. """

  def setUp(self):
    """ Initial set up. """
    self.num_samples = 100
    self.model = None
    self.X = None
    self._test_visualisation = False
    self._child_set_up()

  def test_mean_std(self):
    """ Test mean and std. """
    self.report('Testing sizes of returned mean and std for %s.'%(self.model))
    mean_vals, std_vals = self.model.eval_y_giv_x(self.X, uncert_form='std')
    assert len(mean_vals) == len(self.X)
    assert len(std_vals) == len(self.X)

  def test_sampling(self):
    """ Tests sampling. """
    self.report('Testing returned samples for %s.'%(self.model))
    samples = self.model.sample_y_giv_x(self.num_samples, self.X)
    mean_vals, std_vals = self.model.eval_y_giv_x(self.X, uncert_form='std')
    sample_mean = np.array(samples).mean(axis=0)
    sample_std = np.array(samples).std(axis=0)
    tolerance = 5 * np.linalg.norm(std_vals) / np.sqrt(self.num_samples)
    err_mean = np.linalg.norm(sample_mean - np.array(mean_vals))
    err_std = np.linalg.norm(sample_std - np.array(std_vals))
    self.report(' err_mean=%0.4f, err_std=%0.4f, tol=%0.4f'%(
                err_mean, err_std, tolerance), 'test_result')
    assert err_mean <= tolerance
    assert err_std <= tolerance

  def test_visualisation(self):
    """ Tests visualisation. """
    if self._test_visualisation:
      self.report('Testing visualisation for %s.'%(self.model))
      self.visualise()


class LogisticWithGaussianNoiseTestCase(DiscModelBaseTestCase, BaseTestClass):
  """ Unit tests for LogisticWithGaussianNoise. """

  def _child_set_up(self):
    """ set up. """
    a = 1
    b = 3
    c = 4.5
    eta2 = 0.01
    self.model = dpe.LogisticWithGaussianNoise(a, b, c, eta2)
    self.X = 10 * np.random.random((30, 1))
    self._test_visualisation = True

  def visualise(self):
    """ Visualises the model. """
    X = np.linspace(0, 10, 100)
    mean_vals = self.model.eval_y_mean_giv_x(X)
    samples = self.model.sample_y_giv_x(5, X)
    plt.plot(X, mean_vals)
    plt.plot(X, np.array(samples).T, '.')
    plt.title(str(self.model))
    plt.show()


if __name__ == '__main__':
  execute_tests()

