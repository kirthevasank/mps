"""
  A unit test for calling multiple functions with the same domain.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

import numpy as np
# Local imports
import utils.euclidean_synthetic_functions as esf
from utils.base_test_class import BaseTestClass, execute_tests

_TOL = 1e-5


class EuclideanMultiFunctionSyntheticTestCase(BaseTestClass):
  """ Unit tests for multi-functions. """

  def setUp(self):
    """ prior set up. """
    self.test_function_data = [('hartmann', 3, None, 'no_noise', None, 2),
                               ('hartmann3', None, None, 'no_noise', None, 1),
                               ('hartmann6', None, None, 'no_noise', None, 3),
                               # With noise
                               ('branin', None, None, 'gauss', 0.1, 3),
                               ('borehole', None, None, 'gauss', 10.0, 4),
                               # TODO: add multi-fidelity here
                              ]
    self.num_test_points = 3

  @classmethod
  def _test_ret_val_sizes(cls, multifunc_caller, unnorm_multifunc_caller, num_multifuncs):
    """ Tests the number of function in the function caller. """
    dom_point = np.random.random((multifunc_caller.domain.dim,))
    raw_dom_point = multifunc_caller.get_raw_domain_coords(dom_point)
    norm_caller_val, _ = multifunc_caller.eval_single(dom_point)
    unnorm_caller_val, _ = unnorm_multifunc_caller.eval_single(raw_dom_point)
    assert len(multifunc_caller.funcs) == num_multifuncs
    assert len(unnorm_multifunc_caller.funcs) == num_multifuncs
    assert len(norm_caller_val) == num_multifuncs
    assert len(unnorm_caller_val) == num_multifuncs

  def _test_func_vals(self, multifunc_caller, unnorm_multifunc_caller):
    """ Tests the function value. """
    dom_points = np.random.random((self.num_test_points, multifunc_caller.domain.dim))
    raw_dom_points = multifunc_caller.get_raw_domain_coords(dom_points)
    norm_vals = [multifunc_caller.eval_single(pt)[0] for pt in dom_points]
    unnorm_vals = [unnorm_multifunc_caller.eval_single(pt)[0] for pt in raw_dom_points]
    norm_vals = np.array(norm_vals)
    unnorm_vals = np.array(unnorm_vals)
    err = np.linalg.norm(norm_vals - unnorm_vals)
    num_evals = norm_vals.size
    tolerance = 0 if multifunc_caller.noise_scale is None else \
                3 * multifunc_caller.noise_scale * num_evals
    assert err <= tolerance
    assert norm_vals.shape[1] == len(multifunc_caller.funcs)
    assert unnorm_vals.shape[1] == len(multifunc_caller.funcs)

  def test_all_synthetic_multifunctions(self):
    """ Tests all synthetic functions in a loop. """
    for idx, (func_name, domain_dim, fidel_dim, noise_type, noise_scale, num_multifuncs) \
      in enumerate(self.test_function_data):
      sf_or_mf = 'sf' if fidel_dim is None else 'mf'
      self.report(('Test %d/%d: %s(%s), num_multifuncs: %d, dom_dim:%s, fidel_dim:%s, ' +
                   'noise(%s, %s).')%(idx+1, len(self.test_function_data),
                  func_name, sf_or_mf, num_multifuncs, domain_dim, fidel_dim,
                  noise_type, noise_scale))
      multifunc_caller = esf.get_syn_func_caller(func_name, domain_dim, fidel_dim,
                                                 noise_type, noise_scale,
                                                 to_normalise_domain=True,
                                                 num_multifuncs=num_multifuncs)
      unnorm_multifunc_caller = esf.get_syn_func_caller(func_name, domain_dim, fidel_dim,
                                                         noise_type, noise_scale,
                                                         to_normalise_domain=False,
                                                         num_multifuncs=num_multifuncs)
      self._test_ret_val_sizes(multifunc_caller, unnorm_multifunc_caller, num_multifuncs)
      self._test_func_vals(multifunc_caller, unnorm_multifunc_caller)



if __name__ == '__main__':
  execute_tests()

