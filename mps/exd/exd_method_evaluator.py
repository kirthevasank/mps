"""
  Harness for conducting black box optimisation evaluations.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used
# pylint: disable=abstract-method
# pylint: disable=star-args
# pylint: disable=maybe-no-member

from argparse import Namespace
from datetime import datetime
import numpy as np
import os
# Local imports
from .exd_utils import get_euclidean_initial_qinfos
from ..policies.initialiser import Initialiser
from ..utils.method_evaluator import BaseMethodEvaluator


class ExDMethodEvaluator(BaseMethodEvaluator):
  """ Base class for evaluating methods for optimisation. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, study_name, experiment_caller, worker_manager, max_capital, methods,
               num_trials, save_dir, evaluation_options, save_file_prefix='',
               method_options=None, reporter=None, **kwargs):
    """ Constructor. Also see BasicExperimenter for more args. """
    # pylint: disable=too-many-arguments
    save_file_name = self._get_save_file_name(save_dir, study_name,
      worker_manager.num_workers, save_file_prefix, worker_manager.get_time_distro_info(),
      max_capital)
    super(ExDMethodEvaluator, self).__init__(study_name, num_trials,
                                             save_file_name, reporter=reporter, **kwargs)
    self.experiment_caller = experiment_caller
    self.worker_manager = worker_manager
    self.max_capital = float(max_capital)
    # Methods
    self.methods = methods
    self.num_methods = len(methods)
    self.domain = experiment_caller.domain
    self.method_options = (method_options if method_options else
                           {key: None for key in methods})
    # Experiment options will have things such as if the evaluations are noisy,
    # the time distributions etc.
    self.evaluation_options = evaluation_options
    self._set_up_saving()

  @classmethod
  def _get_save_file_name(cls, save_dir, study_name, num_workers, save_file_prefix,
                          time_distro_str, max_capital):
    """ Gets the save file name. """
    save_file_prefix = save_file_prefix if save_file_prefix else study_name
    save_file_name = '%s-M%d-%s-c%d-%s.mat'%(save_file_prefix, num_workers,
      time_distro_str, int(max_capital), datetime.now().strftime('%m%d-%H%M%S'))
    save_file_name = os.path.join(save_dir, save_file_name)
    return save_file_name

  def _set_up_saving(self):
    """ Runs some routines to set up saving. """
    # Store methods and the options in to_be_saved
    self.to_be_saved.max_capital = self.max_capital
    self.to_be_saved.num_workers = self.worker_manager.num_workers
    self.to_be_saved.methods = self.methods
    self.to_be_saved.method_options = self.method_options # Some error created here.
    self.to_be_saved.time_distro_str = self.worker_manager.get_time_distro_info()
    # Data about the problem
    self.to_be_saved.true_maxval = (self.experiment_caller.maxval
      if self.experiment_caller.maxval is not None else -np.inf)
    self.to_be_saved.true_argmax = (self.experiment_caller.argmax
      if self.experiment_caller.argmax is not None else 'not-known')
    self.to_be_saved.domain_type = self.domain.get_type()
#     # For the results
    self.data_to_be_saved = ['query_step_idxs',
                             'query_points',
                             'query_vals',
                             'query_true_vals',
                             'query_send_times',
                             'query_receive_times',
                             'query_eval_times',
                             'query_worker_ids',
                             'curr_penalties',
                             'curr_best_penalties',
                             'num_jobs_per_worker',
                            ]
    self.data_to_be_saved_if_available = ['query_fidels',
                                          'query_cost_at_fidels',
                                          'query_at_fidel_to_opts']
    self.data_not_to_be_mat_saved.extend(['method_options', 'query_points',
                                          'curr_opt_points', 'curr_true_opt_points'])
    self.data_not_to_be_pickled.extend(['method_options'])
    for data_type in self.data_to_be_saved:
      setattr(self.to_be_saved, data_type, self._get_new_empty_results_array())
    for data_type in self.data_to_be_saved_if_available:
      setattr(self.to_be_saved, data_type, self._get_new_empty_results_array())

  def _get_new_empty_results_array(self):
    """ Returns a new empty arrray to be used for saving results. """
#     return np.empty((self.num_methods, 0), dtype=np.object)
    return np.array([[] for _ in range(self.num_methods)], dtype=np.object)

  def _get_new_iter_results_array(self):
    """ Returns an empty array to be used for saving results of current iteration. """
#     return np.empty((self.num_methods, 1), dtype=np.object)
    return np.array([['-'] for _ in range(self.num_methods)], dtype=np.object)

  def _print_method_header(self, full_method_name):
    """ Prints a header for the current method. """
    trial_header = '-- Exp %d/%d on %s:: %s with cap %0.4f. ----------------------'%(
      self.trial_iter, self.num_trials, self.study_name, full_method_name,
      self.max_capital)
    self.reporter.writeln(trial_header)

  def get_iteration_header(self):
    """ Header for iteration. """
    noisy_str = ('no-noise' if self.experiment_caller.noise_type == 'no_noise' else
                 'noisy(%0.2f)'%(self.experiment_caller.noise_scale))
    maxval_str = ('?' if self.experiment_caller.maxval is None
                       else '%0.5f'%(self.experiment_caller.maxval))
    ret = '%s (M=%d), td: %s, max=%s, max-capital %0.2f, %s'%(self.study_name,
      self.worker_manager.num_workers, self.to_be_saved.time_distro_str, maxval_str,
      self.max_capital, noisy_str)
    return ret

  def _print_method_result(self, method, final_penalty, best_penalty, num_evals):
    """ Prints the result for this method. """
    final_penalty_str = '%0.5f'%(final_penalty) if isinstance(final_penalty, float) \
                        else str(final_penalty)
    best_penalty_str = '%0.5f'%(best_penalty) if isinstance(best_penalty, float) \
                        else str(best_penalty)
    result_str = 'Method: %s achieved penalty final:%s, best: %s in %d evaluations.\n'%(
      method, final_penalty_str, best_penalty_str, num_evals)
    self.reporter.writeln(result_str)

  def _get_prev_eval_qinfos(self):
    """ Gets the initial qinfos for all methods. """
    if self.evaluation_options.prev_eval_points == 'generate':
      init_pool_qinfos = self._get_initial_pool_qinfos()
    else:
      # Load from the file.
      raise NotImplementedError('Not written reading results from file yet.')
    # Create an initialiser
    initialiser = Initialiser(self.experiment_caller, self.worker_manager)
    initialiser.options.get_initial_qinfos = lambda _: init_pool_qinfos
    initialiser.options.max_num_steps = 0
    init_hist = initialiser.initialise()
    return init_hist.query_qinfos

  def _get_initial_pool_qinfos(self):
    """ Returns the intial pool to bootstrap methods in one evaluation. """
    raise NotImplementedError('Implement in a child class.')

  def _run_method_on_experiment_caller(self, method, experiment_caller,
                              worker_manager, max_capital, meth_options, reporter):
    """ Run method on the function caller and return. """
    raise NotImplementedError('Implement in a child class!')

  def run_trial_iteration(self):
    """ Runs each method in self.methods once and stores the results to be saved. """
    curr_iter_results = Namespace()
    for data_type in self.data_to_be_saved:
      setattr(curr_iter_results, data_type, self._get_new_iter_results_array())
    for data_type in self.data_to_be_saved_if_available:
      setattr(curr_iter_results, data_type, self._get_new_iter_results_array())

    # Fetch pre-evaluation points.
    self.worker_manager.reset()
    prev_eval_qinfos = self._get_prev_eval_qinfos()
    prev_eval_vals = [qinfo.val for qinfo in prev_eval_qinfos]
    self.reporter.writeln('Using %d pre-eval points with values. eval: %s.'%(
                          len(prev_eval_qinfos), prev_eval_vals))

    # Will go through each method in this loop.
    for meth_iter in range(self.num_methods):
      curr_method = self.methods[meth_iter]
      curr_meth_options = self.method_options[curr_method]
      # Set prev_eval points and vals
      curr_meth_options.prev_evaluations = Namespace(qinfos=prev_eval_qinfos)
      # Reset worker manager
      self.worker_manager.reset()
      self.reporter.writeln(
        '\nResetting worker manager: worker_manager.experiment_designer:%s'%(
        str(self.worker_manager.experiment_designer)))

      # Call the method here.
      self._print_method_header(curr_method)
      history = self._run_method_on_experiment_caller(curr_method, self.experiment_caller,
                  self.worker_manager, self.max_capital, curr_meth_options, self.reporter)

      # Now save results for current method
      for data_type in self.data_to_be_saved:
        data = getattr(history, data_type)
        data_pointer = getattr(curr_iter_results, data_type)
        data_pointer[meth_iter, 0] = data
      for data_type in self.data_to_be_saved_if_available:
        if hasattr(history, data_type):
          data = getattr(history, data_type)
        else:
          data = ['xx'] * len(history.query_points)
        data_pointer = getattr(curr_iter_results, data_type)
        data_pointer[meth_iter, 0] = data
      # Print out results
      final_penalty = history.curr_penalties[-1]
      best_penalty = history.curr_best_penalties[-1]
      num_evals = len(history.query_points)
      self._print_method_result(curr_method, final_penalty, best_penalty, num_evals)
      # Save results of current iteration
      self.update_to_be_saved(curr_iter_results)
      self.save_pickle()
      self.save_results()
    # Save here
    self.update_to_be_saved(curr_iter_results)
#     self.save_pickle()
    # No need to explicitly save_results() here - it is done by the parent class.

  def update_to_be_saved(self, curr_iter_results):
    """ Updates the results of the data to be saved with curr_iter_results."""
    for data_type in self.data_to_be_saved:
      data = getattr(curr_iter_results, data_type)
      curr_data_to_be_saved = getattr(self.to_be_saved, data_type)
      if curr_data_to_be_saved.shape[1] == self.trial_iter:
        updated_data_to_be_saved = curr_data_to_be_saved
        updated_data_to_be_saved[:, -1] = data.ravel()
      elif curr_data_to_be_saved.shape[1] < self.trial_iter:
        updated_data_to_be_saved = np.concatenate((curr_data_to_be_saved, data), axis=1)
      else:
        raise ValueError('Something wrong with data saving.')
      setattr(self.to_be_saved, data_type, updated_data_to_be_saved)


# A Method Evaluator on Euclidean spaces ---------------------------------------------
class ExDMethodEvaluatorEuclidean(ExDMethodEvaluator):
  """ Constructor. """

  def _get_initial_pool_qinfos(self):
    """ Gets initial pool. """
    # Do all intialisations at the highest fidelity
    init_qinfos = get_euclidean_initial_qinfos('latin_hc',
            self.evaluation_options.initial_pool_size,
            self.experiment_caller.domain.bounds)
    return init_qinfos

