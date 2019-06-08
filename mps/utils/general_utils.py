"""
  A collection of very generic python utilities.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np
from scipy.sparse import dok_matrix
from scipy.linalg import solve_triangular

def map_to_cube(pts, bounds):
  """ Maps bounds to [0,1]^d and returns the representation in the cube. """
  return (pts - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])

def map_to_bounds(pts, bounds):
  """ Given a point in [0,1]^d, returns the representation in the original space. """
  return pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def get_sublist_from_indices(orig_list, idxs):
  """ Returns a sublist from the indices. orig_list can be anthing that can be
      indexed and idxs are a list of indices. """
  return [orig_list[idx] for idx in idxs]


def compute_average_sq_prediction_error(Y1, Y2):
  """ Returns the average prediction error. """
  return np.linalg.norm(np.array(Y1) - np.array(Y2))**2 / len(Y1)

def dist_squared(X1, X2):
  """ If X1 is n1xd and X2 is n2xd, this returns an n1xn2 matrix where the (i,j)th
      entry is the squared distance between X1(i,:) and X2(j,:).
  """
  n1, dim1 = X1.shape
  n2, dim2 = X2.shape
  if dim1 != dim2:
    raise ValueError('Second dimension of X1 and X2 should be equal.')
  dist_sq = (np.outer(np.ones(n1), (X2**2).sum(axis=1))
             + np.outer((X1**2).sum(axis=1), np.ones(n2))
             - 2*X1.dot(X2.T))
  dist_sq = np.clip(dist_sq, 0.0, np.inf)
  return dist_sq


def project_symmetric_to_psd_cone(M, is_symmetric=True, epsilon=0):
  """ Projects the symmetric matrix M to the PSD cone. """
  if is_symmetric:
    try:
      eigvals, eigvecs = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
      eigvals, eigvecs = np.linalg.eig(M)
      eigvals = np.real(eigvals)
      eigvecs = np.real(eigvecs)
  else:
    eigvals, eigvecs = np.linalg.eig(M)
  clipped_eigvals = np.clip(eigvals, epsilon, np.inf)
  return (eigvecs * clipped_eigvals).dot(eigvecs.T)


def stable_cholesky(M, add_to_diag_till_psd=True):
  """ Returns L, a 'stable' cholesky decomposition of M. L is lower triangular and
      satisfies L*L' = M.
      Sometimes nominally psd matrices are not psd due to numerical issues. By adding a
      small value to the diagonal we can make it psd. This is what this function does.
      Use this iff you know that K should be psd. We do not check for errors.
  """
  _printed_warning = False
  if M.size == 0:
    return M # if you pass an empty array then just return it.
  try:
    # First try taking the Cholesky decomposition.
    L = np.linalg.cholesky(M)
  except np.linalg.linalg.LinAlgError as e:
    # If it doesn't work, then try adding diagonal noise.
    if not add_to_diag_till_psd:
      raise e
    diag_noise_power = -11
    max_M = np.diag(M).max()
    diag_noise = np.diag(M).max() * 1e-11
    chol_decomp_succ = False
    while not chol_decomp_succ:
      try:
        diag_noise = (10 ** diag_noise_power) * max_M
        L = np.linalg.cholesky(M + diag_noise * np.eye(M.shape[0]))
        chol_decomp_succ = True
      except np.linalg.linalg.LinAlgError:
        if diag_noise_power > -9 and not _printed_warning:
          from warnings import warn
          warn(('Could not compute Cholesky decomposition despite adding %0.4f to the '
                'diagonal. This is likely because the M is not positive semi-definite.')%(
               (10**diag_noise_power) * max_M))
          _printed_warning = True
        diag_noise_power += 1
      if diag_noise_power >= 5:
        raise ValueError(('Could not compute Cholesky decomposition despite adding' +
                          ' %0.4f to the diagonal. This is likely because the M is not ' +
                          'positive semi-definite or has infinities/nans.')%(diag_noise))
  return L


# Solving triangular matrices -----------------------
def _solve_triangular_common(A, b, lower):
  """ Solves Ax=b when A is a triangular matrix. """
  if A.size == 0 and b.shape[0] == 0:
    return np.zeros((b.shape))
  else:
    return solve_triangular(A, b, lower=lower)

def solve_lower_triangular(A, b):
  """ Solves Ax=b when A is lower triangular. """
  return _solve_triangular_common(A, b, lower=True)

def solve_upper_triangular(A, b):
  """ Solves Ax=b when A is upper triangular. """
  return _solve_triangular_common(A, b, lower=False)


def draw_gaussian_samples(num_samples, mu, K):
  """ Draws num_samples samples from a Gaussian distribution with mean mu and
      covariance K.
  """
  num_pts = len(mu)
  L = stable_cholesky(K)
  U = np.random.normal(size=(num_pts, num_samples))
  V = L.dot(U).T + mu
  return V


# Executing a Pythong string -------------------------------------------------------
def evaluate_strings_with_given_variables(_strs_to_execute, _variable_dict=None):
  """ Executes a list of python strings and returns the results as a list.
      variable_dict is a dictionary mapping strings to values which may be used in
      str_to_execute.
  """
  if _variable_dict is None:
    _variable_dict = {}
  if not isinstance(_strs_to_execute, list):
    _got_list_of_constraints = False
    _strs_to_execute = [_strs_to_execute]
  else:
    _got_list_of_constraints = True
  for _key, _value in _variable_dict.items():
    locals()[_key] = _value
  _ret = [eval(_elem) for _elem in _strs_to_execute]
  if _got_list_of_constraints:
    return _ret
  else:
    return _ret[0]


# Matrix/Array/List utilities ------------------------------------------------------
def get_nonzero_indices_in_vector(vec):
  """ Returns the nonzero indices in the vector vec. """
  if not isinstance(vec, np.ndarray):
    vec = np.asarray(vec.todense()).ravel()
  ret, = vec.nonzero()
  return ret

def reorder_list_or_array(M, ordering):
  """ Reorders a list or array like object. """
  if isinstance(M, list):
    return [M[i] for i in ordering]
  else:
    return M[ordering]

def reorder_rows_and_cols_in_matrix(M, ordering):
  """ Reorders the rows and columns in matrix M. """
  array_type = type(M)
  if array_type == dok_matrix: # Check if a sparse matrix to convert to array
    M = np.asarray(M.todense())
  elif array_type == list:
    M = np.array(M)
  # Now do the reordering
  M = M[:, ordering][ordering]
  # Convert back
  if array_type == dok_matrix: # Check if a sparse matrix for return
    M = dok_matrix(M)
  elif array_type == list:
    M = [list(m) for m in M]
  return M

def _set_coords_to_val(A, coords, val):
  """ Sets the indices in matrix A to value. """
  for coord in coords:
    A[coord[0], coord[1]] = val

def get_dok_mat_with_set_coords(n, coords):
  """ Returns a sparse 0 matrix with the coordinates in coords set to 1. """
  A = dok_matrix((n, n))
  _set_coords_to_val(A, coords, 1)
  return A

def block_augment_array(A, B, C, D):
  """ Given a n1xn2 array A, an n1xn3 array B, an n4xn5 array C, and a
      n4x(n2 + n3 - n5) array D, this returns (n1+n4)x(n2+n3) array of the form
      [A, B; C; D].
  """
  AB = np.hstack((A, B))
  CD = np.hstack((C, D))
  return np.vstack((AB, CD))

def generate_01_grid(size_per_dim, dim=None, as_mesh=False):
  """ Generates a grid in [0,1]^d having sizes size_per_dim[i] on dimension i.
      if as_mesh is True, it returns the grid as a mesh. Otherwise as an nxdim array.
  """
  if not hasattr(size_per_dim, '__iter__'):
    size_per_dim = [size_per_dim] * dim
  grid_on_each_dim = []
  for curr_size in size_per_dim:
    curr_grid = np.linspace(0, 1, curr_size + 1)[:-1]
    curr_grid = curr_grid + 0.5/float(curr_size)
    grid_on_each_dim.append(curr_grid)
  mgrid = np.meshgrid(*grid_on_each_dim)
  if as_mesh:
    ret = mgrid
  else:
    ret = np.array([elem.flatten() for elem in mgrid]).T
    tmp = np.array(ret[:, 1])
    ret[:, 1] = np.array(ret[:, 0])
    ret[:, 0] = tmp
  return ret


# For sampling based on fitness values -------------------------------------------------
# We are using them in the GA and BO algorithms.
def get_exp_probs_from_fitness(fitness_vals, scaling_param=None, scaling_const=None):
  """ Returns sampling probabilities from fitness values; the fitness values are
      exponentiated and used as probabilities.
  """
  fitness_vals = np.array(fitness_vals)
  if scaling_param is None:
    scaling_const = scaling_const if scaling_const is not None else 0.5
    scaling_param = scaling_const * fitness_vals.std()
  mean_param = fitness_vals.mean()
  exp_probs = np.exp((fitness_vals - mean_param)/scaling_param)
  return exp_probs/exp_probs.sum()

def sample_according_to_exp_probs(fitness_vals, num_samples, replace=False,
                                  scaling_param=None, scaling_const=None):
  """ Samples after exponentiating the fitness values. """
  exp_probs = get_exp_probs_from_fitness(fitness_vals, scaling_param, scaling_const)
  return np.random.choice(len(fitness_vals), num_samples, p=exp_probs, replace=replace)

