import numpy as np

from dataclasses import dataclass
from math import comb
from typing import Dict

from .exception import SingularSystem, UnboundTargetFunction
from .util import delete_by_value, get_indexes_of_negative_values, remove_zero, safe_division_by_positive_number


# type aliases
matrix = np.ndarray
vector = np.ndarray
index = int
base_type = Dict[index, index]


@dataclass
class SimplexAnswer:
  """
  Represent the solution
  given by the simplex method.
  """
  optimal_vector: vector
  target_function_value: float


class SimplexSolver():
  """
  Implement Simplex Method
  for solution of Linear Programming Problem
  that is given in a canonical form:
    cx -> max
    Ax = b
    x >= 0
  """

  def __init__(self, c: vector, A: matrix, b: vector, base: base_type = {}):
    """
    Instance variables:
    n: dimension of the space
    m: number of constraints
    max_iteration: maximum number of
                   algorithm's iterations
    base: system's of equations base
    Q: system's simplex table
    """
    m, n = A.shape

    self.n = n
    self.m = m
    self.max_iteration = comb(n, m)

    if base:
      self.base = base
      self.Q = self._init_Q(c, A, b)
    else:
      self.base, self.Q = self._find_artificial_base_and_Q(c, A, b)
      # number of constraints might decrease after ^
      self.m = len(self.Q) - 1


  def get_base(self):
    return self.base


  def get_Q(self):
    return self.Q


  def _init_Q(self, c: vector, A: matrix, b: vector) -> matrix:
    """
    Initialize simplex table using c, A, b:
    Q = | 0  | -c1 | -c2 | .. | -cn |
        | b1 | a11 | a12 | .. | a1n |
        | b2 | a21 | a22 | .. | a2n |
                      ..
        | bm | am1 | am2 | .. | amn |
    """

    Q = np.zeros((self.m + 1, self.n + 1))
    
    Q[0, 1:] = -c
    Q[1:, 0] = b
    Q[1:, 1:] = A

    return Q


  def row_simplex_method(self) -> SimplexAnswer:
    """
    Perform Simplex Method in row form.
    """
    iteration = 0
    self.Q = self._gaussian_elimination(self.Q, self.base)

    while iteration < self.max_iteration:
      negative_q0_indexes = get_indexes_of_negative_values(self.Q[0])
      negative_q0_indexes = remove_zero(negative_q0_indexes)

      # check if the target function is unbound
      for col in negative_q0_indexes:
        if np.all(self.Q[:, col] <= 0):
          raise UnboundTargetFunction

      # check if the stopping criteria is met
      if len(negative_q0_indexes) == 0:
        simplex_answer = SimplexAnswer(
          optimal_vector=self._get_basis_solution(),
          target_function_value=self.Q[0][0]
        )
        return simplex_answer

      # choose column
      s = self._choose_entering_variable(negative_q0_indexes)

      # choose row
      r = self._choose_leaving_variable(s)

      # change base and simplex table for the next iteration
      self.base[r] = s
      self.Q = self._gaussian_elimination(self.Q, {r: s})

      iteration += 1

    # if the algorithm doesn't finish after 'max_iteration'
    # then the system is singular
    raise SingularSystem


  def _gaussian_elimination(self, Q, new_base_items: base_type) -> matrix:
    """
    Perform one step of gaussian elimination method.
    """
    for row, basis_ind in new_base_items.items():
      Q[row] = Q[row] / Q[row][basis_ind]
      for other_row in range(self.m + 1):
        if other_row != row:
          Q[other_row] = Q[other_row] - Q[other_row][basis_ind] * Q[row]

    return Q


  def _choose_entering_variable(self, columns: [index]) -> index:
    """
    Choose the entering variable
    using alpha_3 choice rule
    from http://www.itlab.unn.ru/uploads/opt/optBook1.pdf
    """
    fractions = [
      min([
        safe_division_by_positive_number(self.Q[row][0], self.Q[row][col])
        for row in range(1, self.m + 1)
      ])
      for col in columns
    ]

    max_fraction_index = np.argmax(fractions)

    return columns[max_fraction_index]


  def _choose_leaving_variable(self, column: index) -> index:
    """
    Choose the leaving variable
    using beta_1 choice rule
    from http://www.itlab.unn.ru/uploads/opt/optBook1.pdf
    """
    fractions = [
      safe_division_by_positive_number(self.Q[row][0], self.Q[row][column])
      for row in range(1, self.m + 1)
    ]
    
    if len(fractions) > 0:
      min_fraction_index = np.argmin(fractions) + 1
    else:
      raise UnboundTargetFunction

    return min_fraction_index


  def _get_basis_solution(self) -> vector:
    """
    Find basis solution
    by base and simplex table.
    """
    basis_solution = np.zeros(self.n)
    for row, basis_index in self.base.items():
      basis_solution[basis_index - 1] = self.Q[row][0]

    return basis_solution


  def _find_artificial_base_and_Q(self, c: vector, A: matrix, b: vector) -> base_type:
    """
    Use Artificial Basis Method to find
    initial valid base and related simplex table.
    """
    c_artificial, A_artificial, b_artificial, base_artificial = self._init_artificial_c_A_b_and_base(A, b)

    inner_solver = SimplexSolver(c_artificial, A_artificial, b_artificial, base=base_artificial)
    _ = inner_solver.row_simplex_method()

    artificial_base = inner_solver.get_base()
    artificial_Q = inner_solver.get_Q()
    artificial_target_function_value = artificial_Q[0][0]

    if artificial_target_function_value != 0:
      raise SingularSystem

    initial_base = artificial_base
    initial_Q = artificial_Q

    # exclude extra rows from simplex table
    for row, basis_ind in artificial_base.items():
      if basis_ind <= self.n:
        continue

      q_row_n = initial_Q[row][1 : self.n + 1]
      if all(q_row_n == 0):
        initial_Q = np.delete(initial_Q, row, axis=0)
        initial_base = delete_by_value(initial_base, basis_ind)

      else:
        for j in range(1, self.n + 1):
          if initial_Q[row][j] != 0:
            initial_Q = self._gaussian_elimination(initial_Q, {row: j})
            initial_base[row] = j
            break

    # exclude extra columns from simplex table
    initial_Q = np.delete(
      initial_Q,
      [col for col in range(self.n + 1, self.n + self.m + 1)],
      axis=1
    )

    initial_Q[0] = np.append(0, -c)

    return initial_base, initial_Q


  def _init_artificial_c_A_b_and_base(self, A: matrix, b: vector) -> (vector, matrix, vector, base_type):
    """
    Initialize artificial c, A, b and base
    according to Artificial Basis Method.
    """
    c_artificial = np.append(
      np.repeat(0, self.n),
      np.repeat(-1, self.m)
    )

    A_artificial = np.zeros((self.m, self.n + self.m))
    A_artificial[:, : self.n] = A

    b_artificial = b

    negative_b_indexes = get_indexes_of_negative_values(b)

    for ind in negative_b_indexes:
      b_artificial[ind] = -b_artificial[ind]
      A_artificial[ind] = -A_artificial[ind]

    A_artificial[:, self.n:] = np.eye(self.m)

    base_artificial = {i: self.n + i for i in range(1, self.m + 1)}

    return c_artificial, A_artificial, b_artificial, base_artificial
