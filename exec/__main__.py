# It's necessary to add the path of 'src'
# in 'sys.path' to import 'simplex' module
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('src')))

import numpy as np

from src.simplex import SimplexAnswer, SimplexSolver


def read_c_A_b():
  n = int(input('Enter the dimension of space (n)\n'))
  m = int(input('Enter the number of constraints (m)\n'))

  c = list(map(
    int,
    input(
      'Enter \'n\' coefficients of the target function (c)\n'
    ).split()
  ))

  if len(c) != n:
    raise Exception('The dimension of vector \
representing the number of coefficients must be \'n\'')

  print('Enter the left part of the constraints equations line by line (A)')
  A = []
  for _ in range(m):
    A_i = list(map(int, input().split()))
    if A_i:
      A.append(A_i)
  
  if len(A) != m or len(A[0]) != n:
    raise Exception('The dimension of matrix \
representing the left part of constraints equations must be (\'m\' x \'n\')')
  
  b = list(map(
    int,
    input(
      'Enter the right part of the constraints equations (b)\n'
    ).split()
  ))

  if len(b) != m:
    raise Exception('The dimension of vector \
representing the right part of constraints must be \'m\'')

  return np.array(c), np.array(A), np.array(b)


def pretty_print_simplex_answer(simplex_answer: SimplexAnswer):
  optimal_vector = simplex_answer.optimal_vector
  target_function_value = simplex_answer.target_function_value
  np.set_printoptions(precision=3)
  print('Optimal vector:', optimal_vector)
  print('Target function maximum value:', target_function_value)


if __name__ == '__main__':
  c, A, b = read_c_A_b()

  solver = SimplexSolver(c, A, b)
  solution = solver.row_simplex_method()

  pretty_print_simplex_answer(solution)
