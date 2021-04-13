import numpy as np

from unittest import TestCase

from src.exception import SingularSystem, UnboundTargetFunction
from src.simplex import SimplexAnswer, SimplexSolver, matrix, vector


class SimplexTest(TestCase):
  def _solve(self, c: vector, A: matrix, b: vector) -> SimplexAnswer:
    solver = SimplexSolver(c, A, b)
    return solver.row_simplex_method()
  

  def _are_equal_answers(self, x: SimplexAnswer, y: SimplexAnswer) -> bool:
    are_equal_vectors = np.allclose(x.optimal_vector, y.optimal_vector)
    are_equal_values = np.allclose(x.target_function_value, y.target_function_value)
    return are_equal_vectors and are_equal_values


  def test1(self):
    c = np.array([0, 2, -1, -1])
    A = np.array([
      [2, 1, 1, 0],
      [1, 2, 0, 1]
    ])
    b = np.array([6, 6])

    actual = self._solve(c, A, b)
    expected = SimplexAnswer(
      optimal_vector=[2, 2, 0, 0],
      target_function_value=4
    )

    self.assertEqual(
      self._are_equal_answers(actual, expected), True
    )


  def test2(self):
    c = np.array([-1, -2, -3, 1])
    A = np.array([
      [1, -3, -1, -2],
      [1, -1, 1, 0]
    ])
    b = np.array([-4, 0])

    actual = self._solve(c, A, b)
    expected = SimplexAnswer(
      optimal_vector=[0, 0, 0, 2],
      target_function_value=2
    )

    self.assertEqual(
      self._are_equal_answers(actual, expected), True
    )

  def test3(self):
    c = np.array([-1, -2, -1, 3, -1])
    A = np.array([
      [1, 1, 0, 2, 1],
      [1, 1, 1, 3, 2],
      [0, 1, 1, 2, 1]
    ])
    b = np.array([5, 9, 6])

    actual = self._solve(c, A, b)
    expected = SimplexAnswer(
      optimal_vector=[0, 0, 1, 2, 1],
      target_function_value=4
    )

    self.assertEqual(
      self._are_equal_answers(actual, expected), True
    )

  def test4(self):
    c = np.array([-1, -1, -1, 1, -1])
    A = np.array([
      [1, 1, 2, 0, 0],
      [0, -2, -2, 1, -1],
      [1, -1, 6, 1, 1]
    ])
    b = np.array([4, -6, 12])

    actual = self._solve(c, A, b)
    expected = SimplexAnswer(
      optimal_vector=[0, 4, 0, 9, 7],
      target_function_value=-2
    )

    self.assertEqual(
      self._are_equal_answers(actual, expected), True
    )    


  def test5(self):
    c = np.array([-1, 4, -3, 10])
    A = np.array([
      [1, 1, -1, -10],
      [1, 14, 10, -10]
    ])
    b = np.array([0, 11])

    with self.assertRaises(SingularSystem):
      self._solve(c, A, b)

  def test6(self):
    c = np.array([-1, -1, 1, -1, 2])
    A = np.array([
      [3, 1, 1, 1, -2],
      [6, 1, 2, 3, -4],
      [10, 0, 3, 6, -7]
    ])
    b = np.array([10, 20, 30])

    actual = self._solve(c, A, b)
    expected = SimplexAnswer(
      optimal_vector=[0, 0, 10, 0, 0],
      target_function_value=10
    )

    self.assertEqual(
      self._are_equal_answers(actual, expected), True
    )

  def test7(self):
    c = np.array([-1, 2, -1, -1])
    A = np.array([
      [-10, 1, 1, 0],
      [-10, 2, 0, 1]
    ])
    b = np.array([6, 6])

    with self.assertRaises(UnboundTargetFunction):
      self._solve(c, A, b)
