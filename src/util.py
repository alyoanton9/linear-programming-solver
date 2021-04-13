import numpy as np
import sys


infinity = float(sys.maxsize)


def remove_zero(values):
  is_not_zero = lambda v: v != 0
  return list(filter(is_not_zero, values))


def safe_division_by_positive_number(a, b):
  return a / b if b > 0 else infinity


def get_indexes_of_negative_values(values):
  is_negative_value = values >= 0
  indexes, = np.where(is_negative_value == False)
  return indexes


def delete_by_value(d, value):
  return {k:v for k, v in d.items() if v != value}
