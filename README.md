# Simplex Method

## Description

Implement Simplex Method to solve the Linear Programming Problem that is given in a canonical form:

##### Target function `f`:
```
f = cx → max
```
##### Constraints:
```
Ax = b
x >= 0
```
Where `c` is an n-dimensional vector, `A` is a matrix of (n x m) dimension and `b` is an m-dimensional vector.

## Usage example

```powershell
simplex-method> python exec
Enter the dimension of space (n)
4
Enter the number of constraints (m)
2
Enter 'n' coefficients of the target function (c)
0 2 -1 -1
Enter the left part of the constraints equations line by line (A)
2 1 1 0
1 2 0 1
Enter the right part of the constraints equations (b)
6 6
Optimal vector: [2. 2. 0. 0.]
Target function maximum value: 4.0
```

## Tests

To run all tests execute the `python -m unittest` command in the root of the project.

## Literature cited

http://www.itlab.unn.ru/uploads/opt/optBook1.pdf
