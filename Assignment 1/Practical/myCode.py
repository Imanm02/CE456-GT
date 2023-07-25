import numpy as np
from scipy.optimize import linprog

def process_inputs():
    count = int(input())
    matrix_rows = []
    for _ in range(count):
        matrix_rows.append(input())
    return count, np.array([list(map(int, row.split())) for row in matrix_rows])

def prepare_lp_variables(dim):
    objective, equality_lhs, equality_rhs, ineq_lhs, ineq_rhs, bounds, ineq_lhs_transposed = [], [[1]*dim], [1], [], [0]*dim, [(0, None)]*dim, []
    objective.extend([0]*dim)
    objective.append(1)
    equality_lhs[0].append(0)
    bounds.append((None, None))
    return objective, equality_lhs, equality_rhs, ineq_lhs, ineq_rhs, bounds, ineq_lhs_transposed

def construct_inequalities(ineq_lhs, ineq_lhs_transposed, data, dim):
    for index in range(dim):
        ineq_lhs.append(list(data[index]) + [-1])
        ineq_lhs_transposed.append(list(-data.T[index]) + [-1])

def solve_linprog(objective, ineq_lhs_transposed, ineq_rhs, ineq_lhs, equality_lhs, equality_rhs, bounds):
    res1 = linprog(c=objective, A_ub=ineq_lhs_transposed, b_ub=ineq_rhs, A_eq=equality_lhs, b_eq=equality_rhs, bounds=bounds, method="interior-point")
    res2 = linprog(c=objective, A_ub=ineq_lhs, b_ub=ineq_rhs, A_eq=equality_lhs, b_eq=equality_rhs, bounds=bounds, method="interior-point")
    return res1, res2

def print_results(res1, res2, dim):
    print(" ".join(map(str, res1.x[:dim])))
    print(" ".join(map(str, res2.x[:dim])))

def main():
    dimension, data = process_inputs()
    objective, equality_lhs, equality_rhs, ineq_lhs, ineq_rhs, bounds, ineq_lhs_transposed = prepare_lp_variables(dimension)
    construct_inequalities(ineq_lhs, ineq_lhs_transposed, data, dimension)
    result1, result2 = solve_linprog(objective, ineq_lhs_transposed, ineq_rhs, ineq_lhs, equality_lhs, equality_rhs, bounds)
    print_results(result1, result2, dimension)

main()