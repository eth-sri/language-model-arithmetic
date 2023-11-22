from scipy.optimize import minimize
import numpy as np

def objective(x, acceptance, F1, F2):
    return (1 - acceptance) * ((x - 1) * F1 + F2) / (1 - acceptance ** x)

def optimize(acceptance, F1, F2):
    return minimize(objective, 2, args=(acceptance, F1, F2), bounds=[(1, None)]).x[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", type=float, default=0.5)
    parser.add_argument("--F1", type=float, default=1.0)
    parser.add_argument("--F2", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    optimal = optimize(args.acceptance, args.F1, args.F2)
    optimal_int = np.floor(optimal)
    if objective(optimal_int, args.acceptance, args.F1, args.F2) > objective(optimal_int + 1, args.acceptance, args.F1, args.F2):
        optimal_int += 1
    objective_value = objective(optimal_int, args.acceptance, args.F1, args.F2)
    objective_value_at_1 = objective(1, args.acceptance, args.F1, args.F2)
    print(f"Optimal k: {optimal_int}")
    print(f"Objective value: {objective_value:.6f}")
    print(f"Objective value at 1: {objective_value_at_1:.6f}")
    print(f"Expected speedup: {(objective_value_at_1 - objective_value) / objective_value_at_1:.6f}")
    if args.k is not None:
        print(f"Objective at k={args.k}: {objective(args.k, args.acceptance, args.F1, args.F2):.6f}")
    