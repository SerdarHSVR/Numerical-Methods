import jax
from jax import grad, jit
from jax import numpy as jnp
from typing import Tuple, Callable, List
import numpy as np

def objective(x: jnp.ndarray) -> float:
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def constraints(x: jnp.ndarray):
    return jax.nn.relu(jnp.linalg.norm(x) - 3.6), (x[0] + x[1] - 5)**2

def unit_test_1():
    assert objective(np.array([0.0, 0.0])) == 170.0, "Check the objective function"
    assert objective(np.array([0.0, 1.0])) == 136.0, "Check the objective function"
    assert objective(np.array([1.0, 1.0])) == 106.0, "Check the objective function"

    assert float(constraints(jnp.array([1.0, 4.0]))[0]) > 0.0, "Check the constraint function"
    assert float(constraints(np.array([0.0, 3.0]))[0]) == 0.0, "Check the constraint function"

    assert constraints(np.array([1.0, 4.0]))[1] == 0.0, "Check the constraint function"
    assert constraints(np.array([3.0, 2.0]))[1] == 0.0, "Check the constraint function"

    print("OK Computer!!!")
    return True

unit_test_1()

def minimize(objective: Callable,
             constraints: Callable,
             λ: float,
             μ: float,
             x_init: jnp.ndarray,
             max_iter: int = 100,
             lr: float = 0.001,
             stopping_criterion: float = 1e-4) -> jnp.ndarray:

    x = x_init
    for i in range(max_iter):
        penalty_1, penalty_2 = constraints(x)
        penalized_objective = objective(x) + λ * penalty_1**2 + μ * penalty_2

        grad_obj = grad(lambda x: objective(x) + λ * penalty_1**2 + μ * penalty_2)(x)

        x = x - lr * grad_obj

        if jnp.linalg.norm(grad_obj) < stopping_criterion:
            break

    return x

def unit_test_2():
    print("Testing optimization...")
    for i in range(10000):
        np.random.seed(i)
        x_init = minimize(objective, constraints, 10.0, 10.0,
                          np.random.randn(2), lr=0.00008, max_iter=1000)
        const_1, const_2 = np.linalg.norm(x_init), np.sum(x_init)
        if (np.linalg.norm([3,2]) - 1e-5 < const_1 < np.linalg.norm([3,2]) and
            np.abs(const_2 - 5.0) < 0.01):
            print(f"Success! Optimal point found: {x_init}")
            return 1
    print("Solution not found - check implementation")
    return 0

unit_test_2()