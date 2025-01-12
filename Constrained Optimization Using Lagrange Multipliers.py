import jax
from jax import grad, jit
from jax import numpy as jnp
from typing import Tuple, Callable, List
import numpy as np


def objective(x: jnp.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def constraints(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    norm_constraint = jnp.linalg.norm(x) - 3.6
    equality_constraint = x[0] + x[1]
    return norm_constraint, equality_constraint

def minimize(objective: Callable,
                constraints: Callable,
                λ: float,
                μ: float,
                x_init: jnp.ndarray,
                max_iter: int = 100,
                lr: float = 0.001,
                stopping_criterion: float = 1e-4) -> jnp.ndarray:
    x = x_init
    λ_vec = jnp.array([λ, μ])

    grad_objective = grad(objective)

    def lagrangian(x, λ_vec):
        norm_constraint, equality_constraint = constraints(x)
        return (
            objective(x)
            + λ_vec[0] * norm_constraint
            + λ_vec[1] * equality_constraint
        )

    grad_lagrangian = grad(lagrangian, argnums=0)
    grad_constraints = grad(lambda x: jnp.array(constraints(x)), argnums=0)

    for i in range(max_iter):
        x_grad = grad_lagrangian(x, λ_vec)
        x = x - lr * x_grad

        norm_constraint, equality_constraint = constraints(x)
        λ_vec = λ_vec + lr * jnp.array([norm_constraint, equality_constraint])

        if jnp.linalg.norm(x_grad) < stopping_criterion:
            break

    return x

##Let's give a try
key = jax.random.PRNGKey(0)
x_init = jax.random.uniform(key, (2,), minval=-1.2, maxval=1.2)
x = minimize(objective, constraints, 1.0, 1.0, x_init)
print(f"Optimal x: {x}, the norm of x: {jnp.linalg.norm(x)}, and x+y: {x[0]+x[1]}")