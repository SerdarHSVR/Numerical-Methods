from jax import random
import jax.numpy as jnp
from jax import jit, grad

key = random.PRNGKey(0)

A = random.normal(key, (10, 10))

I = jnp.eye(10)

@jit
def loss(x, A):
    return jnp.sum((jnp.dot(A, x) - I) ** 2)

def optimize(f, x, A, n_iter=1000, lr=0.01):
    for i in range(n_iter):
        grads = jit(grad(loss))(x, A)
        x -= lr * grads
    return x

if __name__ == "__main__":
    A_inv = jnp.linalg.inv(A)
    print("A_inv (using jnp.linalg.inv):", A_inv)
    print("A * A_inv:", jnp.dot(A, A_inv))
    print("A_inv * A:", jnp.dot(A_inv, A))

    A_inv_opt = optimize(loss, jnp.eye(10), A, n_iter=1000, lr=0.01)
    print("A_inv_opt (using optimization):", A_inv_opt)
    print("A * A_inv_opt:", jnp.dot(A, A_inv_opt))
    print("A_inv_opt * A:", jnp.dot(A_inv_opt, A))