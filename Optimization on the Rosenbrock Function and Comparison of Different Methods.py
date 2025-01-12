import jax
from jax import grad, random, numpy as jnp, jit
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from functools import partial
## Let's do the thins in float64 in jax
from jax import config
config.update("jax_enable_x64", True)


# Define the Rosenbrock function
@jit
def rosenbrock(x:jnp.ndarray) -> jnp.ndarray:
    # x is a 2D vector
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2




class TrainState_Lion:
    def __init__(self,
                 initial_pt: jnp.ndarray,
                 lr: float,
                 beta1: float,
                 beta2: float,
                 weight_decay: float):
        self.pt = initial_pt
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.m = jnp.zeros_like(initial_pt)

    def update(self, grad: jnp.ndarray):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        update = self.lr * jnp.sign(self.m)
        self.pt -= update


class TrainState_Momentum:
    def __init__(self,
                 initial_pt: jnp.ndarray,
                 lr: float,
                 beta: float):
        self.pt = initial_pt
        self.lr = lr
        self.beta = beta
        self.m = jnp.zeros_like(initial_pt)

    def update(self, grad: jnp.ndarray):
        self.m = self.beta * self.m + (1 - self.beta) * grad
        self.pt -= self.lr * self.m

class TrainState_Momentum:
    def __init__(self,
                 initial_pt: jnp.ndarray,
                 lr: float,
                 beta: float):
        self.pt = initial_pt
        self.lr = lr
        self.beta = beta
        self.m = jnp.zeros_like(initial_pt)

    def update(self, grad: jnp.ndarray):
        self.m = self.beta * self.m + (1 - self.beta) * grad
        self.pt -= self.lr * self.m


class TrainState_ADAM:
    def __init__(self,
                 initial_pt: jnp.ndarray,
                 lr: float,
                 beta1: float,
                 beta2: float,
                 eps: float = 1e-8):
        self.pt = initial_pt
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = jnp.zeros_like(initial_pt)
        self.v = jnp.zeros_like(initial_pt)
        self.t = 0

    def update(self, grad: jnp.ndarray):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.pt -= self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)


def optimize(optimizer_class, initial_pt, grad_fn, iterations, **kwargs):
    state = optimizer_class(initial_pt, **kwargs)
    points = [initial_pt]
    for _ in range(iterations):
        grad_val = grad_fn(state.pt)
        state.update(grad_val)
        points.append(state.pt)
    return jnp.array(points)

initial_pt = jnp.array([2.0, 2.0])
lr = 0.01
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.01
iterations = 500

grad_fn = grad(rosenbrock)

lion_pts = optimize(TrainState_Lion, initial_pt, grad_fn, iterations, lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)

adam_pts = optimize(TrainState_ADAM, initial_pt, grad_fn, iterations, lr=lr, beta1=beta1, beta2=beta2)

momentum_pts = optimize(TrainState_Momentum, initial_pt, grad_fn, iterations, lr=lr, beta=beta1)

# Plot results
plt.figure(figsize=(10, 8))
plt.plot(lion_pts[:, 0], lion_pts[:, 1], label="Lion Optimizer", alpha=0.8)
plt.plot(adam_pts[:, 0], adam_pts[:, 1], label="ADAM", alpha=0.8)
plt.plot(momentum_pts[:, 0], momentum_pts[:, 1], label="GD with Momentum", alpha=0.8)
plt.scatter([1.0], [1.0], c="red", label="Global Minimum", zorder=5)
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.legend()
plt.title("Optimization Trajectories on Rosenbrock Function")
plt.grid()
plt.show()