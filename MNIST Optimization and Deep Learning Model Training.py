
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import jax
import optax
from flax import linen as nn
from jax import numpy as jnp
from jax import grad, value_and_grad, jit
import wandb

# Disable GPU to prevent potential conflicts
tf.config.experimental.set_visible_devices([], 'GPU')

# Constants for data normalization
MEAN = [0.13]
STD = [0.30]

# Set random seed for reproducibility
tf.random.set_seed(0)

# Load MNIST dataset
ds_train, info = tfds.load('mnist', split='train', shuffle_files=False, as_supervised=True, with_info=True)
ds_test, info = tfds.load('mnist', split='test', shuffle_files=False, as_supervised=True, with_info=True)

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = (x - MEAN) / STD
    x = tf.reshape(x, (-1, 28*28))
    return x, y

def prepare(ds):
    ds = ds.shuffle(5000)
    ds = ds.batch(128, True)
    ds = ds.map(preprocess, tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)

# Prepare datasets
train_ds = prepare(ds_train)
test_ds = prepare(ds_test)

class MLP(nn.Module):
    features: list
    dropout_rate: float

    def setup(self):
        self.layers = [nn.Dense(f) for f in self.features]
        self.dropout = nn.Dropout(self.dropout_rate)
        self.final_layer = nn.Dense(10)

    def __call__(self, x, training=False):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x, deterministic=not training)
            x = nn.relu(x)
        return self.final_layer(x)

def train_and_evaluate(config=None):
    # Initialize W&B
    wandb.init(config=config)
    config = wandb.config

    # Model initialization
    model = MLP(features=config.features, dropout_rate=config.dropout_rate)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)), training=False)

    # Optimizer setup
    optimizer = optax.adamw(learning_rate=config.learning_rate)
    opt_state = optimizer.init(params)

    @jit
    def compute_loss(params, x, y, key):
        y_pred = model.apply(params, x, training=True, rngs={'dropout': key})
        loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean()
        return loss

    @jit
    def update(params, opt_state, x, y, key):
        loss, grads = value_and_grad(compute_loss)(params, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
        return params, opt_state, loss, grad_norm

    # Training loop
    for epoch in range(config.epochs):
        train_loss = []
        grad_norms = []
        main_key = jax.random.PRNGKey(epoch)

        for i, (x, y) in enumerate(train_ds):
            x, y = jnp.array(x), jnp.array(y)
            sub_key = jax.random.fold_in(main_key, i)
            params, opt_state, loss, grad_norm = update(params, opt_state, x, y, sub_key)
            train_loss.append(loss)
            grad_norms.append(grad_norm)

        # Validation accuracy
        correct_predictions = 0
        total_predictions = 0
        for x, y in test_ds:
            x, y = jnp.array(x), jnp.array(y)
            y_pred = model.apply(params, x)
            predictions = jnp.argmax(y_pred, axis=-1)
            correct_predictions += jnp.sum(predictions == y)
            total_predictions += len(y)
        val_accuracy = correct_predictions / total_predictions

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': np.mean(train_loss),
            'val_accuracy': val_accuracy,
            'grad_norm': np.mean(grad_norms)
        })

        print(f"Epoch {epoch}: Train Loss = {np.mean(train_loss):.4f}, Val Accuracy = {val_accuracy:.4f}")

    wandb.finish()

# Experiment configurations
sweep_config = {
    'method': 'grid',
    'parameters': {
        'features': {
            'values': [[128, 64], [256, 128, 64]]
        },
        'dropout_rate': {
            'values': [0.2, 0.5]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'epochs': {
            'values': [10]
        }
    }
}

# Initialize W&B sweep
sweep_id = wandb.sweep(sweep_config, project='mnist_optimization')

# Start sweep
wandb.agent(sweep_id, function=train_and_evaluate)