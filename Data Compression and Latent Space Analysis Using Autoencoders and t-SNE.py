import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
import optax
from flax import linen as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE


def load_dataset():
    ds_train = tfds.load("mnist", split="train", as_supervised=True)
    ds_test = tfds.load("mnist", split="test", as_supervised=True)

    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.reshape(image, (-1,))
        return image, label

    ds_train = ds_train.map(preprocess).batch(128).prefetch(1)
    ds_test = ds_test.map(preprocess).batch(128).prefetch(1)
    return ds_train, ds_test


class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x

class Decoder(nn.Module):
    input_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(128)(z)
        z = nn.relu(z)
        z = nn.Dense(self.input_dim)(z)
        return z


class Autoencoder(nn.Module):
    latent_dim: int
    input_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.input_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed

def extract_latent_codes(model, params, dataset):
    latent_codes = []
    labels = []
    for batch in dataset:
        images, lbls = batch
        z = model.apply(params, images, method=model.encode)
        latent_codes.append(z)
        labels.append(lbls)
    return jnp.concatenate(latent_codes), jnp.concatenate(labels)

def compute_loss(params, model, batch):
    inputs, _ = batch
    reconstructed = model.apply(params, inputs)
    loss = jnp.mean(jnp.square(reconstructed - inputs))
    return loss

def train_step(params, model, batch, optimizer, opt_state):
    loss, grads = jax.value_and_grad(compute_loss)(params, model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def plot_reconstruction(model, params, dataset):
    batch = next(iter(dataset))
    images, _ = batch
    reconstructed = model.apply(params, images)

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 10, i + 11)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()

def apply_tsne(latent_codes):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_codes = tsne.fit_transform(np.array(latent_codes))
    return reduced_codes

def plot_latent_space(reduced_codes, labels):
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(reduced_codes[:, 0], reduced_codes[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("Latent Space Visualization with t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

def main():
    latent_dim = 16
    input_dim = 28 * 28

    ds_train, ds_test = load_dataset()

    model = Autoencoder(latent_dim, input_dim)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, input_dim)))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in tfds.as_numpy(ds_train):
            params, opt_state, loss = train_step(params, model, batch, optimizer, opt_state)
            total_loss += loss
            num_batches += 1
        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}, Loss: {loss}")


    print(f"Average Loss: {average_loss}")
    plot_reconstruction(model, params, tfds.as_numpy(ds_test))
    latent_codes, labels = extract_latent_codes(model, params, tfds.as_numpy(ds_test))
    reduced_codes = apply_tsne(latent_codes)
    plot_latent_space(reduced_codes, labels)

if __name__ == "__main__":
    main()