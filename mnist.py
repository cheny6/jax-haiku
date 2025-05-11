import jax
import optax
import torch
import flax.linen as nn
from tqdm import tqdm
import jax.numpy as jnp
from torchvision import datasets, transforms
from flax.training import train_state


# Define the convolutional neural network model
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


# Instantiate the model
model = SimpleNN()


# Define the loss function
def loss_fn(params, key, x, y):
    logits = model.apply(params, x, rngs={'dropout': key})
    onehot = jax.nn.one_hot(y, 10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=onehot).mean()
    return loss


# Define the training step
@jax.jit
def train_step(state, key, batch):
    x, y = batch

    def loss_fn(params):
        logits = model.apply(params, x, rngs={'dropout': key})
        onehot = jax.nn.one_hot(y, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=onehot).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Define the evaluation function
@jax.jit
def eval_step(params, key, batch):
    x, y = batch
    logits = model.apply(params, x, rngs={'dropout': key})
    pred = jnp.argmax(logits, axis=-1)
    return jnp.sum(pred == y)


# Load the MNIST dataset
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset


def main():
    # Load the data
    trainset, testset = load_mnist()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Initialize the model parameters
    key = jax.random.PRNGKey(0)
    init_key, key = jax.random.split(key)
    x = jnp.ones((64, 28, 28, 1))  # Input shape

    # Fix: Only use one random key parameter
    params = model.init(init_key, x)

    # Create the optimizer and training state
    optimizer = optax.adam(1e-3)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0

        # Training
        with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}") as pbar:
            for x, y in trainloader:
                x, y = x.numpy(), y.numpy()
                x = jnp.transpose(x, (0, 2, 3, 1))  # Adjust the dimension order to NHWC
                x, y = jax.device_put(x), jax.device_put(y)

                key, batch_key = jax.random.split(key)
                state, batch_loss = train_step(state, batch_key, (x, y))

                epoch_loss += batch_loss
                batches += 1
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
                pbar.update(1)

        avg_loss = epoch_loss / batches
        print(f"Average loss in Epoch {epoch + 1}: {avg_loss:.4f}")

        # Evaluation
        correct = 0
        total = 0
        for x_batch, y_batch in testloader:
            x_batch, y_batch = x_batch.numpy(), y_batch.numpy()
            x_batch = jnp.transpose(x_batch, (0, 2, 3, 1))
            x_batch, y_batch = jax.device_put(x_batch), jax.device_put(y_batch)

            key, eval_key = jax.random.split(key)
            batch_correct = eval_step(state.params, eval_key, (x_batch, y_batch))
            correct += batch_correct
            total += len(y_batch)

        accuracy = correct / total
        print(f"Accuracy in Epoch {epoch + 1}: {accuracy:.4f}")


if __name__ == "__main__":
    main()