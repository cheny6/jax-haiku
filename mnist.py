import jax
import optax
import torch
import haiku as hk
from tqdm import tqdm
import jax.numpy as jnp
from torchvision import datasets, transforms

class SimpleNN(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape =(2,2), strides=(2,2), padding='VALID')(x)

        x = hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape =(2,2), strides=(2,2), padding='VALID')(x)

        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)  
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x
    
def apply_update(optimizer, opt_state, params, grad):
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
    
def loss_fn(params, key, x, y):
    logits = CNN_model.apply(params, key, x)
    onehot = jax.nn.one_hot(y, 10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=onehot).mean()
    return loss

def update(params, opt_state, key, x, y):
    loss, grad = jax.value_and_grad(loss_fn)(params, key, x, y)
    return grad, loss  

def forward_fn(x):
    cnn_model = SimpleNN()
    return cnn_model(x)

CNN_model = hk.transform(forward_fn)

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def main():
    trainset, testset = load_mnist()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    key = jax.random.PRNGKey(0)
    init_key, key = jax.random.split(key)
    x = jnp.ones((64, 28, 28, 1))  
    params = CNN_model.init(init_key, x)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    epochs = 10
    for epoch in range(epochs):  
        epoch_loss = 0.0
        batches = 0
        with tqdm(total=len(trainloader), desc=f"Train epoch {epoch+1}", leave=False) as pbar:
            for x, y in trainloader:
                x, y = x.numpy(), y.numpy()  
                x = jnp.transpose(x, (0, 2, 3, 1))  
                x, y = jax.device_put(x), jax.device_put(y)
                
                key, batch_key = jax.random.split(key)
                grad, batch_loss = update(params, opt_state, batch_key, x, y)
                params, opt_state = apply_update(optimizer, opt_state, params, grad)
                
                epoch_loss += batch_loss
                batches += 1
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
                pbar.update(1)
        
        avg_loss = epoch_loss / batches
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        correct = 0
        total = 0
        for x_batch, y_batch in testloader:
            x_batch, y_batch = x_batch.numpy(), y_batch.numpy()
            x_batch = jnp.transpose(x_batch, (0, 2, 3, 1))
            x_batch, y_batch = jax.device_put(x_batch), jax.device_put(y_batch)
            
            key, test_key = jax.random.split(key)
            logits = CNN_model.apply(params, test_key, x_batch)
            pred = jnp.argmax(logits, axis=-1)
            correct += jnp.sum(pred == y_batch)
            total += len(y_batch)
            
        accuracy = correct / total
        print(f"Epoch {epoch+1} accuracy: {accuracy:.4f}")
        
if __name__ == "__main__":
    main()
