import jax
import jax.numpy as jnp

x = jnp.ones(10)
print(x.addressable_data(0).device)
