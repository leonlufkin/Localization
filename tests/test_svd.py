
import numpy as np
import jax.numpy as jnp
from jax import jit
import time
import matplotlib.pyplot as plt

# Define the sizes for the benchmark
sizes = [100, 200, 400]#, 800, 1600]

numpy_times = []
jax_times = []
numpy_nans = []
jax_nans = []


@jax.jit
def generate_matrix(D):
    C = jnp.abs(
        jnp.tile(jnp.arange(D)[:, jnp.newaxis], (1, D))
        - jnp.tile(jnp.arange(D), (D, 1))
    )
    return jnp.exp(-(C**2) / (4.0**2))


@jit
def jax_svd(matrix):
    return jnp.linalg.svd(matrix)


@jit
def jax_cholesky(matrix):
    return jnp.linalg.cholesky(matrix)


for size in sizes:
    # Create a random matrix of the given size
    matrix_jax = generate_matrix(size)
    matrix_np = np.array(matrix_jax)

    # Time NumPy SVD
    start_time = time.time()
    u, s, vh = np.linalg.svd(matrix_np)
    end_time = time.time()
    numpy_times.append(end_time - start_time)
    numpy_nans.append(np.isnan(u).any() or np.isnan(s).any() or np.isnan(vh).any())

    # Time NumPy sampling
    start_time = time.time()
    u, s, vh = np.linalg.svd(matrix_np)
    end_time = time.time()
    numpy_times.append(end_time - start_time)
    numpy_nans.append(np.isnan(u).any() or np.isnan(s).any() or np.isnan(vh).any())

    # Time JAX SVD
    start_time = time.time()
    u, s, vh = jax_svd(matrix_jax)
    u.block_until_ready()  # Ensure we wait until JAX is done
    end_time = time.time()
    jax_times.append(end_time - start_time)
    jax_nans.append(jnp.isnan(u).any() or jnp.isnan(s).any() or jnp.isnan(vh).any())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sizes, numpy_times, label='NumPy', marker='o')
plt.plot(sizes, jax_times, label='JAX', marker='o')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds)')
plt.title('SVD Benchmark: NumPy vs JAX')
plt.legend()
plt.grid(True)
plt.show()

# Report any NaN findings
for size, np_nan, jax_nan in zip(sizes, numpy_nans, jax_nans):
    if np_nan:
        print(f"NumPy SVD resulted in NaN values for matrix size {size}.")
    if jax_nan:
        print(f"JAX SVD resulted in NaN values for matrix size {size}.")

