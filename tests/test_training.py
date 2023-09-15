

if __name__ == '__main__':

    # Compile functions here.

    key = jax.random.PRNGKey(0)
    train_iters = 100000
    for i in range(train_iters):
      key_i = jax.random.fold_in(key, i)
