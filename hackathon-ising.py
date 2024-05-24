import jax
from jax import Array
from jax import numpy as jnp


def E(
    state_internal: Array,
    J_internal: float,
    field_external: Array,
    J_external: float,
) -> Array:
    return -1 * (
        J_internal * ((state_internal * jnp.roll(state_internal, 1)).sum())
        + J_external * (field_external @ state_internal.T)
    )


def pi(
    state_internal: Array,
    J_internal: float,
    field_external: Array,
    J_external: float,
) -> Array:
    return jnp.exp(-E(state_internal, J_internal, field_external, J_external))


def propose(
    key: Array,
    state: Array,
    J_internal: float,
    field_external: Array,
    J_external: float,
) -> Array:
    flip = jax.random.randint(key, shape=(), minval=0, maxval=state.shape[0])
    updated_state = state.at[flip].set(-1 * state[flip])

    pi_prime = pi(updated_state, J_internal, field_external, J_external)
    pi_ref = pi(state, J_internal, field_external, J_external)

    return jnp.where(1.0 < pi_prime / pi_ref, 1.0, pi_prime / pi_ref), updated_state


def step(
    key: Array,
    J: float,
    state: Array,
):
    propose_key, accept_key = jax.random.split(key)
    accept_prob, next_state = propose(
        propose_key, state, J, state, 0.0
    )  # set external field to null
    return jnp.where(jax.random.uniform(accept_key) < accept_prob, next_state, state)


def initialize_state(key: Array, d: int) -> Array:
    return jax.random.bernoulli(key, p=0.5, shape=(d,)) * 2 - 1


def chain(key: Array, num_steps: int, initial_state: Array, J: float):
    def body(n, args):
        del n
        state_i, key_i = args
        return step(key_i, J, state_i), jax.random.split(key_i)[0]

    final_state, _ = jax.lax.fori_loop(
        0,
        num_steps,
        body,
        (
            initial_state,
            jax.random.split(key)[0],
        ),
    )
    return final_state


def simulate(key: Array, D: int, J: float):
    initialize_key, chain_key = jax.random.split(key)
    initial_state = initialize_state(initialize_key, D)
    return chain(chain_key, 10000, initial_state, J)


if __name__ == "__main__":
    D = 40
    J = 0.3 # 1.0
    key = jax.random.PRNGKey(0)
    final_state = simulate(key, D, J)
    print(final_state)

    B = 1000
    final_states = jax.vmap(simulate, in_axes=(0, None, None))(
        jax.random.split(key, B), D, J
    )
    print(final_states)
    
    import matplotlib.pyplot as plt
    x = final_states
    xx = (x.T @ x) / len(x)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    im = ax1.imshow(xx)
    cbar = ax1.figure.colorbar(im, ax=ax1)
    ax2.plot( xx[20] )
    ax3.plot( x[:3].T )
    fig.tight_layout()
    fig.savefig(f'./thoughts/distributions/figs/experimental/ising_{J}.png')
