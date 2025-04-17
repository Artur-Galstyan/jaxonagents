import random
from collections import deque

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree
from tqdm import tqdm


class Policy(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, obs_dim: int, actions_dim: int, key: PRNGKeyArray):
        key, subkey = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(obs_dim, 32, key=key)
        self.fc2 = eqx.nn.Linear(32, actions_dim, key=subkey)

    def __call__(self, x: Array) -> Array:
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        return x


class ReplayMemory:
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition: tuple):
        self.memory.append(transition)

    def sample(self, sample_size: int):
        return random.sample(self.memory, k=sample_size)

    def __len__(self):
        return len(self.memory)


def loss_fn(policy: PyTree, states: Array, actions: Array, targets: Array):
    q_values = eqx.filter_vmap(policy)(states)
    batch_size, *_ = q_values.shape
    batch_indices = jnp.arange(batch_size)
    q_values = q_values[batch_indices, actions]
    loss = jnp.mean((q_values - targets) ** 2)
    return loss


@eqx.filter_jit
def step(
    policy: PyTree,
    states: Array,
    actions: Array,
    targets: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
        policy, states, actions, targets
    )
    updates, opt_state = optimizer.update(grads, opt_state, policy)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss_value


env = gym.make("CartPole-v1")
env.reset(seed=42)
random.seed(42)
obs_dim = 4
actions_dim = 2


policy = Policy(obs_dim, actions_dim, key=jax.random.key(42))
target = Policy(obs_dim, actions_dim, key=jax.random.key(42))

n_episodes = 5000
learning_rate = 0.01
discount_factor = 0.9
network_sync_rate = 10
replay_memory_size = 1000
batch_size = 32

epsilon = 1.0
epsilon_history = []
loss_values = []
memory = ReplayMemory(maxlen=replay_memory_size)

optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

step_count = 0
mean_batch_rewards = []

for i in tqdm(range(n_episodes)):
    state, _ = env.reset()

    terminated, truncated = False, False
    eps_reward = np.array(0.0)
    while not terminated and not truncated:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.array(jnp.argmax(jax.lax.stop_gradient(policy(state))))

        new_state, reward, terminated, truncated, _ = env.step(action)
        eps_reward += reward
        memory.append((state, action, new_state, reward, terminated, truncated))

        state = new_state
        step_count += 1
    mean_batch_rewards.append(eps_reward)
    if len(memory) > batch_size:
        batch = memory.sample(batch_size)

        # optimize
        states = []
        actions = []
        new_states = []
        rewards = []
        terms = []
        truncs = []

        for state, action, new_state, reward, terminated, truncated in batch:
            states.append(state)
            actions.append(action)
            new_states.append(new_state)
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
        states = jnp.stack(states)
        actions = jnp.array(actions)
        new_states = jnp.stack(new_states)
        rewards = jnp.array(rewards)
        terms = jnp.array(terms)
        truncs = jnp.array(truncs)

        targets = eqx.filter_vmap(jax.lax.stop_gradient(target))(new_states)
        targets = rewards + discount_factor * jnp.max(targets, axis=1)
        targets = jnp.where(terms, rewards, targets)

        policy, opt_state, loss_value = step(
            policy, states, actions, targets, optimizer, opt_state
        )
        loss_values.append(loss_value)

    epsilon = max(epsilon - 1 / n_episodes, 0)
    epsilon_history.append(epsilon)

    if step_count > network_sync_rate:
        target = eqx.tree_at(lambda x: x, target, policy)
        step_count = 0


# Plot loss values
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.plot(loss_values)
plt.title("Training Loss")
plt.xlabel("Training steps")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
plt.plot(epsilon_history)
plt.title("Epsilon Decay")
plt.xlabel("Episodes")
plt.ylabel("Epsilon")

plt.subplot(1, 3, 3)
plt.plot(mean_batch_rewards)
plt.title("Mean Batch Rewards")
plt.xlabel("Training steps")
plt.ylabel("Mean Reward")

plt.tight_layout()
plt.show()
