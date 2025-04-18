import random
from collections import deque

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from beartype.typing import Callable, List
from jaxtyping import Array, PRNGKeyArray, PyTree
from tqdm import tqdm

from jaxonagents.base.abstract import AbstractDQNPolicy
from jaxonagents.base.types import ActionArray, StateArray


@eqx.filter_jit
def step(
    policy: PyTree,
    states: Array,
    actions: Array,
    targets: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: Callable[[AbstractDQNPolicy, StateArray, ActionArray, Array], Array],
):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
        policy, states, actions, targets
    )
    updates, opt_state = optimizer.update(grads, opt_state, policy)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss_value


class Policy(AbstractDQNPolicy):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, obs_dim: int, actions_dim: int, key: PRNGKeyArray):
        key, subkey = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(obs_dim, 64, key=key)
        self.fc2 = eqx.nn.Linear(64, actions_dim, key=subkey)

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


class TrainingMetrics:
    """Class to store and process training metrics"""

    def __init__(self, window_size: int = 100):
        self.episode_rewards: List[float] = []
        self.avg_rewards: List[float] = []
        self.losses: List[float] = []
        self.q_values_mean: List[float] = []
        self.q_values_max: List[float] = []
        self.q_values_min: List[float] = []
        self.window_size = window_size

    def update_rewards(self, reward: float):
        self.episode_rewards.append(reward)
        # Calculate moving average
        window = min(self.window_size, len(self.episode_rewards))
        avg_reward = sum(self.episode_rewards[-window:]) / window
        self.avg_rewards.append(avg_reward)

    def update_loss(self, loss: float):
        self.losses.append(loss)

    def update_q_stats(self, q_values: Array):
        self.q_values_mean.append(float(jnp.mean(q_values)))
        self.q_values_max.append(float(jnp.max(q_values)))
        self.q_values_min.append(float(jnp.min(q_values)))

    def plot_metrics(self):
        """Plot all collected metrics"""
        plt.figure(figsize=(15, 12))

        # Plot episode rewards
        plt.subplot(3, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

        # Plot average rewards
        plt.subplot(3, 2, 2)
        plt.plot(self.avg_rewards)
        plt.title(f"Moving Average Rewards (window={self.window_size})")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")

        # Plot loss values
        plt.subplot(3, 2, 3)
        plt.plot(self.losses)
        plt.title("Loss Values")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")

        # Plot Q-value statistics
        plt.subplot(3, 2, 4)
        plt.plot(self.q_values_mean, label="Mean")
        plt.plot(self.q_values_max, label="Max")
        plt.plot(self.q_values_min, label="Min")
        plt.title("Q-Value Statistics")
        plt.xlabel("Training Steps")
        plt.ylabel("Q-Value")
        plt.legend()

        # Plot epsilon decay
        if hasattr(self, "epsilon_history"):
            plt.subplot(3, 2, 5)
            plt.plot(self.epsilon_history)
            plt.title("Epsilon Decay")
            plt.xlabel("Episodes")
            plt.ylabel("Epsilon")

        plt.tight_layout()
        plt.show()


class DQN:
    @staticmethod
    def __call__(
        policy: AbstractDQNPolicy,
        target: AbstractDQNPolicy,
        loss_fn: Callable[[AbstractDQNPolicy, StateArray, ActionArray, Array], Array],
        env: gym.Env,
        optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
        n_episodes: int,
        learning_rate: float,
        discount_factor: float,
        network_sync_rate: int,
        replay_memory_maxlen: int,
        batch_size: int,
        env_seed: int,
        random_seed: int,
        metrics_window_size: int = 100,
    ) -> tuple[PyTree, TrainingMetrics]:
        env.reset(seed=env_seed)
        random.seed(random_seed)

        # Initialize the metrics tracker
        metrics = TrainingMetrics(window_size=metrics_window_size)

        memory = ReplayMemory(maxlen=replay_memory_maxlen)
        epsilon = 1.0  # Start with a high exploration rate
        metrics.epsilon_history = []  # Add epsilon history to metrics
        opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

        step_count = 0
        training_step = 0

        for i in tqdm(range(n_episodes)):
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0  # Track rewards for this episode

            while not terminated and not truncated:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = jax.lax.stop_gradient(policy(state))
                    action = np.array(jnp.argmax(q_values))

                    # Record Q-value statistics (only when not exploring)
                    if (
                        training_step % 10 == 0
                    ):  # Don't record every step to save memory
                        metrics.update_q_stats(q_values)

                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated, truncated))

                state = new_state
                episode_reward += reward  # Accumulate reward
                step_count += 1

            # Episode finished - update metrics
            metrics.update_rewards(episode_reward)
            metrics.epsilon_history.append(epsilon)

            # Training step
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)

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
                    policy,
                    states,
                    actions,
                    targets,
                    optimizer,
                    opt_state,
                    loss_fn,
                )

                # Record the loss value
                metrics.update_loss(float(loss_value))
                training_step += 1

            # Decay epsilon (linearly)
            epsilon = max(epsilon - 1 / n_episodes, 0)

            # Sync the target network periodically
            if step_count > network_sync_rate:
                target = eqx.tree_at(lambda x: x, target, policy)
                step_count = 0

        return policy, metrics


def loss_fn(policy: AbstractDQNPolicy, states: Array, actions: Array, targets: Array):
    q_values = eqx.filter_vmap(policy)(states)
    batch_size, *_ = q_values.shape
    batch_indices = jnp.arange(batch_size)
    q_values = q_values[batch_indices, actions]
    loss = jnp.mean((q_values - targets) ** 2)
    return loss


# Example usage with the enhanced DQN
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v3")
    obs_dim = env.observation_space.shape[0]  # pyright: ignore
    actions_dim = int(env.action_space.n)  # pyright: ignore

    policy = Policy(obs_dim, actions_dim, key=jax.random.key(42))
    target = Policy(obs_dim, actions_dim, key=jax.random.key(42))

    n_episodes = 2000
    learning_rate = 0.01
    discount_factor = 0.9
    network_sync_rate = 10
    replay_memory_size = 1000
    batch_size = 128

    optimizer = optax.adam(learning_rate=learning_rate)

    # Use the enhanced DQN to train the policy and collect metrics
    policy, metrics = DQN.__call__(
        policy=policy,
        target=target,
        loss_fn=loss_fn,
        env=env,
        optimizer=optimizer,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        network_sync_rate=network_sync_rate,
        replay_memory_maxlen=replay_memory_size,
        batch_size=batch_size,
        env_seed=42,
        random_seed=42,
        metrics_window_size=100,  # Window size for moving average
    )

    # Plot all metrics
    metrics.plot_metrics()

    # Evaluate the trained policy
    def evaluate_policy(policy, env, n_episodes=10):
        """Evaluate the policy without exploration"""
        evaluation_rewards = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                q_values = policy(state)
                action = np.array(jnp.argmax(jax.lax.stop_gradient(q_values)))
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            evaluation_rewards.append(episode_reward)

        return evaluation_rewards

    # Run evaluation
    eval_rewards = evaluate_policy(policy, env)
    print(f"Evaluation over {len(eval_rewards)} episodes:")
    print(f"  Mean reward: {np.mean(eval_rewards):.2f}")
    print(f"  Min reward: {np.min(eval_rewards):.2f}")
    print(f"  Max reward: {np.max(eval_rewards):.2f}")
