import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

gym.register(
    id="CustomFrozenLake-v0",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8", "is_slippery": True},
    max_episode_steps=200
)

custom_16x16_map = generate_random_map(size=16, p=0.8)
gym.register(
    id="CustomFrozenLake16x16-v0",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"desc": custom_16x16_map, "is_slippery": True},
    max_episode_steps=1000
)

def compute_policy_v(env, policy, gamma=0.99):
    nS = env.observation_space.n
    v = np.zeros(nS)
    eps = 1e-8
    while True:
        prev_v = np.copy(v)
        for s in range(nS):
            a = int(policy[s])
            v[s] = sum([p * (r + gamma * prev_v[s_] * (not term)) for p, s_, r, term in env.unwrapped.P[s][a]])
        if np.max(np.abs(prev_v - v)) <= eps:
            break
    return v

def extract_policy(env, v, gamma=0.99):
    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_sa = np.zeros(nA)
        for a in range(nA):
            for next_sr in env.unwrapped.P[s][a]:
                p, s_, r, term = next_sr
                q_sa[a] += p * (r + gamma * v[s_] * (not term))
        policy[s] = np.argmax(q_sa)
    return policy

def value_iteration(env, gamma=0.99):
    nS = env.observation_space.n
    nA = env.action_space.n
    v = np.zeros(nS)
    eps = 1e-8
    iterations = 0
    while True:
        iterations += 1
        prev_v = np.copy(v)
        for s in range(nS):
            q_sa = np.zeros(nA)
            for a in range(nA):
                for next_sr in env.unwrapped.P[s][a]:
                    p, s_, r, term = next_sr
                    q_sa[a] += p * (r + gamma * prev_v[s_] * (not term))
            v[s] = np.max(q_sa)
        if np.max(np.abs(prev_v - v)) <= eps:
            break
    policy = extract_policy(env, v, gamma)
    return policy, v, iterations

def policy_iteration(env, gamma=0.99):
    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.random.choice(nA, nS)
    iterations = 0
    while True:
        iterations += 1
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy
    return policy, old_policy_v, iterations
