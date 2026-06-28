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

def get_best_action(q_values):
    return np.random.choice(np.flatnonzero(q_values == q_values.max()))

def q_learning(env, episodes=20000, alpha=0.1, gamma=0.99, epsilon=0.1):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.ones((nS, nA)) * 1.0
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = get_best_action(Q[state])
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] * (not terminated) - Q[state][action])
            
            state = next_state
            
    policy = np.array([get_best_action(Q[s]) for s in range(nS)])
    return Q, policy
