import safe_grid_gym
import gymnasium as gym

env = gym.make("Vase-v0",render_mode='ansi')
action_space = env.action_space

observation, info = env.reset()
print(env.render())
for i in range(10):
    action = action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    print(env.render())
    # render() is called automatically when render_mode="human"
    # but you can still call it manually if needed
    print(info) 
    if terminated or truncated:
        observation, info = env.reset()


env.close()
