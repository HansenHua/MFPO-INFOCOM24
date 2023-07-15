from util import *

class Env:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
    
    def reset(self, ):
        return self.env.reset()
    
    def step(self, action):
        if self.env_name in ['Pong-v4', 'Breakout-v4', 'CartPole-v1']:
            action = action
        else:
            action = action.numpy()
        return self.env.step(action)
    
    def gen_obs(self, action):
        obs, reward, done, _ = self.step(action)
        return obs, reward, done
    