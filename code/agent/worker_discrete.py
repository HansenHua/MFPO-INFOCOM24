from util import *
from env.env import Env

class Worker_discrete(nn.Module):
    def __init__(self, ):
        super(Worker_discrete, self).__init__()
        self.method_conf = get_global_dict_value('method_conf')
        self.fault_type = self.method_conf['fault_type']
        self.env_name = self.method_conf['env_name']
        self.env = Env(self.env_name)
        self.learning_rate_a = self.method_conf['learning_rate_a']
        self.learning_rate_c = self.method_conf['learning_rate_c']
        self.max_step = 1000
        self.gamma = self.method_conf['gamma']
        self.c = self.method_conf['c']
        self.observation_space = self.env.env.observation_space.shape[0]
        self.action_space = self.env.env.action_space.n
        
        if self.env_name == 'CartPole-v1':
            self.network = nn.Sequential(
                nn.Linear(self.observation_space, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            )
            self.old_model = nn.Sequential(
                nn.Linear(self.observation_space, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.observation_space, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            self.target = nn.Sequential(
                nn.Linear(self.observation_space, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        if self.env_name == 'Pong-v4' or self.env_name == 'Breakout-v4':
            self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc4 = nn.Linear(7 * 7 * 64, 512)
            self.head = nn.Linear(512, 14)
        self.my_softmax = nn.Softmax(dim=-1)
        self.optimizer_new = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate_a, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.optimizer_old = torch.optim.Adam(self.old_model.parameters(), lr=self.learning_rate_a, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_c, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.lr_scheduler_new = optim.lr_scheduler.ExponentialLR(self.optimizer_new, self.method_conf['decay_rate'])
        self.lr_scheduler_old = optim.lr_scheduler.ExponentialLR(self.optimizer_old, self.method_conf['decay_rate'])
        self.lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(self.optimizer_critic, self.method_conf['decay_rate'])
    
    def forward(self, state):
        state = torch.from_numpy(state).float()
        if self.env_name == 'Pong-v4' or self.env_name == 'Breakout-v4':
            x = state.float() / 255
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc4(x.view(x.size(0), -1)))
            return self.head(x)
        return self.network(state)
    
    def gen_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob = self.my_softmax(self.network(state))
        action_prob = action_prob.detach().numpy()
        action = np.random.choice(self.action_space, 1, p=action_prob)[0]

        return action, action_prob[action]
    
    def gen_action_prob(self, state, action):
        state = torch.from_numpy(state).float()
        action_prob = self.my_softmax(self.old_model(state))
        action_prob.detach().numpy()[0]
        prob = action_prob[action]
        return prob
    
    def gen_action_prob_new(self, state, action):
        state = torch.from_numpy(state).float()
        action_prob = self.my_softmax(self.network(state))
        action_prob.detach().numpy()[0]
        prob = action_prob[action]
        return prob
    
    def gen_critic(self, state):
        state = torch.from_numpy(state).float()
        v = self.critic(state)

        return v

    def preprocess(image):
        image = image[35:195]
        image = image[::2, ::2, 0]
        image[image == 144] = 0
        image[image == 109] = 0
        image[image != 0] = 1
        return image.astype(np.float).ravel()
    
    def collect_trajectory(self, batch_size):
        state_batch = []
        action_batch = []
        action_prob_batch = []
        batch_weights = []
        critic_batch = []
        state_prime_batch = []
        r_batch = []

        for _ in range(batch_size):
            state, reward, done = self.env.reset(), 0, False
            if self.env_name == 'Pong-v4' or self.env_name == 'Breakout-v4':
                state = self.preprocess(state)
            reward_batch = []
            step = 0
            while True:
                step += 1
                action, action_prob = self.gen_action(state)
                state_batch.append(state)
                v = self.gen_critic(state)
                critic_batch.append(v)
                state, reward, done, _ = self.env.step(action)
                if self.env_name == 'Pong-v4' or self.env_name == 'Breakout-v4':
                    state = self.preprocess(state)
                reward_batch.append(reward)
                r_batch.append(reward)
                action_batch.append(action)
                state_prime_batch.append(state)
                action_prob_batch.append(action_prob)

                if done or step >= self.max_step:
                    returns = []
                    R = 0
                    for r in reward_batch[::-1]:
                        R = r + self.gamma * R
                        returns.insert(0, R)
                    returns = torch.tensor(returns, dtype=torch.float32)
                
                    advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                    batch_weights += advantage
                    break
        
        batch_weights = torch.as_tensor(batch_weights, dtype = torch.float32)
        action_prob_batch = torch.as_tensor(action_prob_batch, dtype = torch.float32)

        critic_batch = torch.as_tensor(critic_batch, dtype = torch.float32)
        state_prime_batch = torch.as_tensor(state_prime_batch, dtype = torch.float32)
        r_batch = torch.as_tensor(r_batch, dtype = torch.float32)

        return batch_weights, state_batch, action_batch, action_prob_batch, critic_batch, state_prime_batch, r_batch
    
    def train(self, batch_size, step):
        returns, state_batch, action_batch, action_prob_batch, critic_batch, state_prime_batch, reward_batch = self.collect_trajectory(batch_size)

        new_logp = []
        for idx, _ in enumerate(state_batch):
            action_prob = self.gen_action_prob_new(state_batch[idx], action_batch[idx])
            new_logp.append(action_prob)
        new_logp = torch.stack(new_logp)

        s_batch = torch.as_tensor(state_batch, dtype = torch.float32)

        critic_prime = self.target(state_prime_batch)
        q = self.gamma * critic_prime
        q = q.squeeze()
        q = q + reward_batch
        
        loss = F.mse_loss(q, critic_batch)
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        advantage = returns - critic_batch

        batch_loss = -(torch.log(new_logp) * advantage).mean()
        batch_loss.backward()

        old_logp = []
        for idx, _ in enumerate(state_batch):
            action_prob = self.gen_action_prob(state_batch[idx], action_batch[idx])
            old_logp.append(action_prob)
        old_logp = torch.stack(old_logp)

        ratios = torch.exp(torch.log(old_logp.detach()) - torch.log(action_prob_batch.detach()))

        loss_old = -(torch.log(old_logp) * advantage * ratios).mean()
        self.optimizer_old.zero_grad()
        loss_old.backward()

        grad_old = [item.grad for item in self.old_model.parameters()]

        for idx,item in enumerate(self.network.parameters()):
            item.grad = item.grad - (1 - self.c * self.learning_rate_a**2) * grad_old[idx]
        
        grad = [item.grad for item in self.network.parameters()]

        self.old_model = copy.deepcopy(self.network)

        self.optimizer_new.step()

        if step > self.method_conf['decay_start_iter_id']:
            self.lr_scheduler_new.step()
            self.lr_scheduler_old.step()
            self.lr_scheduler_critic.step()

        return grad

    def test(self, i):
        file = open('report.txt', 'a')
        sum_r = 0
        for _ in range(10):
            state = self.env.reset()
            while True:
                action, _ = self.gen_action(state)
                state, reward, done, _ = self.env.step(action)
                sum_r += reward
                if done:
                    break
        file.write(str(i) + ':' + str(sum_r / 10) +'\n')
        print('worker test', sum_r / 10)
        return sum_r / 10
