from util import *
from env.env import Env

class policy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(policy, self).__init__()
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.network = nn.Sequential(
                init_(nn.Linear(in_dim, 256)),
                nn.Tanh(),
                init_(nn.Linear(256, 256)),
                nn.Tanh(),
            )
        self.output = init_(nn.Linear(256, out_dim))
        self.output_ = init_(nn.Linear(256, out_dim))
    
    def forward(self, state):
        # state = torch.from_numpy(state).float()
        s = self.network(state)
        mu = self.output(s)
        sigma = self.output_(s)

        return mu, sigma

class critic(nn.Module):
    def __init__(self, in_dim):
        super(critic, self).__init__()
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.network = nn.Sequential(
                init_(nn.Linear(in_dim, 256)),
                nn.Tanh(),
                init_(nn.Linear(256, 256)),
                nn.Tanh(),
                init_(nn.Linear(256, 1))
            )
    
    def forward(self, x):
        c = self.network(x)
        return c

class Worker_continuous(nn.Module):
    def __init__(self,):
        super(Worker_continuous, self).__init__()
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
        self.action_space = self.env.env.action_space.shape[0]
        self.max_action = float(self.env.env.action_space.high[0])
        self.network = policy(self.observation_space, self.action_space)
        self.old_network = policy(self.observation_space, self.action_space)
        self.critic = critic(self.observation_space)
        self.target = critic(self.observation_space)
        self.optimizer_new = torch.optim.Adam(self.network.network.parameters(), lr=self.learning_rate_a, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.optimizer_old = torch.optim.Adam(self.old_network.network.parameters(), lr=self.learning_rate_a, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_c, eps=self.method_conf['eps'], weight_decay=1e-6)
        self.lr_scheduler_new = optim.lr_scheduler.ExponentialLR(self.optimizer_new, self.method_conf['decay_rate'])
        self.lr_scheduler_old = optim.lr_scheduler.ExponentialLR(self.optimizer_old, self.method_conf['decay_rate'])
        self.lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(self.optimizer_critic, self.method_conf['decay_rate'])
        self.pi = Variable(torch.FloatTensor([math.pi]))

    def normal(self, x, mu, sigma_sq):
        a = ( -1 * (Variable(x)-mu).pow(2) / (2*sigma_sq) ).exp()
        b = 1 / ( 2 * sigma_sq * self.pi.expand_as(sigma_sq) ).sqrt()
        return a*b
    
    def gen_action(self, state):
        state = torch.from_numpy(state).float()
        mu, sigma = self.network(state)
        sigma = F.softplus(sigma)

        eps = torch.randn(mu.size())

        action = (mu + sigma.sqrt()*Variable(eps)).clamp(-self.max_action, self.max_action).data
        prob = self.normal(action, mu, sigma)

        log_prob = prob.log()
        return action, log_prob

    def gen_action_prob(self, state, action):
        state = torch.from_numpy(state).float()
        mu, sigma = self.old_network(state)
        sigma = F.softplus(sigma)
        prob = self.normal(action, mu, sigma)
        log_prob = prob.log()
        
        return log_prob
    
    def gen_action_prob_new(self, state, action):
        state = torch.from_numpy(state).float()
        mu, sigma = self.network(state)
        sigma = F.softplus(sigma)
        prob = self.normal(action, mu, sigma)
        log_prob = prob.log()
        
        return log_prob
    
    def gen_critic(self, state):
        state = torch.from_numpy(state).float()
        v = self.critic(state)

        return v
    
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
            reward_batch = []
            step = 0
            while True:
                step += 1
                action, action_prob = self.gen_action(state)

                state_batch.append(state)
                v = self.gen_critic(state)
                critic_batch.append(v)
                state, reward, done, _ = self.env.step(action.numpy())
                reward_batch.append(reward)
                r_batch.append(reward)
                action_batch.append(action)
                state_prime_batch.append(state)
                action_prob_batch.append(action_prob)

                if done:
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

        state_prime_batch = torch.as_tensor(state_prime_batch, dtype = torch.float32)
        r_batch = torch.as_tensor(r_batch, dtype = torch.float32)

        return batch_weights, state_batch, action_batch, action_prob_batch, critic_batch, state_prime_batch, r_batch
    
    def train(self, batch_size, step):
        returns, state_batch, action_batch, action_prob_batch, critic_batch, state_prime_batch, reward_batch = self.collect_trajectory(batch_size)

        s_batch = torch.as_tensor(state_batch, dtype = torch.float32)

        critic_prime = self.target(state_prime_batch)
        q = self.gamma * critic_prime.detach()
        q = q.squeeze()
        q = q + reward_batch.detach()
        
        loss = F.mse_loss(q, torch.stack(critic_batch).squeeze(-1))
        self.optimizer_critic.zero_grad()

        returns = returns.unsqueeze(-1).repeat(1, self.action_space)
        advantage = returns - torch.stack(critic_batch).detach()
        
        grad = [item.grad for item in self.network.parameters()]

        self.optimizer_new.zero_grad()
        batch_loss = -(torch.stack(action_prob_batch) * advantage).mean()
        batch_loss += loss
        batch_loss.backward()
        self.optimizer_critic.step()

        old_logp = []
        for idx, _ in enumerate(state_batch):
            action_prob = self.gen_action_prob(state_batch[idx], action_batch[idx])
            old_logp.append(action_prob)
        old_logp = torch.stack(old_logp)

        ratios = torch.exp(old_logp.detach() - torch.stack(action_prob_batch).detach())

        
        loss_old = -(old_logp * advantage * ratios).mean()
        self.optimizer_old.zero_grad()
        loss_old.backward()

        grad_old = [item.grad for item in self.old_network.parameters()]

        if grad[0] is not None:
            for idx,item in enumerate(self.network.parameters()):
                item.grad = item.grad + (1 - self.c * self.learning_rate_a**2) * (grad[idx] - grad_old[idx])
        
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
                state, reward, done, _ = self.env.step(action.numpy())
                sum_r += reward
                if done:
                    break
        file.write(str(i) + ':' + str(sum_r / 10) +'\n')
        print('worker test', sum_r / 10)
        return sum_r / 10
        
