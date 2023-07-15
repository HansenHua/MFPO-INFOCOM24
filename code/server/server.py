from util import *
from agent.worker_continuous import Worker_continuous
from agent.worker_discrete import Worker_discrete
from env.env import Env

class Server:
    def __init__(self, method):
        self.method_conf = get_global_dict_value('method_conf')
        self.fault_type = self.method_conf['fault_type']
        self.env_name = self.method_conf['env_name']
        self.method = method
        self.c = self.method_conf['c']
        self.worker_list = []
        self.num_worker = self.method_conf['num_worker']
        self.cur_step = 0
        self.batch_size = self.method_conf['batch_size']
        self.average_type = self.method_conf['average_type']
        if self.env_name in ['Pong-v4', 'Breakout-v4', 'CartPole-v1']:
            self.master = Worker_discrete()
            for _ in range(self.num_worker):
                self.worker_list.append(Worker_discrete())
        else:
            self.master = Worker_continuous()
            for _ in range(self.num_worker):
                self.worker_list.append(Worker_continuous())

        self.env = Env(self.env_name)    
        
        self.master = copy.deepcopy(self.worker_list[0])
    
    def gen_dis(self, x, y):
        l = []
        for i in range(len(x)):
            l.append(torch.square(x[i] - y[i]).sum())
        return sum(l)
    
    def share_model(self, step):
        for worker in self.worker_list:
            worker.target = copy.deepcopy(worker.critic)
        
        self.cur_step = step
        g = []
        gradient = []
        for w in self.worker_list:
            grad = w.train(self.batch_size, step)
            gradient.append(grad)

        for idx, item in enumerate(self.master.network.parameters()):
            grad_item = []
            for i in range(self.num_worker):
                grad_item.append(gradient[i][idx])
            g.append(torch.stack(grad_item).mean(0))
        
        local_weights_target = []
        local_weights_critic = []
        local_weights_network = []
        for w in self.worker_list:
            local_weights_target.append(copy.deepcopy(w.target.state_dict()))
            local_weights_network.append(copy.deepcopy(w.network.state_dict()))
            local_weights_critic.append(copy.deepcopy(w.critic.state_dict()))

        global_weights_target = self.average_weights(local_weights_target)
        global_weights_network = self.average_weights(local_weights_network)
        global_weights_critic = self.average_weights(local_weights_critic)

        self.master.target.load_state_dict(global_weights_target)
        self.master.network.load_state_dict(global_weights_network)
        self.master.critic.load_state_dict(global_weights_critic)
        
        for id, w in enumerate(self.worker_list):
            if self.average_type == 'target':
                w.target = copy.deepcopy(self.master.target)
            if self.average_type == 'network':
                w.network = copy.deepcopy(self.master.network)
            if self.average_type == 'critic':
                w.critic = copy.deepcopy(self.master.critic)


    def train(self, step):
        self.cur_step = step
        g = []
        gradient = []
        for w in self.worker_list:
            grad = w.train(self.batch_size, step)
            gradient.append(grad)


    def log(self, ):
        score = self.master.test(self.cur_step)
    
    def test(self, ):
        if self.env_name in ['Pong-v4', 'Breakout-v4', 'CartPole-v1']:
            self.worker_test = Worker_discrete()
        else:
            self.worker_test = Worker_continuous()
        
        file = open('report_test.txt', 'a')
        sum_r = 0
        for _ in range(10):
            state = self.env.reset()
            while True:
                action, _ = self.worker_test.gen_action(state)
                if self.env_name in ['Pong-v4', 'Breakout-v4', 'CartPole-v1']:
                    action = action
                else:
                    action = action.numpy()
                state, reward, done, _ = self.env.step(action)
                sum_r += reward
                if done:
                    break
        file.write('test result :' + str(sum_r / 10) +'\n')
    
    def average_gradient(self, g, good_set):
        if self.fault_type == None:
            good_set = []
            for i in range(self.num_worker):
                good_set.append(i)
        g_avg = copy.deepcopy(g[good_set[0]])
        for key in range(len(g_avg)):
            for i in good_set[1:]:
                g_avg[key] += g[i][key]
            g_avg[key] = torch.div(g_avg[key], len(g))
        return g_avg
    
    def average_weights(self, w):
        set = []
        for i in range(self.num_worker):
            set.append(i)
        w_avg = copy.deepcopy(w[set[0]])
        for key in w_avg.keys():
            for i in set:
                if i == set[0]:
                    continue
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    def reset_grad(self, worker_id, g_list):
        for idx, item in enumerate(self.worker_list[worker_id].network.parameters()):
            item.grad = g_list[idx]
