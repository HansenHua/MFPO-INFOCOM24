from util import *
from server.server import Server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='the name of environment')
    parser.add_argument('--method', type=str, default='MFPO', help='method name')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parser.add_argument('--local_update', type=int, default=20, help='frequency of local update')
    parser.add_argument('--num_worker', type=int, default=10, help='number of federated agents')
    parser.add_argument('--average_type', type=str, default='target', help='average type (target/network/critic)')
    parser.add_argument('--c', type=float, default=3, help='momentum parameter')


    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_a', type=float, default=1e-4, help='learning rate of actor')
    parser.add_argument('--lr_c', type=float, default=1e-4, help='learning rate of critic')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--max_iteration', type=int, default=2000, help='maximum training iteration')

    args = parser.parse_args()

    method_conf = importlib.import_module('config.' + args.env_name + '.conf_temp_' + args.env_name).METHOD_CONF

    global_dict_init()
    set_global_dict_value('method_conf', method_conf)

    method_conf['learning_rate_a'] = args.lr_a
    method_conf['learning_rate_c'] = args.lr_c
    method_conf['env_name'] = args.env_name
    method_conf['gamma'] = args.gamma
    method_conf['average_type'] = args.average_type
    method_conf['num_worker'] = args.num_worker
    method_conf['batch_size'] = args.batch_size
    method_conf['local_update'] = args.local_update
    method_conf['c'] = args.c


    fix_random_seed(args.seed)

    server = Server(args.method)

    config = get_global_dict_value('method_conf')

    if args.mode == 'train':
        for i in range(args.max_iteration):
            if i % args.local_update == 0:
                server.log()
                server.share_model(i)
            else:
                # server.test()
                server.train(i)

    if args.mode == 'test':
        server.test()
