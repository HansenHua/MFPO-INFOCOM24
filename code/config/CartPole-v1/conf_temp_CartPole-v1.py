from util import *

METHOD_CONF = {
    'learning_rate_a': 1e-4,
    'learning_rate_c': 1e-4,
    'gamma': 0.99,
    'eps': 1e-5,
    'env_name': 'CartPole-v1',
    'average_type': 'target',
    'fault_type': None,
    'num_worker': 10,
    'batch_size': 20,
    'local_update': 10,
    'c': 3,
    'decay_rate': 0.99,
    'decay_start_iter_id': 500,
}
