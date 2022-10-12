# 配置常量
CONFIG = {
    'kill_action': 30,  # 和棋回合数
    'dirichlet': 0.2,  # 经验值
    'play_out': 1200,  # 每次移动的模拟次数
    'c_puct': 5,  # u的权重
    'buffer_size': 100000,  # 经验池大小
    'pytorch_model_path': 'current_policy.pkl',  # pytorch模型路径
    'train_data_buffer_path': 'train_data_buffer.pkl',  # 数据容器的路径
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs': 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'train_update_interval': 600,  # 模型更新间隔时间
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
}
