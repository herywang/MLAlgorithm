from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # 初始化游戏环境
        observation = env.reset()

        while True:
            # 刷新游戏环境
            env.render()
            # RL根据当前状态选择一个action
            action = RL.choose_action(observation)
            # RL 选择一个action后, agent有一个新的观测值, 同时获得一个奖励, 和是否结束游戏的标识位: done
            observation_, reward, done = env.step(action)
            # 存储到replay memory中, 用于后面随机选择(s, a, r, s_)进行训练
            RL.store_transition(observation, action, reward, observation_)
            # 探索次数大于200时开始学习是为了防止replay memory中没得训练数据; 然后每探索5步训练一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            # 更改环境当前状态
            observation = observation_
            # 找到宝藏, 游戏结束, 退出这一次玩耍, 进入下一次的训练
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    print("actions num: ", env.n_actions, "\nfeatures: ", env.n_features)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()