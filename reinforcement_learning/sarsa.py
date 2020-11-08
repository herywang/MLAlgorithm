# Sarsa强化学习算法
import numpy as np
import pandas as pd
import tkinter as tk

class RL(object):
    def __init__(self,action_space, learning_rate=0.01, reward_decay=0.9,e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy     #随机选择action的概率,默认0.1的概率随机选择action

        self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)

    def check_stat_exist(self, state):
        if state not in self.q_table.index:
            # 如果当前状态不在q表中,那么在q表中添加一个这个状态
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns,name=state)
            )

    def choose_action(self, observation):
        self.check_stat_exist(observation)  # 首先检查当前状态是否存在
        if np.random.rand() < self.epsilon:
            state_actions = self.q_table.loc[observation, :]
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self,*args):
        pass

# off-policy学习方式
class QLearningTable(RL):
    def __init__(self, actions, learning_rate = 0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable,self).__init__(actions, learning_rate, reward_decay,e_greedy)

    def learn(self,s,a,r,s_):
        self.check_stat_exist(s_)
        q_predict = self.q_table.loc[s,a]   #当前状态采取action a的reward
        if s_ != 'terminal':
            q_target = r+self.gamma*np.max(self.q_table.loc[s_, :])
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr*(q_target - q_predict) #更新q表

# on-policy学习方式
class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self,s, a, r, s_, a_):
        self.check_stat_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr*(q_target - q_predict) #更新q表

class Maze(tk.Tk, object):
    def __init__(self):
        self.UNIT = 40
        self.MAZE_H = 10
        self.MAZE_W = 10
        super(Maze,self).__init__(screenName="强化学习environment")
        self.action_spaces = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_spaces)
        self.title("maze")
        self.geometry("{0}x{1}".format(self.MAZE_H*self.UNIT, self.MAZE_W*self.UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',height=self.MAZE_H * self.UNIT,width=self.MAZE_W * self.UNIT)
        # create grids
        for c in range(0, self.MAZE_W * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_H * self.UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.MAZE_H * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_W * self.UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        # create origin
        origin = np.array([20, 20])
        # hell1
        hell1_center = origin + np.array([self.UNIT * 2, self.UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell2
        hell2_center = origin + np.array([self.UNIT, self.UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        # hell3
        hell3_center = origin + np.array([self.UNIT*5, self.UNIT * 6])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
        # create 创建一个椭圆
        oval1_center = origin + self.UNIT * 9
        self.oval1 = self.canvas.create_oval(
            oval1_center[0] - 15, oval1_center[1] - 15,
            oval1_center[0] + 15, oval1_center[1] + 15,
            fill='yellow')
        oval2_center = origin + self.UNIT*5
        self.oval2 = self.canvas.create_oval(
            oval2_center[0] - 15, oval2_center[1] - 15,
            oval2_center[0] + 15, oval2_center[1] + 15,
            fill='yellow'
        )
        # create red rect
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,origin[0] + 15, origin[1] + 15,fill='red')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > self.UNIT:
                base_action[1] -= self.UNIT
        elif action == 1:  # down
            if s[1] < (self.MAZE_H - 1) * self.UNIT:
                base_action[1] += self.UNIT
        elif action == 2:  # right
            if s[0] < (self.MAZE_W - 1) * self.UNIT:
                base_action[0] += self.UNIT
        elif action == 3:  # left
            if s[0] > self.UNIT:
                base_action[0] -= self.UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)  # next state
        # reward function
        if s_ == self.canvas.coords(self.oval1):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ == self.canvas.coords(self.oval2):
            reward = 1
            done = False
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False
        return s_, reward, done

    def render(self):
        # time.sleep(0.1)
        self.update()

def update():
    for episode in range(100):
        # 初始化观测值
        observation = env.reset()
        # 根据当前观测值从actions中根据当前策略选择一个action
        action = RL.choose_action(str(observation))
        while True:
            # fresh env
            env.render()
            # agent执行action, 获取下一个状态,并获得一个奖励,与是否结束此轮游戏
            observation_, reward, done = env.step(action)
            # agent根据下一个状态值,然后再根据当前策略选择一个action_
            action_ = RL.choose_action(str(observation_))
            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)
            # swap observation and action
            observation = observation_
            action = action_
            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()


