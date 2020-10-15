# 找金币MDP实例
# |-------------------|
# | 1 | 2 | 3 | 4 | 5 |
# | x |   | o |   | x |
# |-------------------|
#   6       7       8

import gym
import time

# class GridEnv(gym.Env):
#
#     def __init__(self):
#         self.states = [1,2,3,4,5,6,7,8]     #全部状态空间
#         self.actions = ['n', 'e', 's', 'w'] #全部动作空间
#         self.rewards = dict()               #回报值
#         self.rewards['1_s'] = -1.0
#         self.rewards['3_s'] = 1.0
#         self.rewards['5_s'] = -1.
#         self.t = dict()                     #状态转移
#         self.t['1_s'] = 6
#         self.t['1_e'] = 2
#         self.t['2_w'] = 1
#         self.t['2_e'] = 3
#         self.t['3_s'] = 7
#         self.t['3_w'] = 2
#         self.t['3_e'] = 4
#         self.t['4_w'] = 3
#         self.t['4_e'] = 5
#         self.t['5_s'] = 8
#         self.t['5_w'] = 4
#         self.terminal_states = dict()       # 终止状态
#         self.terminal_states[6] = 1
#         self.terminal_states[7] = 1
#         self.terminal_states[8] = 1
#
#         self.x = [140, 220, 300, 380, 460, 140, 300, 460]
#         self.y = [250, 250, 250, 250, 250, 150, 150, 150]
#
#         self.state = None
#         self.gamma = 0.8
#         self.viewer = None
#
#     def step(self, action):
#         state = self.state
#         if state in self.terminal_states:
#             return state, 0, True, {}   #返回下一时刻的状态,reward, 是否终止,调试信息
#         key = "%d_%s"%(state,action)    #将状态-动作组成字典的键值
#         if key in self.t:
#             next_state = self.t[key]
#         else:
#             next_state = state
#         if next_state in self.terminal_states:
#             is_terminal = True
#         else:
#             is_terminal = False
#
#         if key not in self.rewards:
#             r = 0
#         else:
#             r = self.rewards[key]
#         return next_state, r, is_terminal, {}
#
#     def reset(self):
#         self.state = self.states[int(random.random() * len(self.states))]
#         return self.state
#
#     def render(self, mode='human', close=False):
#         if close:
#             if self.viewer is not None:
#                 self.viewer.close()
#                 self.viewer = None
#             return
#         screen_width = 600
#         screen_height = 400
#         if self.viewer is None:
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.Viewer(screen_width, screen_height)
#             # 创建网格世界
#             self.line1 = rendering.Line((100, 300), (500, 300))
#             self.line2 = rendering.Line((100, 200), (500, 200))
#             self.line3 = rendering.Line((100, 300), (100, 100))
#             self.line4 = rendering.Line((180, 300), (180, 100))
#             self.line5 = rendering.Line((260, 300), (260, 100))
#             self.line6 = rendering.Line((340, 300), (340, 100))
#             self.line7 = rendering.Line((420, 300), (420, 100))
#             self.line8 = rendering.Line((500, 300), (500, 100))
#             self.line9 = rendering.Line((100, 100), (180, 100))
#             self.line10 = rendering.Line((260, 100), (340, 100))
#             self.line11 = rendering.Line((420, 100), (500, 100))
#             # 创建第一个骷髅
#             self.kulo1 = rendering.make_circle(40)
#             self.circletrans = rendering.Transform(translation=(140, 150))
#             self.kulo1.add_attr(self.circletrans)
#             self.kulo1.set_color(0, 0, 0)
#             # 创建第二个骷髅
#             self.kulo2 = rendering.make_circle(40)
#             self.circletrans = rendering.Transform(translation=(460, 150))
#             self.kulo2.add_attr(self.circletrans)
#             self.kulo2.set_color(0, 0, 0)
#             # 创建金条
#             self.gold = rendering.make_circle(40)
#             self.circletrans = rendering.Transform(translation=(300, 150))
#             self.gold.add_attr(self.circletrans)
#             self.gold.set_color(1, 0.9, 0)
#             # 创建机器人
#             self.robot = rendering.make_circle(30)
#             self.robotrans = rendering.Transform()
#             self.robot.add_attr(self.robotrans)
#             self.robot.set_color(0.8, 0.6, 0.4)
#
#             self.line1.set_color(0, 0, 0)
#             self.line2.set_color(0, 0, 0)
#             self.line3.set_color(0, 0, 0)
#             self.line4.set_color(0, 0, 0)
#             self.line5.set_color(0, 0, 0)
#             self.line6.set_color(0, 0, 0)
#             self.line7.set_color(0, 0, 0)
#             self.line8.set_color(0, 0, 0)
#             self.line9.set_color(0, 0, 0)
#             self.line10.set_color(0, 0, 0)
#             self.line11.set_color(0, 0, 0)
#
#             self.viewer.add_geom(self.line1)
#             self.viewer.add_geom(self.line2)
#             self.viewer.add_geom(self.line3)
#             self.viewer.add_geom(self.line4)
#             self.viewer.add_geom(self.line5)
#             self.viewer.add_geom(self.line6)
#             self.viewer.add_geom(self.line7)
#             self.viewer.add_geom(self.line8)
#             self.viewer.add_geom(self.line9)
#             self.viewer.add_geom(self.line10)
#             self.viewer.add_geom(self.line11)
#             self.viewer.add_geom(self.kulo1)
#             self.viewer.add_geom(self.kulo2)
#             self.viewer.add_geom(self.gold)
#             self.viewer.add_geom(self.robot)
#
#         if self.state is None: return None
#         self.robotrans.set_translation(self.x[self.state - 1], self.y[self.state - 1])
#
#         return self.viewer.render(return_rgb_array=mode == 'rgb_array')

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    env.reset()
    env.render()
    # env.env.close()
    time.sleep(10)
    env.close()


