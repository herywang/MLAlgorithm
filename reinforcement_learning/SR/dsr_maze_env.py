import tkinter as tk
import numpy as np
import time

UNIT = 40  # pixels
MAZE_H = 8  # grid height
MAZE_W = 8  # grid width


class DSRMaze(tk.Tk, object):
    def __init__(self, title='maze'):
        super(DSRMaze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.feature_size = 2
        self.title(title)
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._hells = []
        self._build_maze()

        self.state_size = 2
        self.action_size = 4
        self.done = False

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        # draw line
        for r in range(0, UNIT * MAZE_H, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        # create hell
        hell_origin = np.array([[20 + UNIT * 3, 20 + UNIT * 1],
                                [20 + UNIT * 3, 20 + UNIT * 2],
                                [20 + UNIT * 3, 20 + UNIT * 3],
                                [20 + UNIT * 1, 20 + UNIT * 5],
                                [20 + UNIT * 1, 20 + UNIT * 6],
                                [20 + UNIT * 4, 20 + UNIT * 5],
                                [20 + UNIT * 5, 20 + UNIT * 5],
                                [20 + UNIT * 6, 20 + UNIT * 5],
                                [20 + UNIT * 5, 20]])
        for i in range(hell_origin.shape[0]):
            hell_center = hell_origin[i]
            hell = self.canvas.create_rectangle(hell_center[0] - 15, hell_center[1] - 15,
                                                hell_center[0] + 15, hell_center[1] + 15,
                                                fill='black')
            self._hells.append(hell)

        goal_origin = np.array([20 + UNIT * 7, 20 + UNIT * 1])
        self.goal = self.canvas.create_oval(goal_origin[0] - 15, goal_origin[1] - 15,
                                            goal_origin[0] + 15, goal_origin[1] + 15,
                                            fill='yellow')
        # 智能体初始中心坐标
        agent_origin = np.array([20, 20])
        self.agent = self.canvas.create_rectangle(agent_origin[0] - 15, agent_origin[1] - 15,
                                                  agent_origin[0] + 15, agent_origin[1] + 15, fill='red')
        self.canvas.pack()

    def reset(self):
        self.update()
        self.end = False
        self.canvas.delete(self.agent)
        origin = np.array([20, 20])
        self.agent = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                                  origin[0] + 15, origin[1] + 15,
                                                  fill='red')
        self.done = False
        # return state: the distance about agent position for goal position.
        return (np.array(self.canvas.coords(self.agent)[:2]) - np.array(self.canvas.coords(self.goal)[:2])) / (MAZE_H * UNIT)

    def step(self, action):
        # agent's position
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        reward = 0
        if action == 0 or action == 'u':   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1 or action == 'd':   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2 or action == 'r':   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3 or action == 'l':   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent
        next_coords = self.canvas.coords(self.agent)  # next state
        if next_coords == self.canvas.coords(self.goal):
            reward = 1
            done = True
        # elif self.is_hell(next_coords[0], next_coords[1]):
        #     reward = -1
        #     done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.goal)[:2]))/(MAZE_H*UNIT)
        self.done = done
        self.update()
        return s_, reward, done

    def is_hell(self, x, y):
        for hell in self._hells:
            hell_coord = self.canvas.coords(hell)
            if hell_coord[0] == x and hell_coord[1] == y:
                return True
        return False

    def render(self):
        self.update()

    def get_current_state(self):
        return np.expand_dims((np.array(self.canvas.coords(self.agent)[:2]) - np.array(self.canvas.coords(self.goal)[:2])) / (MAZE_H * UNIT), axis=0)



if __name__ == '__main__':
    q_maze = DSRMaze()
    hell = q_maze.is_hell(0, 0)
    while not hell:
        q_maze.step('r')
        hell = q_maze.is_hell(q_maze.canvas.coords(q_maze.agent)[0], q_maze.canvas.coords(q_maze.agent)[1])
        q_maze.update()
        time.sleep(1)
    q_maze.mainloop()
