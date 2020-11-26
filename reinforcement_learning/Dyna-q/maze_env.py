import tkinter as tk
import numpy as np

UNIT = 40  # pixels
MAZE_H = 8  # grid height
MAZE_W = 8  # grid width


class DynaQMaze(tk.Tk, object):
    def __init__(self):
        super(DynaQMaze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.hell =[]
        self._build_maze()

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
            hell = self.canvas.create_rectangle(hell_center[0] - 19, hell_center[1] - 19,
                                                      hell_center[0] + 19, hell_center[1] + 19,
                                                      fill='black')
            self.hell.append(hell)

        goal_origin = np.array([20 + UNIT * 7, 20 + UNIT * 1])
        self.goal = self.canvas.create_oval(goal_origin[0] - 15, goal_origin[1] - 15,
                                goal_origin[0] + 15, goal_origin[1] + 15,
                                fill='yellow')

        agent_origin = np.array([20, 20])
        self.agent = self.canvas.create_rectangle(agent_origin[0] - 15, agent_origin[1]-15,
                                     agent_origin[0] + 15, agent_origin[1]+15,fill='red')

        self.canvas.pack()


if __name__ == '__main__':
    q_maze = DynaQMaze()
    q_maze.update()
    q_maze.mainloop()
