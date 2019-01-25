import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATE = 6 # the length of the 1 dimensional word.
ACTION = ['left', 'right'] # available actions
EPSILON = 0.9 # greedy police
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # discount factor
MAX_EPISODES = 13 # maximum episodes. (回合)
FRESH_TIME = 0.3 # fresh time of one step of moving.

def build_q_table(n_state, actions):
    table = pd.DataFrame(np.zeros([n_state, len(actions)]), #q_table initial values.
        columns=actions# action's name
    )
    return table

def choose_action(state, q_table):
    # This is how to choose an action.
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTION)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    # This is how agent will interact
    if A == 'right':
        if S == N_STATE - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S+1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S-1
    return S_, R

def update_env(S, episode, step_counter):
    # This is how environment be updated.
    env_list = ['-']*(N_STATE-1) + ['T'] # our environment -----T
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(2)
        print('\r', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATE, ACTION)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 3
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict) # update
            S = S_

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
# if __name__ == '__main__':
#     i = 0
#     len_bar = 20
#     a = 0
#     while i<999999:
#         a = 999999 // 20
#         dot = i // a
#         print('\r' + dot * '=' + '>' + '.'*(20-dot) + '|' + '%.2f'% ((i / 999999) * 100) + '%', end='')
#         i += 1
