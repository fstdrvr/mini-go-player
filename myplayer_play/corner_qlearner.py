import random
import sys
from read import readInput
from write import writeOutput
import pickle
from host import GO
import numpy as np

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0


class QLearner:
    def __init__(self, alpha=.7, gamma=.9, initial_value=0.5, piece_type=None):
        self.type = 'MDP'
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        self.BOARD_SIZE = 5
        # self.board_state = None
        self.board_state_md = None
        self.board_state_tl = None
        self.board_state_tr = None
        self.board_state_bl = None
        self.board_state_br = None
        self.piece_type = piece_type
        self.alpha = alpha
        self.gamma = gamma
        self.initial_value = initial_value
        with open('qtable.pickle', 'rb') as handle:
            self.q_values = pickle.load(handle)
        with open('hist_states.pickle', 'rb') as handle:
            self.history_states = pickle.load(handle)

    def Q(self, state):
        # if state not in self.q_values and state[::-1] not in self.q_values:
        #     q_val = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        #     q_val.fill(self.initial_value)
        #     self.q_values[state] = q_val
        # elif state not in self.q_values:
        #     return np.flip(self.q_values[state[::-1]])
        # return self.q_values[state]
        if state not in self.q_values:
            q_val = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def _encode_state(self, go):
        board = go.board
        self.board_state_md = ''.join(
            [str(board[i][j]) for i in range(1,self.BOARD_SIZE-1) for j in range(1, self.BOARD_SIZE-1)])
        self.board_state_tl = ''.join(
            [str(board[i][j]) for i in range(self.BOARD_SIZE-2) for j in range(self.BOARD_SIZE-2)])
        self.board_state_tr = ''.join(
            [str(board[i][j]) for i in range(2, self.BOARD_SIZE) for j in range(self.BOARD_SIZE-2)])
        self.board_state_bl = ''.join(
            [str(board[i][j]) for i in range(self.BOARD_SIZE-2) for j in range(2, self.BOARD_SIZE)])
        self.board_state_br = ''.join(
            [str(board[i][j]) for i in range(2, self.BOARD_SIZE) for j in range(2, self.BOARD_SIZE)])
        # self.board_state = ''.join(
        #     [str(board[i][j]) for i in range(self.BOARD_SIZE) for j in range(self.BOARD_SIZE)])

    def _select_best_move(self, go):
        self._encode_state(go)
        q_values_dict = {'md': self.Q(self.board_state_md), 'tl': self.Q(self.board_state_tl),
                         'tr': self.Q(self.board_state_tr), 'bl': self.Q(self.board_state_bl),
                         'br': self.Q(self.board_state_br)}
        position, max_q_value, best_i, best_j = None, -1.0, None, None
        for k in ['md','tl', 'tr', 'bl', 'br']:
            for _ in range(9):
                coordinates = np.where(q_values_dict[k] == np.amax(q_values_dict[k]))
                if len(coordinates) == 1:
                    sub_i, sub_j = coordinates[0], coordinates[1]
                else:
                    sub_i, sub_j = coordinates[0][0], coordinates[1][0]
                if k == 'md':
                    i, j = sub_i+1, sub_j+1
                elif k == 'tl':
                    i, j = sub_i, sub_j
                elif k == 'tr':
                    i, j = sub_i+2, sub_j
                elif k == 'bl':
                    i, j = sub_i, sub_j+2
                else:
                    i, j = sub_i+2, sub_j+2
                if go.valid_place_check(i, j, self.piece_type, True):
                    if np.amax(q_values_dict[k]) > max_q_value:
                        max_q_value = np.amax(q_values_dict[k])
                        position = k
                        best_i = i
                        best_j = j
                else:
                    q_values_dict[k][sub_i][sub_j] = -1.0
        if max_q_value != -1.0:
            if position == 'md':
                return best_i, best_j, best_i-1, best_j-1, self.board_state_md
            elif position == "tl":
                return best_i, best_j, best_i, best_j, self.board_state_tl
            elif position == "tr":
                return best_i, best_j, best_i-2, best_j, self.board_state_tr
            elif position == "bl":
                return best_i, best_j, best_i, best_j-2, self.board_state_bl
            else:
                return best_i, best_j, best_i-2, best_j-2, self.board_state_br
        else:
            return -1, -1, -1, -1, None

    def move(self, go):
        row, col, sub_row, sub_col, state = self._select_best_move(go)
        if row >= 0 and col >= 0:
            self.history_states.append((state, (sub_row, sub_col)))
            # with open('hist_states.pickle', 'wb') as handle:
            #     pickle.dump(self.history_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open('qtable.pickle', 'wb') as handle:
            #     pickle.dump(self.q_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return row, col
        else:
            return "PASS"

    def learn(self, result):
        if result == 0:
            reward = DRAW_REWARD
        elif result == self.piece_type:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_states.reverse()
        max_q_value = -1.0
        for hist in self.history_states:
            state, move = hist
            q = self.Q(state)
            if max_q_value < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.amax(q)
        self.history_states = []
        # with open('qtable.pickle', 'wb') as handle:
        #     pickle.dump(self.q_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('hist_states.pickle', 'wb') as handle:
        #     pickle.dump(self.history_states, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = QLearner(piece_type=piece_type)
    action = player.move(go)
    writeOutput(action)