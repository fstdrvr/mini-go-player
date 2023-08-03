from read import readInput
from write import writeOutput
from host import GO
import numpy as np
import random
import pickle


def _global_helper_function():
    return [0, 0]


SIMULATIONS = 10000


class MCTS:
    def __init__(self, go_board, board_size, original, temp, is_terminal, parent_node=None, parent_action=None):
        self.go_board = go_board.copy_board()
        self.BOARD_SIZE = board_size
        self.board_state = self.go_board.board
        self.original_piece_type = original
        self.cur_piece_type = temp
        self.terminal = is_terminal
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.children = []
        self.potential_moves = self.get_moves()
        self.encoded_state = self.encode_state()

        # with open('qtable.pickle', 'rb') as handle:
        #     self.q_values = pickle.load(handle)
        # with open('hist_states.pickle', 'rb') as handle:
        #     self.history_states = pickle.load(handle)

    def encode_state(self):
        encoded_state = ''.join(
            [str(self.board_state[i][j]) for i in range(self.BOARD_SIZE) for j in range(self.BOARD_SIZE)])
        encoded_state += str(self.cur_piece_type)
        return encoded_state

    def get_moves(self):
        legal_moves = []
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.go_board.valid_place_check(i, j, self.cur_piece_type, test_check=True):
                    legal_moves.append((i, j))
        return legal_moves

    def calculate_value(self):
        return trans_table[self.encoded_state][1]

    def is_terminal(self):
        return self.terminal

    def is_fully_expanded(self):
        return not self.potential_moves

    def expand(self):
        move = self.potential_moves.pop()
        self.go_board.place_chess(move[0], move[1], self.cur_piece_type)
        self.go_board.died_pieces = self.go_board.remove_died_pieces(3 - self.cur_piece_type)
        self.go_board.n_move += 1
        child = MCTS(self.go_board, self.BOARD_SIZE, self.original_piece_type, 2 if self.cur_piece_type == 1 else 1,
                     self.go_board.game_end(self.cur_piece_type, move), parent_node=self, parent_action=move)
        self.children.append(child)
        return child

    def rollout(self):
        if self.terminal:
            result = self.go_board.judge_winner()
            if result == self.original_piece_type:
                return 1
            elif result == 0:
                return 0
            else:
                return -1
        while True:
            if self.potential_moves:
                random.shuffle(self.potential_moves)
                move = self.potential_moves.pop()
                # move = random.choice(self.potential_moves)
                self.go_board.place_chess(move[0], move[1], self.cur_piece_type)
                self.go_board.died_pieces = self.go_board.remove_died_pieces(3 - self.cur_piece_type)
            else:
                move = "PASS"
            if self.go_board.game_end(self.cur_piece_type, move):
                self.terminal = True
                result = self.go_board.judge_winner()
                if result == self.original_piece_type:
                    return 1
                elif result == 0:
                    return 0
                else:
                    return -1
            self.cur_piece_type = 2 if self.cur_piece_type == 1 else 1
            self.potential_moves = self.get_moves()
            self.go_board.n_move += 1

    def backpropogate(self, result):
        trans_table[self.encoded_state][0] += 1
        trans_table[self.encoded_state][1] += result
        if self.parent_node:
            self.parent_node.backpropogate(result)

    def select(self):
        current_node = self
        while not current_node.terminal:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                if current_node.children:
                    current_node = current_node.select_best_move()
                else:
                    break
        return current_node

    def select_best_move(self, C=np.sqrt(2)):
        child_encoded = [i.encoded_state for i in self.children]
        total_child_visit = sum([value[0] for key, value in trans_table.items() if key in child_encoded])
        values = [(trans_table[i.encoded_state][1] / trans_table[i.encoded_state][0]) +
                  C * np.sqrt(2 * np.log(1 + total_child_visit) / trans_table[i.encoded_state][0]) for i in
                  self.children]
        return self.children[np.argmax(values)]

    def search(self):
        if self.original_piece_type == 1 and (2, 2) in self.potential_moves and self.go_board.valid_place_check(
                2, 2, self.cur_piece_type, test_check=True):
            return 2, 2
        for i in range(SIMULATIONS):
            leaf_node = self.select()
            result = leaf_node.rollout()
            leaf_node.backpropogate(result)
        if not self.children:
            return "PASS"
        return self.select_best_move(C=0).parent_action


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    with open('trans_table.pickle', 'rb') as handle:
        trans_table = pickle.load(handle)
    player = MCTS(go, N, piece_type, piece_type, False)
    action = player.search()
    writeOutput(action)
    with open('trans_table.pickle', 'wb') as handle:
        pickle.dump(trans_table, handle, protocol=min(pickle.HIGHEST_PROTOCOL, 4))
