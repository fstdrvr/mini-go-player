import time

from host import GO
from random_player import RandomPlayer
# from my_player3 import MCTS
import pickle
import argparse

total_win_x = 0
total_win_o = 0
N = 5
verbose = True
player_1 = RandomPlayer()


def _global_helper_function():
    return [0, 0]


from host import GO
import numpy as np
import random
import pickle

SIMULATIONS = 2500

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


def playonegame(start_player, go):
    go.init_board(N)
    piece_type = 1
    go.set_board(piece_type, go.previous_board, go.board)
    go.n_move = 0

    player_stone = 2 if start_player == player_1 else 1
    if start_player == player_1:
        player = start_player
    else:
        player = 2
    while True:
        if player == player_1:
            # MOVE = player.get_input(go, piece_type)
            player_1.piece_type = piece_type
            # MOVE = player_1.move(go)
            MOVE = player.get_input(go, piece_type)
        else:
            player_2 = MCTS(go, N, piece_type, piece_type, False)
            MOVE = player_2.search()
        if MOVE != "PASS":
            action = "MOVE"
        else:
            action = "PASS"
        if action == "MOVE":
            # go.visualize_board()

            if not go.place_chess(MOVE[0], MOVE[1], piece_type):
                # print('Game end.')
                print('The winner is {}'.format('X' if 3 - piece_type == 1 else 'O'))
                if player_stone == 1 and piece_type == 2:
                    global total_win_x
                    total_win_x += 1
                elif player_stone != 1 and piece_type == 1:
                    global total_win_o
                    total_win_o += 1
                break

            go.died_pieces = go.remove_died_pieces(3 - piece_type)

        if go.game_end(piece_type, action):
            result = go.judge_winner()

            if verbose:
                go.visualize_board()
                print()
                # print('Game end.')
                if result == 0:
                    print('The game is a tie.')
                else:
                    print('The winner is {}'.format('X' if result == 1 else 'O'))
                    if player_stone == 1 and result == 1:
                        # global total_win_x
                        total_win_x += 1
                    elif player_stone != 1 and result == 2:
                        # global total_win_o
                        total_win_o += 1
            break

        piece_type = 2 if piece_type == 1 else 1
        player = 2 if player == player_1 else player_1

        if action == "PASS":
            go.previous_board = go.board

        go.n_move += 1


def playgames(gam_num, go):
    for i in range(0, gam_num, 2):
        print("=====game ", i, "=====")
        start_time = time.perf_counter()
        playonegame(player_1, go)
        print(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        print("=====game ", i + 1, "=====")
        playonegame(2, go)
        print(time.perf_counter() - start_time)

        with open('trans_table.pickle', 'wb') as handle:
            pickle.dump(trans_table, handle, protocol=min(pickle.HIGHEST_PROTOCOL, 4))

    print()
    print("win as X: ", total_win_x)
    print("win as O: ", total_win_o)


if __name__ == "__main__":
    with open('trans_table.pickle', 'rb') as handle:
        trans_table = pickle.load(handle)

    parser = argparse.ArgumentParser()
    parser.add_argument("--move", "-m", type=int, help="number of total moves", default=0)
    args = parser.parse_args()

    go = GO(N)
    playgames(args.move, go)

    with open('trans_table.pickle', 'wb') as handle:
        pickle.dump(trans_table, handle, protocol=min(pickle.HIGHEST_PROTOCOL, 4))
