from utils import *
import numpy as np

KOMI = 2.5
w_SCORE = 0.4
w_LIB = 0.3
w_TER = 0.3


class Node:
    """
    Tree node class for the search tree
    """

    def __init__(self, go, level, parent=None):
        self.go = go.copy_board()
        self.level = level
        self.parent = parent
        self.children = []
        self.move = None


class MINMAX:
    """Minmax player with alpha-beta pruning"""

    def __init__(self, piece_type, depth, step_count, root, bf):
        self.piece_type = piece_type
        self.depth = depth
        self.step_count = step_count
        self.root = root
        self.bf = bf

    def find_all_pieces(self, node, piece_type):
        """Find all pieces of both colors on the board"""
        pieces = []
        for i in range(len(node.go.board)):
            for j in range(len(node.go.board)):
                if node.go.board[i][j] == piece_type:
                    pieces.append((i, j))
        return pieces

    def calculate_current_score(self, node, piece_type):
        """Calculate current scores for both colors on the board"""
        pieces = self.find_all_pieces(node, piece_type)
        score = len(pieces)
        if piece_type == 2:
            return score + KOMI
        else:
            return score

    def calculate_total_liberties(self, node, piece_type):
        """Calculate total liberty counts for both colors on the board"""
        pieces = self.find_all_pieces(node, piece_type)
        liberties = 0
        for piece in pieces:
            for neighbour in node.go.detect_neighbor(piece[0], piece[1]):
                if node.go.board[neighbour[0]][neighbour[1]] == 0:
                    liberties += 1
        return liberties

    def find_biggest_territory(self, node, piece_type):
        """Find the biggest territory formed by both colors on the board"""
        pieces = self.find_all_pieces(node, piece_type)
        biggest_ter = 0
        for piece in pieces:
            allies = node.go.ally_dfs(piece[0], piece[1])
            if len(allies) > biggest_ter:
                biggest_ter = len(allies)
        return biggest_ter

    def evaluate_heuristic(self, node, piece_type):
        """Evaluate the heuristic of the moves for the given node"""
        my_score = self.calculate_current_score(node, piece_type)
        oppo_score = self.calculate_current_score(node, 3 - piece_type)
        my_score_heu, oppo_score_heu = my_score * w_SCORE, oppo_score * -w_SCORE

        my_lib = self.calculate_total_liberties(node, piece_type)
        oppo_lib = self.calculate_total_liberties(node, 3 - piece_type)
        my_lib_heu, oppo_lib_heu = my_lib * w_LIB, oppo_lib * -w_LIB

        my_ter = self.find_biggest_territory(node, piece_type)
        oppo_ter = self.find_biggest_territory(node, 3 - piece_type)
        my_ter_heu, oppo_ter_heu = my_ter * w_TER, oppo_ter * -w_TER

        return my_score_heu + my_lib_heu + my_ter_heu + oppo_score_heu + oppo_lib_heu + oppo_ter_heu

    def minmax(self, direction, node, alpha, beta, piece_type):
        """Calculate the min/max heuristics for minimizer and maximizer layers"""
        if direction == "max":
            score = -np.inf
        else:
            score = np.inf
        if node.level == self.depth:
            return self.evaluate_heuristic(node, 3 - piece_type)
        for i in range(len(node.go.board)):
            for j in range(len(node.go.board)):
                child = Node(node.go, node.level + 1)
                if child.go.valid_place_check(i, j, piece_type, test_check=True):
                    child.go.place_chess(i, j, piece_type)
                    child.go.remove_died_pieces(3 - piece_type)
                    child.move = (i, j)
                    if len(node.children) < self.bf:
                        node.children.append(child)
                        if direction == "max":
                            score = max(score, self.minmax("min", child, alpha, beta, 3 - piece_type))
                            if score >= beta:
                                return score
                            alpha = max(alpha, score)
                        else:
                            score = min(score, self.minmax("max", child, alpha, beta, 3 - piece_type))
                            if score <= alpha:
                                return score
                            beta = min(beta, score)
        if len(node.children) == 0:
            return self.evaluate_heuristic(node, 3 - piece_type)
        return score

    def build_tree(self, node):
        """Perform Alpha-beta pruning while building the search tree recursively"""
        result = None
        alpha, beta = -np.inf, np.inf
        if node.level % 2 == 0:
            piece_type = 3 - self.piece_type
        else:
            piece_type = self.piece_type
        for i in range(len(node.go.board)):
            for j in range(len(node.go.board)):
                child = Node(node.go, node.level + 1)
                if child.go.valid_place_check(i, j, piece_type, test_check=True):
                    child.go.place_chess(i, j, piece_type)
                    child.go.remove_died_pieces(3 - piece_type)
                    child.move = (i, j)
                    if len(node.children) < self.bf:
                        node.children.append(child)
                        best_score = self.minmax("min", child, alpha, beta, 3 - self.piece_type)
                        if best_score > alpha:
                            alpha = best_score
                            result = child.move
        return result

    def move(self):
        """Get player move based on Minmax and starting moves"""
        if self.step_count == 1:
            if self.root.go.board[2][2] == 0 and self.root.go.valid_place_check(2, 2, self.piece_type):
                return 2, 2
            elif self.root.go.board[2][1] == 0 and self.root.go.valid_place_check(2, 1, self.piece_type):
                return 2, 1
        move = self.build_tree(self.root)
        if not move:
            return "PASS"
        else:
            return move


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board, step_count = read_input(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    root = Node(go, 1)
    if step_count < 12:
        player = MINMAX(piece_type, 4, step_count, root, 25)
    else:
        player = MINMAX(piece_type, 6, step_count, root, 25)
    action = player.move()
    writeOutput(action)
