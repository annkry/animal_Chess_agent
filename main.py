#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
An agent that will play against the agent created in the previous task will play based on the results returned by the MINIMAX algorithm with a heuristic function evaluating the situation on the board.
Choosing a move by our minimax agent will be about 3-4 times slower than the agent from the previous task.
The depth of the game tree in the minimax algorithm will be 3.

Heuristic function:
A weighted sum that counts possible moves and, depending on the pieces we have left, counts the values (PIECE_VALUES) for those pieces.
'''


import random
import sys
import copy
from cmath import inf
import time

LICZ_ROZGR = 4
MAX_DEPTH = 3
A = 1
B = 1


class WrongMove(Exception):
    def __init__(self):
        self.message = 'Wrong move'
        super().__init__(self.message)


class Jungle:
    PIECE_VALUES = {
        0: 4,
        1: 1,
        2: 2,
        3: 3,
        4: 5,
        5: 7,
        6: 8,
        7: 10
    }
    MAXIMAL_PASSIVE = 30
    DENS_DIST = 0.1
    MX = 7
    MY = 9
    traps = {(2, 0), (4, 0), (3, 1), (2, 8), (4, 8), (3, 7)}
    ponds = {(x, y) for x in [1, 2, 4, 5] for y in [3, 4, 5]}
    dens = [(3, 8), (3, 0)]
    dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    rat, cat, dog, wolf, jaguar, tiger, lion, elephant = range(8)

    def __init__(self):
        self.board = self.initial_board()
        self.pieces = {0: {}, 1: {}}

        for y in range(Jungle.MY):
            for x in range(Jungle.MX):
                C = self.board[y][x]
                if C:
                    pl, pc = C
                    self.pieces[pl][pc] = (x, y)
        self.curplayer = 0
        self.peace_counter = 0
        self.winner = None

    def initial_board(self):
        pieces = """
        L.....T
        .D...C.
        R.J.W.E
        .......
        .......
        .......
        e.w.j.r
        .c...d.
        t.....l
        """

        B = [x.strip() for x in pieces.split() if len(x) > 0]
        T = dict(zip('rcdwjtle', range(8)))

        res = []
        for y in range(9):
            raw = 7 * [None]
            for x in range(7):
                c = B[y][x]
                if c != '.':
                    if 'A' <= c <= 'Z':
                        player = 1
                    else:
                        player = 0
                    raw[x] = (player, T[c.lower()])
            res.append(raw)
        return res

    def random_move(self, player):
        ms = self.moves(player)
        if ms:
            return random.choice(ms)
        return None

    def can_beat(self, p1, p2, pos1, pos2):
        if pos1 in Jungle.ponds and pos2 in Jungle.ponds:
            return True  # rat vs rat
        if pos1 in Jungle.ponds:
            return False  # rat in pond cannot beat any piece on land
        if p1 == Jungle.rat and p2 == Jungle.elephant:
            return True
        if p1 == Jungle.elephant and p2 == Jungle.rat:
            return False
        if p1 >= p2:
            return True
        if pos2 in Jungle.traps:
            return True
        return False

    def pieces_comparison(self):
        for i in range(7, -1, -1):
            ps = []
            for p in [0, 1]:
                if i in self.pieces[p]:
                    ps.append(p)
            if len(ps) == 1:
                return ps[0]
        return None

    def rat_is_blocking(self, player_unused, pos, dx, dy):
        x, y = pos
        nx = x + dx
        for player in [0, 1]:
            if Jungle.rat not in self.pieces[1-player]:
                continue
            rx, ry = self.pieces[1-player][Jungle.rat]
            if (rx, ry) not in self.ponds:
                continue
            if dy != 0:
                if x == rx:
                    return True
            if dx != 0:
                if y == ry and abs(x-rx) <= 2 and abs(nx-rx) <= 2:
                    return True
        return False

    def draw(self):
        TT = {0: 'rcdwjtle', 1: 'RCDWJTLE'}
        for y in range(Jungle.MY):

            L = []
            for x in range(Jungle.MX):
                b = self.board[y][x]
                if b:
                    pl, pc = b
                    L.append(TT[pl][pc])
                else:
                    L.append('.')
            print(''.join(L))
        print('')

    def moves(self, player):
        res = []
        for p, pos in self.pieces[player].items():
            x, y = pos
            for (dx, dy) in Jungle.dirs:
                pos2 = (nx, ny) = (x+dx, y+dy)
                if 0 <= nx < Jungle.MX and 0 <= ny < Jungle.MY:
                    if Jungle.dens[player] == pos2:
                        continue
                    if pos2 in self.ponds:
                        if p not in (Jungle.rat, Jungle.tiger, Jungle.lion):
                            continue
                        if p == Jungle.tiger or p == Jungle.lion:
                            if dx != 0:
                                dx *= 3
                            if dy != 0:
                                dy *= 4
                            if self.rat_is_blocking(player, pos, dx, dy):
                                continue
                            pos2 = (nx, ny) = (x+dx, y+dy)
                    if self.board[ny][nx] is not None:
                        pl2, piece2 = self.board[ny][nx]
                        if pl2 == player:
                            continue
                        if not self.can_beat(p, piece2, pos, pos2):
                            continue
                    res.append((pos, pos2))
        return res

    def victory(self, player, mes):
        oponent = 1-player
        if len(self.pieces[oponent]) == 0:
            self.winner = player
            return True

        x, y = self.dens[oponent]
        if self.board[y][x]:
            self.winner = player
            return True

        if self.peace_counter >= Jungle.MAXIMAL_PASSIVE:
            r = self.pieces_comparison()
            if r is None:

                self.winner = 0  # draw is second player's victory
            else:
                """if mes == "gl0" or mes == "gl1":
                    print(r)"""
                self.winner = r
            return True
        return False

    def do_move(self, m):
        self.curplayer = 1 - self.curplayer
        if m is None:
            return
        pos1, pos2 = m
        x, y = pos1
        pl, pc = self.board[y][x]

        x2, y2 = pos2
        if self.board[y2][x2]:  # piece taken!
            pl2, pc2 = self.board[y2][x2]
            del self.pieces[pl2][pc2]
            self.peace_counter = 0
        else:
            self.peace_counter += 1

        self.pieces[pl][pc] = (x2, y2)
        self.board[y2][x2] = (pl, pc)
        self.board[y][x] = None

    def update(self, player, move_string, mes):
        self.curplayer = player
        move = tuple(int(m) for m in move_string.split(' '))
        if len(move) != 4:
            raise WrongMove
        possible_moves = self.moves(player)
        if not possible_moves:
            if move != (-1, -1, -1, -1):
                raise WrongMove
            move = None
        else:
            move = ((move[0], move[1]), (move[2], move[3]))
            if move not in possible_moves:
                raise WrongMove
        self.do_move(move)

        if self.victory(player, mes):
            assert self.winner is not None
            return 2 * self.winner - 1
        else:
            return None

    def heur(self):
        licz_1 = 0
        licz_2 = 0
        for i in range(7, -1, -1):
            if i in self.pieces[1]:
                licz_1 += Jungle.PIECE_VALUES[i]
            # else:
            #    licz_2 += Jungle.PIECE_VALUES[i]
            if i in self.pieces[0]:
                licz_2 += Jungle.PIECE_VALUES[i]
            # else:
            #    licz_1 += Jungle.PIECE_VALUES[i]
        licz1_1 = len(self.moves(1))
        licz1_2 = len(self.moves(0))
        return A*(licz_1-licz_2) + B*(licz1_1-licz1_2)

    def utility(self, p):
        if self.victory(p, "n") < 0:
            return (inf, 0, 0, 0, 0)
        if self.victory(p, "n") == 0:
            return (0, 0, 0, 0, 0)
        if self.victory(p, "n") > 0:
            return (-inf, 0, 0, 0, 0)

    def cut_off_tests(self, depth):
        if depth > MAX_DEPTH:
            return True
        return False

    def max_alpha_beta(self, alpha, beta, depth, start):
        px = None
        py = None
        px_2 = None
        py_2 = None
        player = 0
        mov = self.moves(player)
        if self.victory(player, "n"):
            return self.utility(player,)
        if self.cut_off_tests(depth):
            return (self.heur(), 0, 0, 0, 0)
        value = -inf
        moves = mov
        if mov != [None]:
            for iter in moves:
                (a, b), (c, d) = iter
                zbicie = 0
                # affect
                pl, pc = self.board[b][a]
                pl2, pc2 = 0, 0
                p_c = self.peace_counter
                if self.board[d][c]:  # piece taken!
                    pl2, pc2 = self.board[d][c]
                    del self.pieces[pl2][pc2]
                    self.peace_counter = 0
                    zbicie = 1
                else:
                    self.peace_counter += 1
                self.pieces[pl][pc] = (c, d)
                self.board[d][c] = (pl, pc)
                self.board[b][a] = None
                # end of affect
                (m, min_i, in_j, cos, cos1) = self.min_alpha_beta(
                    alpha, beta, depth + 1, False)
                if m > value:
                    value = m
                    px = a
                    py = b
                    px_2 = c
                    py_2 = d
                # undo everything
                if zbicie == 1:
                    self.pieces[pl2][pc2] = (c, d)
                    self.peace_counter = p_c
                    self.board[b][a] = (pl, pc)
                    self.board[d][c] = (pl2, pc2)
                    self.pieces[pl][pc] = (a, b)
                else:
                    self.peace_counter -= 1
                    self.board[b][a] = (pl, pc)
                    self.board[d][c] = None
                    self.pieces[pl][pc] = (a, b)
                # end of undo
                if value >= beta:
                    return (value, px, py, px_2, py_2)

                if value > alpha:
                    alpha = value
        else:
            (m, min_i, in_j, cos, cos1) = self.min_alpha_beta(
                alpha, beta, depth + 1, False)
            if m > value:
                value = m
            if value >= beta:
                return (value, px, py, px_2, py_2)

            if value > alpha:
                alpha = value

        return (value, px, py, px_2, py_2)

    def min_alpha_beta(self, alpha, beta, depth, start):
        player = 1
        qx = None
        qy = None
        qx1 = None
        qy1 = None
        mov = self.moves(player)
        if self.victory(player, "n"):
            return self.utility(player)
        if self.cut_off_tests(depth):
            return (self.heur(), 0, 0, 0, 0)
        minv = inf
        moves = mov
        if mov != [None]:
            for iter in moves:
                # affect
                (a, b), (c, d) = iter
                zbicie = 0
                pl, pc = self.board[b][a]
                pl2, pc2 = 0, 0
                p_c = self.peace_counter
                if self.board[d][c]:  # piece taken!
                    pl2, pc2 = self.board[d][c]
                    del self.pieces[pl2][pc2]
                    self.peace_counter = 0
                    zbicie = 1
                else:
                    self.peace_counter += 1
                self.pieces[pl][pc] = (c, d)
                self.board[d][c] = (pl, pc)
                self.board[b][a] = None
                # end of affect
                (m, max_i, max_j, cos, cos1) = self.max_alpha_beta(
                    alpha, beta, depth + 1, False)
                if m < minv:
                    minv = m
                    qx = a
                    qy = b
                    qx1 = c
                    qy1 = d
                # undo everything
                if zbicie == 1:
                    self.pieces[pl2][pc2] = (c, d)
                    self.peace_counter = p_c
                    self.board[b][a] = (pl, pc)
                    self.board[d][c] = (pl2, pc2)
                    self.pieces[pl][pc] = (a, b)
                else:
                    self.peace_counter -= 1
                    self.board[b][a] = (pl, pc)
                    self.board[d][c] = None
                    self.pieces[pl][pc] = (a, b)
                # koniec undo

                if minv <= alpha:
                    return (minv, qx, qy, qx1, qy1)

                if minv < beta:
                    beta = minv
        else:
            (m, max_i, max_j, cos, cos1) = self.max_alpha_beta(
                alpha, beta, depth + 1, False)
            if m < minv:
                minv = m
            if minv <= alpha:
                return (minv, qx, qy, qx1, qy1)

            if minv < beta:
                beta = minv

        return (minv, qx, qy, qx1, qy1)


class Player(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.game = Jungle()
        self.my_player = 1

    def say(self, what):
        sys.stdout.write(what)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def hear(self):
        line = sys.stdin.readline().split()
        return line[0], line[1:]

    def sym(self, J, move):
        (a, b), (c, d) = move
        move_string = str(a)+' '+str(b)+' '+str(c)+' '+str(d)
        J.curplayer = 1
        kon = J.update(1, move_string, "n")
        if kon != None:
            if kon == 1:
                return True
            else:
                return False
        player = 1
        op = 0
        while True:
            # my agent move - 0
            J.curplayer = 0
            moves = J.moves(0)
            if moves:
                move = random.choice(moves)
                (a, b), (c, d) = move
                move_string = str(a)+' '+str(b)+' '+str(c)+' '+str(d)
                kon = J.update(0, move_string, "n")
                if kon != None:
                    if kon == 1:
                        return True
                    else:
                        return False
                move = (move[0][0], move[0][1], move[1][0], move[1][1])
            else:
                J.do_move(None)
                move = (-1, -1, -1, -1)
            # move of my opponent - 1
            movess = J.moves(1)
            if movess:
                move = random.choice(movess)
                (a, b), (c, d) = move
                move_string = str(a)+' '+str(b)+' '+str(c)+' '+str(d)
                kon = J.update(1, move_string, "n")
                if kon != None:
                    if kon == 1:
                        return True
                    else:
                        return False
                move = (move[0][0], move[0][1], move[1][0], move[1][1])
            else:
                J.do_move(None)
                move = (-1, -1, -1, -1)

    def loop(self):
        player = 1
        op = 0
        while True:
            # self.game.draw()
            mov = self.game.moves(1)
            max_wygr = -1
            max_move = []
            for i in mov:
                licz = 0
                for j in range(LICZ_ROZGR):
                    if self.sym(copy.deepcopy(self.game), i):
                        licz += 1
                if max_wygr <= licz:
                    max_wygr = licz
                    max_move.append(i)
            (a, b), (c, d) = random.choice(max_move)
            move_string = str(a)+' '+str(b)+' '+str(c)+' '+str(d)
            kon = self.game.update(1, move_string, "gl1")
            if kon != None:
                if kon == 1:
                    return True
                else:
                    return False
            if player == 0:
                self.my_player = 0
            st = time.time()
            (m, aa, bb, cc, dd) = self.game.max_alpha_beta(-inf, inf, 0, True)
            en = time.time()
            move_string = str(aa)+' '+str(bb)+' '+str(cc)+' '+str(dd)
            if aa != None:
                self.game.update(0, move_string, "gl0")


if __name__ == '__main__':
    count_win = 0
    for i in range(0, 10):
        player = Player()
        if player.loop():  # agent 1 won
            print("The state in which my agent failed: ")
            player.game.draw()
        else:  # agent 0 won
            count_win += 1
    print("The agent won " + str(count_win) + " times.")
