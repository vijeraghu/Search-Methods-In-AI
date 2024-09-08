import collections
import numpy as np


class State:
    def __init__(self, bx, by, cx, cy, toys_collected):
        self.bx = bx
        self.by = by
        self.cx = cx
        self.cy = cy
        self.toys_collected = toys_collected


    def __eq__(self, other):
        return (self.bx, self.by, self.cx, self.cy, tuple(map(tuple, self.toys_collected))) == \
               (other.bx, other.by, other.cx, other.cy, tuple(map(tuple, other.toys_collected)))


    def __hash__(self):
        return hash((self.bx, self.by, self.cx, self.cy, tuple(map(tuple, self.toys_collected))))


def is_valid_move(x, y):
    return 0 <= x < 8 and 0 <= y < 8


def boy_moves(bx, by):
    moves = [
        (bx+2, by+1), (bx+2, by-1),
        (bx-2, by+1), (bx-2, by-1),
        (bx+1, by+2), (bx+1, by-2),
        (bx-1, by+2), (bx-1, by-2)
    ]
    return [(x, y) for x, y in moves if is_valid_move(x, y)]


def cat_move(cx, cy, bx, by):
    dx = bx - cx
    dy = by - cy
    if abs(dx) > abs(dy):
        return cx + (1 if dx > 0 else -1), cy
    elif abs(dy) > abs(dx):
        return cx, cy + (1 if dy > 0 else -1)
    else:
        return cx + (1 if dx > 0 else -1), cy


def bfs(initial_state, max_iterations=100000):
    queue = collections.deque([(initial_state, [])])
    visited = set()
    iterations = 0


    while queue and iterations < max_iterations:
        current_state, path = queue.popleft()
        iterations += 1
       
        if iterations % 1000 == 0:
            print(f"Iteration {iterations}, Queue size: {len(queue)}, Visited states: {len(visited)}")
       
        if np.all(current_state.toys_collected):
            print(f"Solution found after {iterations} iterations")
            return path
       
        if (current_state.bx, current_state.by) == (current_state.cx, current_state.cy):
            continue
       
        state_tuple = (current_state.bx, current_state.by, current_state.cx, current_state.cy,
                       tuple(map(tuple, current_state.toys_collected)))
       
        if state_tuple in visited:
            continue
       
        visited.add(state_tuple)
       
        for next_bx, next_by in boy_moves(current_state.bx, current_state.by):
            new_toys_collected = np.copy(current_state.toys_collected)
            if not new_toys_collected[next_bx][next_by]:
                new_toys_collected[next_bx][next_by] = 1
           
            next_cx, next_cy = cat_move(current_state.cx, current_state.cy, next_bx, next_by)
           
            if (next_bx, next_by) != (next_cx, next_cy):  # Ensure the boy doesn't move to the cat's position
                next_state = State(next_bx, next_by, next_cx, next_cy, new_toys_collected)
                queue.append((next_state, path + [(next_bx, next_by)]))
   
    print(f"No solution found after {iterations} iterations")
    return None


def solve_toddler_toy_trek(bx, by, cx, cy, toys):
    initial_state = State(bx, by, cx, cy, np.array(toys))
    solution = bfs(initial_state)
   
    if solution:
        print("Solution found!")
        print("Boy's path:", solution)
    else:
        print("No solution found.")


# Example usage
toys = [[0 for _ in range(8)] for _ in range(8)]
toys[1][1] = 1
toys[2][2] = 1
toys[7][7] = 1


solve_toddler_toy_trek(0, 0, 7, 7, toys)




import random


class Problem:
    def _init_(self, n, boy_pos, cat_pos, toys_collected):
        self.n = n
        self.initialState = (boy_pos, cat_pos, toys_collected)
        self.boy_moves = [[2, 1], [2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1]]  # Knight-like moves


    def goalTest(self, bx, by, cx, cy, toys_collected):
        if (bx, by) == (cx, cy):
            return False  # Game over if boy and cat collide
        for row in toys_collected:
            if 0 in row:
                return False
        return True  # all tosy collected


class Node:
    def _init_(self, problem, parent=None, action=None):
        if parent is None:
            self.boy_pos, self.cat_pos, self.toys_collected = problem.initialState
            self.boy_path = [self.boy_pos]  # boy's path
            self.cat_path = [self.cat_pos]  # cat's path
            self.visited_positions = {self.boy_pos}  # visited positions
        else:
            self.boy_pos = (parent.boy_pos[0] + action[0], parent.boy_pos[1] + action[1])
            self.toys_collected = [row[:] for row in parent.toys_collected]
            self.toys_collected[self.boy_pos[0] - 1][self.boy_pos[1] - 1] = 1  
            self.cat_pos = self.moveCat(parent.cat_pos, self.boy_pos)


       
            self.boy_path = parent.boy_path + [self.boy_pos]
            self.cat_path = parent.cat_path + [self.cat_pos]
            self.visited_positions = parent.visited_positions.copy()  # parent's visited positions are copied
            self.visited_positions.add(self.boy_pos)  # current position is marked visited


    def moveCat(self, cat_pos, boy_pos):
        cx, cy = cat_pos
        bx, by = boy_pos
        if cx < bx:
            cx += 1
        elif cx > bx:
            cx -= 1
        if cy < by:
            cy += 1
        elif cy > by:
            cy -= 1
        return cx, cy


