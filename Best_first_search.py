import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import heapq

class Problem:
    def __init__(self, n, boy_pos, cat_pos, toys_collected):
        self.n = n
        self.initialState = (boy_pos, cat_pos, toys_collected)
        self.boy_moves = [[2, 1], [2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1]]  # Knight-like moves

    def goalTest(self, bx, by, cx, cy, toys_collected):
        if (bx, by) == (cx, cy):
            return False  # Game over if boy and cat collide
        for row in toys_collected:
            if 0 in row:
                return False  # Toys are still remaining
        return True  # All toys collected

    def heuristic(self, bx, by, cx, cy, toys_collected):
        # Heuristic function to estimate the cost to reach the goal
        min_toy_dist = float('inf')
        for i in range(self.n):
            for j in range(self.n):
                if toys_collected[i][j] == 0:  # Find uncollected toys
                    toy_dist = abs(bx - (i + 1)) + abs(by - (j + 1))  # Manhattan distance
                    min_toy_dist = min(min_toy_dist, toy_dist)
        
        cat_dist = abs(bx - cx) + abs(by - cy)  # Manhattan distance from the cat

        return min_toy_dist + cat_dist


class Node:
    def __init__(self, problem, parent=None, action=None):
        if parent is None:
            self.boy_pos, self.cat_pos, self.toys_collected = problem.initialState
            self.boy_path = [self.boy_pos]
            self.cat_path = [self.cat_pos]
            self.visited_positions = {(self.boy_pos[0], self.boy_pos[1])}
        else:
            self.boy_pos = (parent.boy_pos[0] + action[0], parent.boy_pos[1] + action[1])
            self.cat_pos = MoveCat(parent.cat_pos[0], parent.cat_pos[1], self.boy_pos[0], self.boy_pos[1])
            self.toys_collected = [row[:] for row in parent.toys_collected]
            self.toys_collected[self.boy_pos[0] - 1][self.boy_pos[1] - 1] = 1  # Collect toy at new position
            self.boy_path = parent.boy_path + [self.boy_pos]
            self.cat_path = parent.cat_path + [self.cat_pos]
            self.visited_positions = parent.visited_positions.copy()
            self.visited_positions.add((self.boy_pos[0], self.boy_pos[1]))
        
        # Calculate heuristic cost for this node
        self.heuristic_cost = problem.heuristic(self.boy_pos[0], self.boy_pos[1], self.cat_pos[0], self.cat_pos[1], self.toys_collected)

    def __lt__(self, other):
        return self.heuristic_cost < other.heuristic_cost


def MoveGen(n, bx, by, cx, cy, toys_collected):
    boy_dx = [2, 2, 1, 1, -1, -1, -2, -2]
    boy_dy = [1, -1, 2, -2, 2, -2, 1, -1]

    valid_moves = []
    for i in range(8):
        new_bx = bx + boy_dx[i]
        new_by = by + boy_dy[i]

        # Ensure the move is within bounds and the toy hasn't been collected
        if 1 <= new_bx <= n and 1 <= new_by <= n and toys_collected[new_bx - 1][new_by - 1] == 0:
            # Move the cat toward the new boy's position
            new_cx, new_cy = MoveCat(cx, cy, new_bx, new_by)
            # Ensure the boy and cat don't collide
            if (new_bx, new_by) != (new_cx, new_cy):
                new_toys_collected = [row[:] for row in toys_collected]
                new_toys_collected[new_bx - 1][new_by - 1] = 1
                valid_moves.append((new_bx, new_by, new_cx, new_cy, new_toys_collected))

    return valid_moves


def MoveCat(cx, cy, bx, by):
    # Move cat toward the boy's position
    if cx < bx:
        cx += 1
    elif cx > bx:
        cx -= 1
    if cy < by:
        cy += 1
    elif cy > by:
        cy -= 1
    return cx, cy


def GoalTest(bx, by, cx, cy, toys_collected):
    if (bx, by) == (cx, cy):
        return False  # Game over if boy and cat collide
    for row in toys_collected:
        if 0 in row:
            return False  # Toys are still remaining
    return True  # All toys collected


def best_first_search(problem):
    # Start from the root node
    node = Node(problem)
    if problem.goalTest(node.boy_pos[0], node.boy_pos[1], node.cat_pos[0], node.cat_pos[1], node.toys_collected):
        return node

    # Priority queue for Best-First Search (using a heap)
    frontier = []
    heapq.heappush(frontier, (node.heuristic_cost, node))
    
    # Set to keep track of explored positions (boy's positions only)
    explored = set()
    explored.add(node.boy_pos)  # Mark initial position as explored
    
    num_expanded_nodes = 0

    while frontier:
        _, node = heapq.heappop(frontier)  # Pop the node with the lowest heuristic cost
        num_expanded_nodes += 1

        # Print the current node's position
        print(f"Expanding node: Boy's position: {node.boy_pos}, Cat's position: {node.cat_pos}")

        # Check if this node is the goal state
        if problem.goalTest(node.boy_pos[0], node.boy_pos[1], node.cat_pos[0], node.cat_pos[1], node.toys_collected):
            return node

        # Generate child nodes based on possible moves
        for action in problem.boy_moves:
            new_bx = node.boy_pos[0] + action[0]
            new_by = node.boy_pos[1] + action[1]

            # Check if the move is within bounds
            if 1 <= new_bx <= problem.n and 1 <= new_by <= problem.n:
                new_pos = (new_bx, new_by)

                # Check if this new position has been visited already
                if new_pos not in explored:
                    child = Node(problem, node, action)

                    # Avoid cat-boy collision
                    if child.boy_pos != child.cat_pos:
                        # Mark this position as explored
                        explored.add(child.boy_pos)
                        
                        # Push the child into the frontier if it's not a goal state
                        heapq.heappush(frontier, (child.heuristic_cost, child))

    return None  # Return None if no solution is found




def checkAllCellsVisited(visited_positions, n):
    """Check if all grid cells were visited."""
    total_cells = n * n
    if len(visited_positions) == total_cells:
        print("All cells of the grid have been visited!")
    else:
        print(f"Only {len(visited_positions)} out of {total_cells} cells were visited.")


def animate_paths(boy_path, cat_path):
    grid_size = 8
    grid = np.zeros((grid_size, grid_size))

    # Set up the figure and axis for the plot
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    # Set grid limits
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Create a matrix to track visited cells
    visited_cells = np.zeros((grid_size, grid_size))

    # Highlight visited cells (initially empty)
    visited_patches = []

    for i in range(grid_size):
        for j in range(grid_size):
            patch = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='none')
            visited_patches.append(ax.add_patch(patch))

    # Markers for boy and cat
    boy_marker, = ax.plot([], [], 'bo', markersize=15, label='Boy')  # Blue circle for the boy
    cat_marker, = ax.plot([], [], 'ro', markersize=15, label='Cat')  # Red circle for the cat

    # Initialize the plot
    def init():
        boy_marker.set_data([], [])
        cat_marker.set_data([], [])
        return boy_marker, cat_marker

    # Update function for the animation
    def update(frame):
        boy_pos = boy_path[frame]
        cat_pos = cat_path[frame]

        # Update boy's and cat's position (place marker in the center of the cells)
        boy_marker.set_data(boy_pos[1] - 0.5, boy_pos[0] - 0.5)
        cat_marker.set_data(cat_pos[1] - 0.5, cat_pos[0] - 0.5)

        # Mark the boy's position as visited
        visited_cells[boy_pos[0] - 1, boy_pos[1] - 1] = 1

        # Update the color of the visited cells
        for i in range(grid_size):
            for j in range(grid_size):
                if visited_cells[i, j] == 1:
                    visited_patches[i * grid_size + j].set_facecolor('lightblue')

        # Print the current position of the boy
        print(f"Animating frame {frame}: Boy's position: {boy_pos}")

        return boy_marker, cat_marker, *visited_patches


# Function to get user inputs for the grid size, boy's position, and cat's position
def get_user_inputs():
    n = int(input("Enter the grid size (n x n): "))
    boy_start_x = int(input(f"Enter the boy's starting row (1 to {n}): "))
    boy_start_y = int(input(f"Enter the boy's starting column (1 to {n}): "))
    cat_start_x = int(input(f"Enter the cat's starting row (1 to {n}): "))
    cat_start_y = int(input(f"Enter the cat's starting column (1 to {n}): "))
    return n, (boy_start_x, boy_start_y), (cat_start_x, cat_start_y)


def find_solution_with_user_input():
    n, boy_start, cat_start = get_user_inputs()
    if boy_start != cat_start:  # Ensure the boy and cat start in different positions
        toys_collected = [[0] * n for _ in range(n)]  # No toys collected initially
        problem = Problem(n, boy_start, cat_start, toys_collected)
        solution_node = best_first_search(problem)
        if solution_node:
            return solution_node, boy_start, cat_start
    return None, boy_start, cat_start


# Example usage
def try_all_start_states(n):
    for boy_start_x in range(1, n+1):
        for boy_start_y in range(1, n+1):
            for cat_start_x in range(1, n+1):
                for cat_start_y in range(1, n+1):
                    boy_start = (boy_start_x, boy_start_y)
                    cat_start = (cat_start_x, cat_start_y)
                    
                    # Ensure the boy and cat don't start in the same position
                    if boy_start != cat_start:
                        print(f"Trying boy starting at {boy_start} and cat starting at {cat_start}...")
                        
                        toys_collected = [[0] * n for _ in range(n)]  # No toys collected initially
                        problem = Problem(n, boy_start, cat_start, toys_collected)
                        
                        solution_node = best_first_search(problem)
                        
                        if solution_node:
                            print(f"Solution found with boy starting at {boy_start} and cat starting at {cat_start}!")
                            print("Boy's path:", solution_node.boy_path)
                            print("Cat's path:", solution_node.cat_path)
                            print("Number of nodes expanded:", len(solution_node.boy_path))
                            checkAllCellsVisited(solution_node.visited_positions, n)
                            animate_paths(solution_node.boy_path, solution_node.cat_path)
                            return solution_node, boy_start, cat_start  # Return when a solution is found
                        else:
                            print(f"No solution found for this start state.")
    
    print("No solution found for any starting configuration.")
    return None, None, None  # No solution found for any start state


# Example usage for an 8x8 grid
solution_node, boy_start, cat_start = try_all_start_states(8)
if solution_node:
    print(f"Solution found with boy starting at {boy_start} and cat starting at {cat_start}!")
else:
    print("No solution was found with any starting positions.")
