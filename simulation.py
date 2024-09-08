import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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


def dfs(problem):
    # Start from the root node
    node = Node(problem)
    if problem.goalTest(node.boy_pos[0], node.boy_pos[1], node.cat_pos[0], node.cat_pos[1], node.toys_collected):
        return node

    # Stack for DFS
    frontier = [node]
    explored = set()
    num_expanded_nodes = 0

    while frontier:
        node = frontier.pop()  # Pop the last node (DFS behavior)
        num_expanded_nodes += 1

        # Add to explored set only if the node is not an edge node or not previously visited
        if not is_edge(node.boy_pos[0], node.boy_pos[1], problem.n):
            explored.add((node.boy_pos, node.cat_pos))

        # Generate child nodes based on possible moves
        for action in problem.boy_moves:
            new_bx = node.boy_pos[0] + action[0]
            new_by = node.boy_pos[1] + action[1]

            # Check if the move is within bounds
            if 1 <= new_bx <= problem.n and 1 <= new_by <= problem.n:
                child = Node(problem, node, action)

                # Avoid cat-boy collision and handle revisiting based on edge condition
                if (child.boy_pos != child.cat_pos) and (
                    is_edge(child.boy_pos[0], child.boy_pos[1], problem.n) or (child.boy_pos, child.cat_pos) not in explored):
                    if problem.goalTest(child.boy_pos[0], child.boy_pos[1], child.cat_pos[0], child.cat_pos[1], child.toys_collected):
                        return child
                    frontier.append(child)

    return None


def is_edge(x, y, n):
    """Check if the position (x, y) is on the edge of the grid."""
    return x == 1 or x == n or y == 1 or y == n


def checkAllCellsVisited(visited_positions, n):
    """Check if all grid cells were visited."""
    total_cells = n * n
    if len(visited_positions) == total_cells:
        print("All cells of the grid have been visited!")
    else:
        print(f"Only {len(visited_positions)} out of {total_cells} cells were visited.")


# Animation code
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

        return boy_marker, cat_marker, *visited_patches

    # Animation settings with slower speed (interval in milliseconds)
    ani = animation.FuncAnimation(fig, update, frames=len(boy_path), init_func=init, blit=True, repeat=False, interval=500)

    # Show plot with labels
    plt.legend()
    plt.show()


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
        solution_node = dfs(problem)
        if solution_node:
            return solution_node, boy_start, cat_start
    return None, boy_start, cat_start


# Example usage
solution_node, boy_start, cat_start = find_solution_with_user_input()
if solution_node:
    print(f"Solution found with boy starting at {boy_start} and cat starting at {cat_start}!")
    print("Boy's path:", solution_node.boy_path)
    print("Cat's path:", solution_node.cat_path)
    print("Number of nodes expanded:", len(solution_node.boy_path))
    checkAllCellsVisited(solution_node.visited_positions, solution_node.boy_path[0][0])
    animate_paths(solution_node.boy_path, solution_node.cat_path)
else:
    print(f"No solution found with boy starting at {boy_start} and cat starting at {cat_start}.")


