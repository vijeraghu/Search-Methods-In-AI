# Search-Methods-In-AI
#Toddler Toy Trek

#Problem Description
In a magical 8x8 play area, a little boy and his pet cat are on a thrilling adventure to collect
all the toys scattered across the grid. However, there's a catch! The cat, mischievous and
playful, can move on the grid too, but unlike the boy, it moves like a rook in chess
(horizontally or vertically).
The boy must move only in an L shape and collect all the toys to win the game. But he must
be careful! If the cat lands on the same square as the boy, the game will be over.

Your mission is to help the boy navigate the play area, gather all the toys, and avoid losing
to the cat.
#Why is this problem interesting?
This problem is interesting because it builds on the classical Knight’s Tour by adding new
constraints. The boy moves like a knight, but he must avoid the cat, which moves like a
rook, and collect toys on the grid. The dynamic interaction between the boy and the cat
adds real-time decision-making complexity. The challenge lies in balancing two
goals—avoiding the cat and collecting toys—while designing effective search algorithms
and heuristics to solve the problem efficiently. This makes it a rich exploration of state
space search, multi-objective optimization, and AI planning.
#State Representation:
The state is represented as a tuple (bx, by, cx, cy, toys_collected), where:
● (bx, by) represents the boy's current position on the 8x8 grid.
● (cx, cy) represents the cat's current position on the 8x8 grid.
● toys_collected is an 8x8 binary matrix, where:
● toys_collected[i][j] = 1 indicates that the toy on tile (i, j) has been collected.
● toys_collected[i][j] = 0 indicates that the toy on tile (i, j) is still there.
#Movement Rules
Boy's Movement:
● The boy moves like a knight in chess: two squares in one direction and one square
in a perpendicular direction.
● The boy does not revisit any visited cell.
#Cat's Movement:
● The cat moves like a rook in chess: it can move one square either horizontally or
vertically(up, down, left, right).
● The cat’s movement is determined by a simple rule: it always moves toward the
boy's current position, but it cannot move diagonally.
● The cat moves after the boy, and if it reaches the boy's position, the game ends.

#Note : Refer to the report for more insights.

