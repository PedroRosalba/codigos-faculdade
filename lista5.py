import random
from collections import deque

def generate_maze(m, n, room = 0, wall = 1, cheese = '.' ):
    # Initialize a (2m + 1) x (2n + 1) matrix with all walls (1)
    maze = [[wall] * (2 * n + 1) for _ in range(2 * m + 1)]

    # Directions: (row_offset, col_offset) for N, S, W, E
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(x, y):
        """Recursive DFS to generate the maze."""
        # Mark the current cell as visited by making it a path (room)
        maze[2 * x + 1][2 * y + 1] = room

        # Shuffle the directions to create a random path
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy  # New cell coordinates
            if 0 <= nx < m and 0 <= ny < n and maze[2 * nx + 1][2 * ny + 1] == wall:
                # Open the wall between the current cell and the new cell
                maze[2 * x + 1 + dx][2 * y + 1 + dy] = room
                # Recursively visit the new cell
                dfs(nx, ny)

    # Start DFS from the top-left corner (0, 0) of the logical grid
    dfs(0, 0)
    count = 0
    while True: # placing the chesse
        i = int(random.uniform(0, 2 * m))
        j = int(random.uniform(0, 2 * m))
        count += 1
        if maze[i][j] == room:
            maze[i][j] = cheese 
            break

    return maze

def print_maze(maze):
    for row in maze:
        print(" ".join(map(str, row)))

def find_path_to_cheese(maze, start=(1, 1), cheese='.'):
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
    
    # Initialize queue for BFS and a dictionary to track the path
    queue = deque([start])
    came_from = {start: None}
    
    while queue:
        x, y = queue.popleft()
        
        # Check if we've found the cheese
        if maze[x][y] == cheese:
            path = []
            # Start backtracking from cheese to start
            while (x, y) != start:
                path.append((x, y))
                x, y = came_from[(x, y)]
            path.append(start)  # Add the start position at the end
            path.reverse()  # Reverse to get path from start to cheese
            return path  # Return the path from start to cheese
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] != 1 and (nx, ny) not in came_from:
                queue.append((nx, ny))
                came_from[(nx, ny)] = (x, y)
    
    return None  # No path found



def print_maze_with_path(maze, path):
    maze_copy = [row[:] for row in maze]
    for (x, y) in path:
        if maze_copy[x][y] != '.':
            maze_copy[x][y] = '*'
    print_maze(maze_copy)

# Example usage:
if __name__ == '__main__':
    m, n = 3, 10
    random.seed(10)
    maze = generate_maze(m, n)
    print("Generated Maze:")
    print_maze(maze)

    # Find path to cheese
    path = find_path_to_cheese(maze)
    if path:
        print("\nMaze with Path to Cheese:")
        print_maze_with_path(maze, path)
    else:
        print("No path found to the cheese.")
