def contains_lake(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_boundary(x, y):
        return x == 0 or y == 0 or x == rows - 1 or y == cols - 1

    def dfs(x, y):
        stack = [(x, y)]
        is_lake = True
        visited[x][y] = True
        while stack:
            cx, cy = stack.pop()
            if is_boundary(cx, cy):
                is_lake = False
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and matrix[nx][ny] == '0':
                    visited[nx][ny] = True
                    stack.append((nx, ny))
        return is_lake

    for i in range(1, rows - 1):  
        for j in range(1, cols - 1):  
            if matrix[i][j] == '0' and not visited[i][j]:
                if dfs(i, j):  
                    return True
    return False

matrix_with_lake = [
    ['1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '1'],
    ['1', '0', '1', '0', '1'],
    ['1', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1']
]

matrix_without_lake = [
    ['1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '1'],
    ['1', '0', '1', '0', '0'],
    ['1', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1']
]

contains_lake_with_lake = contains_lake(matrix_with_lake)
contains_lake_without_lake = contains_lake(matrix_without_lake)

print(contains_lake_with_lake, contains_lake_without_lake)