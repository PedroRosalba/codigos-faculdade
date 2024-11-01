#Bruno Pereira de Paula from Impatech

#Questão 1

'''
Dado a função abaixo

def find_nb(data, point):
    Dt = data - point
    d = np.linalg.norm(Dt, axis=1)
    idt = np.argmin(d)
    return d[idt], idt

Sabemos que:
	"data" é uma lista de pontos da forma (x,y)∈ℝ²
	"point" é um ponto da forma (x,y)∈ℝ², iremos chamar de (x0, y0)

Temos que:
	"Dt" será todos os pontos de "data" subtraidos de "point", ou seja, basicamente essa operação muda o ponto de referência, mudando da origem para o "point".
	Note que Dt não é necessário para calcular a norma entre os pontos de "data" e o ponto "point", podendo ser feito da forma convencional.
	"d" será a lista de todas as distâncias em relação ao "point"
	"idt" achará o indice do menor valor de "d"
	e por último a função retorna o menor valor e o índice do ponto

	Em síntese, dado um conjunto de pontos, a função retorna a distância entre o ponto mais próximo e outro ponto dado.

Análise de complexidade:
	Note que o pior caso é se calcularmos a distância para todos os ponto e achar a menor distância, logo temos dois algorítimos que são O(n)
	Como dado f(x) e g(x), onde f(x) = O(n) e g(x) é O(n), seja h(x) = f(x) + g(x), então h(x) = O(n)
'''

#Questão 2

#Código já implementado

import random

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
		print_maze(maze)

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
		j = int(random.uniform(0, 2 * n))
		count += 1
		if maze[i][j] == room:
			maze[i][j] = cheese 
			break

	return maze

def print_maze(maze):
	for row in maze:
		print(" ".join(map(str, row)))

# //////////// Novo código /////////

def iter_generate_maze(m, n, room = "0", wall = "1", cheese = "."): 
	# Initialize a (2m + 1) x (2n + 1) matrix with all walls (1)
	maze = [[wall] * (2 * n + 1) for _ in range(2 * m + 1)]

	# Directions: (row_offset, col_offset) for N, S, W, E
	directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

	def iter_dfs(x0, y0): #faz o dfs de forma iterativa

		stack = [] #cria a pilha que agirá como o bfs
		stack.append((x0,y0, 0,0)) # para cada ponto, temos o ponto atual e o valor somado que faz ele retornar ao ponto anterior

		while stack:

			atual_room = stack.pop(-1) #tira o último termo

			if maze[2 * atual_room[0] + 1][2 * atual_room[1] + 1] == wall: #verifica a posição atual é uma parede

				maze[2*atual_room[0] + 1][ 2*atual_room[1] + 1] = room #transforma a posição atual em corredor

				maze[2*atual_room[0] + 1 + atual_room[2]][2*atual_room[1] + 1 + atual_room[3]] = room #transforma o caminho de onde veio em corredor

				random.shuffle(directions) #embaralha as direções

				for dx, dy in directions: # percorre as direções possíveis
					nx, ny = atual_room[0] + dx, atual_room[1] + dy #Soma a posição atual a direção, criando uma nova posição
					if 0 <= nx < m and 0 <= ny < n: # se caso essa posição estiver fora do labirinto, ela será eliminada
						stack.append((nx, ny, -dx, -dy)) # adiciona a nova posição na pilha

	#O trecho de código abaixo é equivalente ao mesmo trecho do algorítmo recursivo
	iter_dfs(0,0)
	count = 0

	while True: # placing the chesse
		i = int(random.uniform(0, 2 * m))
		j = int(random.uniform(0, 2 * n))
		count += 1
		if maze[i][j] == room:
			maze[i][j] = cheese 
			break

	return maze

# Examplos comparando a versão recursiva e a iterativa estão no final da questão 3

#Questão 3

def find_cheese(maze, cheese, x0 = 1, y0 = 1, visible_maze = False, iterative = True) -> str: #o parâmetro "visible_maze" irá decidir se o caminho será mostrado com o labirinto ou não

	wall = maze[0][0] #define a variável que será a parede
	room = maze[1][1] #define a variável que será uma sala

	if room == cheese:
		print("you already are in the cheese!")#verifica se o queijo está na posição inicial

	walked = [[0] * len(maze[0]) for _ in range(len(maze))] #cria uma matriz representando se uma local foi visitado ou não

	#versão recursiva
	def dfs_find_cheese(x, y, walk = ""):

		walked[x][y] = 1 #sinaliza para a matriz walked que o ponto ja foi visitado

		if maze[x][y] == cheese: #verifica se a posição atual é um queijo
			return walk
		else:
			walk_now = None #None representará o caminho que leva ao queijo

			#abaixo cada condição verifica uma direção, e aplica o dfs novamente caso essa direção seja válida
			#Note que pela natureja da geração do labirinto, só existe um caminho possível até o queijo, logo se caso umas dessas direções levar ao queijo, ela será única e não tem perigo de uma sobrepor a outra
			if maze[x][y-1] != wall and walked[x][y-1] == 0:
				walk_a = dfs_find_cheese(x, y-1, walk + "←") #A ilustração tem os eixos trocados, por isso as setinhas também estão trocadas
				if walk_a != None:
					walk_now = walk_a
			if maze[x-1][y] != wall and walked[x-1][y] == 0:
				walk_a = dfs_find_cheese(x-1, y, walk + "↑")#A ilustração tem os eixos trocados, por isso as setinhas também estão trocadas
				if walk_a != None:
					walk_now = walk_a
			if maze[x][y+1] != wall and walked[x][y+1] == 0:
				walk_a = dfs_find_cheese(x, y+1, walk + "→") #A ilustração tem os eixos trocados, por isso as setinhas também estão trocadas
				if walk_a != None:
					walk_now = walk_a
			if maze[x+1][y] != wall and walked[x+1][y] == 0:
				walk_a = dfs_find_cheese(x+1, y, walk + "↓") #A ilustração tem os eixos trocados, por isso as setinhas também estão trocadas
				if walk_a != None:
					walk_now = walk_a
			return walk_now

	#versão iterativa
	def iter_find_cheese(x, y):

		stack = [] # cria a pilha
		stack.append([x,y,""]) # adiciona o ponto inicial na pilha
		walked[x][y] = 1 # sinaliza que aquela posição já foi visitada

		while stack: # enquanto a pilha não estiver vazia

			atual_room = stack.pop(-1) #retira o ponto da pilha

			#atribui os valores
			walk = atual_room[2]
			nx = atual_room[0]
			ny = atual_room[1]

			if maze[nx][ny] == cheese: #verifica se achou o queijo
				return walk

			else: 
				#caso negativo cada condição irá verifica se a direção é váida, caso for, ele marca aquela posição como visitada e adiciona na pilha
				if maze[nx][ny-1] != wall and walked[nx][ny-1] == 0:
					walked[nx][ny-1] = 1
					stack.append([nx, ny-1, walk + "←"])
				if maze[nx-1][ny] != wall and walked[nx-1][ny] == 0:
					stack.append([nx-1,ny, walk + "↑"])
					walked[nx-1][ny] = 1
				if maze[nx][ny+1] != wall and walked[nx][ny+1] == 0:
					walked[nx][ny+1] = 1
					stack.append([nx,ny+1, walk + "→"])
				if maze[nx+1][ny] != wall and walked[nx+1][ny] == 0:
					walked[nx+1][ny] = 1
					stack.append([nx+1,ny, walk + "↓"])

	if iterative: # decide qual algorítmo vai usar
		walk_r = iter_find_cheese(x0,y0)
	else:
		walk_r = dfs_find_cheese(x0,y0)

	if visible_maze: #verifica se irá representar o caminho com o labirinto ou apenas o caminho
		maze_aux = maze.copy()
	else:
		maze_aux = [[' '] * len(maze[0]) for _ in range(len(maze))]

	#define x e y auxiliares
	px = x0  
	py = y0

	for arrow in walk_r: #para cada seta no caminho

		maze_aux[px][py] = arrow #atualiza a posição com a seta

		#e de acordo com a seta as variáveis são atualizadas para a nova posição 

		if arrow == "←":
			py-=1
		elif arrow == "↑":
			px-=1
		elif arrow == "→":
			py+=1
		else:
			px+=1

	print_maze(maze_aux) #printa o caminho

#Exemplos da questão 3 e 4

if __name__ == '__main__':
	m, n = 5, 7  # Grid size

	room = ' '
	wall = 'H'
	cheese = '●'

	maze1 = generate_maze(m, n, room, wall, cheese)
	print('\nRecursive Maze')
	print_maze(maze1)

	print("\nWalk 1 - No Maze")
	find_cheese(maze1, cheese)

	print("\nWalk 1 - Maze")
	find_cheese(maze1, cheese, visible_maze = True)

	maze2 = iter_generate_maze(m, n, room, wall, cheese)
	print('\nIterative Maze')
	print_maze(maze2)

	print("\nWalk 2 - DFS")
	find_cheese(maze2, cheese, iterative  = False)

	print("\nWalk2 - No DFS")
	find_cheese(maze2, cheese)

'''

O Algorítmo do exercício é um DFS, de forma geral "ele segue o caminho até o fim", antes de ir para o outro caminho mais próximo.
Esse algorítmo ele é muito extremo, pois ele pode acabar pegando o caminho certo e garantir o tempo optimal, ou ele pode errar o caminho e acabar verificando o labirinto todo "atoa".
De forma geral, o DFS ele é mais extremo em relação ao tempo de execução, enquanto o BFS consegue garantir um tempo o mesmo tempo médio, porém com menos variância.

o labirinto é definido com uma árvore em camadas, logo em geral o bfs é melhor, pois o tempo do algorítmo é proporcional a distância do queijo ao ponto inicial.
'''

#Questão 5

class Graph:
	def __init__(self):
		self.adjacency_list = {}
		self.vertex_values = {}

	# Verifica se há uma aresta de x para y
	def adjacent(self, x, y):
		if x in self.adjacency_list:
			return y in self.adjacency_list[x]
		return False #retorna falso caso x não for encontrado

	# Lista todos os vizinhos do vértice x
	def neighbors(self, x):
		if x in self.adjacency_list:
			return self.adjacency_list[x]
		return [] #retorna uma lista vazia caso x não for encontrado

	# Adiciona um vértice x ao grafo
	def add_vertex(self, x):
		if x not in self.adjacency_list:
			self.adjacency_list[x] = []
			self.vertex_values[x] = None  # Valor padrão None

	# Remove um vértice x do grafo
	def remove_vertex(self, x):
		if x in self.adjacency_list:
		# Remove o vértice da lista de adjacência de outros vértices
			for neighbors in self.adjacency_list.values():
				if x in neighbors:
					neighbors.remove(x)
			# Remover o vértice da lista de adjacência e dos valores de vértice
			del self.adjacency_list[x]
			del self.vertex_values[x]

	# Adiciona uma aresta de x para y
	def add_edge(self, x, y):
		if x in self.adjacency_list and y in self.adjacency_list:
			if y not in self.adjacency_list[x]:
				self.adjacency_list[x].append(y)

	# Remove uma aresta de x para y
	def remove_edge(self, x, y):
		if x in self.adjacency_list and y in self.adjacency_list[x]:
			self.adjacency_list[x].remove(y)

	# Retorna o valor associado ao vértice x
	def get_vertex_value(self, x):
		if x in self.vertex_values:
			return self.vertex_values[x]
		return None #retorna False caso x não for encontrado

	# Define o valor associado ao vértice x como v
	def set_vertex_value(self, x, v):
		if x in self.vertex_values:
			self.vertex_values[x] = v






