from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.optimize import fsolve
import time
# def find_judge(n, trust):
#     # Inicializando dois arrays: 
#     # trust_count - para contar quantas pessoas confiam em uma pessoa
#     # trusted_by - para contar quantas pessoas uma pessoa confia
#     trust_count = [0] * (n + 1)
#     trusted_by = [0] * (n + 1)
    
#     # Percorrendo a lista de trust e atualizando trust_count e trusted_by
#     for a, b in trust:
#         trust_count[b] += 1  # b recebe a confiança de a
#         trusted_by[a] += 1   # a confia em b
    
#     # Procurando pelo juiz
#     for i in range(1, n + 1):
#         # Se uma pessoa é o juiz, ela deve ser confiada por todos (n-1) e não deve confiar em ninguém
#         if trust_count[i] == n - 1 and trusted_by[i] == 0:
#             return i
    
#     # Se nenhum juiz foi encontrado, retorna -1
#     return -1

# # Testes
# t = [[1, 2], [1, 3], [2, 3]]
# n = 3
# print(find_judge(n, t))  # Saída: 3

# t = [[1, 3], [2, 3], [3, 1]]
# n = 3
# print(find_judge(n, t))  # Saída: -1

# Questao 2


#Questao 3

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import lagrange

# # Dados fornecidos
# altura = np.array([200, 400, 600, 800, 1000, 1200, 1400])
# temperatura = np.array([15, 9, 5, 3, -2, -5, -15])

# # Calculando o polinômio interpolador usando Lagrange
# polinomio = lagrange(altura, temperatura)

# # Exibindo o polinômio
# print("Polinômio interpolador:")
# print(polinomio)

# # (a) Encontrar a altura onde a temperatura é 0 graus Celsius
# # Usaremos uma busca numérica para encontrar a raiz do polinômio em que a temperatura é igual a 0
# from scipy.optimize import fsolve
# altura_zero_graus = fsolve(lambda x: polinomio(x), 0)
# print(f"A altura onde a temperatura é provavelmente 0 graus Celsius: {altura_zero_graus[0]:.2f} metros")

# # (b) Encontrar a temperatura a 700 metros
# temperatura_700m = polinomio(700)
# print(f"A temperatura a 700 metros de altura: {temperatura_700m:.2f} graus Celsius")

# # Plot para visualizar a interpolação
# x_vals = np.linspace(200, 1400, 1000)
# y_vals = polinomio(x_vals)

# plt.plot(x_vals, y_vals, label='Polinômio Interpolador')
# plt.scatter(altura, temperatura, color='red', label='Pontos Dados')
# plt.xlabel('Altura (metros)')
# plt.ylabel('Temperatura (Celsius)')
# plt.axhline(0, color='gray', linestyle='--')
# plt.title('Interpolação: Altura vs Temperatura')
# plt.legend()
# plt.grid(True)
# plt.show()


# class Domain:
#     min = None
#     max = None

#     def __contains__(self, x):
#         raise NotImplementedError
    
#     def __repr__(self):
#         raise NotImplementedError

#     def __str__(self):
#         return self.__repr__()
    
#     def copy(self):
#         raise NotImplementedError 


# class Interval(Domain):
#     def __init__(self, p1, p2):
#         self.inff, self.supp = min(p1, p2), max(p1, p2)
    
#     @property
#     def min(self):
#         return self.inff

#     @property
#     def max(self):
#         return self.supp
    
#     @property
#     def size(self):
#         return (self.max - self.min)
    
#     @property
#     def haf(self):
#         return (self.max + self.min)/2.0
    
#     def __contains__(self, x):
#         return  np.all(np.logical_and(self.inff <= x, x <= self.supp))

#     def __str__(self):
#         return f'[{self.inff:2.4f}, {self.supp:2.4f}]' 

#     def __repr__(self):
#         return f'[{self.inff!r:2.4f}, {self.supp!r:2.4f}]'
    
#     def copy(self):
#         return Interval(self.inff, self.supp)


# class RealFunction:
#     f = None
#     prime = None
#     domain = None
    
#     def eval_safe(self, x):
#         if self.domain is None or x in self.domain:
#             return self.f(x)
#         else:
#             raise Exception("The number is out of the domain")

#     def prime_safe(self, x):
#         if self.domain is None or x in self.domain:
#             return self.prime(x)
#         else:
#             raise Exception("The number is out of the domain")
        
#     def __call__(self, x) -> float:
#         return self.eval_safe(x)
    
#     def plot(self):
#         fig, ax = plt.subplots()
#         X = np.linspace(self.domain.min, self.domain.max, 100)
#         Y = self(X)
#         ax.plot(X,Y)
#         return fig, ax


# def bissect(f: RealFunction, 
#             search_space: Interval, 
#             erroTol: float = 1e-4, 
#             maxItr: int = 1e4, 
#             eps: float = 1e-6 ) -> Interval:
#     count = 0
#     ss = search_space.copy()
#     err = ss.size/2.0
#     fa, fb = f(ss.min), f(ss.max)
#     if fa * fb > -eps:
#         if abs(fa) < eps:
#             return Interval(ss.min, ss.min)
#         elif abs(fb) < eps:
#             return Interval(ss.max, ss.max)
#         else:
#             raise Exception("The interval extremes share the same signal;\n employ the grid search method to locate a valid interval.")
#     while count <= maxItr and err > erroTol:
#         count += 1
#         a, b, m =  ss.min, ss.max, ss.haf
#         fa, fb, fm = f(a), f(b), f(m)
#         if abs(fm) < eps:
#             return Interval(m, m)
#         elif fa * fm < -eps:
#             ss = Interval(a, m)
#         elif fb * fm < -eps:
#             ss = Interval(m, b)
#     return ss


# def grid_search(f: RealFunction, domain: Interval = None, grid_freq = 8) -> Interval:
#     if domain is not None:
#         D = domain.copy()
#     else:
#         D = f.domain.copy()
#     L1 = np.linspace(D.min, D.max, grid_freq)
#     FL1 = f(L1)
#     TI = FL1[:-1]*FL1[1:]
#     VI = TI <= 0
#     if not np.any(VI):
#         return None
#     idx = np.argmax(VI)
#     return Interval(L1[idx], L1[idx+1])


# def newton_root(f: RealFunction, x0: float, err: float = 1e-4, maxItr: int = 100, eps: float = 1e-6) -> float:
#     x = x0
#     for i in range(maxItr):
#         fx = f.eval_safe(x)
#         if abs(fx) < err:
#             return x
#         f_prime_x = f.prime_safe(x)
#         if abs(f_prime_x) < eps:
#             raise Exception("Derivada perto de zero, nao converge.")
#         x = x - fx / f_prime_x
#     raise Exception("Numero max de iteracoes atingidas, nao converge")


# if __name__ == '__main__':
#     d = Interval(-1.0, 2.0)
#     print(d)

#     nt = np.linspace(d.min-.1, d.max+1, 5)

#     for n in nt:
#         sts = 'IN' if n in d else 'OUT'
#         print(f'{n} is {sts} of {d}')

#     class funcTest(RealFunction):
#         f = lambda self, x : np.power(x, 2) - 1
#         prime = lambda self, x : 2*x
#         domain = Interval(-2, 2)

#     ft = funcTest()
#     ND = grid_search(ft, grid_freq=12)
#     print("Resultado biseccao", bissect(ft, search_space=ND))
#     print("Resultado metodo de newton:", newton_root(ft, x0=0.5))

#     fig, ax = ft.plot()
#     ax.axhline(0, color='gray', linestyle='--')
#     plt.show()

# class interpolater:

#     def evaluate(self, X):
#         raise NotImplementedError

#     def __call__(self,  X):
#         return self.evaluate(X)

# class VandermondeMatrix(interpolater):
#     def __init__(self, x, y):
#         if len(x) != len(y):
#             raise RuntimeError(f"Dimensions must be equal len(x) = {len(x)} != len(y) = {len(y)}")
#         self.data = [x, y]
#         self._degree = len(x) - 1
#         self._buildMatrix()
#         self._poly = np.linalg.solve(self.matrix, self.data[1])

#     def _buildMatrix(self):
#         self.matrix = np.ones([self._degree + 1, self._degree + 1])
#         for i, x in enumerate(self.data[0]):
#             self.matrix[i, 1:] = np.multiply.accumulate(np.repeat(x, self._degree))

#     def evaluate(self, X):
#         r = 0.0
#         for c in self._poly[::-1]:
#             r = c + r * X
#         return r


# class LagrangePolynomial(interpolater):
#     def __init__(self, x, y):
#         if len(x) != len(y):
#             raise RuntimeError(f"Dimensions must be equal len(x) = {len(x)} != len(y) = {len(y)}")
#         self.data = [x, y]
#         self.poly = lagrange(x, y)

#     def evaluate(self, X):
#         return self.poly(X)


# def random_sample(intv, N):
#     r = np.random.uniform(intv[0], intv[1], N - 2)
#     r.sort()
#     return np.array([intv[0]] + list(r) + [intv[1]])


# def error_pol(f, P, intv, n=1000):
#     x = random_sample(intv, n)
#     vectError = np.abs(f(x) - P(x))
#     return np.sum(vectError) / n, np.max(vectError)


# if __name__ == '__main__':
#     DataX = [10.7, 11.075, 11.45, 11.825, 12.2, 12.5]
#     DataY = [-0.25991903, 0.04625002, 0.16592075, 0.13048074, 0.13902777, 0.2]

#     start_vander = time.time()
#     Pvm = VandermondeMatrix(DataX, DataY)
#     end_vander = time.time()
#     time_vander = end_vander - start_vander

#     start_lagrange = time.time()
#     Plg = LagrangePolynomial(DataX, DataY)
#     end_lagrange = time.time()
#     time_lagrange = end_lagrange - start_lagrange

#     print(f"Tempo para Vandermonde {time_vander:.6f} s")
#     print(f"Tempo para Lagrange{time_lagrange:.6f} s")

#     X = np.linspace(min(DataX) - 0.2, max(DataX) + 0.2, 100)
#     Y_vander = Pvm(X)
#     Y_lagrange = Plg(X)

#     _, ax = plt.subplots(1)
#     ax.plot(X, Y_vander, label='Vandermonde Matrix')
#     ax.plot(X, Y_lagrange, linestyle='--', label='Lagrange Polynomial')
#     ax.axis('equal')
#     ax.plot(DataX, DataY, 'o', label='Data Points')
#     ax.legend()
#     plt.show()

def generate_random_points(num_points, range_min=-10, range_max=10):
    return np.random.uniform(range_min, range_max, (num_points, 2))


def calculate_mst(points):
    num_points = len(points)
    visited = [False] * num_points
    mst_edges = []
    edge_count = 0

    visited[0] = True
    edges = []
    for i in range(1, num_points):
        dist = np.linalg.norm(points[0] - points[i])
        heappush(edges, (dist, 0, i))

    while edges and edge_count < num_points - 1:
        dist, u, v = heappop(edges)
        if not visited[v]:
            visited[v] = True
            mst_edges.append((u, v, dist))
            edge_count += 1

            for i in range(num_points):
                if not visited[i]:
                    dist = np.linalg.norm(points[v] - points[i])
                    heappush(edges, (dist, v, i))
    
    return mst_edges


def plot_mst(points, mst_edges):
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], 'bo')  

    for u, v, dist in mst_edges:
        x_coords = [points[u, 0], points[v, 0]]
        y_coords = [points[u, 1], points[v, 1]]
        ax.plot(x_coords, y_coords, 'r-')

    ax.axis('equal')
    plt.title('Minimum-spanning Tree')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    num_points = 10
    points = generate_random_points(num_points)

    mst_edges = calculate_mst(points)

    plot_mst(points, mst_edges)

