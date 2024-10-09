import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def gerar_pontos(N, distribuicao):
    if distribuicao == 'uniforme':
        x = np.random.uniform(-1, 1, N)
        y = np.random.uniform(-1, 1, N)
    elif distribuicao == 'normal':
        x = np.random.normal(0, 0.5, N)
        y = np.random.normal(0, 0.5, N)
    elif distribuicao == 't':
        x = np.random.standard_t(0.5, N)  # Nota: a distribuição t de Student não tem uma média definida para todos os graus de liberdade
        y = np.random.standard_t(0.5, N)
    else:
        raise ValueError("Distribuição inválida. Escolha 'uniforme', 'normal' ou 't'.")

    return np.array(list(zip(x, y)))

def calcular_fecho_convexo(pontos):
    hull = ConvexHull(pontos)
    return pontos[hull.vertices]

def main():
    N = int(input("Digite a quantidade de pontos: "))

    distribuicao = input("Diga qual distribuição deseja (uniforme, normal, t): ").lower()
    
    try:
        pontos = gerar_pontos(N, distribuicao)
        print("Pontos gerados:")
        print(pontos)

        pontos_fecho = calcular_fecho_convexo(pontos)
        print("Pontos que pertencem ao fecho convexo:")
        print(pontos_fecho)

        plt.figure()
        plt.plot(pontos[:, 0], pontos[:, 1], 'o', label='Pontos gerados')
        plt.plot(pontos_fecho[:, 0], pontos_fecho[:, 1], 'r-', lw=2, label='Fecho Convexo')
        plt.fill(pontos_fecho[:, 0], pontos_fecho[:, 1], 'r', alpha=0.3)
        plt.title("Fecho Convexo dos Pontos Gerados")
        plt.legend()
        plt.grid()
        plt.show()
    except ValueError as e:
        print(e)    

if __name__ == "__main__":
    main()




# import random

# class RandomStack:
#     def __init__(self):
#         self.stack = []
    
#     def push(self, item):
#         self.stack.append(item)
    
#     def pop(self):
#         if not self.stack:
#             raise IndexError("pop from empty stack")
        
#         random_index = random.randint(0, len(self.stack) - 1)
        
#         self.stack[random_index], self.stack[-1] = self.stack[-1], self.stack[random_index]
        
#         return self.stack.pop()

#     def is_empty(self):
#         """Verifica se a pilha está vazia"""
#         return len(self.stack) == 0

#     def size(self):
#         """Retorna o tamanho da pilha"""
#         return len(self.stack)

# def test_random_stack():
#     rs = RandomStack()
    
#     rs.push(10)
#     rs.push(20)
#     rs.push(30)
#     rs.push(40)
#     rs.push(50)
#     assert rs.size() == 5
    
#     popped_item = rs.pop()
#     for i in range(len(rs.stack)):
#         print(rs.stack[i])
#     assert popped_item in [10, 20, 30, 40, 50]
#     assert rs.size() == 4
    
#     rs.pop()
#     rs.pop()
#     rs.pop()
#     rs.pop()
#     assert rs.is_empty() == True
    
#     try:
#         rs.pop()
#     except IndexError as e:
#         assert str(e) == "pop from empty stack"
    
#     print("Todos os testes foram bem sucedidos")

# test_random_stack()

# class TreeNode:
#     def __init__(self, val=0):
#         self.val = val
#         self.left = None
#         self.right = None

# def check_height(node):
#     if not node:
#         return 0

#     left_height = check_height(node.left)
#     right_height = check_height(node.right)

#     if left_height == -1 or right_height == -1:
#         return -1
    
#     if abs(left_height - right_height) > 1:
#         return -1

#     return max(left_height, right_height) + 1

# def is_balanced(root):
#     return check_height(root) != -1

# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)
# root.right.right = TreeNode(6)
# # root.right.left = TreeNode(7)
# # root.right.left.left = TreeNode(8)
# # root.right.left.left.left = TreeNode(9)
# # root.right.left.left.right = TreeNode(10)
# # root.right.left.left.right.right = TreeNode(11)
# # root.left.left.left = TreeNode(12)
# # root.left.left.right = TreeNode(13)
# # root.left.left.left.right = TreeNode(14)
# # root.left.left.left.left = TreeNode(15)
# # root.left.left.left.right.right = TreeNode(16)  

# #testar essa porra no papel depois 
# #     
# print(is_balanced(root))  