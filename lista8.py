import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class PolynomialDegreeSelector:
    def __init__(self, max_degree):
        self.max_degree = max_degree
        self.best_degree = None
        self.errors = []

    def find_best_degree(self, x, y):
        
        # Dividimos os dados em conjunto de treinamento e validação (30% - 70%)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
        
        self.errors = []
        
        for degree in range(1, self.max_degree + 1):
            A_train = np.vander(x_train, degree + 1)
            coefficients = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
            
            A_val = np.vander(x_val, degree + 1)
            y_pred = A_val @ coefficients
            
            # Calcular o erro no conjunto de validação
            mse = mean_squared_error(y_val, y_pred)
            self.errors.append(mse)
        
        self.best_degree = np.argmin(self.errors) + 1
        return self.best_degree

    def plot_errors(self):
        if not self.errors:
            raise ValueError("Nenhum erro calculado. Execute `find_best_degree` primeiro.")
        
        plt.plot(range(1, self.max_degree + 1), self.errors, marker='o')
        plt.xlabel("Grau do Polinômio")
        plt.ylabel("Erro Quadrático Médio (MSE)")
        plt.title("Erro de Validação vs. Grau do Polinômio")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 3 * x**3 - 2 * x**2 + x + np.random.normal(scale=100, size=x.shape)

    max_degree = 10

    selector = PolynomialDegreeSelector(max_degree)
    best_degree = selector.find_best_degree(x, y)
    print(f"Melhor grau de polinômio encontrado: {best_degree}")

    selector.plot_errors()

    fitter = PolynomialFitter(best_degree)
    fitter.fit(x, y)
    fitter.plot(x, y)
