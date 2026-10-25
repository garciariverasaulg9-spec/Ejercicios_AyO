import numpy as np
import scipy.sparse.csgraph as csgraph
import scipy.spatial.distance as sp_dist
import matplotlib.pyplot as plt
import itertools
import random
import os

# 1. Función para calcular la longitud del tour TSP
def route_length(route, dist_matrix):
    return sum(dist_matrix[route[i], route[i+1]] for i in range(len(route)-1)) + dist_matrix[route[-1], route[0]]

# 2. Función de Búsqueda Local 2-opt (retorna 3 valores: ruta, costo, iteraciones)
def two_opt_local_search(initial_route, dist_matrix):
    best_route = list(initial_route)
    best_dist = route_length(best_route, dist_matrix)
    improved = True
    iterations = 0
    
    while improved:
        improved = False
        n = len(best_route)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                iterations += 1
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_length(new_route, dist_matrix)
                if new_dist < best_dist - 1e-6:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
                
    return best_route, best_dist, iterations

# 3. Función Exacta (Fuerza Bruta)
def exact_tsp_bruteforce(dist_matrix):
    n = len(dist_matrix)
    min_cost = float('inf')
    for p in itertools.permutations(range(n)):
        c = route_length(p, dist_matrix)
        if c < min_cost:
            min_cost = c
    return min_cost

# 4. Algoritmo de Aproximación basado en MST
def mst_tsp_approx(dist_matrix):
    mst = csgraph.minimum_spanning_tree(dist_matrix).toarray()
    mst_adj = mst + mst.T
    visited = []
    
    def dfs(u):
        visited.append(u)
        for v in range(len(dist_matrix)):
            if mst_adj[u, v] > 0 and v not in visited:
                dfs(v)
    dfs(0)
    
    cost = route_length(visited, dist_matrix)
    return visited, cost

# Asegurar la creación de la carpeta para guardar gráficas
os.makedirs('graphs', exist_ok=True)

# Experimento sobre 100 instancias
np.random.seed(7)
random.seed(7)
n_instances = 100
n_cities = 9

ratios_pure = []
ratios_post_2opt = []

for _ in range(n_instances):
    coords = np.random.rand(n_cities, 2) * 100
    dist_matrix = sp_dist.cdist(coords, coords)
    
    c_opt = exact_tsp_bruteforce(dist_matrix)
    route_mst, c_mst = mst_tsp_approx(dist_matrix)
    ratios_pure.append(c_mst / c_opt)
    
    # Línea 52 funcional con los 3 valores de retorno
    _, c_2opt, _ = two_opt_local_search(route_mst, dist_matrix)
    ratios_post_2opt.append(c_2opt / c_opt)

# Generación y guardado de la gráfica
plt.figure(figsize=(8.5, 4.3))
plt.hist(ratios_pure, bins=15, alpha=0.7, label=f'MST Puro (Promedio: {np.mean(ratios_pure):.3f})', color='#5DADE2', edgecolor='black')
plt.hist(ratios_post_2opt, bins=15, alpha=0.75, label=f'MST + 2-opt (Promedio: {np.mean(ratios_post_2opt):.3f})', color='#F4D03F', edgecolor='black')
plt.axvline(2.0, color='#C0392B', linestyle='--', linewidth=2.5, label='Límite Teórico Upper Bound (ρ ≤ 2.0)')

plt.xlabel('Razón de Aproximación (C_obtenido / C_óptimo)')
plt.ylabel('Frecuencia (Número de Instancias)')
plt.title('Distribución de la Razón de Aproximación en 100 Instancias de 9 Ciudades')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

#plt.savefig('graphs/mst_approx_ratio.png', dpi=300)
plt.show()
