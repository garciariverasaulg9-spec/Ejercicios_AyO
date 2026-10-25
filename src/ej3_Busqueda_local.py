import numpy as np
import matplotlib.pyplot as plt
import random

def route_length(route, dist_matrix):
    return sum(dist_matrix[route[i], route[i+1]] for i in range(len(route)-1)) + dist_matrix[route[-1], route[0]]

def two_opt_local_search(initial_route, dist_matrix):
    best_route = list(initial_route)
    best_dist = route_length(best_route, dist_matrix)
    improvements = 0
    improved = True

    while improved:
        improved = False
        n = len(best_route)
        for i in range(1, n - 1):
            for j in range(i + 1, n):

                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_length(new_route, dist_matrix)
                
                if new_dist < best_dist - 1e-6:
                    best_route = new_route
                    best_dist = new_dist
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
    return best_route, best_dist, improvements

def nearest_neighbor_tour(dist_matrix, start_node):
    n = len(dist_matrix)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]
    curr = start_node

    while unvisited:
        nxt = min(unvisited, key=lambda node: dist_matrix[curr][node])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    return tour


np.random.seed(42)
num_instances = 50
num_cities = 30

imp_random, cost_random = [], []
imp_nn, cost_nn = [], []

for _ in range(num_instances):
    coords = np.random.rand(num_cities, 2) * 1000
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)


    init_rand = list(range(num_cities))
    random.shuffle(init_rand)
    _, dist_r, count_r = two_opt_local_search(init_rand, dist_matrix)
    imp_random.append(count_r)
    cost_random.append(dist_r)


    start_city = random.randint(0, num_cities - 1)
    init_nn = nearest_neighbor_tour(dist_matrix, start_city)
    _, dist_n, count_n = two_opt_local_search(init_nn, dist_matrix)
    imp_nn.append(count_n)
    cost_nn.append(dist_n)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.boxplot([imp_random, imp_nn], tick_labels=['Inicio Aleatorio', 'Vecino Más Cercano (NN)'], patch_artist=True)
ax1.set_ylabel('Número de Mejoras 2-opt')
ax1.set_title('Esfuerzo de Convergencia (Iteraciones)')
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.boxplot([cost_random, cost_nn], labels=['Inicio Aleatorio', 'Vecino Más Cercano (NN)'], patch_artist=True)
ax2.set_ylabel('Distancia del Óptimo Local Alcanzado')
ax2.set_title('Calidad de la Solución Final')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
#plt.savefig('graphs/tsp_2opt_boxplot.png', dpi=300)
plt.show()
