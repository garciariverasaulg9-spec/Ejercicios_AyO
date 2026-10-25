import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

def route_length(route, dist_matrix):
    return sum(dist_matrix[route[i], route[i+1]] for i in range(len(route)-1)) + dist_matrix[route[-1], route[0]]

def simulated_annealing(dist_matrix, schedule_type='geometric_slow', T0=1000.0, T_min=1e-2, max_iter=1500):
    n = len(dist_matrix)
    current_route = list(range(n))
    random.shuffle(current_route)
    current_cost = route_length(current_route, dist_matrix)
    
    best_route = list(current_route)
    best_cost = current_cost
    
    history = []
    T = T0
    t_start = time.perf_counter()

    for k in range(1, max_iter + 1):
        # Operador de movimiento: 2-opt estocástico
        i, j = sorted(random.sample(range(n), 2))
        candidate = current_route[:i] + current_route[i:j+1][::-1] + current_route[j+1:]
        candidate_cost = route_length(candidate, dist_matrix)
        
        delta = candidate_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_route = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_route = list(current_route)
                best_cost = current_cost
        
        history.append(best_cost)
        
        # Regla de enfriamiento
        if schedule_type == 'geometric_slow':
            T *= 0.995
        elif schedule_type == 'geometric_fast':
            T *= 0.85
        elif schedule_type == 'logarithmic':
            T = T0 / (1 + math.log(1 + k))

    exec_time = time.perf_counter() - t_start
    return best_cost, exec_time, history

# Instancia de Evaluación (50 Ciudades)
np.random.seed(101)
coords = np.random.rand(50, 2) * 1000
dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)

cost_slow, time_slow, hist_slow = simulated_annealing(dist_matrix, 'geometric_slow')
cost_fast, time_fast, hist_fast = simulated_annealing(dist_matrix, 'geometric_fast')
cost_log,  time_log,  hist_log  = simulated_annealing(dist_matrix, 'logarithmic')

# Gráfica de Convergencia
plt.figure(figsize=(9, 5))
plt.plot(hist_slow, label=f'Lento (Geométrico $\\alpha=0.995$) -> Min: {cost_slow:.1f}', color='darkgreen', linewidth=1.5)
plt.plot(hist_fast, label=f'Rápido (Quenching $\\alpha=0.85$) -> Min: {cost_fast:.1f}', color='purple', linewidth=1.5)
plt.plot(hist_log,  label=f'Diferente (Logarítmico) -> Min: {cost_log:.1f}', color='orange', linewidth=1.5)

plt.xlabel('Iteración')
plt.ylabel('Distancia del Mejor Recorrido')
plt.title('Simulated Annealing TSP: Calidad del Mejor Recorrido vs. Iteraciones')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
#plt.savefig('graphs/sa_convergence.png', dpi=300)
plt.show()
