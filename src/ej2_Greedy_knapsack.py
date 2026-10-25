import matplotlib.pyplot as plt

def greedy_knapsack(weights, values, capacity, criterion='ratio'):
    n = len(weights)
    if criterion == 'ratio':
        indices = sorted(range(n), key=lambda i: values[i]/weights[i], reverse=True)
    elif criterion == 'min_weight':
        indices = sorted(range(n), key=lambda i: weights[i])
    
    selected = []
    total_weight = 0
    total_value = 0

    for i in indices:
        if total_weight + weights[i] <= capacity:
            selected.append(i)
            total_weight += weights[i]
            total_value += values[i]

    return selected, total_weight, total_value

# Instancia de Prueba
capacity = 100
weights = [10, 20, 90]
values  = [2, 2, 150]
names   = ['Objeto A', 'Objeto B', 'Objeto C']

sel_w, weight_w, val_w = greedy_knapsack(weights, values, capacity, 'min_weight')
sel_r, weight_r, val_r = greedy_knapsack(weights, values, capacity, 'ratio')

print(f"Menor Peso -> Objetos: {[names[i] for i in sel_w]} | Peso: {weight_w} | Valor Total: {val_w}")
print(f"Valor/Peso -> Objetos: {[names[i] for i in sel_r]} | Peso: {weight_r} | Valor Total: {val_r}")

# Visualización Comparativa
methods = ['Menor Peso Primero', 'Razón Valor / Peso', 'Óptimo Global (DP)']
total_values = [val_w, val_r, 152]
used_capacities = [weight_w, weight_r, 100]

fig, ax1 = plt.subplots(figsize=(8, 4.5))

bars = ax1.bar(methods, total_values, color=['#e74c3c', '#2ecc71', '#3498db'], width=0.4)
ax1.set_ylabel('Valor Total Obtenido', fontsize=11)
ax1.set_title('Mochila 0/1: Trampa del Óptimo Local por Criterio de Menor Peso', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"v={yval}", ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
#plt.savefig('graphs/knapsack_comparison.png', dpi=300)
plt.show()
