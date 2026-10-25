import multiprocessing as mp
import random
import time
import matplotlib.pyplot as plt

def estimate_pi_chunk(samples: int) -> int:
    """Cuenta el número de puntos aleatorios dentro del círculo unitario."""
    inside = 0
    for _ in range(samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    return inside

def run_experiment():
    total_samples = 40_000_000
    processes_list = [1, 2, 4, 8]
    times = []

    for p in processes_list:
        samples_per_process = total_samples // p
        t0 = time.perf_counter()
        
        with mp.Pool(processes=p) as pool:
            results = pool.map(estimate_pi_chunk, [samples_per_process] * p)
            
        total_inside = sum(results)
        pi_estimate = 4 * total_inside / total_samples
        elapsed = time.perf_counter() - t0
        
        times.append(elapsed)
        print(f"Procesos: {p} | Tiempo: {elapsed:.4f}s | Estimación Pi: {pi_estimate:.6f}")

    # Cálculo de Speedup
    t1 = times[0]
    speedups = [t1 / tp for tp in times]

    # Generación de la Gráfica
    plt.figure(figsize=(10, 4.5))
    
    plt.subplot(1, 2, 1)
    plt.plot(processes_list, times, 'o-', color='crimson', label='Tiempo de Ejecución')
    plt.xlabel('Número de Procesos')
    plt.ylabel('Tiempo de Pared (s)')
    plt.title('Procesos vs. Tiempo de Ejecución')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(processes_list, speedups, 's-', color='navy', label='Speedup Real')
    plt.plot(processes_list, processes_list, '--', color='gray', label='Speedup Lineal Ideal')
    plt.xlabel('Número de Procesos')
    plt.ylabel('Speedup ($T_1 / T_p$)')
    plt.title('Speedup vs. Número de Procesos')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    #plt.savefig('graphs/speedup_pi.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    run_experiment()
