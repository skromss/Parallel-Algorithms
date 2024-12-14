#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def floyd_warshall_serial(graph):
    """
    Implements the Floyd-Warshall algorithm serially.

    Args:
        graph: A 2D NumPy array representing the graph.

    Returns:
        A 2D NumPy array containing the shortest distances between all pairs of nodes.
    """
    n = len(graph)
    dist = np.array(graph, dtype=np.float64)  # Avoids errors if using python lists initially

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def floyd_warshall_parallel(graph, num_processes=mp.cpu_count()):
    """
    Implements the Floyd-Warshall algorithm in parallel using joblib.

    Args:
        graph: A 2D NumPy array representing the graph.
        num_processes: The number of processes to use.

    Returns:
        A 2D NumPy array containing the shortest distances between all pairs of nodes.
    """
    n = len(graph)
    dist = np.array(graph, dtype=np.float64)  # Copy so no shared memory modifications

    for k in range(n):
        results = Parallel(n_jobs=num_processes)(delayed(_process_row)(i, k, dist[i,:].copy(), dist[k,:].copy()) for i in range(n))
        for row_idx, updated_row in results:
            dist[row_idx,:] = updated_row
    return dist

def _process_row(row_idx, k, row_copy, k_row_copy):
    """
    Processes a single row of the distance matrix.

    Args:
        row_idx: The index of the row to process.
        k: The intermediate node.
        row_copy: A copy of the current row.
        k_row_copy: A copy of the k-th row.

    Returns:
        A tuple containing the row index and the updated row.
    """
    n = len(row_copy)
    for j in range(n):
        row_copy[j] = min(row_copy[j], row_copy[k] + k_row_copy[j])
    return row_idx, row_copy

def generate_random_graph(num_nodes, connection_probability):
    """
    Generates a random graph with specified number of nodes and connection probability.

    Args:
        num_nodes: The number of nodes in the graph.
        connection_probability: The probability of a connection between two nodes.

    Returns:
        A 2D NumPy array representing the graph.
    """
    graph = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(graph, 0)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < connection_probability:
                weight = np.random.randint(1, 10) # random weight
                graph[i][j] = weight
                graph[j][i] = weight
    return graph

def benchmark(func, graph, iterations=5):
    """
    Measures the average execution time of a function.

    Args:
        func: The function to benchmark.
        graph: The input to the function.
        iterations: The number of iterations to run the benchmark.

    Returns:
        The average execution time in seconds.
    """
    times = []
    for _ in range(iterations):
        start_time = time.time()
        func(graph)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)


# In[6]:


# Parameters
num_nodes = 500  # Minimum requirement
connection_probability = 0.5  # Minimum requirement
num_iterations = 3

# Generate random graph
graph = generate_random_graph(num_nodes, connection_probability)

# Benchmark serial execution
serial_time = benchmark(floyd_warshall_serial, graph, num_iterations)
print(f"Serial execution time: {serial_time:.4f} seconds")


# In[7]:


# Benchmark parallel execution
speedups = []
num_processes_list = [1, 2, 4, 8, 12] 

for num_processes in num_processes_list:
    parallel_time = benchmark(lambda g: floyd_warshall_parallel(g, num_processes), graph, num_iterations)
    speedup = serial_time / parallel_time
    speedups.append(speedup)
    print(f"Parallel execution time ({num_processes} processes): {parallel_time:.4f} seconds, Speedup: {speedup:.2f}")


# In[8]:


# Plot speedup
plt.plot(num_processes_list, speedups, marker='o')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.title('Floyd-Warshall Speedup')
plt.grid(True)
plt.show()


# In[ ]:




