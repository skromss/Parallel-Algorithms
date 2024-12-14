# Parallel Floyd-Warshall Algorithm

This repository implements the Floyd-Warshall algorithm for finding all-pairs shortest paths in a graph, with parallelization using the `joblib` library.

**1. Algorithm and Parallelization:**

* **Algorithm:** Floyd-Warshall algorithm for finding shortest paths between all pairs of nodes in a weighted graph.
* **Parallelization Method:** The algorithm is parallelized by distributing the computation of each row of the distance matrix across multiple processes using `joblib.Parallel` and `joblib.delayed`.

**2. Running the Code:**

1. **Data Preparation:**
   - **Generate Random Graph:**
      - The code includes a function `generate_random_graph(num_nodes, connection_probability)` to generate a random graph with specified parameters.
      - Modify `num_nodes` and `connection_probability` as needed.
   - **Use Existing Graph:**
      - If you have a graph in a suitable format (e.g., adjacency matrix), replace the `generate_random_graph` call with code to load your graph data.

2. **Run the Code:**
   - Execute the Python script. 
   - The script will run the serial and parallel versions of the algorithm, calculate speedup, and generate a plot of speedup vs. number of processes.

**3. Parallelized Part:**

The parallelization is implemented by distributing the computation of each row of the distance matrix across multiple processes. Each process calculates the updated values for a single row using the `_process_row` function.

**4. Speedup Calculation and Results:**

- The script calculates speedup as `speedup = serial_time / parallel_time`.
- A plot is generated showing the relationship between the number of processes and the achieved speedup.

![image](https://github.com/user-attachments/assets/d01c3c79-c38b-4967-b39c-54a595ea0df1)
