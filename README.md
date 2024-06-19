# Delaunay Triangulation Algorithms

## Overview
This repository contains implementations of both sequential and parallel Delaunay triangulation algorithms. The primary objective is to compare the performance and efficiency of the sequential algorithm against its parallel counterpart, especially when handling larger datasets.

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithms](#algorithms)
    - [Sequential Implementation](#sequential-implementation)
    - [Parallel Implementation](#parallel-implementation)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Performance Comparison](#performance-comparison)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction
Delaunay triangulation is a fundamental problem in computational geometry with applications in graphics, mesh generation, and spatial data analysis. This project provides implementations of Delaunay triangulation algorithms designed to run both sequentially and in parallel, utilizing modern parallel computing techniques.

## Algorithms

### Sequential Implementation
The sequential implementation of the Delaunay triangulation algorithm is straightforward and processes one point at a time to construct the triangulation. This approach is simple but can become inefficient with larger datasets due to its computational complexity.

### Parallel Implementation
The parallel implementation leverages multi-threading to divide the work among multiple processors. It follows a strategy based on the Bowyer-Watson algorithm, with enhancements for parallel execution. Key features include:
- **Partitioning**: Uses Hilbert curves for partitioning the dataset to ensure spatial locality and load balancing.
- **Concurrency**: Manages concurrent insertion of points while maintaining the integrity of the triangulation.
- **Efficiency**: Demonstrates significant performance improvements over the sequential version, especially for large datasets.

## Installation
To install and run the project, follow these steps:

1. **Clone the repository**:
    ```bash
    [git clone https://github.com/KonstantinosGalanis/Parallel-Mesh-Generation-Algorithm.git]
    cd Diploma Thesis
    ```

2. **Set up the environment**:
    - Ensure you have Python installed (preferably version 3.8 or above).
    - Install the required dependencies using `pip`:
      ```bash
      pip install -r requirements.txt
      ```

3. **CUDA Setup**:
    - For the parallel implementation, ensure you have CUDA installed and properly configured on your machine. Refer to the [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads) for more details.

## Usage
To run the sequential implementation:
```bash
python sequential_delaunay.py
