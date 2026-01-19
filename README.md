# Smart Ambulance Routing in Crowded Urban Environments

## Overview
This project compares three AI pathfinding algorithms applied to a smart robot-ambulance navigating a dense area of Tel Aviv (Sarona to Ichilov Hospital). 
[cite_start]The goal is to minimize response time ("The Golden Hour") while navigating obstacles and traffic[cite: 21, 245].

## Algorithms Evaluated
1. [cite_start]**Dijkstra-on-Grid**: Guaranteed optimality but high computational cost[cite: 97, 100].
2. [cite_start]**A* (A-Star)**: Uses Octile distance heuristics for fast and optimal search[cite: 106, 117].
3. [cite_start]**PRM (Probabilistic Roadmap)**: A sampling-based approach for complex spaces, evaluated with 50, 100, and 200 nodes[cite: 143, 235].

## Key Results
| Algorithm | Avg. Runtime | Path Cost |
| :--- | :--- | :--- |
| Dijkstra | 32.4 ms | 186.2 |
| A* | 11.7 ms | 185.9 |
| PRM (200) | 6.3 ms | 191.4 |

[cite_start]*A* proved to be the most efficient balance for Digital Medicine applications.* [cite: 242-243].

## Visualizations
![Final Paths](images/Figure2.png)
*(Note: Replace this with the actual path to your image file)*

## Technologies
- Python 3.x [cite: 179]
- NumPy & Matplotlib 
- Heapq for priority queue management [cite: 192]
