import numpy as np
import math
import heapq
import time
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. Global Constants & Configuration
# ==========================================
OBST = 1
FREE = 0
SQRT2 = math.sqrt(2)

# ==========================================
# 2. Map Generation (Tel Aviv Scenario)
# ==========================================
def build_tel_aviv_map(n=70):
    """
    Simulates the urban layout of Central Tel Aviv (Sarona to Ichilov).
    Returns a grid where 0=Road, 1=Building/Obstacle.
    """
    grid = np.zeros((n, n), dtype=int)
    grid[:, :] = OBST  # Start with buildings
    
    # Major Avenues (Kaplan, Shaul HaMelech, etc.)
    avenues = [10, 25, 45, 60]
    for r in avenues:
        grid[r-2:r+2, :] = FREE
        
    # Major Streets (Ibn Gabirol, Weizmann, etc.)
    streets = [10, 30, 50, 65]
    for c in streets:
        grid[:, c-2:c+2] = FREE
        
    # Strategic Alleyways and Shortcuts
    grid[10:25, 20] = FREE
    grid[45:60, 40] = FREE
    
    # Define Sarona (Start) and Ichilov (Goal)
    start = (60, 10) 
    goal = (10, 50)  
    
    # Clear start/goal areas
    grid[start[0], start[1]] = FREE
    grid[goal[0], goal[1]] = FREE
    
    return grid, start, goal

def crowd_cost(n=70, seed=42):
    """Generates dynamic traffic costs based on congestion hotspots."""
    rng = np.random.default_rng(seed)
    cost = np.ones((n, n))
    for _ in range(12): # 12 Traffic hotspots
        cr, cc = rng.integers(10, n-10, 2)
        rad = rng.integers(8, 15)
        intensity = rng.uniform(4, 8)
        for r in range(n):
            for c in range(n):
                d = math.hypot(r-cr, c-cc)
                if d < rad:
                    cost[r, c] += intensity * (1 - d/rad)
    return cost

# ==========================================
# 3. Pathfinding Helper Functions
# ==========================================
def neighbors8(p, grid):
    r, c = p
    for dr, dc, base in [(-1,0,1), (1,0,1), (0,-1,1), (0,1,1),
                         (-1,-1,SQRT2), (-1,1,SQRT2), (1,-1,SQRT2), (1,1,SQRT2)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if grid[nr, nc] != OBST:
                yield (nr, nc), base

def octile_heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy + (SQRT2 - 2) * min(dx, dy)

def reconstruct_path(parent, start, goal):
    if goal not in parent and goal != start: return []
    path, cur = [goal], goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path

# ==========================================
# 4. Algorithms Implementation
# ==========================================

def dijkstra(grid, cost, start, goal):
    pq = [(0, start)]
    dist = {start: 0}; parent = {}; expanded = 0
    t0 = time.perf_counter()
    while pq:
        g, u = heapq.heappop(pq)
        if u == goal: break
        if g > dist.get(u, float('inf')): continue
        expanded += 1
        for v, b in neighbors8(u, grid):
            ng = g + b * cost[v]
            if ng < dist.get(v, float('inf')):
                dist[v] = ng; parent[v] = u
                heapq.heappush(pq, (ng, v))
    return reconstruct_path(parent, start, goal), time.perf_counter()-t0, expanded, dist.get(goal, np.inf)

def astar(grid, cost, start, goal):
    pq = [(octile_heuristic(start, goal), start)]
    gval = {start: 0}; parent = {}; expanded = 0
    t0 = time.perf_counter()
    while pq:
        f, u = heapq.heappop(pq)
        if u == goal: break
        expanded += 1
        for v, b in neighbors8(u, grid):
            ng = gval[u] + b * cost[v]
            if ng < gval.get(v, float('inf')):
                gval[v] = ng; parent[v] = u
                heapq.heappush(pq, (ng + octile_heuristic(v, goal), v))
    return reconstruct_path(parent, start, goal), time.perf_counter()-t0, expanded, gval.get(goal, np.inf)

def line_free(a, b, grid):
    steps = max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        r, c = int(round(a[0] + t*(b[0]-a[0]))), int(round(a[1] + t*(b[1]-a[1])))
        if grid[r, c] == OBST: return False
    return True

def prm(grid, start, goal, n_samples=100, rad=15):
    t0 = time.perf_counter()
    samples = [start, goal]
    while len(samples) < n_samples:
        r, c = random.randint(0, grid.shape[0]-1), random.randint(0, grid.shape[1]-1)
        if grid[r, c] == FREE: samples.append((r, c))
    
    adj = {p: [] for p in samples}
    for i, a in enumerate(samples):
        for j, b in enumerate(samples[i+1:], i+1):
            d = math.hypot(a[0]-b[0], a[1]-b[1])
            if d <= rad and line_free(a, b, grid):
                adj[a].append((b, d)); adj[b].append((a, d))
    
    pq = [(0, start)]; dist = {start: 0}; parent = {}; expanded = 0
    while pq:
        g, u = heapq.heappop(pq)
        if u == goal: break
        expanded += 1
        for v, d in adj[u]:
            ng = g + d
            if ng < dist.get(v, float('inf')):
                dist[v] = ng; parent[v] = u
                heapq.heappush(pq, (ng, v))
    return reconstruct_path(parent, start, goal), time.perf_counter()-t0, expanded, dist.get(goal, np.inf)

# ==========================================
# 5. Execution and Comparison
# ==========================================
def run_experiment():
    n = 70
    grid, start, goal = build_tel_aviv_map(n)
    cost = crowd_cost(n)
    
    results = {}
    results['Dijkstra'] = dijkstra(grid, cost, start, goal)
    results['A*'] = astar(grid, cost, start, goal)
    
    # PRM Iterations as requested by Professor Levner
    prm_configs = [50, 100, 200]
    for count in prm_configs:
        results[f'PRM-{count}'] = prm(grid, start, goal, n_samples=count)

    print(f"{'Algorithm':<15} | {'Time(ms)':<10} | {'Expanded':<10} | {'Cost':<10}")
    print("-" * 55)
    for name, (path, t, exp, c) in results.items():
        print(f"{name:<15} | {t*1000:<10.2f} | {exp:<10} | {c:<10.2f}")

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='gray_r')
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, (name, (path, t, exp, c)) in enumerate(results.items()):
        if path:
            py, px = zip(*path)
            plt.plot(px, py, label=f"{name} Path", linewidth=2, color=colors[i%len(colors)])
    
    plt.scatter(start[1], start[0], color='lime', s=200, label='Sarona (Start)', edgecolors='black')
    plt.scatter(goal[1], goal[0], color='red', s=200, label='Ichilov (Goal)', edgecolors='black')
    plt.title("Tel Aviv Smart Ambulance Routing Comparison")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment()
