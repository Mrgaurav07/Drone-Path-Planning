import random
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.geometry import Point
import numpy as np

def generate_random_regular_polygon(area):
    num_sides = random.randint(3, 8)
    min_radius = 1
    max_radius = 3

    while True:
        radius = random.uniform(min_radius, max_radius)
        x = random.uniform(area.bounds[0] + radius, area.bounds[2] - radius)
        y = random.uniform(area.bounds[1] + radius, area.bounds[3] - radius)
        center = Point(x, y)

        vertices = []
        for i in range(num_sides):
            dist = radius + random.uniform(-0.5, 0.5)
            angle = 2 * math.pi * i / num_sides
            vertex_x = x + dist * math.cos(angle)
            vertex_y = y + dist * math.sin(angle)
            vertices.append((vertex_x, vertex_y))

        candidate_polygon = Polygon(vertices)
        if area.contains(candidate_polygon):
            return candidate_polygon

def divide_area(area, divisions):
    subareas = []
    min_x, min_y, max_x, max_y = area.bounds
    width = (max_x - min_x) / divisions
    height = (max_y - min_y) / divisions

    for i in range(divisions):
        for j in range(divisions):
            subarea = Polygon([(min_x + i * width, min_y + j * height),
                               (min_x + (i + 1) * width, min_y + j * height),
                               (min_x + (i + 1) * width, min_y + (j + 1) * height),
                               (min_x + i * width, min_y + (j + 1) * height)])
            subareas.append(subarea)

    return subareas

def lawn_mower_motionflr(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    traversal_path = []

    current_y = min_y

    while current_y <= max_y:
        if current_y % 2 == 0:  # Move from left to right
            current_x = max_x
            while current_x >= min_x:
                current_point = Point(current_x, current_y)
                if current_point.within(polygon):
                    traversal_path.append((current_x, current_y))
                current_x -= 0.5  # Increment by a smaller step for better coverage
        else:  # Move from right to left
            current_x = min_x
            while current_x <= max_x:
                current_point = Point(current_x, current_y)
                if current_point.within(polygon):
                    traversal_path.append((current_x, current_y))
                current_x += 0.5  # Increment by a smaller step for better coverage

        current_y += 1  # Move to the next row

    return traversal_path

def lawn_mower_motionfrl(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    traversal_path = []

    current_y = min_y

    while current_y <= max_y:
        if current_y % 2 == 0:  # Move from left to right
            current_x = min_x
            while current_x <= max_x:
                current_point = Point(current_x, current_y)
                if current_point.within(polygon):
                    traversal_path.append((current_x, current_y))
                current_x += 0.5  # Increment by a smaller step for better coverage
        else:  # Move from right to left
            current_x = max_x
            while current_x >= min_x:
                current_point = Point(current_x, current_y)
                if current_point.within(polygon):
                    traversal_path.append((current_x, current_y))
                current_x -= 0.5  # Increment by a smaller step for better coverage

        current_y += 1  # Move to the next row

    return traversal_path

def lawn_mower_motionful(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    traversal_path = []

    current_x = min_x

    while current_x <= max_x:
        if current_x % 2 == 0:  # Move from left to right
            current_y = min_y
            while current_y <= max_y:
                current_point = Point(current_x, current_y)
                if current_point.within(polygon):
                    traversal_path.append((current_x, current_y))
                current_y += 0.5  # Increment by a smaller step for better coverage
        else:  # Move from right to left
            current_y = max_y
            while current_y >= min_y:
                current_point = Point(current_x, current_y)
                if current_point.within(polygon):
                    traversal_path.append((current_x, current_y))
                current_y -= 0.5  # Increment by a smaller step for better coverage

        current_x += 1  # Move to the next row

    return traversal_path

def lawn_mower_motionflu(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    traversal_path = []

    current_x = min_x

    while current_x <= max_x:
        if current_x % 2 == 0:  # Move from left to right
            current_y = max_y
            while current_y >= min_y:
                current_point = Point(current_x, current_y)
                if current_point.within(polygon):
                    traversal_path.append((current_x, current_y))
                current_y -= 0.5  # Increment by a smaller step for better coverage
        else:  # Move from right to left
            current_y = min_y
            while current_y <= max_y:
                current_point = Point(current_x, current_y)
                if current_point.within(polygon):
                    traversal_path.append((current_x, current_y))
                current_y += 0.5  # Increment by a smaller step for better coverage

        current_x += 1  # Move to the next row

    return traversal_path

class Ant:
    def __init__(self, start_point):
        self.visited = [start_point]
        self.distance_travelled = 0

    def move(self, current_point, pheromone_matrix, alpha, beta):
        available_points = [point for point in pheromone_matrix.keys() if point not in self.visited]
        probabilities = []
        total = 0
        for point in available_points:
            pheromone_level = pheromone_matrix[current_point][point]
            distance = current_point.distance(point)
            total += (pheromone_level ** alpha) * ((1 / distance) ** beta)
        for point in available_points:
            pheromone_level = pheromone_matrix[current_point][point]
            distance = current_point.distance(point)
            probability = ((pheromone_level ** alpha) * ((1 / distance) ** beta)) / total
            probabilities.append((point, probability))
        chosen_point = self.choose_point(probabilities)
        self.visited.append(chosen_point)
        self.distance_travelled += current_point.distance(chosen_point)
        return chosen_point

    def choose_point(self, probabilities):
        cumulative_probabilities = np.cumsum([p[1] for p in probabilities])
        random_value = random.random()
        for i, prob in enumerate(cumulative_probabilities):
            if random_value <= prob:
                return probabilities[i][0]

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def ant_colony_tsp(coords, num_ants, num_iterations, evaporation_rate=0.5, alpha=1, beta=2):
    # Create a list of all unique centroids as Point objects
    unique_coords = [Point(coord) for coord in set(coords)]

    # Initialize the pheromone matrix with all possible combinations of unique centroids
    pheromone_matrix = {(point1, point2): 1 for point1 in unique_coords for point2 in unique_coords if point1 != point2}

    best_route = None
    best_distance = float('inf')

    for _ in range(num_iterations):
        ants = [Ant(random.choice(unique_coords)) for _ in range(num_ants)]
        for ant in ants:
            current_point = ant.visited[-1]
            while len(ant.visited) < len(unique_coords):
                available_points = [point for point in unique_coords if point not in ant.visited]
                if not available_points:
                    break
                probabilities = []
                total = 0
                for point in available_points:
                    if (current_point, point) in pheromone_matrix:
                        pheromone_level = pheromone_matrix[current_point, point]
                    else:
                        pheromone_level = 1
                    distance = current_point.distance(point)
                    total += (pheromone_level ** alpha) * ((1 / distance) ** beta)
                for point in available_points:
                    if (current_point, point) in pheromone_matrix:
                        pheromone_level = pheromone_matrix[current_point, point]
                    else:
                        pheromone_level = 1
                    distance = current_point.distance(point)
                    probability = ((pheromone_level ** alpha) * ((1 / distance) ** beta)) / total
                    probabilities.append((point, probability))
                chosen_point = ant.choose_point(probabilities)
                ant.visited.append(chosen_point)
                ant.distance_travelled += current_point.distance(chosen_point)
                current_point = chosen_point
            ant.distance_travelled += current_point.distance(ant.visited[0])
            if ant.distance_travelled < best_distance:
                best_route = ant.visited
                best_distance = ant.distance_travelled
        for i in range(len(best_route)):
            start_point = best_route[i]
            end_point = best_route[(i + 1) % len(best_route)]
            pheromone_matrix[start_point, end_point] += 1 / best_distance
            pheromone_matrix[end_point, start_point] += 1 / best_distance
        for key in pheromone_matrix.keys():
            pheromone_matrix[key] *= (1 - evaporation_rate)
    return best_route, best_distance


def plot_polygons_with_paths(random_polygons, best_route):
    if not best_route:
        print("No valid route found to plot.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Random Regular Polygon with TSP Path')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Plot the starting point
    ax.scatter(0, 0, 0, color='green', label='Starting Point (0, 0)')

    for polygon in random_polygons:
        x, y = polygon.exterior.xy
        z = [0] * len(x)
        ax.plot(x, y, z, alpha=0.3)

    for i in range(len(best_route)):
        start_point = best_route[i]
        end_point = best_route[(i + 1) % len(best_route)]
        ax.plot([start_point.x, end_point.x], [start_point.y, end_point.y], [0, 0], [0, 0], linestyle='dashed', color='red')

    drawing_pointer = best_route[0]
    for polygon in random_polygons:
        nearest_point = nearest_points(polygon, drawing_pointer)[0]
        path = lawn_mower_motionfrl(polygon) if abs(polygon.bounds[2] - polygon.bounds[0]) >= abs(polygon.bounds[3] - polygon.bounds[1]) else lawn_mower_motionflu(polygon)
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, [0] * len(path_x), marker='o')
            drawing_pointer = Point(path_x[-1], path_y[-1])

    ax.set_zlabel('Z-axis')
    ax.set_zlim(-1, 1)

    # Add legend
    ax.legend()

    plt.show()


# Generate random polygons
area = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
divisions = random.randint(3, 5)
sub_areas = divide_area(area, divisions)
random_polygons = []
while len(random_polygons) < len(sub_areas):
    for sub_area in sub_areas:
        if sub_area not in random_polygons:
            random_polygon = generate_random_regular_polygon(sub_area)
            random_polygons.append(random_polygon)

# Get centroids of the polygons
centroids = [list(polygon.centroid.coords)[0] for polygon in random_polygons]

# Solve TSP using Ant Colony Optimization
best_route, best_distance = ant_colony_tsp(centroids, num_ants=10, num_iterations=100)

# Plot polygons with TSP path and lawn mower paths
plot_polygons_with_paths(random_polygons, best_route)