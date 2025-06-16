import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_random_points(num_points, width=1000, height=1000):
    """Generate random points in 2D space."""
    points = np.random.uniform(0, [width, height], (num_points, 2))
    return points

def plot_voronoi(points, width=1000, height=1000):
    """Plot Voronoi diagram using scipy.spatial.Voronoi with proper boundary clipping."""
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Plot points
    ax.plot(points[:,0], points[:,1], 'bo')
    
    # Plot ridges
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-')
    
    # Plot infinite ridges
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 1000
            
            # Clip the line to the boundary
            x1, y1 = vor.vertices[i]
            x2, y2 = far_point
            
            # Check intersection with boundaries
            intersections = []
            if x1 != x2:  # not vertical
                # Intersection with left boundary
                y_left = y1 + (0 - x1) * (y2 - y1) / (x2 - x1)
                if 0 <= y_left <= height:
                    intersections.append((0, y_left))
                # Intersection with right boundary
                y_right = y1 + (width - x1) * (y2 - y1) / (x2 - x1)
                if 0 <= y_right <= height:
                    intersections.append((width, y_right))
            if y1 != y2:  # not horizontal
                # Intersection with bottom boundary
                x_bottom = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                if 0 <= x_bottom <= width:
                    intersections.append((x_bottom, 0))
                # Intersection with top boundary
                x_top = x1 + (height - y1) * (x2 - x1) / (y2 - y1)
                if 0 <= x_top <= width:
                    intersections.append((x_top, height))
            
            # Sort intersections by distance from start point
            intersections.sort(key=lambda p: (p[0] - x1)**2 + (p[1] - y1)**2)
            
            # Plot clipped line
            if intersections:
                ax.plot([x1, intersections[0][0]], [y1, intersections[0][1]], 'k-')
    
    # Set plot limits
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    plt.title('Voronoi Diagram')
    plt.show()

def main():
    # Generate random points
    num_points = 10
    points = generate_random_points(num_points)
    
    # Plot the Voronoi diagram
    plot_voronoi(points, width=1000, height=1000)

if __name__ == "__main__":
    main()
