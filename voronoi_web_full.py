from collections import defaultdict
from math import pi
import os
from flask import Flask, request, jsonify, render_template_string
import numpy as np
from scipy.spatial import Voronoi
import matplotlib
matplotlib.use('Agg')  # Set matplotlib to non-interactive mode
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set matplotlib backend explicitly
os.environ['MPLBACKEND'] = 'Agg'

app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Voronoi Diagram Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .form-container { flex: 1; }
        .diagram-container { flex: 2; }
        .diagram { max-width: 100%; }
        input { width: 100px; }
        button { padding: 8px 16px; }
    </style>
</head>
<body>
    <h1>Voronoi Diagram Generator</h1>
    <div class="container">
        <div class="form-container">
            <form id="voronoiForm">
                <label for="points">Number of Points:</label>
                <input type="number" id="points" name="points" min="2" max="100" value="10">
                <button type="submit">Generate Diagram</button>
            </form>
        </div>
        <div class="diagram-container">
            <img id="diagram" class="diagram" src="{{ diagram_data }}" alt="Voronoi Diagram">
        </div>
    </div>
    <script>
        document.getElementById('voronoiForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const points = document.getElementById('points').value;
            const response = await fetch(`/generate?points=${points}`);
            const data = await response.json();
            document.getElementById('diagram').src = data.diagram;
        });
    </script>
</body>
</html>
'''


def angle_between_vectors(v1, v2):
    """Calculate the angle in radians between two vectors v1 and v2."""
    # Compute dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Clamp the value of cos_theta to the range [-1, 1] to prevent floating point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle = np.arccos(cos_theta)
    return angle


def calculate_angles(V1, V2, V3):
    """Calculate the angles between point P and three other points P1, P2, and P3."""
    # # Vectors from P to P1, P2, and P3
    # V1 = np.array([P1[0] - P[0], P1[1] - P[1]])
    # V2 = np.array([P2[0] - P[0], P2[1] - P[1]])
    # V3 = np.array([P3[0] - P[0], P3[1] - P[1]])
    
    # Calculate the angles between the vectors
    angle_P1_P2 = angle_between_vectors(V1, V2)
    angle_P2_P3 = angle_between_vectors(V2, V3)
    angle_P1_P3 = angle_between_vectors(V1, V3)
    
    return angle_P1_P2, angle_P2_P3, angle_P1_P3

def generate_voronoi_diagram(points, width=1000, height=1000):
    """Generate Voronoi diagram and return it as a base64 encoded image."""
    # Create random points
    points_data = np.random.uniform(0+10, [width-10, height-10], (points, 2))
    
    # Create Voronoi diagram
    vor = Voronoi(points_data)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot points
    ax.plot(points_data[:,0], points_data[:,1], 'bo')
    
    # Number the points_data
    print("DATA POITNS:")
    for i, (x, y) in enumerate(points_data):
        print("\tPoint ", i, ": (", x, y, ")")
        ax.text(x, y, str(i), color='blue', fontsize=12, fontweight='bold', ha='right', va='bottom')
    
    # Plot vertices with different colors
    colors = ['r', 'g', 'c', 'm', 'y', 'k']
    print("Vertices (color coding):")
    for i, vert in enumerate(vor.vertices):
        color = colors[i % len(colors)]
        print(f"\t{i} : {vert} (color: {color})")
        ax.plot(vert[0], vert[1], color + 'o')
        ax.text(vert[0], vert[1], str(i), color=color, fontsize=10, fontweight='bold', ha='right', va='bottom')
    
    
    ridge_dict = defaultdict(list)
    print("RIDGES:")
    # Plot ridges
    for i, simplex in enumerate(vor.ridge_vertices):
        ridge_pts = vor.ridge_points[i]
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            print("\t", simplex, '-', vor.vertices[simplex[0]], ", ", vor.vertices[simplex[1]])
            ridge_dict[simplex[0]].append(simplex[1])
            ridge_dict[simplex[1]].append(simplex[0])
            ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-')
        else:
            print("\t", simplex)
            if ridge_pts[1] < 0:
                ridge_dict[simplex[0]].append(simplex[1])
            else:
                ridge_dict[simplex[1]].append(simplex[0])
        print("\t\tRidge Points:", ridge_pts)
    
    print("RIDGE DICT:")
    for ridge in ridge_dict:
        print(ridge, np.asarray(ridge_dict[ridge]))
    
        

    new_vertices = defaultdict(list)
    new_vertices_index = -2
    # -2 ->  edge point1, edge point2
    # pointed to by X, Y, Z
    # -3 ->  edge point1, edge point2
    # pointed to by X, Y, Z

    # Plot infinite ridges with clipping
    center = points_data.mean(axis=0)
    #if 0:
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        print("\n\nVertex Points:", simplex, " for Points:", pointidx)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]
            print("Point", i, "->", vor.vertices[i])
            print("\tAdjacent Verts:", ridge_dict[i])
            adjacent_verts = ridge_dict[i] 
            t = points_data[pointidx[1]] - points_data[pointidx[0]]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = points_data[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 1000
            
            x1, y1 = vor.vertices[i]
            x2, y2 = far_point
            
            print("\tMidpoint:", midpoint)
            print("\tFar Point:", far_point)
            # Find intersection with boundaries
            intersections = []
            if x1 != x2:  # horizontal line check
                y_left = y1 + (0 - x1) * (y2 - y1) / (x2 - x1)
                if 0 <= y_left <= height:
                    intersections.append((0, y_left))
                    print("\tLEFT SIDE INTERSECTION")
                y_right = y1 + (width - x1) * (y2 - y1) / (x2 - x1)
                if 0 <= y_right <= height:
                    intersections.append((width, y_right))
                    print("\tRIGHT SIDE INTERSECTION")
            if y1 != y2:  # vertical line check
                x_bottom = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                if 0 <= x_bottom <= width:
                    intersections.append((x_bottom, 0))
                    print("\tBOTTOM SIDE INTERSECTION")
                x_top = x1 + (height - y1) * (x2 - x1) / (y2 - y1)
                if 0 <= x_top <= width:
                    intersections.append((x_top, height))
                    print("\tTOP SIDE INTERSECTION")
            
            """
                Todo:
                    -Sorting by proximity to midpoint doesn't fix interesction issue
                        -Need to make sure line runs THROUGH midpoint
                        -Proximity doesn't guarntee it runs through midpoint
                    -Determine how to handle when entire line bisects graph 
                        -Example midpoint has two closer lines
                        -OOP - proximity to lines?

            """


            intersections.sort(key=lambda p: (p[0] - x1)**2 + (p[1] - y1)**2)
            # Sort intersections based on distance to midpoint
            #intersections.sort(key=lambda p: (p[0] - midpoint[0])**2 + (p[1] - midpoint[1])**2)
            print("\tIntersections:", np.asarray(intersections))
            intersect_point = 0



            # Find Angles
            center_point = vor.vertices[i]
            if 0 <= center_point[0] <= width and 0 <= center_point[1] <= height: 
                vectors = list()
                print("Point", i, ":", np.asarray(center_point))
                infinite_count = 0
                for vert in adjacent_verts:
                    if vert == -1:
                        infinite_count += 1
                        other_point = intersections[0]
                    elif vert < -1:
                        print("\tVert:", vert, "->", np.asarray(new_vertices[vert]))
                        other_point = new_vertices[vert][0]
                    else:
                        other_point = vor.vertices[vert]
                    vectors.append(np.array([other_point[0] - center_point[0], other_point[1] - center_point[1]]))
                    print("\tto point", vert, ":", np.asarray(other_point))

                if infinite_count == 2:
                    for inter in intersections:
                        new_vertices[new_vertices_index].append(inter)
                    print("Adding new vertex:", new_vertices_index, " to system")
                    for i in range(len(adjacent_verts)):
                        if adjacent_verts[i] == -1:
                            adjacent_verts[i] = new_vertices_index
                            break
                    print("Updated adjacent vertices for vertex:", i, ":", adjacent_verts)
                    new_vertices_index -= 1

                if infinite_count <= 1:
                    angles = calculate_angles(vectors[0], vectors[1], vectors[2])
                    # angle1 = angle_between_vectors(vectors[0], vectors[1])
                    # angle2 = angle_between_vectors(vectors[1], vectors[2])
                    # angle3 = angle_between_vectors(vectors[2], vectors[0])
                    # angles = [angle1, angle2, angle3]
                    angles_deg = [np.degrees(a) for a in angles]
                    anglesum = sum(angles)
                    anglesum_deg = sum(angles_deg)

                    # Print angles in radians and degrees, and denote which two vertex points they are between
                    angle_pairs = [
                        (adjacent_verts[0], adjacent_verts[1]),
                        (adjacent_verts[1], adjacent_verts[2]),
                        (adjacent_verts[2], adjacent_verts[0])
                    ]
                    for idx, ((v1, v2), a, ad) in enumerate(zip(angle_pairs, angles, angles_deg), 1):
                        print(f"\tAngle {idx} (between vertices {v1} and {v2}): {a:.6f} rad, {ad:.2f} deg")
                    print(f"\tAngle Sum: {anglesum:.6f} rad, {anglesum_deg:.2f} deg")

                    # Check if sum is close to 360 deg (2*pi radians)
                    if abs(anglesum - 2 * np.pi) > 1e-4:
                        print("\t[WARNING] Angles do not sum to 360 degrees (2π radians)!")

                    # If sum is less than 360 degrees, try switching the intersection point
                    tolerance = 1e-4
                    if anglesum < 2 * np.pi - tolerance: # TODO can I delete this: and len(intersections) > 1:
                        print("\t[INFO] Angle sum is less than 360 degrees. Switching intersection point for infinite vertex.")
                        # Rebuild vectors using intersections[1] for vert == -1
                        new_vectors = []
                        for vert in adjacent_verts:
                            if vert == -1:
                                other_point = intersections[1]
                                intersect_point = 1
                            else:
                                other_point = vor.vertices[vert]
                            new_vectors.append(np.array([other_point[0] - center_point[0], other_point[1] - center_point[1]]))
                        # Recalculate angles
                        
                        angles = calculate_angles(vectors[0], vectors[1], vectors[2])
                        # angle1 = angle_between_vectors(new_vectors[0], new_vectors[1])
                        # angle2 = angle_between_vectors(new_vectors[1], new_vectors[2])
                        # angle3 = angle_between_vectors(new_vectors[2], new_vectors[0])
                        # angles = [angle1, angle2, angle3]
                        angles_deg = [np.degrees(a) for a in angles]
                        anglesum = sum(angles)
                        anglesum_deg = sum(angles_deg)
                        # Print recalculated angles
                        for idx, ((v1, v2), a, ad) in enumerate(zip(angle_pairs, angles, angles_deg), 1):
                            print(f"\t[RECALC] Angle {idx} (between vertices {v1} and {v2}): {a:.6f} rad, {ad:.2f} deg")
                        print(f"\t[RECALC] Angle Sum: {anglesum:.6f} rad, {anglesum_deg:.2f} deg")
                        if abs(anglesum - 2 * np.pi) > tolerance:
                            print("\t[WARNING] Even after switching intersection, angles do not sum to 360 degrees!")
                            return generate_voronoi_diagram(points, width, height)
                            print("FUCK START OVER")


            if intersections:
                ax.plot([x1, intersections[intersect_point][0]], [y1, intersections[intersect_point][1]], 'k-')
    
    # Set plot limits and aspect
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    plt.title('Voronoi Diagram')
    
    # Save figure to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    
    # Encode image as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

@app.route('/')
def index():
    # Generate initial diagram with 10 points
    diagram_data = generate_voronoi_diagram(10)
    return render_template_string(HTML_TEMPLATE, diagram_data=diagram_data)

@app.route('/generate')
def generate():
    points = int(request.args.get('points', 10))
    diagram_data = generate_voronoi_diagram(points)
    return jsonify({'diagram': diagram_data})

if __name__ == '__main__':
    from flask_cors import CORS
    CORS(app)
    app.run(debug=True, host='127.0.0.1', port=5000)
