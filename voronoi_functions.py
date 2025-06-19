from collections import defaultdict
from math import pi
import numpy as np
from scipy.spatial import Voronoi
import matplotlib
matplotlib.use('Agg')  # Set matplotlib to non-interactive mode
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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


def find_border_intersections(x1, y1, x2, y2, width, height):
    intersections = []
    if x1 != x2:  # horizontal line check
        y_left = y1 + (0 - x1) * (y2 - y1) / (x2 - x1)
        if 0 <= y_left <= height:
            intersections.append((0, y_left))
            #print("\tLEFT SIDE INTERSECTION")
        y_right = y1 + (width - x1) * (y2 - y1) / (x2 - x1)
        if 0 <= y_right <= height:
            intersections.append((width, y_right))
            #print("\tRIGHT SIDE INTERSECTION")
    if y1 != y2:  # vertical line check
        x_bottom = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
        if 0 <= x_bottom <= width:
            intersections.append((x_bottom, 0))
            #print("\tBOTTOM SIDE INTERSECTION")
        x_top = x1 + (height - y1) * (x2 - x1) / (y2 - y1)
        if 0 <= x_top <= width:
            intersections.append((x_top, height))
            #print("\tTOP SIDE INTERSECTION")
    return intersections

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

            intersections.sort(key=lambda p: (p[0] - x1)**2 + (p[1] - y1)**2)
            # Sort intersections based on distance to midpoint
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
                    if anglesum < 2 * np.pi - tolerance: 
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

def generate_voronoi_data(points, width=1000, height=1000):
    print("\n\n_"*50)
    print("generate_voronoi_data()")
    """Generate Voronoi diagram and return it as a base64 encoded image."""
    # Create random points
    points_data = np.random.uniform(0+10, [width-10, height-10], (points, 2))
    
    # Create Voronoi diagram
    vor = Voronoi(points_data)

    ret_points = points_data
    ret_vertices = list()
    print("Vertices:")
    for i, vertex in enumerate(vor.vertices):
        print(i, ":", vertex)
        ret_vertices.append(vertex)
    #ret_vertices = np.asarray(vor.vertices)
    ret_ridges = list()

    
    ridge_dict = defaultdict(list)
    print("RIDGES:")
    # Plot ridges
    for i, simplex in enumerate(vor.ridge_vertices):
        ridge_pts = vor.ridge_points[i]
        simplex = np.asarray(simplex)
        
        if np.all(simplex >= 0):
            print("RIDGE VERTEX PAIR:", simplex, '-', vor.vertices[simplex[0]], ", ", vor.vertices[simplex[1]])
            # Check if vertices are within frame boundaries
            x1, y1 = vor.vertices[simplex[0]]
            x2, y2 = vor.vertices[simplex[1]]
            ob_count = 0
            replace_simplex = None
            if x1 < 0 or x1 > width or y1 < 0 or y1 > height:
                ob_count += 1
                replace_simplex = 0
            if x2 < 0 or x2 > width or y2 < 0 or y2 > height:
                ob_count += 1
                replace_simplex = 1
            if ob_count > 0:
                intersections = find_border_intersections(x1, y1, x2, y2, width, height)
            # if both of the vertices are outside the frame, use both intersections
            if ob_count == 2:
                #continue # this is done out precaution.  It seems like this behavior can't be fixed
                print("\t\tBoth vertices are outside frame boundaries")
                if abs(intersections[0][0] - intersections[1][0]) == 1000 or abs(intersections[0][1] - intersections[1][1]) == 1000 or \
                    (x1 >= 1000 and x2 >= 1000) or \
                    (x1 >= 1000 and x2 >= 1000) or \
                    (y1 < 0 and y2 < 0) or \
                    (y1 > 0 and y2 > 0):
                    print("\tSKIPPING THIS ONE")
                    continue
                    
                for intersection in intersections:
                    print("\t1).  NEW RIDGE VERTEX #", len(ret_vertices), ":", np.asarray(intersection))
                    ret_vertices.append(np.asarray(intersection))
                    vor.vertices = np.append(vor.vertices, [intersection], axis=0)
                    vor.ridge_vertices[i] = [len(ret_vertices)-2, len(ret_vertices)-1]
                print("\t1).  NEW RIDGE Between Ridge Vertices", len(ret_vertices)-1, "and", len(ret_vertices))  
                ret_ridges.append([len(ret_vertices)-2, len(ret_vertices)-1])
                ridge_dict[len(ret_vertices)-1].append(len(ret_vertices)-2)
                ridge_dict[len(ret_vertices)-2].append(len(ret_vertices)-1)
            elif ob_count == 1:
                print(f"\t\tOne of Vertex {vertex} at ({x1:.2f}, {y1:.2f}) or ({x2:.2f}, {y2:.2f}) is outside frame boundaries")
                for intersect in intersections:
                    if  (x1 < intersect[0] < x2) or (x2 < intersect[0] < x1) and \
                        (y1 < intersect[1] < y2) or (y2 < intersect[1] < y1):
                        print("\t2).  NEW RIDGE VERTEX #", len(ret_vertices), ":", np.asarray(intersect))  
                        ret_vertices.append(np.asarray(intersect))  

                        vor.vertices = np.append(vor.vertices, [intersect], axis=0)
                        vor.ridge_vertices[i] = [simplex[replace_simplex], len(ret_vertices)-1]
                        if replace_simplex == 0:  # if the OB vertex is the first one
                            print("\t2). NEW RIDGE Between Ridge Vertices: ", simplex[1], "to", len(ret_vertices)-1)
                            vor.ridge_vertices[i] = [simplex[1], len(ret_vertices)-1]
                            ret_ridges.append([simplex[1], len(ret_vertices)-1])
                            ridge_dict[simplex[1]].append(len(ret_vertices)-1)
                            ridge_dict[len(ret_vertices)-1].append(simplex[1])
                        else:  # if the OB vertex is the second one
                            print("\t2). NEW RIDGE Between Ridge Vertices: ", simplex[0], "to", len(ret_vertices)-1)
                            vor.ridge_vertices[i] = [simplex[0], len(ret_vertices)-1]
                            ret_ridges.append([simplex[0], len(ret_vertices)-1])
                            ridge_dict[simplex[0]].append(len(ret_vertices)-1)
                            ridge_dict[len(ret_vertices)-1].append(simplex[0])
                    else:  # intersection is on the other side and not between desired points
                        print("\t\tOutside Range - Skipping ", intersect)
            else:    # if there both vertices are inside the frame    
                ridge_dict[simplex[0]].append(simplex[1])
                ridge_dict[simplex[1]].append(simplex[0])
                ret_ridges.append(simplex)
                
        else:
            print("\t", simplex)
            if ridge_pts[1] < 0:
                ridge_dict[simplex[0]].append(simplex[1])
            else:
                ridge_dict[simplex[1]].append(simplex[0])
        print("\t\tRidge Points:", ridge_pts)

    print("Updated Vertices:")
    for i, vertex in enumerate(np.asarray(vor.vertices)):
        print(i, " ", vertex)

    print("Updated RIDGES:")
    # Plot ridges
    for i, simplex in enumerate(vor.ridge_vertices):
        ridge_pts = vor.ridge_points[i]
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            print("\t", simplex, '-', vor.vertices[simplex[0]], ", ", vor.vertices[simplex[1]])
        else:
            print("\t", simplex, '-', vor.vertices[simplex[0]], ", ", vor.vertices[simplex[1]])

    print("RIDGE DICT:")
    for ridge in ridge_dict:
        print(ridge, np.asarray(ridge_dict[ridge]))


    new_vertices = defaultdict(list)  # TODO:  Asesss need for this

    # Plot infinite ridges with clipping
    center = points_data.mean(axis=0)
    #if 0:
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        print("\n\nRidge Vertices:", simplex, " for Ridge Points:", pointidx)
        if np.any(simplex < 0):
            index = simplex[simplex >= 0][0]
            print("\tPoint", index, "->", vor.vertices[index])
            print("\t\tAdjacent Ridge Verts:", np.asarray(ridge_dict[index]))
            adjacent_verts = ridge_dict[index] 
            t = points_data[pointidx[1]] - points_data[pointidx[0]]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = points_data[pointidx].mean(axis=0)
            far_point = vor.vertices[index] + np.sign(np.dot(midpoint - center, n)) * n * 1000
            
            x1, y1 = vor.vertices[index]
            x2, y2 = far_point

            # if the valid vertex is off frame, skip
            if not (0 < x1 < width and 0 < y1 < height):
                continue 
            
            print("\tMidpoint:", midpoint)
            print("\tFar Point:", far_point)
            # Find intersection with boundaries
            intersections = find_border_intersections(x1, y1, x2, y2, width, height)

            intersections.sort(key=lambda p: (p[0] - x1)**2 + (p[1] - y1)**2)
            print("\tIntersections:", np.asarray(intersections))
            intersect_point = 0



            # Find Angles
            center_point = vor.vertices[index]
            if 0 <= center_point[0] <= width and 0 <= center_point[1] <= height: 
                vectors = list()
                print("Point", index, ":", np.asarray(center_point))
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
                    new_vertices[len(vor.vertices)] = intersections
                    print("Adding new vertex:", len(vor.vertices), " to system")
                    for j in range(len(adjacent_verts)):
                        if adjacent_verts[j] == -1:
                            adjacent_verts[j] = len(vor.vertices)
                            break
                    # print("Updated adjacent vertices for vertex:", index, ":", adjacent_verts)
                    # print("Creating NEW RIDGE:", index, len(vor.vertices))
                    # print("NEW VERTEX:", intersections[0], "index:", len(vor.vertices))
                    #ret_ridges.append(i, len(vor.vertices))
                    #vor.vertices.append(intersections[0])
                    #ret_vertices.append(intersections[0])

                if infinite_count <= 1:
                    angles = calculate_angles(vectors[0], vectors[1], vectors[2])
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
                    if anglesum < 2 * np.pi - tolerance: 
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
                        angles = calculate_angles(new_vectors[0], new_vectors[1], new_vectors[2])
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
            
            print("NEW RIDGE VERTEX #", len(ret_vertices), ":", np.asarray(intersections[intersect_point]))  
            print("NEW RIDGE Between Ridge Vertices", index, "and",len(ret_vertices))  
            ret_vertices.append(np.asarray(intersections[intersect_point]))
            vor.vertices = np.append(vor.vertices, [intersections[intersect_point]], axis=0)
            ret_ridges.append([index, len(ret_vertices)-1])


    # TODO:  Assess why this sometimes doesn't return 3 elements
    # TODO:  See if we can return the voronoi objects instead of maintaining both simultaneously
    # TODO:  See if we can convert this into polygon shapes 
    print("RETURNING:\nPOINTS", ret_points, "\nVERTICES", ret_vertices, "\nRIDGES", ret_ridges)
    return (ret_points, ret_vertices, ret_ridges)

def generate_voronoi_plot(points, vertices, ridges, width=1000, height=1000):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot points
    ax.plot(points[:,0], points[:,1], 'bo')
    
    # Number the points_data
    print("DATA POINTS:")
    for i, (x, y) in enumerate(points):
        print("\tPoint ", i, ": (", x, y, ")")
        ax.text(x, y, str(i), color='blue', fontsize=12, fontweight='bold', ha='right', va='bottom')
    
    # Plot vertices with different colors
    colors = ['r', 'g', 'c', 'm', 'y', 'k']
    print("Vertices (color coding):")
    for i, vert in enumerate(vertices):
        color = colors[i % len(colors)]
        print(f"\t{i} : {vert} (color: {color})")
        ax.plot(vert[0], vert[1], color + 'o')
        ax.text(vert[0], vert[1], str(i), color=color, fontsize=10, fontweight='bold', ha='right', va='bottom')
    
    
    print("RIDGES:")
    # Plot ridges
    for i, simplex in enumerate(ridges):
        ridge_pts = ridges[i]
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            print("\t", simplex, '-', vertices[simplex[0]], ", ", vertices[simplex[1]])
            x1, y1 = vertices[simplex[0]]
            x2, y2 = vertices[simplex[1]]
            if abs(x1 - x2) == 1000 or abs(y1 - y2) == 1000 or \
                x1 >= 1000 and x2 >= 1000 or \
                y1 >= 1000 and y2 >= 1000 or \
                x1 < 0 and x2 < 0 or \
                y1 < 0 and y2 < 0:
                print("\tPSYCH! I AINT PLOTTING - ", simplex)
                continue
            else:
                ax.plot([vertices[simplex[0]][0], vertices[simplex[1]][0]], [vertices[simplex[0]][1], vertices[simplex[1]][1]], 'k-')




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
