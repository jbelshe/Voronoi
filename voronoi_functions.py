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

def add_frame_edges(vertices, ridges):
    vertices.append([0, 0])
    vertices.append([1, 0])
    vertices.append([1, 1])
    vertices.append([0, 1])
    edges = [[],[],[],[]]  
    print("VERTEX:")
    for i, vert in enumerate(vertices):
        x, y = np.asarray(vert)
        print('\t', i, ':', float(vert[0]), float(vert[1]))
        if x == 0:
            edges[0].append(i)
        if x == 1:
            edges[1].append(i)
        if y == 0:
            edges[2].append(i)
        if y == 1:
            edges[3].append(i)
        

    print(edges)
    # sort vertical access by y access 
    edges[0] = sorted(edges[0], key=lambda x: vertices[x][1])
    edges[1] = sorted(edges[1], key=lambda x: vertices[x][1])
    # sort horizontal access by x access
    edges[2] = sorted(edges[2], key=lambda x: vertices[x][0])
    edges[3] = sorted(edges[3], key=lambda x: vertices[x][0])



    for edge in edges:
        print("EDGE:", edge)
    return edges

def add_edges_to_vertices(vertex_dict, edges):
    for edge in edges:
        for i in range(len(edge)-1):
            vertex_dict[edge[i]].append(edge[i+1])
            vertex_dict[edge[i+1]].append(edge[i])
    return vertex_dict


def assign_regions_to_vertices(vertex_region_dict, vertex_dict, regions, ridge_points_dict):
    ret_regions = [[] for _ in range(len(regions))]
    for i, region in enumerate(regions):
        region_set = set(region) # turn region into a set
        print("Assessing Region:", i, region, region_set)
        for vertex in region: # for each point in the region
            print("\tChecking Vertex:", vertex)
            if vertex >= 0 and vertex in vertex_dict: # if the point is a valid vertex
                adjacent_vertices = vertex_dict[vertex]
                print("\t\tAdjacent Vertices:", adjacent_vertices)
                if len(adjacent_vertices) == 3: # if its an internal vertex
                    possible_points = []
                    for av in adjacent_vertices: # get vertex's adjacency vector (3 points)
                        print("\t\t\tChecking Adjacent Vertex:", av)
                        if av in region_set:  # if the adjacent vertex is in the region
                            ridge_pair = tuple(sorted((av, vertex)))
                            print("\t\t\tSearching Ridge Points Dict for:", ridge_pair)
                            if ridge_pair in ridge_points_dict:
                                possible_points.append(ridge_points_dict[ridge_pair])
                                print("\t\t\tAdding Point:", ridge_points_dict[ridge_pair])
                    if possible_points and len(possible_points) == 2:
                        print("\t\tPOSSIBLE POINTS:", possible_points)
                        intersection = list(possible_points[0] & possible_points[1])
                        print("\t\tIntersection POINT:", intersection)
                        if len(intersection) == 1:
                            point = intersection[0]
                            ret_regions[point] = region
                        else:
                            print("\t\t\tIntersection Issue:", intersection)
                    else:
                        print("\t\t\tNo Possible Points", possible_points)
    print("DID THIS SHIT WORK")
    for i, region in enumerate(ret_regions):
        if len(region) > 0:
            print("REGION:", i, region)
    return ret_regions

                
            
            

def generate_voronoi_data(points, width=1, height=1):
    print("\n\n", ("_"*50))
    print("generate_voronoi_data()")
    """Generate Voronoi diagram and return it as a base64 encoded image."""
    # Create random points
    points_data = np.random.uniform(0.01, [width-(width*0.10), height-(height*0.10)], (points, 2))
    
    # Create Voronoi diagram
    vor = Voronoi(points_data)
    

    vertex_region_dict = defaultdict(list)
    vertex_dict = defaultdict(list) 
    ridge_points_dict = defaultdict(set)

    print("Regions:")
    for i, region in enumerate(vor.regions):
        print(i, " : ", region)

    print("Vertex REGIONS:")
    for i, region in enumerate(vor.regions):
        for p in region:
            vertex_region_dict[p].append(i)


    for v in vertex_region_dict:
        print("\t", v, " : ", vertex_region_dict[v])



    ret_points = points_data
    ret_vertices = list()
    print("Vertices:")
    for i, vertex in enumerate(vor.vertices):
        print(i, ":", vertex)
        vertex_dict[i] = list()
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
            print("Corresponding Points:", vor.ridge_points[i])
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
                # if abs(intersections[0][0] - intersections[1][0]) == 1000 or abs(intersections[0][1] - intersections[1][1]) == 1000:
                #     print("\t\tSKIPING REASON: Edge to edge")
                #     print("\tSKIPPING THIS ONE")
                #     continue
                if (x1 >= width and x2 >= width) or \
                    (y1 >= height and y2 >= height):
                    print("\t\tSKIPING REASON: Both X or Y above max border")
                    print("\tSKIPPING THIS ONE")
                    continue
                elif (x1 < 0 and x2 < 0) or \
                    (y1 < 0 and y2 < 0):
                    print("\t\tSKIPING REASON: Both X or Y below min border")
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
                vertex_dict[len(ret_vertices)-1].append(len(ret_vertices)-2)
                vertex_dict[len(ret_vertices)-2].append(len(ret_vertices)-1)
                point_x, point_y = tuple(sorted(vor.ridge_points[i]))
                ridge_points_dict[len(ret_vertices)-2, len(ret_vertices)-1] = { int(point_x), int(point_y)}
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
                            vertex_dict[simplex[1]].append(len(ret_vertices)-1)
                            vertex_dict[len(ret_vertices)-1].append(int(simplex[1]))

                            point_x, point_y = tuple(sorted(vor.ridge_points[i]))
                            ridge_points_dict[tuple(sorted((simplex[1], len(ret_vertices)-1)))] = {int(point_x), int(point_y)}
                        else:  # if the OB vertex is the second one
                            print("\t2). NEW RIDGE Between Ridge Vertices: ", simplex[0], "to", len(ret_vertices)-1)
                            vor.ridge_vertices[i] = [simplex[0], len(ret_vertices)-1]
                            ret_ridges.append([simplex[0], len(ret_vertices)-1])
                            ridge_dict[simplex[0]].append(len(ret_vertices)-1)
                            ridge_dict[len(ret_vertices)-1].append(simplex[0])
                            vertex_dict[simplex[0]].append(len(ret_vertices)-1)
                            vertex_dict[len(ret_vertices)-1].append(int(simplex[0]))
                            point_x, point_y = tuple(sorted(vor.ridge_points[i]))
                            ridge_points_dict[tuple(sorted((simplex[0], len(ret_vertices)-1)))] = {int(point_x), int(point_y)}
                    else:  # intersection is on the other side and not between desired points
                        print("\t\tOutside Range - Skipping ", intersect)
            else:    # if there both vertices are inside the frame    
                ridge_dict[simplex[0]].append(simplex[1])
                ridge_dict[simplex[1]].append(simplex[0])

                point_x, point_y = tuple(sorted(vor.ridge_points[i]))
                ridge_points_dict[tuple(sorted(simplex))] = {int(point_x), int(point_y)}
                vertex_dict[simplex[0]].append(int(simplex[1]))
                vertex_dict[simplex[1]].append(int(simplex[0]))
                ret_ridges.append(simplex)
                
        else:
            print("\t", simplex)
            if ridge_pts[1] < 0:
                ridge_dict[simplex[0]].append(simplex[1])
            else:
                ridge_dict[simplex[1]].append(simplex[0])
        print("\t\tRidge Points:", ridge_pts)



    print("RIDGE Dictionary:")
    for ridge in ridge_points_dict:
        print("\tRidge", np.asarray(ridge), "connects points:", np.asarray(ridge_points_dict[ridge]))

    ret_regions = assign_regions_to_vertices(vertex_region_dict, vertex_dict, vor.regions,ridge_points_dict)


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


    #print("VD:", vertex_dict)
    new_vertices = defaultdict(list)  # TODO:  Asesss need for this

    # Plot infinite ridges with clipping
    center = points_data.mean(axis=0)

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

            # TODO: Figure out how to determine when an offrame vertex can send a line all the way across
                # I think you need to do interesection calculations with all other lines
                # not worth the pain imo, I'm going to calculate # of polygons and redo if #'s don't match up
            # # if the valid vertex is off frame, skip
            if not (0 < x1 < width and 0 < y1 < height):
                print("\t\tValid Index is completely off frame -> SKIPPING", x1, width, y1, height)
                continue 
            # if (0 < x1 < width and 0 < y1 < height):
            #     print("NEW RIDGE VERTEX #", len(ret_vertices)-1, ":", np.asarray(intersections[0]))  
            #     ret_vertices.append(np.asarray(intersections[0]))
            #     print("NEW RIDGE VERTEX #", len(ret_vertices)-1, ":", np.asarray(intersections[1])) 
            #     ret_vertices.append(np.asarray(intersections[1]))
            #     vor.vertices = np.append(vor.vertices, [intersections[0]], axis=0)
            #     vor.vertices = np.append(vor.vertices, [intersections[1]], axis=0)
            #     print("NEW RIDGE Between Ridge Vertices", len(ret_vertices)-2, "and",len(ret_vertices)-1)  
            #     ret_ridges.append([len(ret_vertices)-2, len(ret_vertices)-1]) 


            
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
                            return generate_voronoi_data(points, width, height)
            
            print("NEW RIDGE VERTEX #", len(ret_vertices), ":", np.asarray(intersections[intersect_point]))  
            print("NEW RIDGE Between Ridge Vertices", index, "and",len(ret_vertices))  
            vertex_dict[index].append(len(ret_vertices))  
            vertex_dict[len(ret_vertices)].append(index)
            ret_vertices.append(np.asarray(intersections[intersect_point]))
            vor.vertices = np.append(vor.vertices, [intersections[intersect_point]], axis=0)
            ret_ridges.append([index, len(ret_vertices)-1])

    edges = add_frame_edges(ret_vertices, ret_ridges)
    vertex_dict = add_edges_to_vertices(vertex_dict, edges)


    print("Vertex Adjacency Matrix:")
    for vertex in vertex_dict:
        print("\t", vertex, np.asarray(vertex_dict[vertex]))



    # TODO:  Assess why this sometimes doesn't return 3 elements
    # TODO:  See if we can return the voronoi objects instead of maintaining both simultaneously
    # TODO:  See if we can convert this into polygon shapes 
    print("RETURNING:\nPOINTS", ret_points, "\nVERTICES", ret_vertices, "\nRIDGES", ret_ridges)
    return (ret_points, ret_vertices, ret_ridges, vor.regions)

def generate_voronoi_plot(points, vertices, ridges, regions, width=1000, height=1000):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot points
    ax.plot(points[:,0]*width, points[:,1]*height, 'bo')
    
    # Number the points_data
    print("DATA POINTS:")
    for i, (x_raw, y_raw) in enumerate(points):
        x, y = x_raw * width, y_raw * height
        print("\tPoint ", i, ": (", x, y, ")")
        ax.text(x, y, str(i), color='blue', fontsize=12, fontweight='bold', ha='right', va='bottom')
    
    # Plot vertices with different colors
    colors = ['r', 'g', 'c', 'm', 'y', 'k']
    print("Vertices (color coding):")
    for i, vert in enumerate(vertices):
        x, y = vert[0] * width, vert[1] * height
        color = colors[i % len(colors)]
        print(f"\t{i} : {x, y} (color: {color})")
        ax.plot(x, y, color + 'o')
        ax.text(x, y, str(i), color=color, fontsize=10, fontweight='bold', ha='right', va='bottom')
    
    
    print("RIDGES:")
    # Plot ridges
    for i, simplex in enumerate(ridges):
        ridge_pts = ridges[i]
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            x1, y1 = vertices[simplex[0]][0] * width, vertices[simplex[0]][1] * height
            x2, y2 = vertices[simplex[1]][0] * width, vertices[simplex[1]][1] * height
            print("\t", simplex, '-', f"({x1}, {y1}), ({x2}, {y2})")
            if    x1 >= width and x2 >= width or \
                y1 >= height and y2 >= height or \
                x1 <= 0 and x2 <= 0 or \
                y1 <= 0 and y2 <= 0:
                print("\tPSYCH! I AINT PLOTTING - ", simplex)
                continue
            else:
                ax.plot([x1, x2], [y1, y2], 'k-')


    print("REGIONS:")
    for region in regions:
        print("\tRegion:", region)
        if len(region) > 0:  # Avoid empty regions
           # Scale vertices by width and height
           polygon = [(vertices[i][0] * width, vertices[i][1] * height) for i in region]
           #print(polygon)
           # Generate a random color for each region
           color = np.random.rand(3,)  # Random RGB color
           ax.fill(*zip(*polygon), facecolor=color, alpha=0.3)

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
