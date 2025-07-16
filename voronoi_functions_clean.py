from collections import defaultdict
import enum
from math import pi
import numpy as np
from scipy.spatial import Voronoi
import matplotlib
matplotlib.use('Agg')  # Set matplotlib to non-interactive mode
import base64
import MyVoronoi as mv
import matplotlib.pyplot as plt
from io import BytesIO



#TODO:
# Create ridges between edges for MyVoronoi
# Remove vertices out of frame from adjacency measurements.  Never add?


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


def add_frame_edges(my_voronoi):
    my_voronoi.add_vertex(0, 0)
    my_voronoi.add_vertex(1, 0)
    my_voronoi.add_vertex(1, 1)
    my_voronoi.add_vertex(0, 1)
    edges = [[],[],[],[]]  
    print("VERTEX:")
    for i in range(len(my_voronoi.vertices)):
        x, y = my_voronoi.get_vertex_xy(i)
        print('\t', i, ':', x, y)
        if x == 0:
            edges[0].append(i)
        if x == 1:
            edges[1].append(i)
        if y == 0:
            edges[2].append(i)
        if y == 1:
            edges[3].append(i)
        
    # sort vertical access by y access 
    edges[0] = sorted(edges[0], key=lambda k: my_voronoi.get_vertex_xy(k)[1])
    edges[1] = sorted(edges[1], key=lambda k: my_voronoi.get_vertex_xy(k)[1])
    # sort horizontal access by x access
    edges[2] = sorted(edges[2], key=lambda k: my_voronoi.get_vertex_xy(k)[0])
    edges[3] = sorted(edges[3], key=lambda k: my_voronoi.get_vertex_xy(k)[0])


    # for edge in edges:
    #     for i in range(len(edge)-1):
    #         print("Adding Ridge from ", edge[i], "to ", edge[i+1])
    #         my_voronoi.add_ridge([[my_voronoi.get_vertex_xy(edge[i]), my_voronoi.get_vertex_xy(edge[i+1])])
    # for edge in edges:
    #     print("EDGE:", edge)
    
    return 1


# def add_edges_to_vertices(vertex_dict, edges, vertex_region_dict):
#     print("Vertex-to-region")
#     for vertex in vertex_region_dict:
#         print("\tVertex:", vertex, "region:", vertex_region_dict[vertex])
#     for edge in edges:
#         for i in range(len(edge)-1):
#             vertex_dict[edge[i]].append(edge[i+1])
#             vertex_dict[edge[i+1]].append(edge[i])
#     return vertex_dict


# def add_ridges_to_point_regions(ridge_vertex1, ridge_vertex2, ridge_points, my_polygon_data, my_polygon_data_advanced, vertex_region_dict):
#     polygon_point1 = ridge_points[0]
#     polygon_point2 = ridge_points[1]
#     ridge_vertex1 = int(ridge_vertex1)
#     ridge_vertex2 = int(ridge_vertex2)
    
    
#     # Index = Voronoi Point, value = set of vertices belonging to point's polygon
#     my_polygon_data[polygon_point1].add(ridge_vertex1)
#     my_polygon_data[polygon_point1].add(ridge_vertex2)
#     my_polygon_data[polygon_point2].add(ridge_vertex1)
#     my_polygon_data[polygon_point2].add(ridge_vertex2)

#     # index = vertex, value = set of polygons point connects to
#     vertex_region_dict[int(ridge_vertex1)].add(int(polygon_point1))
#     vertex_region_dict[int(ridge_vertex2)].add(int(polygon_point1))
#     vertex_region_dict[int(ridge_vertex1)].add(int(polygon_point2))
#     vertex_region_dict[int(ridge_vertex2)].add(int(polygon_point2))

#     # my_polygon_data_advanced[polygon_point1] = defaultdict(list)
#     # my_polygon_data_advanced[polygon_point2] = defaultdict(list)
    
#     my_polygon_data_advanced[polygon_point1][ridge_vertex1].append((ridge_vertex1, ridge_vertex2))
#     my_polygon_data_advanced[polygon_point1][ridge_vertex2].append((ridge_vertex2, ridge_vertex1))

#     my_polygon_data_advanced[polygon_point2][ridge_vertex1].append((ridge_vertex1, ridge_vertex2))
#     my_polygon_data_advanced[polygon_point2][ridge_vertex2].append((ridge_vertex2, ridge_vertex1))



def correct_finite_ridges(my_voronoi):
    print("\n\ncorrect_finite_ridges()")
    for i, ridge in enumerate(my_voronoi.ridges):
        ridge_pts = ridge.points
        ridge_vts = np.asarray(ridge.vertices)
        
        if np.all(ridge_vts >= 0):
            print("RIDGE VERTEX PAIR:", ridge_vts, '-', my_voronoi.vertices[ridge_vts[0]], ", ", my_voronoi.vertices[ridge_vts[1]])
            print("Corresponding Points:", ridge_pts)
            # Check if vertices are within frame boundaries
            
            x1, y1 = my_voronoi.get_vertex_xy(ridge_vts[0])
            x2, y2 = my_voronoi.get_vertex_xy(ridge_vts[1])
            ob_count = 0
            replace_ridge_vts = None
            if x1 < 0 or x1 > my_voronoi.width or y1 < 0 or y1 > my_voronoi.height:
                ob_count += 1
                replace_ridge_vts = 0
            if x2 < 0 or x2 > my_voronoi.width or y2 < 0 or y2 > my_voronoi.height:
                ob_count += 1
                replace_ridge_vts = 1
            if ob_count > 0:
                intersections = find_border_intersections(x1, y1, x2, y2, my_voronoi.width, my_voronoi.height)
            
            # if both of the vertices are outside the frame, use both intersections
            if ob_count == 2:
                print("\t\tBoth vertices are outside frame boundaries")
                if ((x1 >= my_voronoi.width and x2 >= my_voronoi.width) or (y1 >= my_voronoi.height and y2 >= my_voronoi.height)) or \
                (x1 < 0 and x2 < 0) or (y1 < 0 and y2 < 0):
                    print("\t\tSKIPING REASON: Both X or Y below min border")
                    print("\tSKIPPING THIS ONE")
                    continue
                    
                # Add both intersections to the list of vertices and ridge lists
                vertex_id1 = my_voronoi.add_vertex(intersections[0][0], intersections[0][1])
                vertex_id2 = my_voronoi.add_vertex(intersections[1][0], intersections[1][1])
                #my_voronoi.update_ridge_in_region(ridge_pts, ridge_vts[1], ridge_vts[0])
                #my_voronoi.update_ridge_in_region(ridge_pts, ridge_vts[0], ridge_vts[1])
                my_voronoi.add_ridge([int(vertex_id1), int(vertex_id2)], ridge_pts)

                # for intersection in intersections:
                #     print("\t1).  NEW RIDGE VERTEX #", len(ret_vertices), ":", np.asarray(intersection))
                #     vor.vertices = np.append(vor.vertices, [intersection], axis=0)
                print("\t1).  NEW RIDGE Between Ridge Vertices", vertex_id1, "and", vertex_id2)  


                # TODO:  Worry about regions after confirmed, we are getting correct points, vertices, and ridges
                #point_x, point_y = tuple(sorted(vor.ridge_points[i]))
                #ridge_points_dict[new_vertex_index1, new_vertex_index2] = { int(point_x), int(point_y)}
                #add_ridges_to_point_regions(new_vertex_index1, new_vertex_index2, vor.ridge_points[i], my_polygon_data, my_polygon_data_advanced, vertex_region_dict)
            
            # If only one vertex is out of the frame 
            elif ob_count == 1:
                print(f"\t\tOne of Vertex {ridge_vts} at ({x1:.2f}, {y1:.2f}) or ({x2:.2f}, {y2:.2f}) is outside frame boundaries")
                for intersect in intersections:
                    if  (x1 < intersect[0] < x2) or (x2 < intersect[0] < x1) and (y1 < intersect[1] < y2) or (y2 < intersect[1] < y1):
                        #ret_vertices.append(np.asarray(intersect))  

                        #vor.vertices = np.append(vor.vertices, [intersect], axis=0)
                        #vor.ridge_vertices[i] = [ridge_vts[replace_ridge_vts], len(ret_vertices)-1]
                        vertex_id = my_voronoi.add_vertex(intersect[0], intersect[1])
                        # TODO:  Determine if we keep line below
                        #my_voronoi.set_ridge_vertices(i, [ridge_vts[replace_ridge_vts], vertex_id])
                        print("\t2).  NEW (RIDGE) VERTEX #", vertex_id, ":", np.asarray(intersect))  
                        
                        #if replacement vertex is index 0, then the new vertex is index 1 and vice versa
                        safe_vertex_id = 1 - replace_ridge_vts 
                        
                        print("Changing Ridge Vertex: ", int(ridge_vts[safe_vertex_id]), "Before:", int(vertex_id))
                        my_voronoi.set_ridge_vertices(i, [ridge_vts[safe_vertex_id], vertex_id])
                        print("Changing Ridge Vertex: ", int(ridge_vts[safe_vertex_id]), "After:", int(vertex_id))
                        my_voronoi.add_ridge([int(ridge_vts[safe_vertex_id]), int(vertex_id)], ridge_pts)
                        my_voronoi.remove_vertex_from_ridge_adj_dict(ridge_vts[safe_vertex_id], ridge_vts[replace_ridge_vts])
                        my_voronoi.remove_vertex_from_ridge_adj_dict(ridge_vts[replace_ridge_vts], ridge_vts[safe_vertex_id])
                        print("CHECK: Adding Ridge", [int(ridge_vts[safe_vertex_id]), int(vertex_id)], "with points", ridge_pts)
                        


                        #ridge_dict[ridge_vts[0]].append(new_vertex_index)
                        #ridge_dict[new_vertex_index].append(ridge_vts[0])
                        #vertex_dict[ridge_vts[0]].append(new_vertex_index)
                        #vertex_dict[new_vertex_index].append(int(ridge_vts[0]))
                        #point_x, point_y = tuple(sorted(my_voronoi.ridges[i].points))
                        #point_x, point_y = tuple(sorted(vor.ridge_points[i]))
                        #ridge_points_dict[tuple(sorted((ridge_vts[1], new_vertex_index)))] = {int(point_x), int(point_y)}
                        #add_ridges_to_point_regions(ridge_vts[1], new_vertex_index, vor.ridge_points[i], my_polygon_data, my_polygon_data_advanced, vertex_region_dict)
                        

                    else:  # intersection is on the other side and not between desired points
                        print("\t\tOutside Range - Skipping ", intersect)
            else:    # if there both vertices are inside the frame    
                pass
                # ridge_dict[ridge_vts[0]].append(ridge_vts[1])
                # ridge_dict[ridge_vts[1]].append(ridge_vts[0])
                # add_ridges_to_point_regions(ridge_vts[0], ridge_vts[1], vor.ridge_points[i], my_polygon_data, my_polygon_data_advanced, vertex_region_dict)
                # point_x, point_y = tuple(sorted(vor.ridge_points[i]))
                # ridge_points_dict[tuple(sorted(ridge_vts))] = {int(point_x), int(point_y)}
                # vertex_dict[ridge_vts[0]].append(int(ridge_vts[1]))
                # vertex_dict[ridge_vts[1]].append(int(ridge_vts[0]))
                # ret_ridges.append(ridge_vts)

                
        else:
            print("\t", ridge_vts)
            # if ridge_pts[1] < 0:
            #     ridge_dict[ridge_vts[0]].append(ridge_vts[1])
            # else:
            #     ridge_dict[ridge_vts[1]].append(ridge_vts[0])
        #print("\t\tRidge Points:", ridge_pts)


def correct_infinite_vertices(my_voronoi):
    print("\n\ncorrect_infinite_vertices()")
    my_voronoi.print_ridges()
     # Plot infinite ridges with clipping
    center = my_voronoi.get_points_all_np().mean(axis=0)
    for ridge in my_voronoi.ridges:
        ridge_pts = np.asarray(ridge.points)
        ridge_vts = np.asarray(ridge.vertices)
        print("\n\nRidge Vertices:", ridge_vts, " for Ridge Points:", ridge_pts)
        if np.any(ridge_vts < 0):
            index = ridge_vts[ridge_vts >= 0][0] # set index to the positive vertex index
            print("\tVertex", index, "->", my_voronoi.get_vertex_xy(index))
            #print("\t\tAdjacent Ridge Verts:", np.asarray(ridge_dict[index]))
            t = my_voronoi.get_point_np(ridge_pts[1]) - my_voronoi.get_point_np(ridge_pts[0])

            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            
            midpoint = (my_voronoi.get_point_np(ridge_pts[0]) + my_voronoi.get_point_np(ridge_pts[1])) / 2
            far_point = my_voronoi.get_vertex_np(index) + np.sign(np.dot(midpoint - center, n)) * n * 1000
            
            x1, y1 = my_voronoi.get_vertex_xy(index)
            x2, y2 = far_point
    
            if not (0 < x1 < my_voronoi.width and 0 < y1 < my_voronoi.height):
                print("\t\tValid Index is completely off frame -> SKIPPING", x1, my_voronoi.width, y1, my_voronoi.height)
                continue 

            # Find intersection with boundaries
            intersections = find_border_intersections(x1, y1, x2, y2, my_voronoi.width, my_voronoi.height)
            intersections.sort(key=lambda p: (p[0] - x1)**2 + (p[1] - y1)**2)

            print("\tIntersections:", np.asarray(intersections))
            intersect_point = 0
            
            # ret_ridges.append([index, len(ret_vertices)-1])
            #my_voronoi.add_ridge((index, len(my_voronoi.vertices)), ridge_pts)
            new_vertex_id = my_voronoi.add_vertex(intersections[0][0], intersections[0][1])
            #print("Setting Rdige:", index)
            #my_voronoi.set_ridge_vertices(i, [ridge_vts[safe_vertex_id], vertex_id])
            my_voronoi.add_ridge((index, new_vertex_id), ridge_pts)
            print("UDPATED DICT for", index, ":", my_voronoi.ridge_dict[index])


            # TODO THIS IS A BUG, it doesn't project a line at all
            if len( my_voronoi.ridge_dict[index] ) == 2:
                continue

            center_point = my_voronoi.get_vertex_xy(index)
            if 0 <= center_point[0] <= my_voronoi.width and 0 <= center_point[1] <= my_voronoi.height: 
                vectors = list()
                print("Point", index, ":", np.asarray(center_point))
                infinite_count = 0
                print("CHECK ADJACENT VERTS:", my_voronoi.ridge_dict[index])
                for vert in my_voronoi.ridge_dict[index]:
                    if vert == -1:
                        infinite_count += 1
                        other_point = intersections[0]
                    else:
                        other_point = my_voronoi.get_vertex_np(vert) #vor.vertices[vert]
                    # TODO:  Optimize repeat calcualtions with vectors object in my_voronoi
                    vectors.append(np.array([other_point[0] - center_point[0], other_point[1] - center_point[1]]))
                    print("\tto point", vert, ":", np.asarray(other_point))


                # if one vertex has two adjacent infinite vertices, add the most suggest version and exit
                # the calculations will be done when the second infinite vertex is added, the infinite count will be 1 next time
                if infinite_count == 2:
                    temp_vertex_id = my_voronoi.add_vertex(intersections[0][0], intersections[0][1])
                    my_voronoi.add_ridge((index, temp_vertex_id), ridge_pts)
                    my_voronoi.ridge_dict[index].append(temp_vertex_id)
                    my_voronoi.ridge_dict[temp_vertex_id].append(index)
                    #my_voronoi.update_ridge_in_region(ridge_pts, index, )
                    print("Adding new vertex:", temp_vertex_id, " to system")
                    break

                if infinite_count <= 1:
                    angles = calculate_angles(vectors[0], vectors[1], vectors[2])
                    angles_deg = [np.degrees(a) for a in angles]
                    anglesum = sum(angles)
                    anglesum_deg = sum(angles_deg)
                    adjacent_vertices = my_voronoi.ridge_dict[index]
                    print("\tAdjacent Vertices:", adjacent_vertices)
                    #if len(adjacent_vertices) == 3:
                        
                    # Print angles in radians and degrees, and denote which two vertex points they are between
                    angle_pairs = [
                        (adjacent_vertices[0], adjacent_vertices[1]),
                        (adjacent_vertices[1], adjacent_vertices[2]),
                        (adjacent_vertices[2], adjacent_vertices[0])
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
                        my_voronoi.set_vertex_xy(new_vertex_id, intersections[1])
                        for vert in adjacent_vertices:
                            other_vertex = my_voronoi.get_vertex_xy(vert)
                            # if vert == new_vertex_id:
                            #     other_point = intersections[1]
                            #     intersect_point = 1
                            # else:
                            #     other_vertex = vor.get_vertex_xy(vert)
                            #     #other_point = vor.vertices[vert]
                            #     other_vertex = my_voronoi.get_vertex_xy(vert)
                            new_vectors.append(np.array([other_vertex[0] - center_point[0], other_vertex[1] - center_point[1]]))
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
                            points = np.random.uniform(0.01, [width-(width*0.10), height-(height*0.10)], (len(points), 2))
                            return generate_voronoi_data_clean(points, width, height)
            
            print("NEW RIDGE VERTEX #", new_vertex_id, ":", np.asarray(intersections[intersect_point]))  
            print("NEW RIDGE Between Ridge Vertices", index, "and", new_vertex_id)  
            #vertex_dict[index].append(len(ret_vertices))  
            #vertex_dict[len(ret_vertices)].append(index)
            
            #add_ridges_to_point_regions(len(ret_vertices), index, ridge_pts, my_polygon_data, my_polygon_data_advanced, vertex_region_dict)
            #ret_vertices.append(np.asarray(intersections[intersect_point]))
            #vor.vertices = np.append(vor.vertices, [intersections[intersect_point]], axis=0)
            
    return my_voronoi

     





def generate_voronoi_data_clean(points, width=1, height=1):
    print("\n\n", ("_"*50))
    print("generate_voronoi_data()")
    """Generate Voronoi diagram and return it as a base64 encoded image."""
    # Create random points
    points_data = points #np.random.uniform(0.01, [width-(width*0.10), height-(height*0.10)], (points, 2))
    
    # Create Voronoi diagram
    vor = Voronoi(points_data)
    
    my_voronoi = mv.MyVoronoi(width, height, vor.points, vor.vertices, vor.ridge_vertices, vor.ridge_points)



    correct_finite_ridges(my_voronoi)

    correct_infinite_vertices(my_voronoi)

    edges = add_frame_edges(my_voronoi)
    # add_edges_to_vertices(my_voronoi, edges)
    #vertex_dict = add_edges_to_vertices(vertex_dict, edges, vertex_region_dict)

    #edges = add_frame_edges(ret_vertices)
    #vertex_dict = add_edges_to_vertices(vertex_dict, edges, vertex_region_dict)


    my_voronoi.print_voronoi()

    points_arr = list()
    vertices_arr = list()
    ridges_arr = list()

    for i in range(len(my_voronoi.points)):
        points_arr.append(my_voronoi.get_point_xy(i))

    for i in range(len(my_voronoi.vertices)):
        vertices_arr.append(my_voronoi.get_vertex_xy(i))

    for i in range(len(my_voronoi.ridges)):
        ridges_arr.append(my_voronoi.get_ridge_vertices(i))

    diagram_data_json = generate_voronoi_plot_clean(my_voronoi)

    return points_arr, vertices_arr, ridges_arr, diagram_data_json





def generate_voronoi_plot_clean(my_voronoi):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot points
    print("DATA POINTS:")
    for point in my_voronoi.points:
        ax.plot(point.x*my_voronoi.width, point.y*my_voronoi.height, 'bo')
        print("\tPoint ", point.index, ": (", point.x, point.y, ")")
        ax.text(point.x, point.y, str(point.index), color='blue', fontsize=12, fontweight='bold', ha='right', va='bottom')

    
    # # Number the points_data
    # print("DATA POINTS:")
    # for i, (x_raw, y_raw) in enumerate(my_voronoi.points):
    #     x, y = x_raw * my_voronoi.width, y_raw * my_voronoi.height
    #     print("\tPoint ", i, ": (", x, y, ")")
    #     ax.text(x, y, str(i), color='blue', fontsize=12, fontweight='bold', ha='right', va='bottom')
    
    # Plot vertices with different colors
    colors = ['r', 'g', 'c', 'm', 'y', 'k']
    print("Vertices (color coding):")
    for i in range(len(my_voronoi.vertices)):
        vert = my_voronoi.get_vertex_xy(i)
        x, y = vert[0] * my_voronoi.width, vert[1] * my_voronoi.height
        color = colors[i % len(colors)]
        print(f"\t{i} : {x, y} (color: {color})")
        ax.plot(x, y, color + 'o')
        ax.text(x, y, str(i), color=color, fontsize=10, fontweight='bold', ha='right', va='bottom')
    
    
    print("RIDGES:")
    # Plot ridges
    #for i, simplex in enumerate(ridges):
    for i in range(len(my_voronoi.ridges)):
        ridge_vts = my_voronoi.get_ridge_vertices(i)
        if np.all(np.asarray(ridge_vts) >= 0):
            x1, y1 = my_voronoi.get_vertex_xy(ridge_vts[0])
            x2, y2 = my_voronoi.get_vertex_xy(ridge_vts[1])
            if x1 >= my_voronoi.width and x2 >= my_voronoi.width or \
                y1 >= my_voronoi.height and y2 >= my_voronoi.height or \
                x1 <= 0 and x2 <= 0 or \
                y1 <= 0 and y2 <= 0:
                print("\tPSYCH! I AINT PLOTTING - ", ridge_vts)
                continue
            else:
                ax.plot([x1, x2], [y1, y2], 'k-')
        # simplex = np.asarray(simplex)
        # if np.all(simplex >= 0):
        #     x1, y1 = my_voronoi.get_ridg
        #     vertices[simplex[0]][0] * width, vertices[simplex[0]][1] * height
        #     x2, y2 = vertices[simplex[1]][0] * width, vertices[simplex[1]][1] * height
        #     print("\t", simplex, '-', f"({x1}, {y1}), ({x2}, {y2})")
        #     if    x1 >= width and x2 >= width or \
        #         y1 >= height and y2 >= height or \
        #         x1 <= 0 and x2 <= 0 or \
        #         y1 <= 0 and y2 <= 0:
        #         print("\tPSYCH! I AINT PLOTTING - ", simplex)
        #         continue
        #     else:
        #         ax.plot([x1, x2], [y1, y2], 'k-')


    # print("REGIONS:")
    # for j, region in enumerate(regions):
    #     print("\tRegion", j, ":", region)
    #     ordered_region = list()
    #     #ordered_region = sort_polygon_data(region, vertices)
    #     if len(region) > 0:  # Avoid empty regions
    #        # Scale vertices by width and height
    #        polygon = [(vertices[i][0] * width, vertices[i][1] * height) for i in region]
    #        #print("\t\t", polygon)
    #        # Generate a random color for each region
    #        color = np.random.rand(3,)  # Random RGB color
    #        ax.fill(*zip(*polygon), facecolor=color, alpha=0.3)

    # Set plot limits and aspect
    ax.set_xlim(0, my_voronoi.width)
    ax.set_ylim(0, my_voronoi.height)
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
