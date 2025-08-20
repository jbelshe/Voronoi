from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass
class Vertex:
    """Represents a vertex in the Voronoi diagram."""
    x: float
    y: float
    adjacent_points: Set[int]
    index: int
    
@dataclass
class Point:
    """Represents an input point that generates the Voronoi cell."""
    x: float
    y: float
    index: int

@dataclass
class Ridge:
    """Represents a ridge (edge) between two Voronoi cells."""
    vertices: Tuple[int, int]
    points: List[int]

@dataclass
class Region:
    """Represents a region in the Voronoi diagram."""
    id: int
    vertices: Set[int]
    ridge_adjacency: Dict[int, Set[int]]
    deleted_vertices: Set[int]
    ordered_vertices: List[int]


class MyVoronoi:
    """Class representing a Voronoi diagram."""
    def __init__(self, width, height, points, vertices, ridge_vertices, ridge_points):
        self.width = width
        self.height = height
        self.vertices: List[Vertex] = []  # List of vertices
        self.ridges: List[Ridge] = []     # List of ridges
        self.points: List[Point] = []     # List of input points
        self.regions: List[Region] = []   # List of regions
        self.ridge_dict = defaultdict(list)
        for point in points:
            self.add_point(point[0], point[1])
            
        for vertex in vertices:
            self.add_vertex(vertex[0], vertex[1])
            
        for rv, rp in zip(ridge_vertices, ridge_points):
            self.add_ridge(rv, rp)
            self.add_vertex_points(rv[0], rp)
            self.add_vertex_points(rv[1], rp)
            
        self.print_voronoi()
        

    # def __init__(self):
    #     """Initialize an empty Voronoi diagram."""
    #     self.vertices: List[Vertex] = []  # List of vertices
    #     self.ridges: List[Ridge] = []     # List of ridges
    #     self.points: List[Point] = []     # List of input points
    #     self.regions: List[Region] = []   # List of regions
        
    def add_point(self, x: float, y: float) -> int:
        """Add a new point to the diagram and return its index."""
        point = Point(x=x, y=y, index=len(self.points))
        self.points.append(point)
        self.add_region(point.index)
        return point.index

    def add_region(self, id: int) -> int:
        region = Region(id=id, vertices=set(), ridge_adjacency=defaultdict(set), deleted_vertices=set(), ordered_vertices=list())
        self.regions.append(region)
        return region.id

        
    def add_vertex(self, x: float, y: float, points_indices: Tuple[int, int] = None) -> int:
        """Add a new vertex to the diagram and return its index."""
        vertex = Vertex(x=x, y=y, index=len(self.vertices), adjacent_points=set())
        self.vertices.append(vertex)
        if points_indices is not None:
            self.add_vertex_points(len(self.vertices) - 1, points_indices)
        return len(self.vertices) - 1

    def add_vertex_points(self, vertex_index: int, point_indices: Tuple[int, int]):
        """Add a new vertex to the diagram and return its index."""
        self.vertices[vertex_index].adjacent_points.add(int(point_indices[0]))
        self.vertices[vertex_index].adjacent_points.add(int(point_indices[1]))
        
    def add_ridge(self, vertex_indices: Tuple[int, int], point_indices: List[int]) -> int:
        """Add a new ridge between two vertices."""

        ridge = Ridge(
            vertices = vertex_indices,
            points = point_indices
        )
        print("add_ridge -- ADDED RIDGE with vertices:", vertex_indices, "and points:", point_indices)

        if vertex_indices[0] != -1 and vertex_indices[1] != -1:
            self.upload_ridge_to_region(vertex_indices, point_indices)
            self.ridge_dict[int(vertex_indices[0])].append(int(vertex_indices[1]))
            self.ridge_dict[int(vertex_indices[1])].append(int(vertex_indices[0]))
        self.ridges.append(ridge)
        return len(self.ridges) - 1



    def sort_region_vertices(self) -> int:
        for region in self.regions:
            ret_val = self.order_region_vertices(region)
            if ret_val != 1:
                return -1
        return 1


    def order_region_vertices(self, region: Region) -> int:
        ordered_region = list()
        visited = set()
        curr_node = region.vertices.pop()
        region.vertices.add(curr_node)
        region_size = len(region.vertices) - len(region.deleted_vertices)

        while len(ordered_region) < region_size:
            x, y = self.get_vertex_xy(curr_node)

            # if the current vertex is deleted or OOB, move to back of set
            if curr_node in region.deleted_vertices or self.is_out_of_bounds_xy(x, y):
                curr_node = region.vertices.pop()
                region.vertices.add(curr_node)  
                continue
            
            #print("Checking:", curr_node, "Visited:", visited, "Curr_Size:", len(ordered_region), "Neighbours:", region.ridge_adjacency[curr_node])
            # add node to ordered region and mark as visited
            ordered_region.append(curr_node)
            visited.add(curr_node)

            # look at its neighbors and move to the next one that hasn't been visited
            for neighbor in region.ridge_adjacency[curr_node]:
                if neighbor not in visited:
                    curr_node = neighbor
                    continue   
            
            if len(region.ridge_adjacency[curr_node]) > 2:
                print("ERROR - Region", region.id, "has more than 2 adjacents at vertex", curr_node)
                ordered_region = []
                break
            #elif len(region.ridge_adjacency[curr_node]) == 1:
                #print("ERROR - Region", region.id, "has only 1 adjacent at vertex", curr_node)
                #exit(1)

        region.ordered_vertices = ordered_region
        return 1




    def upload_ridge_to_region(self, vertex_indices: Tuple[int, int], point_indices: Tuple[int, int]):
        for point_index in point_indices: # Either one or two points in point indices 
            self.regions[point_index].ridge_adjacency[vertex_indices[0]].add(int(vertex_indices[1]))
            self.regions[point_index].ridge_adjacency[vertex_indices[1]].add(int(vertex_indices[0]))
            self.regions[point_index].vertices.add(int(vertex_indices[0]))
            self.regions[point_index].vertices.add(int(vertex_indices[1]))


    def update_ridge_in_region(self, point_indices: Tuple[int, int], vertex_index: int, old_index: int):
        self.regions[point_indices[0]].ridge_adjacency[vertex_index].remove(old_index)
        self.regions[point_indices[1]].ridge_adjacency[vertex_index].remove(old_index)
        self.regions[point_indices[0]].deleted_vertices.add(int(old_index))
        self.regions[point_indices[1]].deleted_vertices.add(int(old_index))

    def get_point(self, index: int) -> Point:
        """Get point by index."""
        return self.points[index]

    def get_point_xy(self, index: int) -> Tuple[float, float]:
        """Get point x and y coordinates by index."""
        return self.points[index].x, self.points[index].y

    def get_point_np(self, index: int) -> np.ndarray:
        """Get point x and y coordinates by index."""
        return np.array([self.points[index].x, self.points[index].y])

    def get_points_all_np(self) -> np.ndarray:
        """Get all points as a numpy array."""
        return np.array([[point.x, point.y] for point in self.points])
        
    def get_vertex(self, index: int) -> Vertex:
        """Get vertex by index."""
        return self.vertices[index]

    def set_vertex_xy(self, index: int, xy: Tuple[float, float]):
        """Set vertex by index."""
        self.vertices[index].x = xy[0]
        self.vertices[index].y = xy[1]

    def get_vertex_xy(self, index: int) -> Tuple[float, float]:
        """Get vertex x and y coordinates by index."""
        return self.vertices[index].x, self.vertices[index].y

    def get_vertex_np(self, index: int) -> np.ndarray:
        """Get vertex x and y coordinates by index."""
        return np.array([self.vertices[index].x, self.vertices[index].y])

    def get_vertex_adjacent_points(self, index: int) -> Set[int]:
        """Get vertex adjacent points by index."""
        return self.vertices[index].adjacent_points
        
    def get_ridge(self, index: int) -> Ridge:
        """Get ridge by index."""
        return self.ridges[index]

    def get_ridge_vertices(self, index: int) -> Tuple[int, int]:
        """Get ridge vertices by index."""
        return self.ridges[index].vertices
        
    def get_ridges_for_point(self, point_index: int) -> List[Ridge]:
        """Get all ridges associated with a given point."""
        return [ridge for ridge in self.ridges if point_index in ridge.point_indices]


    def set_ridge_vertices(self, index: int, vertices: Tuple[int, int]):
        """Set the vertices for a given ridge."""
        self.ridges[index].vertices = vertices

    def is_valid_ridge(self, ridge: Ridge) -> bool:
        x1, y1 = self.get_vertex_xy(ridge.vertices[0])
        x2, y2 = self.get_vertex_xy(ridge.vertices[1])
        if x1 >= self.width and x2 >= self.width or \
            y1 >= self.height and y2 >= self.height or \
            x1 <= 0 and x2 <= 0 or \
            y1 <= 0 and y2 <= 0:
            return False
        return True

    def get_min_max_points(self):
        min_x_pt = -1
        min_y_pt = -1
        max_x_pt = -1
        max_y_pt = -1
        min_x = 1
        min_y = 1
        max_x = 0
        max_y = 0
        for point in self.points:
            if point.x < min_x:
                min_x = point.x
                min_x_pt = point.index
            if point.x > max_x:
                max_x = point.x
                max_x_pt = point.index
            if point.y < min_y:
                min_y = point.y
                min_y_pt = point.index
            if point.y > max_y:
                max_y = point.y
                max_y_pt = point.index
        return min_x_pt, min_y_pt, max_x_pt, max_y_pt

    def remove_vertex_from_ridge_adj_dict(self, index: int, vertex: int):
        """Remove the adjacent vertex for a given ridge."""
        print("Removing Vertex", vertex, "from Ridge", index, "->", self.ridge_dict[index])
        self.ridge_dict[index].remove(vertex)

    def is_out_of_bounds_xy(self, x: float, y: float) -> bool:
        return x < 0 or x > self.width or y < 0 or y > self.height


    def print_points(self):
        """Print the points in the diagram."""
        print("\n\nprint_points()")
        for point in self.points:
            print(f"\t{point.index}: ({point.x}, {point.y})")

    def print_vertices(self):
        """Print the vertices in the diagram."""
        print("\n\nprint_vertices()")
        for vertex in self.vertices:
            #print("\t",vertex)
            print(f"\t{vertex.index}: ({vertex.x}, {vertex.y})")

    def print_ridges(self):
        """Print the ridges in the diagram."""
        print("\n\nprint_ridges()")
        print("\tRidge Count:", len(self.ridges))
        for i, ridge in enumerate(self.ridges):
            if len(ridge.points) == 2:
                print(f"\tRIDGE {i}). [{ridge.vertices[0]} to {ridge.vertices[1]}] - for points {ridge.points[0]} and {ridge.points[1]}")
            elif len(ridge.points) == 1:
                print(f"\tRIDGE {i}). [{ridge.vertices[0]} to {ridge.vertices[1]}] - for points {ridge.points[0]}")
            else:
                print(f"\tRIDGE {i}). [{ridge.vertices[0]} to {ridge.vertices[1]}] - NO POINTS")


    def print_regions(self):
        """Print the regions in the diagram."""
        print("\n\nprint_regions()")
        for i, region in enumerate(self.regions):
            print(f"\tREGION {i} - Point: {region.id} Vertices:{region.vertices}")
            for vertex in region.ridge_adjacency:
                print(f"\t\tVertex: {vertex} -> {region.ridge_adjacency[vertex]}")
            print(f"\t\tDeleted Vertices: {region.deleted_vertices}")

    def print_voronoi(self):
        """Print the Voronoi diagram."""
        print("\n\nprint_voronoi()")
        self.print_points()
        self.print_vertices()
        self.print_ridges()
        self.print_regions()