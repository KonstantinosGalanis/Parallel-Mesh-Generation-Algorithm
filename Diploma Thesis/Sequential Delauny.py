import open3d as o3d
import numpy as np
import torch
from hilbertcurve.hilbertcurve import HilbertCurve
from itertools import combinations
from scipy.spatial import ConvexHull
from numba import cuda
from numba.typed import List
import time

class Tetrahedron:
    def __init__(self, vertices, index):
        self.vertices = vertices  # List of vertex indices
        self.index = index  # Index of the tetrahedron in the DelaunayTriangulation list
        self.neighbors = [None, None, None, None]  # One neighbor for each face
        self.edges = self.compute_edges()

    def compute_edges(self):
        # Computes all edges of the tetrahedron as a list of vertex index pairs
        return [
            (self.vertices[i], self.vertices[j]) for i in range(4) for j in range(i+1, 4)
        ]

    def find_shared_face(self, other_tetra):
        faces_self = [
            {self.vertices[1], self.vertices[2], self.vertices[3]},  # Face opposite to vertex 0
            {self.vertices[0], self.vertices[2], self.vertices[3]},  # Face opposite to vertex 1
            {self.vertices[0], self.vertices[1], self.vertices[3]},  # Face opposite to vertex 2
            {self.vertices[0], self.vertices[1], self.vertices[2]},  # Face opposite to vertex 3
        ]

        # Dynamically compute faces for the other tetrahedron
        faces_other = [
            {other_tetra.vertices[1], other_tetra.vertices[2], other_tetra.vertices[3]},  # Face opposite to vertex 0
            {other_tetra.vertices[0], other_tetra.vertices[2], other_tetra.vertices[3]},  # Face opposite to vertex 1
            {other_tetra.vertices[0], other_tetra.vertices[1], other_tetra.vertices[3]},  # Face opposite to vertex 2
            {other_tetra.vertices[0], other_tetra.vertices[1], other_tetra.vertices[2]},  # Face opposite to vertex 3
        ]

        for i, face_self in enumerate(faces_self):
            for j, face_other in enumerate(faces_other):
                if face_self == face_other:
                    return i, j  # Return the face indices for self and other tetrahedron
        return None  # No shared face found

    def set_neighbor_with(self, other_tetra, shared_face_indices):
        self_face_index, other_face_index = shared_face_indices
        self.neighbors[self_face_index] = other_tetra.index
        other_tetra.neighbors[other_face_index] = self.index

class DelaunayTriangulation:
    def __init__(self):
        self.tetrahedra = []
        self.vertices = []
        self.free_indices = []  # Stack of indices of removed tetrahedra for reuse

    @staticmethod
    def volume_of_tetrahedron(points):
        p0, p1, p2, p3 = points
        return np.abs(np.dot((p0 - p3), np.cross((p1 - p3), (p2 - p3)))) / 6.0

    @staticmethod
    def is_coplanar(points, tolerance=1e-10):
        return np.abs(DelaunayTriangulation.volume_of_tetrahedron(points)) < tolerance

    @staticmethod
    def find_non_coplanar_points(points):
        num_points = len(points)
        for i in range(num_points - 3):
            for j in range(i + 1, num_points - 2):
                for k in range(j + 1, num_points - 1):
                    for l in range(k + 1, num_points):
                        if not DelaunayTriangulation.is_coplanar(points[[i, j, k, l]]):
                            return points[[i, j, k, l]]
        raise ValueError("No four non-coplanar points found.")

    def init(self, S):
        non_coplanar_points = self.find_non_coplanar_points(S)
        if non_coplanar_points is not None:
            # Directly return the Tetrahedron object created by add_tetrahedron
            return self.add_tetrahedron(non_coplanar_points)
        else:
            raise ValueError("Unable to find four non-coplanar points for initialization.")
        
    @staticmethod
    def compute_hilbert_indices(points, n):
        p = points.shape[1]  # Dimensions
        n = max(1, min(n, 2))  # Ensure n is at least 1 and cap the resolution to avoid overflow
        hilbert_curve = HilbertCurve(p, n)
        min_point = np.amin(points, axis=0)
        max_point = np.amax(points, axis=0)
        
        # Check if min_point and max_point are very close or identical
        diff = max_point - min_point
        if np.any(diff <= 1e-9):  # Adjust the threshold as necessary
            # Handle the case where all points are identical or very close to each other
            normalized_points = np.zeros_like(points)
        else:
            normalized_points = (points - min_point) / diff

        try:
            # Ensure that the operation doesn't result in invalid values
            int_points = np.clip((normalized_points * (2**n - 1)), 0, 2**n - 1).astype(np.int64)
            indices = [hilbert_curve.distance_from_point(tuple(p)) for p in int_points]
            return np.array(indices)
        except OverflowError as e:
            print(f"OverflowError encountered with n={n}: {e}")
            raise

    @staticmethod
    def SoRt(S):
        k = 10  # Tuning parameter for Hilbert curve resolution
        if len(S) > 1:
            n = int(np.ceil(k * np.log2(max(1, len(S))))) 
        else:
            n = 1  
        np.random.shuffle(S)  # Shuffle S to prevent pathological cases
        hilbert_indices = DelaunayTriangulation.compute_hilbert_indices(S, n)
        sorted_indices = np.argsort(hilbert_indices)
        return S[sorted_indices]
    
    @staticmethod
    def signed_volume_of_tetrahedron(a, b, c, d):
        a, b, c, d = map(np.array, [a, b, c, d])  # Ensure all are NumPy arrays
        return np.dot(np.cross(b - a, c - a), d - a) / 6.0

    def is_visible_from_facet(self, tetrahedron, facet_index, point):
        # Extract the vertices of the tetrahedron except the one opposite to the facet
        vertex_indices = [i for i in range(4) if i != facet_index]
        vertices = [self.vertices[idx] for idx in [tetrahedron.vertices[i] for i in vertex_indices]]

        # Calculate the signed volume for the tetrahedron formed by the facet and the point
        volume = self.signed_volume_of_tetrahedron(vertices[0], vertices[1], vertices[2], point)

        # If the signed volume is negative, the point is visible from this facet
        return volume < 0
    
    def walk(self, start_tetrahedron, point):
        current_tetrahedron = start_tetrahedron
        visited = set()

        while True:
            if current_tetrahedron is None:
                break

            visited.add(current_tetrahedron.index)
            found_visible_neighbor = False
            for i, neighbor_index in enumerate(current_tetrahedron.neighbors):
                if neighbor_index is not None and neighbor_index not in visited:
                    neighbor_tetrahedron = self.tetrahedra[neighbor_index]
                    if neighbor_tetrahedron is not None:
                        visible = self.is_visible_from_facet(current_tetrahedron, i, point)
                        if visible:
                            current_tetrahedron = neighbor_tetrahedron
                            found_visible_neighbor = True
                            return current_tetrahedron

            if not found_visible_neighbor:
                return current_tetrahedron

    def compute_circumsphere(self, tetrahedron):
        # Convert points to NumPy arrays for consistent mathematical operations
        A = np.array(self.vertices[tetrahedron.vertices[0]])
        B = np.array(self.vertices[tetrahedron.vertices[1]])
        C = np.array(self.vertices[tetrahedron.vertices[2]])
        D = np.array(self.vertices[tetrahedron.vertices[3]])

        # Use coordinates relative to point 'a' of the tetrahedron.
        ba = B - A
        ca = C - A
        da = D - A
        
        # Squares of lengths of the edges incident to 'a'.
        len_ba = np.dot(ba, ba)
        len_ca = np.dot(ca, ca)
        len_da = np.dot(da, da)
        
        # Cross products of these edges.
        cross_cd = np.cross(ca, da)
        cross_db = np.cross(da, ba)
        cross_bc = np.cross(ba, ca)
        
        # Calculate the denominator of the formula.
        denominator = 0.5 / np.dot(ba, cross_cd)
        
        # Calculate offset (from 'a') of circumcenter.
        circ = (len_ba * cross_cd + len_ca * cross_db + len_da * cross_bc) * denominator
        
        # Calculate the coordinates of the circumcenter
        circumcenter = A + circ
        radius = np.linalg.norm(ba + circ)
        return circumcenter, radius

    def is_in_circumsphere(self, tetrahedron, point):
        center, radius = self.compute_circumsphere(tetrahedron)
        
        # Calculate the distance from the center of the circumsphere to the point
        distance_to_point = np.linalg.norm(center - point)
        
        # Determine if the point is inside the circumsphere
        inside = distance_to_point <= radius
        return inside

    def find_cavity(self, start_tetrahedron, point):
        queue = [start_tetrahedron]
        cavity_tetrahedra = set()
        visited = set()
        cavity_tetrahedra_indices = []

        while queue:
            tetrahedron = queue.pop(0)
            if tetrahedron.index in visited:  # Skip if already visited
                continue
            visited.add(tetrahedron.index)  # Mark as visited immediately
            if self.is_in_circumsphere(tetrahedron, point):
                cavity_tetrahedra.add(tetrahedron)
                cavity_tetrahedra_indices.append(tetrahedron.index)
                for neighbor_index in tetrahedron.neighbors:
                    if neighbor_index is not None and neighbor_index not in visited:
                        queue.append(self.tetrahedra[neighbor_index])

        # Mark the tetrahedra in the cavity for deletion
        for tetra_index in cavity_tetrahedra_indices:
            self.remove_tetrahedron(tetra_index)

        sorted_cavity_tetrahedra = sorted(list(cavity_tetrahedra), key=lambda tetra: tetra.index)
        return sorted_cavity_tetrahedra

    def find_boundary_facets(self, cavity_tetrahedra):
        boundary_facets = set()

        if len(self.tetrahedra) == 1 and len(cavity_tetrahedra) == 1:
            initial_tetrahedron = next(iter(cavity_tetrahedra))
            for combination in combinations(initial_tetrahedron.vertices, 3):
                face_points = [self.vertices[i] for i in combination]
                boundary_facets.add(frozenset(tuple(point) for point in face_points))
        else:
            for tetrahedron in cavity_tetrahedra:
                for face_index, neighbor_index in enumerate(tetrahedron.neighbors):
                    if neighbor_index is None or self.tetrahedra[neighbor_index] not in cavity_tetrahedra:
                        combination = [tetrahedron.vertices[i] for i in range(4) if i != face_index]
                        face_points = [self.vertices[i] for i in combination]
                        boundary_facets.add(frozenset(tuple(point) for point in face_points))

        return list(boundary_facets)

    def create_delaunay_ball(self, cavity_tetrahedra, new_point):
        Facets = self.find_boundary_facets(cavity_tetrahedra)
        print("All Facets:", Facets)
        new_tetrahedra = []
        new_point_tuple = tuple(new_point) if not isinstance(new_point, tuple) else new_point
        for face_vertices in Facets:
            new_tetrahedron_vertices = list(face_vertices) + [new_point_tuple]
            new_tetrahedron = self.add_tetrahedron(new_tetrahedron_vertices)
            new_tetrahedra.append(new_tetrahedron)

        sorted_new_tetrahedra = sorted(new_tetrahedra, key=lambda x: x.index)
        return sorted_new_tetrahedra
    
    def sort_neighbors(self):
        for tetra in self.tetrahedra:
            if tetra is not None:
                tetra.neighbors.sort(key=lambda x: (x is not None, x))

    def remove_tetrahedron(self, index):
        tetrahedron = self.tetrahedra[index]
        if tetrahedron is None:
            return

        # Update neighbors' references to this tetrahedron
        for neighbor_index in tetrahedron.neighbors:
            if neighbor_index is not None:
                neighbor = self.tetrahedra[neighbor_index]
                if neighbor is not None:
                    # Remove the current tetrahedron's index from the neighbor's neighbors list
                    neighbor.neighbors = [n if n != index else None for n in neighbor.neighbors]

        # Mark the tetrahedron as removed by setting it to None
        self.tetrahedra[index] = None
        # Add this index to the list of free indices
        self.free_indices.append(index)
        self.sort_neighbors()

    def add_tetrahedron(self, vertex_coords):
        vertex_indices = [self.add_vertex(coord) for coord in vertex_coords]

        # Check if there's an available free index to reuse
        if self.free_indices:
            # Reuse the first available free index
            index = self.free_indices.pop(0)
            new_tetrahedron = Tetrahedron(vertex_indices, index)
            self.tetrahedra[index] = new_tetrahedron
        else:
            # No free index, append a new tetrahedron to the list
            index = len(self.tetrahedra)
            new_tetrahedron = Tetrahedron(vertex_indices, index)
            self.tetrahedra.append(new_tetrahedron)
        
        # Check for neighbors among existing tetrahedra
        for existing_tetra in self.tetrahedra[:-1]:  # Exclude the newly added tetrahedron
            if existing_tetra is not None:  # Check if existing_tetra is not None
                shared_face = new_tetrahedron.find_shared_face(existing_tetra)
                if shared_face:
                    new_tetrahedron.set_neighbor_with(existing_tetra, shared_face)
        self.sort_neighbors()
        return new_tetrahedron

    def add_vertex(self, coord):
        # Try to find an existing vertex with the same coordinates
        for i, existing_coord in enumerate(self.vertices):
            if np.allclose(coord, existing_coord, atol=1e-8):
                return i  # Return the index of the existing vertex
        
        # If no existing vertex is found, add the new vertex
        self.vertices.append(coord)
        return len(self.vertices) - 1

def visualize_triangulation(dt):
    geometries = []
    # Assuming each Tetrahedron now has an 'edges' attribute, which is a list of edge tuples.
    # Collect all unique edges across all tetrahedra to visualize the mesh structure.
    all_edges = set()
    for tetrahedron in dt.tetrahedra:
        if tetrahedron is None:
            continue
        for edge in tetrahedron.edges:
            all_edges.add(edge)

    # Convert unique edge vertex indices to line segments for visualization.
    lines = []
    for edge in all_edges:
        lines.append(list(edge))

    # Convert vertex indices to actual points.
    points = np.array(dt.vertices)
    lines = np.array(lines)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )

    # Set the color of the lines. Here, using a single color for simplicity.
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines))])

    geometries.append(line_set)

    # Visualize the unique edges as a line set.
    o3d.visualization.draw_geometries(geometries)

def split_by_3_pytorch(x, bits=21, max_bit_length=63):
    # Ensure x is a PyTorch tensor
    x = torch.tensor(x, dtype=torch.int64)

    # Mask the input x to only keep the number of bits we are interested in
    x = x & ((1 << bits) - 1)

    # Split bits and interleave with zeros
    x = (x | (x << 2 * max_bit_length)) & torch.tensor(0x1f00000000ffff, dtype=torch.int64)
    x = (x | (x << max_bit_length)) & torch.tensor(0x1f0000ff0000ff, dtype=torch.int64)
    x = (x | (x << max_bit_length // 2)) & torch.tensor(0x100f00f00f00f00f, dtype=torch.int64)
    x = (x | (x << max_bit_length // 4)) & torch.tensor(0x10c30c30c30c30c3, dtype=torch.int64)
    x = (x | (x << max_bit_length // 8)) & torch.tensor(0x1249249249249249, dtype=torch.int64)
    return x

def compute_moore_index_pytorch(points, bits=21):
    # Normalize points to be non-negative and within the range [0, 2^bits - 1]
    min_vals = torch.min(points, 0, keepdim=True).values
    max_vals = torch.max(points, 0, keepdim=True).values
    points = ((points - min_vals) * ((1 << bits) - 1) / (max_vals - min_vals)).int()

    # Compute Morton codes (Moore indices) for the normalized points
    morton_codes = (split_by_3_pytorch(points[:, 0], bits, 21) | 
                    split_by_3_pytorch(points[:, 1], bits, 21) << 1 | 
                    split_by_3_pytorch(points[:, 2], bits, 21) << 2)
    return morton_codes

def partition_points(points, n_threads):
    # Compute Moore indices for each point. Since compute_moore_index_pytorch is vectorized,
    # it can handle all points at once without the need for a list comprehension.
    moore_indices = compute_moore_index_pytorch(points)

    # Sort points by Moore indices
    sorted_indices, _ = torch.sort(moore_indices)

    # Partition sorted indices into approximately equal chunks for each thread
    partitions = torch.array_split(sorted_indices, n_threads)
    return partitions

def Sequential_Delaunay(S):
    DT = DelaunayTriangulation()  # Initialize the DelaunayTriangulation instance
    tau = DT.init(S)  # tau is now a Tetrahedron object
    counter = 0
    initial_vertices = [DT.vertices[idx] for idx in tau.vertices]  # Directly using tau's vertices
    S_prime = DT.SoRt(np.array([point for point in S if not any(np.array_equal(point, vert) for vert in initial_vertices)]))
    for p in S_prime:
        print(f"\nProcessing point {counter}: {p}")
        tau = DT.walk(tau, p)
        C = DT.find_cavity(tau, p)
        B = DT.create_delaunay_ball(C, p)
        if B:  
            tau = B[-1]  # Update tau to the last tetrahedron in B
        
        counter += 1
    return DT

mesh = o3d.io.read_triangle_mesh("stanford-bunny.obj")
mesh.compute_vertex_normals()  # Optional: Compute vertex normals if not already present

# Sample points on the mesh
# You can adjust the number of points and the radius for Poisson disk sampling
point_cloud = mesh.sample_points_poisson_disk(number_of_points=40)

# Alternatively, use uniform sampling (comment out the above and uncomment below)
# point_cloud = mesh.sample_points_uniformly(number_of_points=1000)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])

points = np.asarray(point_cloud.points)
#points_tensor = torch.from_numpy(points).float()

#if torch.cuda.is_available():
#    points_tensor = points_tensor.to('cuda')

n_threads = 64
#partitions = partition_points(points_tensor, n_threads)
start_time = time.time()
DT = Sequential_Delaunay(points)
end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")
#print(f"Number of tetrahedra: {len(DT.tetrahedra)}")
visualize_triangulation(DT)
