import open3d as o3d
import numpy as np
import math
import time
from hilbertcurve.hilbertcurve import HilbertCurve
from numba import cuda
from numba import int32, float64, boolean

MAX_TETRAHEDRA = 3000
MAX_QUEUE_SIZE = 2500
MAX_VERTICES = 100

@cuda.jit(device=True)
def sort_neighbors(tetrahedra):
    # Example: Sorting each tetrahedron's neighbors to ensure -1s are at the end
    for i in range(tetrahedra.shape[0]):
        # Simple bubble sort for the neighbors of tetrahedron i, for demonstration
        for j in range(3):  # Only need to go up to the third neighbor, as the last one has no next neighbor to compare
            for k in range(3 - j):
                # Assuming neighbors are stored in the second row (index 1)
                if tetrahedra[i, 1, k] < 0 and tetrahedra[i, 1, k + 1] >= 0:
                    # Swap neighbors to move -1 (or any invalid marker) towards the end
                    tetrahedra[i, 1, k], tetrahedra[i, 1, k + 1] = tetrahedra[i, 1, k + 1], tetrahedra[i, 1, k]

@cuda.jit(device=True)
def faces_are_equal(face_a, face_b):
    """
    Check if two triangular faces are equal by ensuring each vertex in one face can be matched
    to a unique vertex in the other face, considering all vertices.
    """
    used = cuda.local.array(3, dtype=boolean)  # Track used vertices in face_b
    for i in range(3):
        used[i] = False

    # Attempt to match each vertex in face_a to a unique vertex in face_b
    for i in range(3):  # For each vertex in face_a
        matched = False
        for j in range(3):  # For each vertex in face_b
            if not used[j]:
                vertices_match = True
                for k in range(3):  # Check each coordinate
                    if face_a[i, k] != face_b[j, k]:
                        vertices_match = False
                        break
                if vertices_match:
                    used[j] = True  # Mark this vertex in face_b as matched
                    matched = True
                    break
        if not matched:
            return False  # If any vertex in face_a can't find a match in face_b, faces are not equal

    return True

@cuda.jit(device=True)
def update_neighbors_with_shared_faces(tetrahedra, idx_self, new_vertex_coords, vertices, idx_other):
    combination_self = cuda.local.array((4, 3), dtype=int32)
    idx = cuda.grid(1)
    # Populate the array with combination values
    combination_self[0, :] = (0, 1, 2)
    combination_self[1, :] = (0, 1, 3)
    combination_self[2, :] = (0, 2, 3)
    combination_self[3, :] = (1, 2, 3)
    shared_count = cuda.local.array((1,), dtype=int32)
    shared_count[0] = 0
    #print("Thread ID:", idx, "| new_vertex_coords:", new_vertex_coords[0, 0], new_vertex_coords[1, 0], new_vertex_coords[2, 0], new_vertex_coords[3, 0])
    for face_idx_self in range(4):
        current_combination = combination_self[face_idx_self]
        vertices_self = cuda.local.array(shape=(3, 3), dtype=np.float64)  # 3 vertices per face, 3 coordinates each
        
        #print("Thread ID:", idx, "| Processing self face index:", face_idx_self)
        for i, vertex_idx in enumerate(current_combination):
            global_idx = int(tetrahedra[idx_self, 0, vertex_idx])
            for coord_idx in range(3):
                vertices_self[i, coord_idx] = vertices[global_idx, coord_idx]
        
        #    print("Thread ID", idx, "| Vertex ", i, "of self face", face_idx_self, vertices_self[i, 0])
        
        for face_idx_other  in range(4):
            new_face_combination = combination_self[face_idx_other ]
            new_face_vertices = cuda.local.array(shape=(3, 3), dtype=np.float64)
            
            # Fetch vertices for the face to compare against
            for i, vertex_idx in enumerate(new_face_combination):
                for coord_idx in range(3):
                    new_face_vertices[i, coord_idx] = new_vertex_coords[vertex_idx, coord_idx]

        #    print("Thread ID", idx, "| Comparing to new face",face_idx_other ,new_face_vertices[0, 0])
            
            if faces_are_equal(vertices_self, new_face_vertices):
                #print("Thread ID", idx, "| idx_self",idx_self ,"face_idx_self", face_idx_self, "idx_other", idx_other, "face_idx_other", face_idx_other)
                tetrahedra[idx_self, 1, face_idx_self] = idx_other
                tetrahedra[idx_other, 1, face_idx_other] = idx_self

@cuda.jit(device=True)
def is_allclose(a, b, atol):
    idx = cuda.grid(1)
    close = True
    for i in range(3):
        #print("Thread ID:", idx, "| Comparing:", a[i], "and", b[i]) 
        if abs(a[i] - b[i]) > atol:
            close = False
            break
    return close

@cuda.jit(device=True)
def add_vertex(vertices, vertex_count, coord):
    idx = cuda.grid(1)
    atol = 1e-10
    for i in range(vertex_count[0]):
        #print("Thread ID:", idx, "| vertices[i]", vertices[i][0], vertices[i][1], vertices[i][2]) 
        if is_allclose(vertices[i], coord, atol):
            return i  # Vertex already exists

    if vertex_count[0] < MAX_VERTICES:
        index = cuda.atomic.add(vertex_count, 0, 1)
        vertices[index, 0] = coord[0] 
        vertices[index, 1] = coord[1]  
        vertices[index, 2] = coord[2]  
        return index
    else:
        return -1
    
@cuda.jit(device=True)
def add_tetrahedron(vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, vertex_coords):
    idx = cuda.grid(1)
    #print("Active Thread ID _ ADD:", idx)
    vertex_indices = cuda.local.array(4, dtype=int32)
    index = cuda.local.array(1, dtype=int32)
    for i in range(4):
        vertex_indices[i] = add_vertex(vertices, vertex_count, vertex_coords[i])
        #print("Thread ID:", idx, "| Vertex", i, "index:", vertex_indices[i])
        #print("Thread", idx, "vertex_count", vertex_count[0])

    index = -1 
    if free_indices_count[0] > 0:
        # Atomically decrement the free_indices_count and get the previous value
        index = cuda.atomic.sub(free_indices_count, 0, 1)
        #print("Thread ID:", idx, "prev_free_idx", index) 
        if index > 0:
            # Safely use the decremented index
            index = free_indices[index - 1]
            # Atomically increment the tetrahedron count if a valid index is used
            cuda.atomic.add(tetrahedron_count, idx, 1)
        #    print("Thread ID:", idx, "used index", index)  
        else:
            index = cuda.atomic.add(tetrahedron_count, idx, 1)
    else:
        index = cuda.atomic.add(tetrahedron_count, idx, 1)  # Increment first, then use the updated count
    
    #print("Thread ID:", idx, "tetrahedron_count", tetrahedron_count[idx])   
    #print("Thread ID:", idx, "index", index)
    # Initialize the tetrahedron's data
    for i in range(4):
        tetrahedra[idx, index, 1, i] = -1
        tetrahedra[idx, index, 0, i] = vertex_indices[i]
        #print("Thread ID:", idx, "| Tetrahedron index:", index, "| Vertex index at", i, ":", vertex_indices[i])
        #print("Thread ID:", idx, "| Vertices[i, vertex_indices[i]]", vertices[vertex_indices[i], 0], vertices[vertex_indices[i], 1], vertices[vertex_indices[i], 2])

    tetrahedra[idx, index, 2, 0] = index

    edge_combinations = cuda.local.array((6, 2), dtype=int32)
    edge_combinations[0, :] = (0, 1)
    edge_combinations[1, :] = (0, 2)
    edge_combinations[2, :] = (0, 3)
    edge_combinations[3, :] = (1, 2)
    edge_combinations[4, :] = (1, 3)
    edge_combinations[5, :] = (2, 3)
    # Iterate over edge_combinations to set edges
    for i in range(6):
        start_vertex = vertex_indices[edge_combinations[i, 0]]
        end_vertex = vertex_indices[edge_combinations[i, 1]]
        tetrahedra[idx, index, 3, i] = start_vertex
        tetrahedra[idx, index, 4, i] = end_vertex
    
    # Iterate over existing tetrahedra to find shared faces and set neighbors
    for idx_existing in range(tetrahedron_count[idx]):
        if idx_existing != index and tetrahedra[idx, idx_existing, 2, 0] != -1:
            #print("Thread ID:", idx,"idx_existing",idx_existing)
            update_neighbors_with_shared_faces(tetrahedra[idx], idx_existing, vertex_coords, vertices, index)
    
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 2, 1, 0], tetrahedra[idx, 2, 1, 1],tetrahedra[idx, 2, 1, 2], tetrahedra[idx, 2, 1, 3])
    return index

@cuda.jit
def init_kernel(all_points, num_points_per_partition, vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, d_add_result, d_tau):
    cross_result = cuda.local.array(3, dtype=float64)
    idx = cuda.grid(1)
    if idx >= all_points.shape[0]:
        return

    points = all_points[idx]
    n = num_points_per_partition[idx]
    tolerance = 1e-10
    found = False

    for i in range(n - 3):
        if found:
            break
        for j in range(i + 1, n - 2):
            if found:
                break
            for k in range(j + 1, n - 1):
                if found:
                    break
                for l in range(k + 1, n):
                    p0, p1, p2, p3 = points[i], points[j], points[k], points[l]
                    # Inline cross product calculation
                    cross_result[0] = (p1[1] - p3[1]) * (p2[2] - p3[2]) - (p1[2] - p3[2]) * (p2[1] - p3[1])
                    cross_result[1] = (p1[2] - p3[2]) * (p2[0] - p3[0]) - (p1[0] - p3[0]) * (p2[2] - p3[2])
                    cross_result[2] = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
                    # Inline dot product calculation for volume
                    dot = (p0[0] - p3[0]) * cross_result[0] + (p0[1] - p3[1]) * cross_result[1] + (p0[2] - p3[2]) * cross_result[2]
                    volume = abs(dot) / 6.0
                    if volume > tolerance:
                        for index, point_idx in enumerate((i, j, k, l)):
                            d_tau[idx, index, 0] = points[point_idx, 0]  # X-coordinate
                            #print("points[point_idx, 0]",points[point_idx, 0])
                            d_tau[idx, index, 1] = points[point_idx, 1]  # Y-coordinate
                            #print("points[point_idx, 1]",points[point_idx, 1])
                            d_tau[idx, index, 2] = points[point_idx, 2]  # Z-coordinate
                            #print("points[point_idx, 2]",points[point_idx, 2])
                        found = True
                        break
    
    print("d_tau[0,0,0]", d_tau[idx,0,0], d_tau[idx,0,1], d_tau[idx,0,2])
    print("d_tau[0,0,0]", d_tau[idx,1,0], d_tau[idx,1,1], d_tau[idx,1,2])
    print("d_tau[0,0,0]", d_tau[idx,2,0], d_tau[idx,2,1], d_tau[idx,2,2])
    print("d_tau[0,0,0]", d_tau[idx,3,0], d_tau[idx,3,1], d_tau[idx,3,2])
    d_add_result[idx] = add_tetrahedron(vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, d_tau[idx])

@cuda.jit
def mark_filtered_points_kernel(points, initial_vertices, passed_points):
    insert_idx = 0
    idx = cuda.grid(1)

    for point_idx in range(points[idx].shape[0]):
        point_passes = True
        current_point = points[idx, point_idx]  # Correctly access the current point
        for vert_idx in range(initial_vertices[idx].shape[0]):
            vert = initial_vertices[idx, vert_idx]
            all_equal = True
            for dim in range(current_point.shape[0]):  # Use current_point here
                if current_point[dim] != vert[dim]:
                    all_equal = False
                    break
            if all_equal:
                point_passes = False
                break
        if point_passes:
            for dim in range(3):  # Assign each coordinate individually
                passed_points[idx, insert_idx, dim] = points[idx, point_idx, dim]
            insert_idx += 1

@cuda.jit(device=True)
def signed_volume_of_tetrahedron(a, b, c, d):
    ab = cuda.local.array(3, dtype=float64)
    ac = cuda.local.array(3, dtype=float64)
    ad = cuda.local.array(3, dtype=float64)

    for i in range(3):
        ab[i] = b[i] - a[i]
        ac[i] = c[i] - a[i]
        ad[i] = d[i] - a[i]

    cross_ab_ac = cuda.local.array(3, dtype=float64)
    cross_ab_ac[0] = ab[1] * ac[2] - ab[2] * ac[1]
    cross_ab_ac[1] = ab[2] * ac[0] - ab[0] * ac[2]
    cross_ab_ac[2] = ab[0] * ac[1] - ab[1] * ac[0]

    dot = cross_ab_ac[0] * ad[0] + cross_ab_ac[1] * ad[1] + cross_ab_ac[2] * ad[2]

    volume = dot / 6.0
    return volume

@cuda.jit(device=True)
def is_point_inside_tetrahedron(tetrahedron, vertices, point):
    idx = cuda.grid(1)
    vertex_indices = cuda.local.array(4, dtype=int32)
    
    for i in range(4):
        vertex_indices[i] = int(tetrahedron[i])
    
    a = vertices[vertex_indices[0]]
    b = vertices[vertex_indices[1]]
    c = vertices[vertex_indices[2]]
    d = vertices[vertex_indices[3]]
    
    # Calculate signed volumes
    v1 = signed_volume_of_tetrahedron(a, b, c, point)
    v2 = signed_volume_of_tetrahedron(a, b, d, point)
    v3 = signed_volume_of_tetrahedron(a, c, d, point)
    v4 = signed_volume_of_tetrahedron(b, c, d, point)
    total_volume = signed_volume_of_tetrahedron(a, b, c, d)

    # Compute absolute values and their sum
    sum_volumes = abs(v1) + abs(v2) + abs(v3) + abs(v4)
    abs_total_volume = abs(total_volume)

    # Check if the point is inside
    is_inside = abs(sum_volumes - abs_total_volume) < 1e-7

    return is_inside

@cuda.jit(device=True)
def find_neighbor_by_facet(vertex_indices, tetrahedra, idx_current):
    idx = cuda.grid(1)
    
    # Check each possible neighbor
    for neighbor_idx in range(4):
        neighbor_id = int(tetrahedra[idx, idx_current, 1, neighbor_idx])
        if neighbor_id == -1:
            continue  # No neighbor at this facet

        # Check if neighbor shares the same vertices as the facet
        shared_vertices = 0
        # We need to check all four vertices because the neighbor's connection could be based on any facet.
        for j in range(4):  # Checking all four vertices of the neighbor
            neighbor_vertex = tetrahedra[idx, neighbor_id, 0, j]
            for k in range(3):  # Checking vertices of the facet
                if neighbor_vertex == vertex_indices[k]:
                    shared_vertices += 1
                    break  # Stop checking if we find a match

        if shared_vertices == 3:
            #print("Thread ID:", idx, "| Neighbor found at index:", neighbor_id, "sharing vertices with facet indices.")
            return neighbor_id

    #print("Thread ID:", idx, "| No neighbor found for tetrahedron at index:", idx_current)
    return idx_current

@cuda.jit(device=True)
def orientation3D(p0, p1, p2, p):
    # Dummy implementation of the orientation3D function
    # This should compute the determinant of the matrix to determine the orientation
    # Assuming p0, p1, p2, and p are points with .x, .y, .z attributes
    mat = cuda.local.array((4, 4), dtype=float64)

    # Fill the matrix with point coordinates and the extra required values for determinant calculation
    # Assuming p0, p1, p2, and p are each arrays of [x, y, z]
    mat[0, 0], mat[0, 1], mat[0, 2], mat[0, 3] = p0[0], p0[1], p0[2], p0[0]**2 + p0[1]**2 + p0[2]**2
    mat[1, 0], mat[1, 1], mat[1, 2], mat[1, 3] = p1[0], p1[1], p1[2], p1[0]**2 + p1[1]**2 + p1[2]**2
    mat[2, 0], mat[2, 1], mat[2, 2], mat[2, 3] = p2[0], p2[1], p2[2], p2[0]**2 + p2[1]**2 + p2[2]**2
    mat[3, 0], mat[3, 1], mat[3, 2], mat[3, 3] = p[0], p[1], p[2], p[0]**2 + p[1]**2 + p[2]**2

    return determinant(mat)  # Placeholder for determinant computation

@cuda.jit(device=True)
def determinant(mat):
    # Calculate the determinant of a 4x4 matrix using the explicit expansion formula
    return (mat[0, 0] * (mat[1, 1] * mat[2, 2] * mat[3, 3] + mat[1, 2] * mat[2, 3] * mat[3, 1] + mat[1, 3] * mat[2, 1] * mat[3, 2]
                        - mat[1, 3] * mat[2, 2] * mat[3, 1] - mat[1, 1] * mat[2, 3] * mat[3, 2] - mat[1, 2] * mat[2, 1] * mat[3, 3])
           - mat[0, 1] * (mat[1, 0] * mat[2, 2] * mat[3, 3] + mat[1, 2] * mat[2, 3] * mat[3, 0] + mat[1, 3] * mat[2, 0] * mat[3, 2]
                        - mat[1, 3] * mat[2, 2] * mat[3, 0] - mat[1, 0] * mat[2, 3] * mat[3, 2] - mat[1, 2] * mat[2, 0] * mat[3, 3])
           + mat[0, 2] * (mat[1, 0] * mat[2, 1] * mat[3, 3] + mat[1, 1] * mat[2, 3] * mat[3, 0] + mat[1, 3] * mat[2, 0] * mat[3, 1]
                        - mat[1, 3] * mat[2, 1] * mat[3, 0] - mat[1, 0] * mat[2, 3] * mat[3, 1] - mat[1, 1] * mat[2, 0] * mat[3, 3])
           - mat[0, 3] * (mat[1, 0] * mat[2, 1] * mat[3, 2] + mat[1, 1] * mat[2, 2] * mat[3, 0] + mat[1, 2] * mat[2, 0] * mat[3, 1]
                        - mat[1, 2] * mat[2, 1] * mat[3, 0] - mat[1, 0] * mat[2, 2] * mat[3, 1] - mat[1, 1] * mat[2, 0] * mat[3, 2]))

@cuda.jit(device=True)
def signed_volume_of_tetrahedron(a, b, c, d):
    # Create local arrays for vectors ab, ac, and ad
    ab = cuda.local.array(3, dtype=float64)
    ac = cuda.local.array(3, dtype=float64)
    ad = cuda.local.array(3, dtype=float64)
    
    # Compute vectors ab, ac, and ad
    for i in range(3):
        ab[i] = b[i] - a[i]
        ac[i] = c[i] - a[i]
        ad[i] = d[i] - a[i]
    
    # Compute the cross product of ab and ac
    cross_ab_ac = cuda.local.array(3, dtype=float64)
    cross_ab_ac[0] = ab[1] * ac[2] - ab[2] * ac[1]
    cross_ab_ac[1] = ab[2] * ac[0] - ab[0] * ac[2]
    cross_ab_ac[2] = ab[0] * ac[1] - ab[1] * ac[0]
    
    # Compute the dot product of cross_ab_ac and ad
    dot = cross_ab_ac[0] * ad[0] + cross_ab_ac[1] * ad[1] + cross_ab_ac[2] * ad[2]
    
    # Return the signed volume of the tetrahedron
    volume = dot / 6.0
    return volume

@cuda.jit(device=True)
def is_visible_from_facet(vertices, tetrahedron_data, facet_index, point):
    # Initialize a local array to store the indices of the three vertices forming the facet
    vertex_indices = cuda.local.array(3, dtype=int32)

    idx = 0
    for i in range(4):
        if i != facet_index:
            # Assuming the vertex indices are stored as floats, convert them to integers
            # This is the corrected line, casting the floating-point index to an integer
            vertex_indices[idx] = int(tetrahedron_data[0, i])
            idx += 1

    # Retrieve the actual vertex coordinates using the indices
    a = vertices[vertex_indices[0]]
    b = vertices[vertex_indices[1]]
    c = vertices[vertex_indices[2]]
    
    volume = signed_volume_of_tetrahedron(a, b, c, point)
    return volume < 0

@cuda.jit(device=True)
def walk_kernel(start_index, point, vertices, tetrahedra, d_walk_results, is_inside_tetrahedron):    
    idx = cuda.grid(1)
    sort_neighbors(tetrahedra[idx])
    if is_point_inside_tetrahedron(tetrahedra[idx, start_index, 0], vertices, point):
        is_inside_tetrahedron[idx] = True
        return start_index
    visited = cuda.local.array(MAX_QUEUE_SIZE, dtype=int32)
    visited_count = cuda.local.array(1, dtype=int32)  
    vertex_indices = cuda.local.array(3, dtype=int32)
    vertex_count = 0
    is_inside_tetrahedron[idx] = False
    for i in range(visited.shape[0]):
        visited[i] = -1 

    d_walk_results[idx] = start_index
    visited_count[0] = 0
    visited[visited_count[0]] = d_walk_results[idx]
    visited_count[0]  += 1
    print("Thread ID:", idx, "| Start Index:", start_index, "| Initial Visited:", visited[0])
    neighbor_idx = 0

    while True:
        found_visible_facet = False
        
        # Check visibility from each of the 4 facets of the current tetrahedron
        for facet_index in range(4):
            vertex_count = 0
            if neighbor_idx != -1:
                already_visited = False
                for i in range(4):
                    if i != facet_index:
                        vertex_indices[vertex_count] = int(tetrahedra[idx, d_walk_results[idx], 0, i])
                        vertex_count += 1
                #print("Thread ID:", idx, "|vertex_indices",vertex_indices[0], vertex_indices[1], vertex_indices[2])
                a = vertices[vertex_indices[1]]
                b = vertices[vertex_indices[0]]
                c = vertices[vertex_indices[2]]

                #print("Thread ID:", idx, "| Vertex A coordinates:", a[0])
                #print("Thread ID:", idx, "| Vertex B coordinates:", b[0])
                #print("Thread ID:", idx, "| Vertex C coordinates:", c[0])
                #print("Thread ID:", idx, "| point:", point[0])
                orientation = orientation3D(a, b, c, point)
                #print("Thread ID:", idx, "|volume:",orientation)
                neighbor_idx = find_neighbor_by_facet(vertex_indices, tetrahedra, d_walk_results[idx])
                print("Thread ID:", idx, "| Checking Neighbor at Index:", neighbor_idx)
                for j in range(visited_count[0]):
                    #print("Thread ID:", idx, "|visited",visited[j])
                    #print("Thread ID:", idx, "|neighbor_idx",neighbor_idx)
                    if visited[j] == neighbor_idx:
                        already_visited = True
                        break
                if neighbor_idx != -1 and not already_visited:
                    visited[visited_count[0]] = neighbor_idx
                    visited_count[0] += 1
                    if orientation < 0:
                        print("Thread ID:", idx, "| Point is visible from facet", facet_index, "of tetrahedron", d_walk_results[idx], ". Moving to neighbor", neighbor_idx)
                        d_walk_results[idx] = neighbor_idx
                        found_visible_facet = True
                        if is_point_inside_tetrahedron(tetrahedra[idx, d_walk_results[idx], 0], vertices, point):
                            return d_walk_results[idx]
                        break  # Break to restart the check with the new current tetrahedron

        # If the point is not visible from any facet of the current tetrahedron, it means the point is inside it
        if not found_visible_facet:
            if is_point_inside_tetrahedron(tetrahedra[idx, d_walk_results[idx], 0], vertices, point):
                print("Point is inside the tetrahedron")
            else:
                print("Point is outside the tetrahedron")
            break
    return d_walk_results[idx]

@cuda.jit(device=True)
def remove_tetrahedron(tetrahedron_index, tetrahedra, free_indices, tetrahedron_count, free_indices_count, delauny_ball_num, d_create_delaunay_ball_results, vertices, vertex_count):
    idx = cuda.grid(1)
    shift = 0
    stop = 0
    # Ensure the thread ID is within bounds
    if idx >= tetrahedra.shape[0]:
        return
   
    # Extract neighbors for the tetrahedron
    neighbors = tetrahedra[idx, tetrahedron_index, 1] 
    # Iterate through neighbors to update their references
    for i in range(4):  # Assuming each tetrahedron has 4 neighbors
        neighbor_idx = neighbors[i]
        if neighbor_idx != -1:  # Check if the neighbor exists
            # Attempt to atomically update the neighbor's reference if it points back to the current tetrahedron
            for j in range(4):  # Check all neighbor references
                if tetrahedra[idx, tetrahedron_index, 1, j]  == tetrahedron_index:
                    tetrahedra[idx, tetrahedron_index, 1, j] = -1 
    
    insert_pos = cuda.atomic.add(free_indices_count, 0, 1)
    #print("Thread ID:", idx, "insert_pos", insert_pos)
    if insert_pos < MAX_TETRAHEDRA:  
        free_indices[insert_pos] = tetrahedron_index
        #print("Thread ID:", idx, "tetrahedron_index APO REMOVE", tetrahedron_index)

    #print("Thread ID:", idx, "delauny_ball_num APO REMOVE", delauny_ball_num[idx])
    for i in range(delauny_ball_num[idx]):
        if d_create_delaunay_ball_results[i] == tetrahedron_index:
            shift = 1
            stop = i

    if shift == 1:
        for i in range(stop, delauny_ball_num[idx]):
            d_create_delaunay_ball_results[i] = d_create_delaunay_ball_results[i + 1]
        cuda.atomic.add(delauny_ball_num, idx, -1)

    tetrahedra[idx, tetrahedron_index, 2, 0] = -1
    cuda.atomic.add(tetrahedron_count, idx, -1)

@cuda.jit(device=True)
def compute_circumsphere(A, B, C, D):
    # Manually perform vector subtraction to get coordinates relative to A
    ba = cuda.local.array(3, dtype=float64)
    ca = cuda.local.array(3, dtype=float64)
    da = cuda.local.array(3, dtype=float64)

    for i in range(3):
        ba[i] = B[i] - A[i]
        ca[i] = C[i] - A[i]
        da[i] = D[i] - A[i]

    # Manually compute squares of lengths of the edges incident to 'A'
    len_ba = ba[0]**2 + ba[1]**2 + ba[2]**2
    len_ca = ca[0]**2 + ca[1]**2 + ca[2]**2
    len_da = da[0]**2 + da[1]**2 + da[2]**2
    
    # Manually compute cross products of these edges
    cross_cd = cuda.local.array(3, dtype=float64)
    cross_db = cuda.local.array(3, dtype=float64)
    cross_bc = cuda.local.array(3, dtype=float64)
    
    # Cross product of CA and DA
    cross_cd[0] = ca[1]*da[2] - ca[2]*da[1]
    cross_cd[1] = ca[2]*da[0] - ca[0]*da[2]
    cross_cd[2] = ca[0]*da[1] - ca[1]*da[0]
    
    # Cross product of DA and BA
    cross_db[0] = da[1]*ba[2] - da[2]*ba[1]
    cross_db[1] = da[2]*ba[0] - da[0]*ba[2]
    cross_db[2] = da[0]*ba[1] - da[1]*ba[0]
    
    # Cross product of BA and CA
    cross_bc[0] = ba[1]*ca[2] - ba[2]*ca[1]
    cross_bc[1] = ba[2]*ca[0] - ba[0]*ca[2]
    cross_bc[2] = ba[0]*ca[1] - ba[1]*ca[0]
    
    # Calculate the denominator of the formula
    denominator = 0.5 / (ba[0]*cross_cd[0] + ba[1]*cross_cd[1] + ba[2]*cross_cd[2])
    
    # Calculate offset (from 'A') of circumcenter
    circ = cuda.local.array(3, dtype=float64)
    for i in range(3):
        circ[i] = (len_ba * cross_cd[i] + len_ca * cross_db[i] + len_da * cross_bc[i]) * denominator
    
    # Calculate the coordinates of the circumcenter
    circumcenter = cuda.local.array(3, dtype=float64)
    for i in range(3):
        circumcenter[i] = A[i] + circ[i]
    
    # Compute radius manually
    radius = math.sqrt((ba[0] + circ[0])**2 + (ba[1] + circ[1])**2 + (ba[2] + circ[2])**2)
    
    return circumcenter, radius

@cuda.jit(device=True)
def inSphere(A, B, C, D, E):
    # Compute the matrix components relative to point E
    def calc_components(p, E):
        px, py, pz = p[0] - E[0], p[1] - E[1], p[2] - E[2]
        p_squared = px**2 + py**2 + pz**2
        return px, py, pz, p_squared

    # Calculate matrix components for A, B, C, D relative to E
    ax, ay, az, a_squared = calc_components(A, E)
    bx, by, bz, b_squared = calc_components(B, E)
    cx, cy, cz, c_squared = calc_components(C, E)
    dx, dy, dz, d_squared = calc_components(D, E)

    # Compute the determinant of the matrix
    det = (
        ax * (by * (cz * d_squared - cz * dz) + bz * (cy * dz - cz * dy) + dy * (bz * cy - by * cz)) -
        ay * (bx * (cz * d_squared - cz * dz) + bz * (cx * dz - cz * dx) + dx * (bz * cx - bx * cz)) +
        az * (bx * (cy * d_squared - cy * dy) + by * (cx * dy - cy * dx) + dx * (by * cx - bx * cy)) -
        a_squared * (bx * (cy * dz - cz * dy) - by * (cx * dz - cz * dx) + bz * (cx * dy - cy * dx))
    )

    return det

@cuda.jit(device=True)
def is_in_circumsphere(idx, center, radius, point):
    """
    Check if 'point' is within the circumsphere defined by 'center' and 'radius'.
    This version prints debug information for the thread with index `idx`.
    """
    # Compute vector from center to point
    v = (center[0] - point[0], center[1] - point[1], center[2] - point[2])

    # Compute the norm of vector v
    dist = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    #Print debugging information including the thread index and scalar details only
    #print("Thread ID:", idx, "| Checking point x:", point[0], "y:", point[1], "z:", point[2])
    #print("Thread ID:", idx, "| Center x:", center[0], "y:", center[1], "z:", center[2])
    #print("Thread ID:", idx, "| Radius:", radius)
    #print("Thread ID:", idx, "| Distance from center to point:", dist)

    # Check if the point lies within the circumsphere
    return dist <= radius

@cuda.jit(device=True)
def find_cavity_kernel(point, tetrahedra, vertices, free_indices, free_indices_count, tetrahedron_count, cavity_count, start_tetrahedron_idx, find_cavity_results, delauny_ball_num, d_create_delaunay_ball_results, vertex_count, is_inside_tetrahedron):
    idx = cuda.grid(1)
    # Shared memory for queue and visited flags.
    visited = cuda.local.array(MAX_QUEUE_SIZE, dtype=int32)  # Each thread gets its own 'visited' array
    queue = cuda.local.array(MAX_QUEUE_SIZE, dtype=int32)  # Each thread gets its own 'queue'
    queue_size = cuda.local.array(1, dtype=int32)  # Each thread gets its own 'queue_size'
    actuall_index = cuda.local.array(4, dtype=int32) 
    local_cavity_count = cuda.local.array(1, dtype=int32)
    local_cavity_count = 0
    cavity_count[idx] = 0
    A = cuda.local.array(3, dtype=float64)
    B = cuda.local.array(3, dtype=float64)
    C = cuda.local.array(3, dtype=float64)
    D = cuda.local.array(3, dtype=float64)
    print("Thread ID:", idx, "start_tetrahedron_idx", start_tetrahedron_idx)
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 0, 1, 0], tetrahedra[idx, 0, 1, 1],tetrahedra[idx, 0, 1, 2], tetrahedra[idx, 0, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 1, 1, 0], tetrahedra[idx, 1, 1, 1],tetrahedra[idx, 1, 1, 2], tetrahedra[idx, 1, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 2, 1, 0], tetrahedra[idx, 2, 1, 1],tetrahedra[idx, 2, 1, 2], tetrahedra[idx, 2, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 3, 1, 0], tetrahedra[idx, 3, 1, 1],tetrahedra[idx, 3, 1, 2], tetrahedra[idx, 3, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 4, 1, 0], tetrahedra[idx, 4, 1, 1],tetrahedra[idx, 4, 1, 2], tetrahedra[idx, 4, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 5, 1, 0], tetrahedra[idx, 5, 1, 1],tetrahedra[idx, 5, 1, 2], tetrahedra[idx, 5, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 6, 1, 0], tetrahedra[idx, 6, 1, 1],tetrahedra[idx, 6, 1, 2], tetrahedra[idx, 6, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 7, 1, 0], tetrahedra[idx, 7, 1, 1],tetrahedra[idx, 7, 1, 2], tetrahedra[idx, 7, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 8, 1, 0], tetrahedra[idx, 8, 1, 1],tetrahedra[idx, 8, 1, 2], tetrahedra[idx, 8, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 9, 1, 0], tetrahedra[idx, 9, 1, 1],tetrahedra[idx, 9, 1, 2], tetrahedra[idx, 9, 1, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 10, 1, 0], tetrahedra[idx, 10, 1, 1],tetrahedra[idx, 10, 1, 2], tetrahedra[idx, 10, 1, 3])
    queue[0] = start_tetrahedron_idx
    queue_size[0] = 1
    for i in range(MAX_QUEUE_SIZE):
        visited[i] = 0
    #print("Thread ID:", idx, "queue[0]", queue[0])
    while queue_size[0] > 0:
    #    print("Thread ID:", idx, "local_idx", queue[0])
    #    print("Thread ID:", idx, "visited[local_idx]", visited[queue[0]])
        if visited[queue[0]] == 0:
            visited[queue[0]] = 1  # Mark as visited
            
            for i in range(4):
                actuall_index[i] = tetrahedra[idx, queue[0] , 0 , i]
                #print("Thread ID:", idx, "Read vertex index at local_idx", local_idx, "pos", i, ":", actuall_index[i])
                
            A = vertices[actuall_index[0], :]
            B = vertices[actuall_index[1], :]
            C = vertices[actuall_index[2], :]
            D = vertices[actuall_index[3], :]
            #print("Thread ID:", idx, "Vertex A x,y,z:", A[0], A[1], A[2], "Vertex B x,y,z:", B[0], B[1], B[2],  "Vertex C x,y,z:", C[0], C[1], C[2],  "Vertex D x,y,z:", D[0], D[1], D[2])
            center, radius = compute_circumsphere(A, B, C, D)
            #print("Thread ID:", idx, "Center x:", center[0], "y:", center[1], "z:", center[2], "Radius:", radius)
            #print("Thread ID:", idx, "point", point[0])
            if inSphere(A, B, C, D, point) or is_inside_tetrahedron[idx] == True:
                find_cavity_results[idx, local_cavity_count] = queue[0]
                local_cavity_count += 1
                #print("Thread ID:", idx, "Cavity Num:", local_cavity_count)
                for n in range(4):  
                    neighbor_idx = int(tetrahedra[idx, queue[0], 1, n])
                #    print("Thread ID:", idx, "neighbor_idx:", neighbor_idx)
                    if neighbor_idx != -1 and visited[neighbor_idx] == 0:
                            queue[queue_size[0]] = neighbor_idx
                            queue_size[0] += 1
                            #print("Thread ID:", idx, "Enqueuing neighbor idx:", neighbor_idx, "Queue Size:", queue_size[0])

        # Shift queue to the left
        for j in range(1, queue_size[0]):
            queue[j-1] = queue[j]
        queue_size[0] -= 1
        #print("queue_size", queue_size[0])
    # Clear any potential residue in the find_cavity_results beyond cavity_num
    for i in range(local_cavity_count):  # Clear out results beyond cavity_num
        if i >= local_cavity_count:
            find_cavity_results[idx, i] = -1

    for i in range(local_cavity_count):  # num_tetrahedra is the total number of tetrahedra
        remove_tetrahedron(find_cavity_results[idx, i], tetrahedra, free_indices, tetrahedron_count, free_indices_count, delauny_ball_num, d_create_delaunay_ball_results[idx], vertices, vertex_count)

    #print("vertex_count", vertex_count[0])
    cuda.atomic.add(cavity_count, idx, local_cavity_count)
    return find_cavity_results

@cuda.jit(device=True)
def get_vertices_of_face(tetrahedron_index, face_index, tetrahedra, vertices):
    idx = cuda.grid(1)
    """
    For a given tetrahedron and face index, retrieves the coordinates of the vertices forming that face.
    """
    # Mapping from face index to vertex indices within the tetrahedron
    # Adjust the mapping based on your tetrahedra structure
    face_vertices = cuda.local.array((3,), dtype=int32)
    if face_index == 0:
        face_vertices[0] = tetrahedra[idx, tetrahedron_index, 0, 1]
        face_vertices[1] = tetrahedra[idx, tetrahedron_index, 0, 2]
        face_vertices[2] = tetrahedra[idx, tetrahedron_index, 0, 3]
    elif face_index == 1:
        face_vertices[0] = tetrahedra[idx, tetrahedron_index, 0, 0]
        face_vertices[1] = tetrahedra[idx, tetrahedron_index, 0, 2]
        face_vertices[2] = tetrahedra[idx, tetrahedron_index, 0, 3]
    elif face_index == 2:
        face_vertices[0] = tetrahedra[idx, tetrahedron_index, 0, 0]
        face_vertices[1] = tetrahedra[idx, tetrahedron_index, 0, 1]
        face_vertices[2] = tetrahedra[idx, tetrahedron_index, 0, 3]
    elif face_index == 3:
        face_vertices[0] = tetrahedra[idx, tetrahedron_index, 0, 0]
        face_vertices[1] = tetrahedra[idx, tetrahedron_index, 0, 1]
        face_vertices[2] = tetrahedra[idx, tetrahedron_index, 0, 2]

    # Retrieve vertex indices for the face from the tetrahedron
    v1 = vertices[face_vertices[0], :]
    v2 = vertices[face_vertices[1], :]
    v3 = vertices[face_vertices[2], :]
    #print("Tetrahedron index:", tetrahedron_index, "Face index:", face_index)
    #print("Vertex 1 index:", face_vertices[0], "coordinates:", v1[0], v1[1], v1[2])
    #print("Vertex 2 index:", face_vertices[1], "coordinates:", v2[0], v2[1], v2[2])
    #print("Vertex 3 index:", face_vertices[2], "coordinates:", v3[0], v3[1], v3[2])
    return v1, v2, v3

@cuda.jit(device=True)
def add_boundary_facet(boundary_facets, boundary_facets_count, v1, v2, v3, idx):
    """
    Adds a new boundary facet if there's space, represented by the vertices v1, v2, and v3.
    """
    pos = cuda.atomic.add(boundary_facets_count, idx, 1)
    #print("Thread ID:", idx, "| pos:", pos)
    if pos < MAX_TETRAHEDRA:
        for j in range(3):
            if j == 0:
                boundary_facets[pos, j, 0] = v1[j]
                boundary_facets[pos, j, 1] = v1[j+1]
                boundary_facets[pos, j, 2] = v1[j+2]
            if j == 1:
                boundary_facets[pos, j, 0] = v2[j-1]
                boundary_facets[pos, j, 1] = v2[j]
                boundary_facets[pos, j, 2] = v2[j+1]
            if j == 2:
                boundary_facets[pos, j, 0] = v3[j-2]
                boundary_facets[pos, j, 1] = v3[j-1]
                boundary_facets[pos, j, 2] = v3[j]

@cuda.jit(device=True)
def is_neighbor_in_cavity(neighbor_idx, cavity_num, cavity_tetrahedra_indices):
    """
    Checks if the neighbor tetrahedron is part of the cavity.
    """
    idx = cuda.grid(1)  # Obtain the thread ID for printing relevant debugging information.
    #print("Thread ID:", idx, "| Checking neighbor_idx:", neighbor_idx, "against cavity list.")

    for i in range(cavity_num):
    #    print("Thread ID:", idx, "| Comparing with cavity tetrahedron index:", cavity_tetrahedra_indices[i])
        if cavity_tetrahedra_indices[i] == neighbor_idx:
    #        print("Thread ID:", idx, "| Neighbor found in cavity at index:", i)
            return True

    #print("Thread ID:", idx, "| Neighbor not found in cavity.")
    return False

@cuda.jit(device=True)
def is_visible_from_facet_delauny(v1, v2, v3, point):
    # Compute normal of the facet
    vec1 = cuda.local.array(3, dtype=float64)  # v1 to v2
    vec2 = cuda.local.array(3, dtype=float64)  # v1 to v3
    normal = cuda.local.array(3, dtype=float64)
    # Vector components
    for i in range(3):
        vec1[i] = v2[i] - v1[i]
        vec2[i] = v3[i] - v1[i]
    # Cross product to get normal
    normal[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    normal[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    normal[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    # Point to vertex vector
    point_to_vertex = cuda.local.array(3, dtype=float64)
    for i in range(3):
        point_to_vertex[i] = point[i] - v1[i]
    # Dot product to determine visibility
    dot_product = 0.0
    for i in range(3):
        dot_product += point_to_vertex[i] * normal[i]
    # Facet is visible if dot product is positive
    return dot_product > 0

@cuda.jit(device=True)
def create_delaunay_ball_kernel(new_point, tetrahedra, vertices, cavity_tetrahedra, cavity_num, vertex_count, free_indices, free_indices_count, tetrahedron_count, d_last_tetra_index, create_delaunay_ball_results, delauny_ball_num, boundary_facets, boundary_facets_count, num_partitions, d_walk_results):
    idx = cuda.grid(1)
    local_delauny_ball_num = cuda.local.array(1, dtype=int32)
    local_delauny_ball_num = 0
    A = cuda.local.array(3, dtype=float64)
    B = cuda.local.array(3, dtype=float64)
    C = cuda.local.array(3, dtype=float64)
    D = cuda.local.array(3, dtype=float64)
    actuall_index = cuda.local.array(4, dtype=int32)

    #print("Thread ID:", idx, "| Cavity Tetrahedra Count:", cavity_num[idx])

    cavity_tetrahedra_indices = cavity_tetrahedra[idx]
    #for i in range(cavity_num[0]):
        #print("Thread ID:", idx, "cavity_tetrahedra_indices:", cavity_tetrahedra_indices[i])

    boundary_facets_count[idx] = 0
    new_tetrahedron_index = d_walk_results[idx]
    # Iterate through each tetrahedron in the cavity to find boundary facets
    for i in range(cavity_num[idx]):
        tetrahedron_index = cavity_tetrahedra[idx, i]
        if tetrahedron_index == -1:  # Skip if tetrahedron index is not valid
            continue

        for face_index in range(4):
            v1, v2, v3 = get_vertices_of_face(tetrahedron_index, face_index, tetrahedra, vertices)
            neighbor_idx = tetrahedra[idx, tetrahedron_index, 1, face_index]
            if neighbor_idx == -1 or not is_neighbor_in_cavity(neighbor_idx, cavity_num[idx], cavity_tetrahedra_indices):
                add_boundary_facet(boundary_facets[idx], boundary_facets_count, v1, v2, v3, idx)

    #print("Thread",idx, "Number of boundary facets found",boundary_facets_count[0])
    for i in range(num_partitions) :
        if i == idx:
            continue
        for j in range(cavity_num[i]):
            tetrahedron_index = cavity_tetrahedra[i, j]
            for k in range(4):
                actuall_index[k] = tetrahedra[i, tetrahedron_index , 0 , k]
                #print("Thread ID:", idx, "Read vertex index at local_idx", local_idx, "pos", i, ":", actuall_index[i])
                
            A = vertices[actuall_index[0], :]
            B = vertices[actuall_index[1], :]
            C = vertices[actuall_index[2], :]
            D = vertices[actuall_index[3], :]

            if inSphere(A, B, C, D, new_point[i]):
                for face_index in range(4):
                    v1, v2, v3 = get_vertices_of_face(tetrahedron_index, face_index, tetrahedra, vertices)
                    neighbor_idx = tetrahedra[i, tetrahedron_index, 1, face_index]
                    if neighbor_idx == -1 or not is_neighbor_in_cavity(neighbor_idx, cavity_num[i], cavity_tetrahedra_indices):
                        add_boundary_facet(boundary_facets[i], boundary_facets_count, v1, v2, v3, i)

    start_index = delauny_ball_num[idx] 
    #print("Thread ID:", idx, "start_index", start_index)
    for i in range(boundary_facets_count[idx]):
        vertex_coords = cuda.local.array((4, 3), dtype=np.float64)  
        #print("Thread",idx, "i",i)
        for j in range(3):  # Three vertices in the facet
            vertex_coords[j, 0] = boundary_facets[idx, i, j, 0]
            #print("Thread",idx, "vertex_coords[j, 0]",vertex_coords[j, 0])
            vertex_coords[j, 1] = boundary_facets[idx, i, j, 1]
            #print("Thread",idx, "vertex_coords[j, 0]",vertex_coords[j, 1])
            vertex_coords[j, 2] = boundary_facets[idx, i, j, 2]
            #print("Thread",idx, "vertex_coords[j, 0]",vertex_coords[j, 2])

        # Add the new point as the fourth vertex
        vertex_coords[3, 0] = new_point[idx, 0]
        vertex_coords[3, 1] = new_point[idx, 1]
        vertex_coords[3, 2] = new_point[idx, 2]
        #print("Thread",idx, "vertex_coords[j, 0]",vertex_coords[3, 0])    
        # Use the 'add_tetrahedron' device function to add the new tetrahedron
        new_tetrahedron_index = add_tetrahedron(vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, vertex_coords)
        #print("Thread ID:", idx, "d_add_result_delauny_ball", new_tetrahedron_index)
        local_delauny_ball_num += 1

    delauny_ball_num[idx] = local_delauny_ball_num + start_index
    d_last_tetra_index[idx] = new_tetrahedron_index

    #print("Thread ID:", idx, "d_last_tetra_index[idx]", d_last_tetra_index[idx])
    return create_delaunay_ball_results

def compute_hilbert_indices(points, dimensions, bits):
    # Normalize points
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    points_normalized = (points - points_min) / (points_max - points_min)
    
    # Scale points to the appropriate integer range
    scale = (1 << bits) - 1
    points_scaled = np.floor(points_normalized * scale).astype(np.int32)

    # Initialize Hilbert curve calculator
    hilbert_curve = HilbertCurve(p=bits, n=dimensions)
    
    # Calculate Hilbert indices
    hilbert_indices = np.array([hilbert_curve.distance_from_point(pt) for pt in points_scaled])
    
    return hilbert_indices


def partition_points(points, desired_partitions, bits=10):
    if points.size == 0:
        return [] 
    dimensions = 3  # 3D points
    bits = 10  # Number of iterations in the Hilbert curve (depth)
    hilbert_indices = compute_hilbert_indices(points, dimensions, bits)
    
    sorted_indices = np.argsort(hilbert_indices)
    sorted_points = points[sorted_indices]
    partitions = []

    # Determine the number of points per partition
    # Assuming all partitions are of equal size for simplicity
    partition_size = len(points) // desired_partitions
    
    for _ in range(desired_partitions - 1):
        # Select the first partition_size elements for the current partition
        partitions.append(sorted_points[:partition_size])
        # Remove these points from sorted_points
        sorted_points = sorted_points[partition_size:]
    
    # Add the remaining points to the last partition
    partitions.append(sorted_points)

    return partitions


def remove_processed_points(all_points, triangulated_points):
    # Ensure triangulated_points can be broadcasted correctly
    # If triangulated_points is just a single point (1, 3), it is already ready for broadcasting
    # If it's multiple points, no need to change the dimensionality (num_tri_points, 3)

    # Calculate the mask by checking where all points do not equal any of the triangulated points
    # This will broadcast triangulated_points across the number of points in all_points
    mask = ~np.any(np.all(all_points[:, np.newaxis] == triangulated_points, axis=-1), axis=-1)

    # Apply the mask to filter out the triangulated points from all points
    filtered_points = all_points[mask]

    return filtered_points

@cuda.jit
def delaunay_kernel(vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, d_add_result, d_walk_results, d_find_cavity_results, cavity_num, d_create_delaunay_ball_results, delauny_ball_num, d_last_tetra_index, boundary_facets, boundary_facets_count, d_passed_points, d_triangulated_point, num_partitions, is_inside_tetrahedron):
    idx = cuda.grid(1)
    for j in range(d_triangulated_point.shape[1]):  # Assuming 2D arrays
        d_triangulated_point[idx, j] = d_passed_points[idx, 0, j]
    #print("vertex_count",vertex_count[0])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 0, 0, 0], tetrahedra[idx, 0, 0, 1],tetrahedra[idx, 0, 0, 2], tetrahedra[idx, 0, 0, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 1, 0, 0], tetrahedra[idx, 1, 0, 1],tetrahedra[idx, 1, 0, 2], tetrahedra[idx, 1, 0, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 2, 0, 0], tetrahedra[idx, 2, 0, 1],tetrahedra[idx, 2, 0, 2], tetrahedra[idx, 2, 0, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 3, 0, 0], tetrahedra[idx, 3, 0, 1],tetrahedra[idx, 3, 0, 2], tetrahedra[idx, 3, 0, 3])
    #print("Thread ID:", idx, "tetrahedron", tetrahedra[idx, 4, 0, 0], tetrahedra[idx, 4, 0, 1],tetrahedra[idx, 4, 0, 2], tetrahedra[idx, 4, 0, 3])
    d_walk_results[idx] = walk_kernel(d_add_result[idx], d_triangulated_point[idx], vertices, tetrahedra, d_walk_results, is_inside_tetrahedron) 
    cuda.syncthreads()
    print("thread_id", idx, "d_walk_results", d_walk_results[idx])
    d_find_cavity_results = find_cavity_kernel(d_triangulated_point[idx], tetrahedra, vertices, free_indices, free_indices_count, tetrahedron_count, cavity_num, d_walk_results[idx], d_find_cavity_results, delauny_ball_num, d_create_delaunay_ball_results, vertex_count, is_inside_tetrahedron) 
    cuda.syncthreads()
    print("thread_id", idx, "cavity_num", cavity_num[idx])
    create_delaunay_ball_kernel(d_triangulated_point, tetrahedra, vertices, d_find_cavity_results, cavity_num, vertex_count, free_indices, free_indices_count, tetrahedron_count, d_last_tetra_index, d_create_delaunay_ball_results[idx], delauny_ball_num, boundary_facets, boundary_facets_count, num_partitions, d_walk_results)  
    cuda.syncthreads()
    print("Thread ID:", idx, "d_last_tetra_index[idx]", d_last_tetra_index[idx])
    print("Thread ID:", idx, "delauny_ball_num[idx]", delauny_ball_num[idx])
    if delauny_ball_num[idx] > 0:
        d_add_result[idx] = d_last_tetra_index[idx]
    print("vertex_count",vertex_count[0])

def parallel_delaunay(partitions):
    max_size = max(part.shape[0] for part in partitions)
    d_partitions = cuda.device_array((len(partitions), max_size, 3), dtype=np.float64)
    num_points_per_partition = np.array([part.shape[0] for part in partitions])
    d_passed_points = cuda.device_array((len(partitions), max_size - 4 , 3), dtype=np.float64)
    d_triangulated_point = cuda.device_array((len(partitions), 3), dtype=np.float64)

    for idx, part in enumerate(partitions):
        num_rows = part.shape[0]
        d_partitions[idx, :num_rows, :] = cuda.to_device(part)
        
    d_tau = cuda.device_array((len(partitions), 4, 3), dtype=np.float64)

    d_add_result = cuda.device_array(len(partitions), dtype=np.int32)

    d_walk_results = cuda.device_array(len(partitions), dtype=np.int32)
    is_inside_tetrahedron = cuda.device_array(len(partitions), dtype=np.bool_)

    d_find_cavity_results = cuda.device_array((len(partitions), MAX_TETRAHEDRA), dtype=np.int32)

    cavity_num = cuda.device_array(len(partitions), dtype=np.int32)

    boundary_facets = cuda.device_array((len(partitions), MAX_TETRAHEDRA, 3, 3), dtype=np.float64)
    boundary_facets_count = cuda.device_array(len(partitions), dtype=np.int32)

    d_create_delaunay_ball_results = cuda.device_array((len(partitions), MAX_TETRAHEDRA), dtype=np.int32)

    delauny_ball_num = cuda.device_array(len(partitions), dtype=np.int32)

    d_last_tetra_index = cuda.device_array(len(partitions), dtype=np.int32)

    tetrahedra = cuda.device_array((len(partitions), MAX_TETRAHEDRA, 5, 6), dtype=np.float64)
    tetrahedra[:, 2, 0] = -1
    vertices = cuda.device_array((MAX_VERTICES, 3), dtype=np.float64)
    tetrahedron_count = cuda.device_array(len(partitions), dtype=np.int32) 
    num_partitions = cuda.device_array(1, dtype=np.int32) 
    num_partitions = len(partitions)
    free_indices = cuda.device_array(MAX_TETRAHEDRA, dtype=np.int32)
    free_indices_count = cuda.device_array(1, dtype=np.int32)

    vertex_count = cuda.device_array(1, dtype=np.int32)
    threads_per_block = 4
    blocks_per_grid = 1
    counter = 0
    init_kernel[blocks_per_grid, threads_per_block](d_partitions, num_points_per_partition, vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, d_add_result, d_tau)
    mark_filtered_points_kernel[blocks_per_grid, threads_per_block](d_partitions, d_tau, d_passed_points)
    for i in range(max_size-3):
        delaunay_kernel[blocks_per_grid, threads_per_block](vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, d_add_result, d_walk_results, d_find_cavity_results, cavity_num, d_create_delaunay_ball_results, delauny_ball_num, d_last_tetra_index, boundary_facets, boundary_facets_count, d_passed_points, d_triangulated_point, num_partitions, is_inside_tetrahedron)  
        remaining_points = d_passed_points.copy_to_host()
        triangulated_points = d_triangulated_point.copy_to_host()
        print("remaining_points",remaining_points)
        print("triangulated_points",triangulated_points)
        remaining_points = np.concatenate(remaining_points, axis=0)
        remaining_points = remove_processed_points(remaining_points, triangulated_points)
        d_passed_points = cuda.device_array((num_partitions, max_size - 5 - i, 3), dtype=np.float64)
        partitions = partition_points(remaining_points, num_partitions)
        for idx, part in enumerate(partitions):
            num_rows = part.shape[0]
            d_passed_points[idx, :num_rows, :] = cuda.to_device(part)
        print("counter", counter)
        counter += 1
        if counter == max_size - 4:
            break
    host_vertices = vertices.copy_to_host()
    host_tetrahedra = tetrahedra.copy_to_host()
    host_tetrahedron_count = tetrahedron_count.copy_to_host()[0]
    
    print(f"tetrahedron_count_main {host_tetrahedron_count}")
    host_vertex_count = vertex_count.copy_to_host()[0]
    
    print(f"host_vertex_count_main {host_vertex_count}")

    #for i in range(4):  # Assuming you are checking four vertices
    #   print("Vertex", i, "coordinates:", host_vertices[i, :])
    #    print("tetra", 0, "vertex indexes:", host_tetrahedra[0, 0, i])
        
    # Print the contents of d_tau
    #print("d_tau contents:")
    # Assuming d_tau is a single device array with shape (num_partitions, 4, 3)
    #d_tau_host = d_tau.copy_to_host()  # Copy the entire array at once
    #for i, partition_tau in enumerate(d_tau_host):
    #    print(f"Partition {i} d_tau:")
    #    for row in partition_tau:
    #        print(f"  {row}")

    # Print the contents of d_add_result
    #print("\nd_add_result contents:")
    #d_add_result_host = d_add_result.copy_to_host()
    #for i in range(len(d_add_result_host)):
    #    print(f"Partition {i} d_add_result:", d_add_result_host[i])

    # Print the contents of d_passed_points
    #print("\nd_passed_points contents:")
    #for i, d_passed_points_part in enumerate(d_passed_points):
    #    d_passed_points_host = d_passed_points_part.copy_to_host()
    #    print(f"Partition {i} d_passed_points:", d_passed_points_host)

    #d_walk_results_host = d_walk_results.copy_to_host()
    #print("\nd_walk_results contents:")
    #for i, d_walk_results_part in enumerate(d_walk_results_host):
    #    print(f"Partition {i} d_walk_results:", d_walk_results_part)

    #print("\nd_find_cavity_results contents:")
    #d_find_cavity_results_host = d_find_cavity_results.copy_to_host()
    #cavity_num_host = cavity_num.copy_to_host()

    #for i in range(len(d_find_cavity_results_host)):
    #    cavity_count_host = cavity_num_host[i]
    #    print(f"Partition {i} Number of Cavities Found: {cavity_count_host}")
    #    if cavity_count_host > 0:
    #        print(f"Partition {i} d_find_cavity_results for detected cavities:")
    #        for j in range(cavity_count_host):
    #            print(f"  Cavity {j}: {d_find_cavity_results_host[i][j]}")
    #    else:
    #        print("  No cavities found in this partition.")

    #print("\nd_last_tetra_indices contents:")
    #d_last_tetra_indices_host = d_last_tetra_index.copy_to_host()  # Copy the whole array to host
    #for i, tetra_index in enumerate(d_last_tetra_indices_host):
    #    print(f"Partition {i} d_last_tetra_indices:", tetra_index)

    #print("\ndelauny_ball_num contents:")
    #delauny_ball_num_host = delauny_ball_num.copy_to_host()  # Copy the whole array to host
    #for i, ball_num in enumerate(delauny_ball_num_host):
    #    print(f"Partition {i} delauny_ball_num:", ball_num)
    
    # Print details of d_create_delaunay_ball_results
    #print("\nd_create_delaunay_ball_results contents:")
    #for i, d_create_delaunay_ball_results_part in enumerate(d_create_delaunay_ball_results):
    #    d_create_delaunay_ball_results_host = d_create_delaunay_ball_results_part.copy_to_host()
    #    print(f"Partition {i} d_create_delaunay_ball_results:", d_create_delaunay_ball_results_host)
    
    return host_tetrahedra, host_vertices, vertex_count

def visualize_tetrahedra_and_all_points(tetrahedra, vertices, vertex_count, all_points):
    # Create a LineSet object for tetrahedra
    lines = o3d.geometry.LineSet()
    if isinstance(vertex_count, cuda.cudadrv.devicearray.DeviceNDArray):
        vertex_count = int(vertex_count.copy_to_host()[0])
    # Define a list of distinct colors for visualization
    distinct_colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1]   # Cyan
    ]

    # Placeholder for points and lines of tetrahedra
    tetra_points = []
    edges = []

    # Visualize each tetrahedron
    num_partitions = tetrahedra.shape[0]
    for i in range(num_partitions):
        for j in range(tetrahedra.shape[1]):
            if tetrahedra[i, j, 2, 0] != -1:  # Check if the tetrahedron is active
                for k in range(6):  # There are 6 edges in a tetrahedron
                    start_vertex = int(tetrahedra[i, j, 3, k])
                    end_vertex = int(tetrahedra[i, j, 4, k])

                    if start_vertex not in tetra_points:
                        tetra_points.append(start_vertex)
                    if end_vertex not in tetra_points:
                        tetra_points.append(end_vertex)

                    start_index = tetra_points.index(start_vertex)
                    end_index = tetra_points.index(end_vertex)
                    edges.append([start_index, end_index])

    # Set points and lines to the LineSet
    tetra_point_coords = [vertices[p] for p in tetra_points if p < vertex_count]  # Fetch the vertex coordinates
    lines.points = o3d.utility.Vector3dVector(tetra_point_coords)
    lines.lines = o3d.utility.Vector2iVector(edges)
    lines.paint_uniform_color([1, 0, 0])  # Red for tetrahedron edges

    # Create point cloud for all points, ensuring each has a distinct color
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)  # Use all vertices, not just those in tetrahedra
    pcd.colors = o3d.utility.Vector3dVector([distinct_colors[i % len(distinct_colors)] for i in range(len(vertices))])  # Distinct colors for each vertex

    # Print vertex information with colors
    for i, point in enumerate(vertices[:vertex_count]):
        color_name = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan'][i % len(distinct_colors)]
        print(f"Vertex {i}: {point} - Color: {color_name}")

    # Visualization
    print("Starting visualization...")
    o3d.visualization.draw_geometries([lines, pcd], window_name='Tetrahedra and All Points Visualization')

mesh = o3d.io.read_triangle_mesh("stanford-bunny.obj")
mesh.compute_vertex_normals()
point_cloud = mesh.sample_points_poisson_disk(number_of_points=40)
# point_cloud = mesh.sample_points_uniformly(number_of_points=1000)
o3d.visualization.draw_geometries([point_cloud])
points = np.asarray(point_cloud.points)
desired_partitions = 4
partitions = partition_points(points, desired_partitions)
for i, partition in enumerate(partitions):
    print(f"Partition {i} with {len(partition)} points:")
    print(partition)
    print("\n---")

colored_point_clouds = []

# Define a color for each partition, if you want the same color for all use one line
colors = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [0, 1, 1],  # Cyan
    [1, 0, 1],  # Magenta
    [0.5, 0.5, 0],  # Olive
    [0, 0.5, 0.5],  # Teal
    [0.5, 0, 0.5],  # Purple
    [0.3, 0.3, 0.3]  # Dark Gray
]

for i, partition in enumerate(partitions):
    # Create a point cloud for each partition
    part_pc = o3d.geometry.PointCloud()
    part_pc.points = o3d.utility.Vector3dVector(partition)  # Convert array to suitable format
    part_pc.colors = o3d.utility.Vector3dVector(np.tile(colors[i % len(colors)], (len(partition), 1)))  # Assign color
    
    colored_point_clouds.append(part_pc)

#o3d.visualization.draw_geometries(colored_point_clouds)
start_time = time.time()
tetrahedra, vertices, vertex_count = parallel_delaunay(partitions)
end_time = time.time()
elapsed_time = end_time - start_time 
print(f"Elapsed time for parallel Delaunay triangulation: {elapsed_time} seconds")
print("Length of the first dimension of tetrahedra:", tetrahedra.shape[1])
#print(f"Number of tetrahedra: {len(DT.tetrahedra)}")
visualize_tetrahedra_and_all_points(tetrahedra, vertices, vertex_count, points)

