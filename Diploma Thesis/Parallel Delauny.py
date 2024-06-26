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
    for i in range(tetrahedra.shape[0]):
        for j in range(3): 
            for k in range(3 - j):
                if tetrahedra[i, 1, k] < 0 and tetrahedra[i, 1, k + 1] >= 0:
                    tetrahedra[i, 1, k], tetrahedra[i, 1, k + 1] = tetrahedra[i, 1, k + 1], tetrahedra[i, 1, k]

@cuda.jit(device=True)
def faces_are_equal(face_a, face_b):
    used = cuda.local.array(3, dtype=boolean)
    for i in range(3):
        used[i] = False

    for i in range(3):
        matched = False
        for j in range(3):
            if not used[j]:
                vertices_match = True
                for k in range(3):
                    if face_a[i, k] != face_b[j, k]:
                        vertices_match = False
                        break
                if vertices_match:
                    used[j] = True 
                    matched = True
                    break
        if not matched:
            return False 

    return True

@cuda.jit(device=True)
def update_neighbors_with_shared_faces(tetrahedra, idx_self, new_vertex_coords, vertices, idx_other):
    combination_self = cuda.local.array((4, 3), dtype=int32)
    idx = cuda.grid(1)
    combination_self[0, :] = (0, 1, 2)
    combination_self[1, :] = (0, 1, 3)
    combination_self[2, :] = (0, 2, 3)
    combination_self[3, :] = (1, 2, 3)
    shared_count = cuda.local.array((1,), dtype=int32)
    shared_count[0] = 0
    for face_idx_self in range(4):
        current_combination = combination_self[face_idx_self]
        vertices_self = cuda.local.array(shape=(3, 3), dtype=np.float64)
        
        for i, vertex_idx in enumerate(current_combination):
            global_idx = int(tetrahedra[idx_self, 0, vertex_idx])
            for coord_idx in range(3):
                vertices_self[i, coord_idx] = vertices[global_idx, coord_idx]
        
        for face_idx_other  in range(4):
            new_face_combination = combination_self[face_idx_other ]
            new_face_vertices = cuda.local.array(shape=(3, 3), dtype=np.float64)
            
            for i, vertex_idx in enumerate(new_face_combination):
                for coord_idx in range(3):
                    new_face_vertices[i, coord_idx] = new_vertex_coords[vertex_idx, coord_idx]
            
            if faces_are_equal(vertices_self, new_face_vertices):
                tetrahedra[idx_self, 1, face_idx_self] = idx_other
                tetrahedra[idx_other, 1, face_idx_other] = idx_self

@cuda.jit(device=True)
def is_allclose(a, b, atol):
    idx = cuda.grid(1)
    close = True
    for i in range(3):
        if abs(a[i] - b[i]) > atol:
            close = False
            break
    return close

@cuda.jit(device=True)
def add_vertex(vertices, vertex_count, coord):
    idx = cuda.grid(1)
    atol = 1e-10
    for i in range(vertex_count[0]):
        if is_allclose(vertices[i], coord, atol):
            return i

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
    vertex_indices = cuda.local.array(4, dtype=int32)
    index = cuda.local.array(1, dtype=int32)
    for i in range(4):
        vertex_indices[i] = add_vertex(vertices, vertex_count, vertex_coords[i])

    index = -1 
    if free_indices_count[0] > 0:
        index = cuda.atomic.sub(free_indices_count, 0, 1)
        if index > 0:
            index = free_indices[index - 1]
            cuda.atomic.add(tetrahedron_count, idx, 1)
        else:
            index = cuda.atomic.add(tetrahedron_count, idx, 1)
    else:
        index = cuda.atomic.add(tetrahedron_count, idx, 1)

    for i in range(4):
        tetrahedra[idx, index, 1, i] = -1
        tetrahedra[idx, index, 0, i] = vertex_indices[i]

    tetrahedra[idx, index, 2, 0] = index

    edge_combinations = cuda.local.array((6, 2), dtype=int32)
    edge_combinations[0, :] = (0, 1)
    edge_combinations[1, :] = (0, 2)
    edge_combinations[2, :] = (0, 3)
    edge_combinations[3, :] = (1, 2)
    edge_combinations[4, :] = (1, 3)
    edge_combinations[5, :] = (2, 3)
    for i in range(6):
        start_vertex = vertex_indices[edge_combinations[i, 0]]
        end_vertex = vertex_indices[edge_combinations[i, 1]]
        tetrahedra[idx, index, 3, i] = start_vertex
        tetrahedra[idx, index, 4, i] = end_vertex
    
    for idx_existing in range(tetrahedron_count[idx]):
        if idx_existing != index and tetrahedra[idx, idx_existing, 2, 0] != -1:
            update_neighbors_with_shared_faces(tetrahedra[idx], idx_existing, vertex_coords, vertices, index)
    
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
                    cross_result[0] = (p1[1] - p3[1]) * (p2[2] - p3[2]) - (p1[2] - p3[2]) * (p2[1] - p3[1])
                    cross_result[1] = (p1[2] - p3[2]) * (p2[0] - p3[0]) - (p1[0] - p3[0]) * (p2[2] - p3[2])
                    cross_result[2] = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
                    dot = (p0[0] - p3[0]) * cross_result[0] + (p0[1] - p3[1]) * cross_result[1] + (p0[2] - p3[2]) * cross_result[2]
                    volume = abs(dot) / 6.0
                    if volume > tolerance:
                        for index, point_idx in enumerate((i, j, k, l)):
                            d_tau[idx, index, 0] = points[point_idx, 0]
                            d_tau[idx, index, 1] = points[point_idx, 1]
                            d_tau[idx, index, 2] = points[point_idx, 2]
                        found = True
                        break

    d_add_result[idx] = add_tetrahedron(vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, d_tau[idx])

@cuda.jit
def filter_points_kernel(points, initial_vertices, passed_points):
    insert_idx = 0
    idx = cuda.grid(1)

    for point_idx in range(points[idx].shape[0]):
        point_passes = True
        current_point = points[idx, point_idx]
        for vert_idx in range(initial_vertices[idx].shape[0]):
            vert = initial_vertices[idx, vert_idx]
            all_equal = True
            for dim in range(current_point.shape[0]):
                if current_point[dim] != vert[dim]:
                    all_equal = False
                    break
            if all_equal:
                point_passes = False
                break
        if point_passes:
            for dim in range(3):
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
    
    v1 = signed_volume_of_tetrahedron(a, b, c, point)
    v2 = signed_volume_of_tetrahedron(a, b, d, point)
    v3 = signed_volume_of_tetrahedron(a, c, d, point)
    v4 = signed_volume_of_tetrahedron(b, c, d, point)
    total_volume = signed_volume_of_tetrahedron(a, b, c, d)

    sum_volumes = abs(v1) + abs(v2) + abs(v3) + abs(v4)
    abs_total_volume = abs(total_volume)

    is_inside = abs(sum_volumes - abs_total_volume) < 1e-7

    return is_inside

@cuda.jit(device=True)
def find_neighbor_by_facet(vertex_indices, tetrahedra, idx_current):
    idx = cuda.grid(1)
    
    for neighbor_idx in range(4):
        neighbor_id = int(tetrahedra[idx, idx_current, 1, neighbor_idx])
        if neighbor_id == -1:
            continue 

        shared_vertices = 0
        for j in range(4):
            neighbor_vertex = tetrahedra[idx, neighbor_id, 0, j]
            for k in range(3):
                if neighbor_vertex == vertex_indices[k]:
                    shared_vertices += 1
                    break

        if shared_vertices == 3:
            return neighbor_id

    return idx_current

@cuda.jit(device=True)
def orientation3D(p0, p1, p2, p):
    mat = cuda.local.array((4, 4), dtype=float64)
    mat[0, 0], mat[0, 1], mat[0, 2], mat[0, 3] = p0[0], p0[1], p0[2], p0[0]**2 + p0[1]**2 + p0[2]**2
    mat[1, 0], mat[1, 1], mat[1, 2], mat[1, 3] = p1[0], p1[1], p1[2], p1[0]**2 + p1[1]**2 + p1[2]**2
    mat[2, 0], mat[2, 1], mat[2, 2], mat[2, 3] = p2[0], p2[1], p2[2], p2[0]**2 + p2[1]**2 + p2[2]**2
    mat[3, 0], mat[3, 1], mat[3, 2], mat[3, 3] = p[0], p[1], p[2], p[0]**2 + p[1]**2 + p[2]**2

    return determinant(mat)

@cuda.jit(device=True)
def determinant(mat):
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
def is_visible_from_facet(vertices, tetrahedron_data, facet_index, point):
    vertex_indices = cuda.local.array(3, dtype=int32)

    idx = 0
    for i in range(4):
        if i != facet_index:
            vertex_indices[idx] = int(tetrahedron_data[0, i])
            idx += 1

    a = vertices[vertex_indices[0]]
    b = vertices[vertex_indices[1]]
    c = vertices[vertex_indices[2]]
    
    volume = signed_volume_of_tetrahedron(a, b, c, point)
    return volume < 0

@cuda.jit(device=True)
def move_kernel(start_index, point, vertices, tetrahedra, d_walk_results, is_inside_tetrahedron):    
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
    neighbor_idx = 0

    while True:
        found_visible_facet = False
        
        for facet_index in range(4):
            vertex_count = 0
            if neighbor_idx != -1:
                already_visited = False
                for i in range(4):
                    if i != facet_index:
                        vertex_indices[vertex_count] = int(tetrahedra[idx, d_walk_results[idx], 0, i])
                        vertex_count += 1
                a = vertices[vertex_indices[1]]
                b = vertices[vertex_indices[0]]
                c = vertices[vertex_indices[2]]
                orientation = orientation3D(a, b, c, point)
                neighbor_idx = find_neighbor_by_facet(vertex_indices, tetrahedra, d_walk_results[idx])
                for j in range(visited_count[0]):
                    if visited[j] == neighbor_idx:
                        already_visited = True
                        break
                if neighbor_idx != -1 and not already_visited:
                    visited[visited_count[0]] = neighbor_idx
                    visited_count[0] += 1
                    if orientation < 0:
                        d_walk_results[idx] = neighbor_idx
                        found_visible_facet = True
                        if is_point_inside_tetrahedron(tetrahedra[idx, d_walk_results[idx], 0], vertices, point):
                            return d_walk_results[idx]
                        break

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
    if idx >= tetrahedra.shape[0]:
        return
   
    neighbors = tetrahedra[idx, tetrahedron_index, 1] 
    for i in range(4):
        neighbor_idx = neighbors[i]
        if neighbor_idx != -1:
            for j in range(4):
                if tetrahedra[idx, tetrahedron_index, 1, j]  == tetrahedron_index:
                    tetrahedra[idx, tetrahedron_index, 1, j] = -1 
    
    insert_pos = cuda.atomic.add(free_indices_count, 0, 1)
    if insert_pos < MAX_TETRAHEDRA:  
        free_indices[insert_pos] = tetrahedron_index

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
def inSphere(A, B, C, D, E):
    def calc_components(p, E):
        px, py, pz = p[0] - E[0], p[1] - E[1], p[2] - E[2]
        p_squared = px**2 + py**2 + pz**2
        return px, py, pz, p_squared

    ax, ay, az, a_squared = calc_components(A, E)
    bx, by, bz, b_squared = calc_components(B, E)
    cx, cy, cz, c_squared = calc_components(C, E)
    dx, dy, dz, d_squared = calc_components(D, E)

    det = (
        ax * (by * (cz * d_squared - cz * dz) + bz * (cy * dz - cz * dy) + dy * (bz * cy - by * cz)) -
        ay * (bx * (cz * d_squared - cz * dz) + bz * (cx * dz - cz * dx) + dx * (bz * cx - bx * cz)) +
        az * (bx * (cy * d_squared - cy * dy) + by * (cx * dy - cy * dx) + dx * (by * cx - bx * cy)) -
        a_squared * (bx * (cy * dz - cz * dy) - by * (cx * dz - cz * dx) + bz * (cx * dy - cy * dx))
    )

    return det

@cuda.jit(device=True)
def cavity_kernel(point, tetrahedra, vertices, free_indices, free_indices_count, tetrahedron_count, cavity_count, start_tetrahedron_idx, find_cavity_results, delauny_ball_num, d_create_delaunay_ball_results, vertex_count, is_inside_tetrahedron):
    idx = cuda.grid(1)
    visited = cuda.local.array(MAX_QUEUE_SIZE, dtype=int32)
    queue = cuda.local.array(MAX_QUEUE_SIZE, dtype=int32)
    queue_size = cuda.local.array(1, dtype=int32)
    actuall_index = cuda.local.array(4, dtype=int32) 
    local_cavity_count = cuda.local.array(1, dtype=int32)
    local_cavity_count = 0
    cavity_count[idx] = 0
    A = cuda.local.array(3, dtype=float64)
    B = cuda.local.array(3, dtype=float64)
    C = cuda.local.array(3, dtype=float64)
    D = cuda.local.array(3, dtype=float64)

    queue[0] = start_tetrahedron_idx
    queue_size[0] = 1
    for i in range(MAX_QUEUE_SIZE):
        visited[i] = 0
    while queue_size[0] > 0:
        if visited[queue[0]] == 0:
            visited[queue[0]] = 1
            
            for i in range(4):
                actuall_index[i] = tetrahedra[idx, queue[0] , 0 , i]
                
            A = vertices[actuall_index[0], :]
            B = vertices[actuall_index[1], :]
            C = vertices[actuall_index[2], :]
            D = vertices[actuall_index[3], :]
            if inSphere(A, B, C, D, point) or is_inside_tetrahedron[idx] == True:
                find_cavity_results[idx, local_cavity_count] = queue[0]
                local_cavity_count += 1
                for n in range(4):  
                    neighbor_idx = int(tetrahedra[idx, queue[0], 1, n])
                    if neighbor_idx != -1 and visited[neighbor_idx] == 0:
                            queue[queue_size[0]] = neighbor_idx
                            queue_size[0] += 1

        for j in range(1, queue_size[0]):
            queue[j-1] = queue[j]
        queue_size[0] -= 1

    for i in range(local_cavity_count):
        if i >= local_cavity_count:
            find_cavity_results[idx, i] = -1

    for i in range(local_cavity_count):
        remove_tetrahedron(find_cavity_results[idx, i], tetrahedra, free_indices, tetrahedron_count, free_indices_count, delauny_ball_num, d_create_delaunay_ball_results[idx], vertices, vertex_count)

    cuda.atomic.add(cavity_count, idx, local_cavity_count)
    return find_cavity_results

@cuda.jit(device=True)
def get_vertices_of_face(tetrahedron_index, face_index, tetrahedra, vertices):
    idx = cuda.grid(1)

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

    v1 = vertices[face_vertices[0], :]
    v2 = vertices[face_vertices[1], :]
    v3 = vertices[face_vertices[2], :]
    return v1, v2, v3

@cuda.jit(device=True)
def add_boundary_facet(boundary_facets, boundary_facets_count, v1, v2, v3, idx):
    pos = cuda.atomic.add(boundary_facets_count, idx, 1)
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
    idx = cuda.grid(1)

    for i in range(cavity_num):
        if cavity_tetrahedra_indices[i] == neighbor_idx:
            return True
    return False

@cuda.jit(device=True)
def is_visible_from_facet_delauny(v1, v2, v3, point):
    vec1 = cuda.local.array(3, dtype=float64)
    vec2 = cuda.local.array(3, dtype=float64)
    normal = cuda.local.array(3, dtype=float64)

    for i in range(3):
        vec1[i] = v2[i] - v1[i]
        vec2[i] = v3[i] - v1[i]

    normal[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    normal[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    normal[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    point_to_vertex = cuda.local.array(3, dtype=float64)

    for i in range(3):
        point_to_vertex[i] = point[i] - v1[i]
    dot_product = 0.0

    for i in range(3):
        dot_product += point_to_vertex[i] * normal[i]

    return dot_product > 0

@cuda.jit(device=True)
def tetrahedra_kernel(new_point, tetrahedra, vertices, cavity_tetrahedra, cavity_num, vertex_count, free_indices, free_indices_count, tetrahedron_count, d_last_tetra_index, create_delaunay_ball_results, delauny_ball_num, boundary_facets, boundary_facets_count, num_partitions, d_walk_results):
    idx = cuda.grid(1)
    local_delauny_ball_num = cuda.local.array(1, dtype=int32)
    local_delauny_ball_num = 0
    A = cuda.local.array(3, dtype=float64)
    B = cuda.local.array(3, dtype=float64)
    C = cuda.local.array(3, dtype=float64)
    D = cuda.local.array(3, dtype=float64)
    actuall_index = cuda.local.array(4, dtype=int32)

    cavity_tetrahedra_indices = cavity_tetrahedra[idx]

    boundary_facets_count[idx] = 0
    new_tetrahedron_index = d_walk_results[idx]

    for i in range(cavity_num[idx]):
        tetrahedron_index = cavity_tetrahedra[idx, i]
        if tetrahedron_index == -1:
            continue

        for face_index in range(4):
            v1, v2, v3 = get_vertices_of_face(tetrahedron_index, face_index, tetrahedra, vertices)
            neighbor_idx = tetrahedra[idx, tetrahedron_index, 1, face_index]
            if neighbor_idx == -1 or not is_neighbor_in_cavity(neighbor_idx, cavity_num[idx], cavity_tetrahedra_indices):
                add_boundary_facet(boundary_facets[idx], boundary_facets_count, v1, v2, v3, idx)

    for i in range(num_partitions) :
        if i == idx:
            continue
        for j in range(cavity_num[i]):
            tetrahedron_index = cavity_tetrahedra[i, j]
            for k in range(4):
                actuall_index[k] = tetrahedra[i, tetrahedron_index , 0 , k]
                
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
    for i in range(boundary_facets_count[idx]):
        vertex_coords = cuda.local.array((4, 3), dtype=np.float64)  
        for j in range(3):
            vertex_coords[j, 0] = boundary_facets[idx, i, j, 0]
            vertex_coords[j, 1] = boundary_facets[idx, i, j, 1]
            vertex_coords[j, 2] = boundary_facets[idx, i, j, 2]

        vertex_coords[3, 0] = new_point[idx, 0]
        vertex_coords[3, 1] = new_point[idx, 1]
        vertex_coords[3, 2] = new_point[idx, 2]
        new_tetrahedron_index = add_tetrahedron(vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, vertex_coords)
        local_delauny_ball_num += 1

    delauny_ball_num[idx] = local_delauny_ball_num + start_index
    d_last_tetra_index[idx] = new_tetrahedron_index

    return create_delaunay_ball_results

def compute_hilbert_indices(points, dimensions, bits):
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    points_normalized = (points - points_min) / (points_max - points_min)
    
    scale = (1 << bits) - 1
    points_scaled = np.floor(points_normalized * scale).astype(np.int32)

    hilbert_curve = HilbertCurve(p=bits, n=dimensions)
    
    hilbert_indices = np.array([hilbert_curve.distance_from_point(pt) for pt in points_scaled])
    
    return hilbert_indices


def partition_points(points, desired_partitions, bits=10):
    if points.size == 0:
        return [] 
    dimensions = 3
    bits = 10
    hilbert_indices = compute_hilbert_indices(points, dimensions, bits)
    
    sorted_indices = np.argsort(hilbert_indices)
    sorted_points = points[sorted_indices]
    partitions = []

    partition_size = len(points) // desired_partitions
    
    for _ in range(desired_partitions - 1):
        partitions.append(sorted_points[:partition_size])
        sorted_points = sorted_points[partition_size:]
    
    partitions.append(sorted_points)

    return partitions


def remove_processed_points(all_points, triangulated_points):
    mask = ~np.any(np.all(all_points[:, np.newaxis] == triangulated_points, axis=-1), axis=-1)

    filtered_points = all_points[mask]

    return filtered_points

@cuda.jit
def delaunay_kernel(vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, d_add_result, d_walk_results, d_find_cavity_results, cavity_num, d_create_delaunay_ball_results, delauny_ball_num, d_last_tetra_index, boundary_facets, boundary_facets_count, d_passed_points, d_triangulated_point, num_partitions, is_inside_tetrahedron):
    idx = cuda.grid(1)
    for j in range(d_triangulated_point.shape[1]):
        d_triangulated_point[idx, j] = d_passed_points[idx, 0, j]
    d_walk_results[idx] = move_kernel(d_add_result[idx], d_triangulated_point[idx], vertices, tetrahedra, d_walk_results, is_inside_tetrahedron) 
    cuda.syncthreads()
    print("thread_id", idx, "d_walk_results", d_walk_results[idx])
    d_find_cavity_results = cavity_kernel(d_triangulated_point[idx], tetrahedra, vertices, free_indices, free_indices_count, tetrahedron_count, cavity_num, d_walk_results[idx], d_find_cavity_results, delauny_ball_num, d_create_delaunay_ball_results, vertex_count, is_inside_tetrahedron) 
    cuda.syncthreads()
    print("thread_id", idx, "cavity_num", cavity_num[idx])
    tetrahedra_kernel(d_triangulated_point, tetrahedra, vertices, d_find_cavity_results, cavity_num, vertex_count, free_indices, free_indices_count, tetrahedron_count, d_last_tetra_index, d_create_delaunay_ball_results[idx], delauny_ball_num, boundary_facets, boundary_facets_count, num_partitions, d_walk_results)  
    cuda.syncthreads()
    print("Thread ID:", idx, "d_last_tetra_index[idx]", d_last_tetra_index[idx])
    print("Thread ID:", idx, "delauny_ball_num[idx]", delauny_ball_num[idx])
    if delauny_ball_num[idx] > 0:
        d_add_result[idx] = d_last_tetra_index[idx]
    print("vertex_count",vertex_count[0])

def parallel_delaunay(partitions, desired_partitions):
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
    threads_per_block = desired_partitions
    blocks_per_grid = 1
    counter = 0
    init_kernel[blocks_per_grid, threads_per_block](d_partitions, num_points_per_partition, vertices, vertex_count, tetrahedra, free_indices, free_indices_count, tetrahedron_count, d_add_result, d_tau)
    filter_points_kernel[blocks_per_grid, threads_per_block](d_partitions, d_tau, d_passed_points)
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
    
    return host_tetrahedra, host_vertices, vertex_count

def visualize_tetrahedra_and_all_points(tetrahedra, vertices, vertex_count, all_points):
    lines = o3d.geometry.LineSet()
    if isinstance(vertex_count, cuda.cudadrv.devicearray.DeviceNDArray):
        vertex_count = int(vertex_count.copy_to_host()[0])
    distinct_colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1]   # Cyan
    ]

    tetra_points = []
    edges = []

    num_partitions = tetrahedra.shape[0]
    for i in range(num_partitions):
        for j in range(tetrahedra.shape[1]):
            if tetrahedra[i, j, 2, 0] != -1:
                for k in range(6):
                    start_vertex = int(tetrahedra[i, j, 3, k])
                    end_vertex = int(tetrahedra[i, j, 4, k])

                    if start_vertex not in tetra_points:
                        tetra_points.append(start_vertex)
                    if end_vertex not in tetra_points:
                        tetra_points.append(end_vertex)

                    start_index = tetra_points.index(start_vertex)
                    end_index = tetra_points.index(end_vertex)
                    edges.append([start_index, end_index])

    tetra_point_coords = [vertices[p] for p in tetra_points if p < vertex_count]
    lines.points = o3d.utility.Vector3dVector(tetra_point_coords)
    lines.lines = o3d.utility.Vector2iVector(edges)
    lines.paint_uniform_color([1, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector([distinct_colors[i % len(distinct_colors)] for i in range(len(vertices))])

    for i, point in enumerate(vertices[:vertex_count]):
        color_name = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan'][i % len(distinct_colors)]
        print(f"Vertex {i}: {point} - Color: {color_name}")

    print("Starting visualization...")
    o3d.visualization.draw_geometries([lines, pcd], window_name='Tetrahedra and All Points Visualization')

number_of_points = int(input("Enter the number of points to sample: "))
desired_partitions = int(input("Enter the number of partitions: "))

mesh = o3d.io.read_triangle_mesh("stanford-bunny.obj")
mesh.compute_vertex_normals()
point_cloud = mesh.sample_points_poisson_disk(number_of_points)
# point_cloud = mesh.sample_points_uniformly(number_of_points=1000)
o3d.visualization.draw_geometries([point_cloud])
points = np.asarray(point_cloud.points)
partitions = partition_points(points, desired_partitions)
for i, partition in enumerate(partitions):
    print(f"Partition {i} with {len(partition)} points:")
    print(partition)
    print("\n---")

colored_point_clouds = []

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
    part_pc = o3d.geometry.PointCloud()
    part_pc.points = o3d.utility.Vector3dVector(partition)
    part_pc.colors = o3d.utility.Vector3dVector(np.tile(colors[i % len(colors)], (len(partition), 1)))
    
    colored_point_clouds.append(part_pc)

#o3d.visualization.draw_geometries(colored_point_clouds)
start_time = time.time()
tetrahedra, vertices, vertex_count = parallel_delaunay(partitions, desired_partitions)
end_time = time.time()
elapsed_time = end_time - start_time 
print(f"Elapsed time for parallel Delaunay triangulation: {elapsed_time} seconds")
visualize_tetrahedra_and_all_points(tetrahedra, vertices, vertex_count, points)

