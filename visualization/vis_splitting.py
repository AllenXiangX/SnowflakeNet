import numpy as np
import open3d as o3d
import h5py

colors_points = (240 / 255, 183 / 255, 117 / 255)
colors_path1 = [0.7, 0.7, 0.7]
colors_path2 = [1, 0, 0]


def load_h5(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    f = h5py.File(path, 'r')
    cloud_data = np.array(f['data'])
    f.close()

    return cloud_data.astype(np.float64)


def load_pcd(path):
    pc = o3d.io.read_point_cloud(path)
    ptcloud = np.array(pc.points)
    return ptcloud


def show_points(points, color=None):
    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        test_pcd.paint_uniform_color(color)
    else:
        test_pcd.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([test_pcd], window_name="Open3D2", point_show_normal=True)


def create_sphere_at_xyz(xyz, colors=None, radius=0.006, resolution=4):
    """create a mesh sphere at xyz
    Args:
        xyz: arr, (3,)
        colors: arr, (3, )

    Returns:
        sphere: mesh sphere
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    # sphere.compute_vertex_normals()
    if colors is None:
        sphere.paint_uniform_color([0.7, 0.1, 0.1])  # To be changed to the point color.
    else:
        sphere.paint_uniform_color(colors)
    sphere = sphere.translate(xyz)
    return sphere


def create_pcd_mesh(point_cloud, colors=None):
    """create a mesh spheres for all coordinates in point_cloud
    Args:
        point_cloud: arr, (m, 3)
        colors: arr, (3, )

    Returns:
        mesh_pcd: obj, mesh point cloud
    """
    mesh = []
    for i in range(point_cloud.shape[0]):
        mesh.append(create_sphere_at_xyz(point_cloud[i], colors=colors))

    mesh_pcd = mesh[0]
    for i in range(1, len(mesh)):
        mesh_pcd += mesh[i]
    return mesh_pcd


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_line(point1, point2, radius=0.0003, resolution=7, colors=None):
    """get mesh of the line between two points
    Args:
        point1: arr (3, )
        point2: arr (3, )
        radius
        resolution
        colors

    Returns:
        cylinder: mesh object, the line is represented as open3d cylinder
    """

    height = np.sqrt(np.sum((point1 - point2) ** 2))
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=height,
        resolution=resolution)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(colors)

    mid = (point1 + point2) / 2

    vec1 = np.array([0, 0, height / 2])
    vec2 = point2 - mid

    T = np.eye(4)
    T[:3, :3] = rotation_matrix_from_vectors(vec1, vec2)
    T[:3, 3] = mid

    cylinder = cylinder.transform(T)

    return cylinder


def get_line_set(pcd1, pcd2, radius=0.001, resolution=10, colors=[0.7, 0.7, 0.7]):
    """get line set between two point clouds that have point correspondence
    Args:
        point_cloud1: arr, (n, 3)
        point_cloud2: arr, (n, 3)

    Returns:
        mesh: mesh of line set
    """
    lines_mesh = []
    for i in range(pcd1.shape[0]):
        lines_mesh.append(get_line(pcd1[i], pcd2[i], radius=radius, resolution=resolution, colors=colors))

    mesh = lines_mesh[0]
    for i in range(1, len(lines_mesh)):
        mesh += lines_mesh[i]

    return mesh


def splitting_paths(pcd1, pcd2, inds=None, colors_points=colors_points, colors_paths=colors_path2):
    """splitting paths between pcd1 and pcd2
    Args:
        pcd1: arr, (n1, 3)
        pcd2: arr, (n2, 3), pcd2 should split from pcd1, n2 = n1 * up_factor
        inds: list of point indices to visualize splittings

    Returns:
        mesh object of pcd1 and the specific splitting paths
    """
    n1 = pcd1.shape[0]
    n2 = pcd2.shape[0]
    up_factor = n2 // n1
    pcd1 = np.tile(pcd1, (1, up_factor)).reshape((n2, 3))
    mesh_point1 = create_pcd_mesh(pcd1, colors=colors_points)
    displacements = None
    if inds is None:
        inds = np.arange(n1)
    for i in inds:
        new_dispacements = get_line_set(pcd1[i * up_factor: (i + 1) * up_factor],
                                        pcd2[i * up_factor: (i + 1) * up_factor], colors=colors_paths)
        if displacements is None:
            displacements = new_dispacements
        else:
            displacements += new_dispacements

    mesh_out = mesh_point1 + displacements
    return mesh_out


def splitting_paths_triple(pcd1, pcd2, pcd3, inds=None, colors_points=colors_points, colors_path1=colors_path1,
                           colors_path2=colors_path2):
    """splitting path of p1 to p2 to p3
    Args:
        pcd1: arr, (n1, 3)
        pcd2: arr, (n2, 3),
        pcd3: arr, (n3, 3),
        inds: list of indices to visualize
        pcd0: optional, arr

    Returns:
        mesh

    """
    n1 = pcd1.shape[0]
    n2 = pcd2.shape[0]
    n3 = pcd3.shape[0]
    up_factor_1 = n2 // n1
    up_factor_2 = n3 // n2
    up_factor = up_factor_1 * up_factor_2
    pcd1_to_2 = np.tile(pcd1, (1, up_factor_1)).reshape((n2, 3))
    pcd2_to_3 = np.tile(pcd2, (1, up_factor_2)).reshape((n3, 3))

    if inds is None:
        inds = np.arange(n1)

    mesh_point1 = create_pcd_mesh(pcd1, colors=colors_points)
    displacements = None

    for j, i in enumerate(inds):
        new_dispacements = get_line_set(pcd1_to_2[i * up_factor_1: (i + 1) * up_factor_1],
                                        pcd2[i * up_factor_1: (i + 1) * up_factor_1], colors=colors_path1)
        new_dispacements += get_line_set(pcd2_to_3[i * up_factor: (i + 1) * up_factor],
                                         pcd3[i * up_factor: (i + 1) * up_factor], colors=colors_path2)

        if displacements is None:
            displacements = new_dispacements
        else:
            displacements += new_dispacements

    mesh_out = mesh_point1 + displacements
    return mesh_out


def Indx_of_range(pcd, rg=[0, 1], axis=0):
    """get point indices according to the range on the specific axis
    Args:
        pcd: arr, (n, 3)
        rg: range, (low, high)
        axis

    Returns:
        list of indices
    """

    n = pcd.shape[0]
    arg_idx = np.argsort(pcd[:, axis])

    low = int(n * rg[0])
    high = int(n * rg[1])

    return set(list(arg_idx[low: high]))


def splittings_by_range(pcd1, pcd2, pcd3,
                        range_x=(0, 0.5),
                        range_y=(0, 0.1),
                        range_z=(0, 0.5)):
    """get splitting paths by specifying the range on each axis
    Args:
        pcd1: arr, (n1, 3)
        pcd2: arr, (n2, 3)
        pcd3: arr, (n3, 3)
        range_x: tuple (low, high), range of points on axis x to visualize
        range_y: tuple (low, high)
        range_z: tuple (low, high)

    Returns:
        mesh
    """

    idx = Indx_of_range(pcd1, range_x, axis=0)
    idx_1 = Indx_of_range(pcd2, range_y, axis=1)
    idx = idx & idx_1

    idx_2 = Indx_of_range(pcd1, range_z, axis=2)
    idx = idx & idx_2
    print(idx)

    mesh_out = splitting_paths_triple(pcd1, pcd2, pcd3, idx)
    return mesh_out