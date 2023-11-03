import numpy as np
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as trans
# import open3d as o3d as o3d
from utils.visualize import *
import functools

def generate_box_point_cloud(args, center, quat, size, np_color=np.array([1, 0, 0]), resolution=0.015, n_particles=1000, static=False, visualize=False, crop=True):
    # Convert quaternion to rotation matrix
    """
    Generates a dense point cloud with box shape given box center, quaternion, and size.

    Parameters:
        center (numpy.ndarray): Center of the box.
        quat (numpy.ndarray): Quaternion representing the orientation of the box.
        size (numpy.ndarray): Size of the box along each axis.
        resolution (float): Resolution of the point cloud.

    Returns:
        point_cloud (open3d.geometry.PointCloud): Point cloud representing the box shape.
    """        
    if args.evenly_spaced == True:
        if static == True:
            return evenly_spaced_static_box_points(args, center, size, quat, n_particles)
        else:
            return evenly_spaced_dynamic_box_points(args, center, size, quat, n_particles)
    else:
        is_tensor = isinstance(center, torch.Tensor)
        if is_tensor:
            device = center.device
            dtype = center.dtype
            center = center.cpu().numpy()
            quat = quat.cpu().numpy()
            size = size.cpu().numpy()
            np_color = torch.tensor(np_color, device=device, dtype=dtype).cpu().numpy()
        else:
            np_color = np.asarray(np_color)
            center = np.array(center)
            size = np.array(size)

        # Generate box vertices in local coordinates
        l, w, h = size / 2
        vertices_local = np.array([[-l, -w, -h], [l, -w, -h], [l, w, -h],
                                [-l, w, -h], [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]])

        # Rotate box vertices by quaternion
        rotation_matrix = trans.quat2mat(quat)
        vertices_local = vertices_local.dot(rotation_matrix.T)
        # Translate box vertices to center
        vertices_local += center

        # Generate voxel centers in local coordinates
        x = np.arange(-l, l + resolution, resolution)
        y = np.arange(-w, w + resolution, resolution)
        z = np.arange(-h, h + resolution, resolution)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        voxel_centers_local = np.array(
            (xv.flatten(), yv.flatten(), zv.flatten())).T

        # Rotate voxel centers by quaternion
        voxel_centers_local = voxel_centers_local.dot(rotation_matrix.T)

        # Translate voxel centers to center
        voxel_centers_local += center

        # # Check if voxel centers are inside the box and generate point cloud
        # indices_inside_box = [i for i in range(voxel_centers_local.shape[0]) if np.all(
        #     np.abs(voxel_centers_local[i] - center) < size / 2)]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(
            voxel_centers_local)

        if args.crop_bound is not None and crop==True:
            # Define the bounding box
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=args.crop_bound[0], max_bound=args.crop_bound[1])

            # Applying a crop function to all values in the ordered dictionary
            point_cloud = point_cloud.crop(bbox)

        if args.farthest_point_sample:
            fps_pcd = point_cloud.farthest_point_down_sample(n_particles)

            # fps_points = fps(np.asarray(point_cloud.points), n_particles)
            # fps_pcd = o3d.geometry.PointCloud()
            # fps_pcd.points = o3d.utility.Vector3dVector(fps_points)
            point_cloud = fps_pcd

        point_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(np_color, (len(point_cloud.points), 1)))

        if visualize:
            visualize_o3d([point_cloud], title='box_point_cloud')

        return point_cloud

def box_vertices(center, size, quaternion):
    # Calculate the half size of the box
    half_size = np.array(size) / 2

    # Define the eight offsets from the center to the vertices
    offsets = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ]) * half_size

    # Convert the quaternion to a rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Rotate the offsets and add them to the center
    vertices = np.dot(offsets, rotation_matrix.T) + center

    return vertices

def box_point_cloud(center, size, quaternion, num_points):
    rotation = R.from_quat(quaternion)
    
    points = np.random.rand(num_points, 3)  # Create a 3D point cloud with random points in the range [0, 1)
    points = (points * size) - (size / 2)  # Scale and translate the points to fit the box dimensions
    points = rotation.apply(points)  # Rotate the points
    points += center  # Translate the points to the box's center
    
    return points

def generate_box_surf_particles(center, size, quaternion, num_points=1000):
    half_size = np.array(size) / 2
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    points = []
    for axis in range(3):
        n_points_per_face = num_points // 6

        for sign in [-1, 1]:
            face_points = np.random.rand(n_points_per_face, 3) * 2 - 1
            face_points[:, axis] = sign
            face_points *= half_size
            face_points = np.dot(face_points, rotation_matrix.T) + center
            points.append(face_points)

    points = np.vstack(points)

    return points



def evenly_spaced_surf_box_points(center, size, quaternion, num_points):
    points_per_edge = int(round(np.cbrt(num_points)))
    if points_per_edge < 2:
        raise ValueError("Number of points must be at least 8 to generate an evenly spaced point cloud.")

    x_space = np.linspace(-size[0] / 2, size[0] / 2, points_per_edge)
    y_space = np.linspace(-size[1] / 2, size[1] / 2, points_per_edge)
    z_space = np.linspace(-size[2] / 2, size[2] / 2, points_per_edge)

    rotation = R.from_quat(quaternion)

    points = []
    for x in x_space:
        for y in y_space:
            for z in z_space:
                if (x == x_space[0] or x == x_space[-1] or
                        y == y_space[0] or y == y_space[-1] or
                        z == z_space[0] or z == z_space[-1]):
                    point = rotation.apply([x, y, z]) + center
                    points.append(point)

    return np.array(points)


def evenly_spaced_static_box_points(args, center, size, quaternion, num_points):
    if args.add_corners_points:
        num_points = num_points - args.num_corners_points
    else:
        pass

    size = np.array(size)

    nonzero_size = size[np.nonzero(size)]  # Exclude the dimensions with size 0
    volume = np.prod(nonzero_size)
    points_per_unit_volume = num_points / volume

    # Calculate num_points_per_axis and find total_points_generated
    num_points_per_axis = (points_per_unit_volume * size)
    total_points_generated = np.prod(num_points_per_axis[np.nonzero(num_points_per_axis)])

    # Adjust num_points_per_axis iteratively to achieve the desired total number of points
    for _ in range(100): 
        if total_points_generated != num_points:
            adjustment_factor = np.cbrt(num_points / total_points_generated)
            num_points_per_axis = np.round(num_points_per_axis * adjustment_factor).astype(int)
            num_points_per_axis = np.maximum(num_points_per_axis, 1)  # Ensure at least 1 point per axis

            total_points_generated = np.prod(num_points_per_axis[np.nonzero(num_points_per_axis)])
        else:
            break
    # If the total number of points still doesn't match, adjust the axis with the largest size
    if total_points_generated != num_points:
        max_axis = np.argmax(size)
        num_points_per_axis[max_axis] = num_points // np.prod(num_points_per_axis[np.arange(3) != max_axis])
        total_points_generated = np.prod(num_points_per_axis)
    
    if total_points_generated != num_points:
        raise ValueError(f"Could not generate the desired number of points. Generated: {total_points_generated}, Desired: {num_points}")

    x_space = np.linspace(-size[0] / 2, size[0] / 2, num_points_per_axis[0])
    y_space = np.linspace(-size[1] / 2, size[1] / 2, num_points_per_axis[1])
    z_space = np.linspace(-size[2] / 2, size[2] / 2, num_points_per_axis[2])

    rotation = R.from_quat(quaternion)
    
    # Create a meshgrid for each axis
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_space, y_space, z_space, indexing='ij')

    # Combine the meshgrid points into a single array
    points = np.stack((x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()), axis=-1)

    # Apply the rotation and translation
    points = rotation.apply(points) + center

    if args.add_corners_points:
        return add_box_corners(points, size, center, rotation)
    else:
        return points

    # points = []
    # for x in x_space:
    #     for y in y_space:
    #         for z in z_space:
    #             point = rotation.apply([x, y, z]) + center
    #             points.append(point)

    # return np.array(points)

def find_factors(num_points, size_ratio):
    factors = []
    for i in range(1, num_points + 1):
        if num_points % i == 0:
            factors.append(i)
    factors = np.array(factors)

    best_factors = [1, 1, 1]
    best_error = np.inf
    for x in factors:
        for y in factors:
            for z in factors:
                if x * y * z == num_points:
                    valid_axes = np.nonzero(size_ratio)
                    error = np.linalg.norm(np.array([x, y, z])[valid_axes] / size_ratio[valid_axes] - 1)
                    if error < best_error:
                        best_factors = [x, y, z]
                        best_error = error
    return best_factors

def evenly_spaced_dynamic_box_points(args, center, size, quaternion, num_points):
    num_points = num_points - 8
    is_tensor = isinstance(center, torch.Tensor)
    if is_tensor:
        device = center.device

    nonzero_size = size[np.nonzero(size)]
    size_ratio = np.zeros_like(size, dtype=np.float32)
    size_ratio[np.nonzero(size)] = nonzero_size / np.min(nonzero_size)

    num_points_per_axis = find_factors(num_points, size_ratio)
    if is_tensor:
        x_space = torch.linspace(-size[0] / 2, size[0] / 2, num_points_per_axis[0], device=device).float()
        y_space = torch.linspace(-size[1] / 2, size[1] / 2, num_points_per_axis[1], device=device).float()
        z_space = torch.linspace(-size[2] / 2, size[2] / 2, num_points_per_axis[2], device=device).float()

        rotation = R.from_quat(quaternion.cpu().numpy())
        rotation = torch.from_numpy(rotation.as_matrix()).to(device=device).float()

        points = []
        for x in x_space:
            for y in y_space:
                for z in z_space:
                    point = torch.matmul(rotation, torch.tensor([x, y, z], device=device)) + center
                    points.append(point)

    else:
        x_space = np.linspace(-size[0] / 2, size[0] / 2, num_points_per_axis[0])
        y_space = np.linspace(-size[1] / 2, size[1] / 2, num_points_per_axis[1])
        z_space = np.linspace(-size[2] / 2, size[2] / 2, num_points_per_axis[2])

        rotation = R.from_quat(quaternion)

        points = []
        for x in x_space:
            for y in y_space:
                for z in z_space:
                    point = rotation.apply([x, y, z]) + center
                    points.append(point)


    if is_tensor:
        points = torch.stack(points, dim=0)
    else:
        points = np.array(points)

    
    if args.add_corners_points:
        return add_box_corners(points, size, center, rotation)
    else:
        return points



def add_box_corners(points, size, center, rotation):
    half_size = size / 2
    corners = np.array([[-half_size[0], -half_size[1], -half_size[2]],
                        [half_size[0], -half_size[1], -half_size[2]],
                        [-half_size[0], half_size[1], -half_size[2]],
                        [half_size[0], half_size[1], -half_size[2]],
                        [-half_size[0], -half_size[1], half_size[2]],
                        [half_size[0], -half_size[1], half_size[2]],
                        [-half_size[0], half_size[1], half_size[2]],
                        [half_size[0], half_size[1], half_size[2]]])

    corner_points = []
    is_tensor = isinstance(points, torch.Tensor)

    for corner in corners:
        if is_tensor:
            device = rotation.device
            corner_tensor = torch.tensor(corner, dtype=torch.float32, device=device)
            point = torch.matmul(rotation, corner_tensor) + center
        else:
            point = rotation.apply(corner) + center
        corner_points.append(point)

    corner_points = torch.stack(corner_points) if is_tensor else np.array(corner_points)

    if is_tensor:
        return torch.cat((points, corner_points))
    else:
        return np.concatenate((points, corner_points))

def recover_box_properties(points):
    # The last 8 points are the corners of the box
    corners = points[-8:]

    # The center of the box is the average of the corner points
    center = np.mean(corners, axis=0)

    # Calculate the direction vectors of the box edges
    edges = np.array([
        corners[1] - corners[0],  # x-axis
        corners[2] - corners[0],  # y-axis
        corners[4] - corners[0]   # z-axis
    ])

    # Perform Gram-Schmidt process to get orthonormal basis
    basis = np.zeros_like(edges)
    zero_norm_indices = []
    for i in range(3):
        if np.linalg.norm(edges[i]) == 0:
            zero_norm_indices.append(i)
        else:
            basis[i] = edges[i] / np.linalg.norm(edges[i])

    if len(zero_norm_indices) == 1:
        # If only one edge has zero length
        if zero_norm_indices[0] == 1:
            basis[1] = np.cross(basis[0], basis[2])
        elif zero_norm_indices[0] == 2:
            basis[2] = np.cross(basis[0], basis[1])
    elif len(zero_norm_indices) == 2:
        # If two edges have zero length
        if 0 not in zero_norm_indices:
            # If x-axis is not zero
            basis[1] = np.array([0, 1, 0]) if np.allclose(basis[0], [0, 1, 0]) else [0, 1, 0]
            basis[2] = np.cross(basis[0], basis[1])
        elif 1 not in zero_norm_indices:
            # If y-axis is not zero
            basis[2] = np.array([0, 0, 1]) if np.allclose(basis[1], [0, 0, 1]) else [0, 0, 1]
            basis[0] = np.cross(basis[1], basis[2])
        elif 2 not in zero_norm_indices:
            # If z-axis is not zero
            basis[0] = np.array([1, 0, 0]) if np.allclose(basis[2], [1, 0, 0]) else [1, 0, 0]
            basis[1] = np.cross(basis[2], basis[0])

    # Convert the basis vectors into a rotation matrix
    rotation_matrix = basis.T

    # Convert the rotation matrix into a quaternion
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    return center, quaternion


def apply_homogeneous_transformation(point_cloud, transformation_matrix):
    """
    Apply a homogeneous transformation to a point cloud.

    Args:
        point_cloud (numpy.ndarray or torch.Tensor): Matrix of shape (n, d) representing the point cloud,
            where n is the number of points and d is the number of dimensions.
        transformation_matrix (numpy.ndarray or torch.Tensor): Homogeneous transformation matrix of shape (d+1, d+1).

    Returns:
        numpy.ndarray or torch.Tensor: Matrix of shape (n, d) representing the transformed point cloud.
    """
    is_tensor = isinstance(transformation_matrix, torch.Tensor)
    n, d = point_cloud.shape
    assert transformation_matrix.shape == (d+1, d+1), "Transformation matrix must be of shape (d+1, d+1)"

    # Append a column of ones to the point cloud matrix to represent homogeneous coordinates
    ones = np.ones((n, 1)) if not is_tensor else torch.ones((n, 1), device=point_cloud.device, dtype=point_cloud.dtype)
    point_cloud_homogeneous = np.hstack((point_cloud, ones)) if not is_tensor else torch.cat((point_cloud, ones), dim=1).float()

    # Apply the transformation using matrix multiplication
    transformed_homogeneous = np.dot(point_cloud_homogeneous, transformation_matrix.T) if not is_tensor else torch.matmul(point_cloud_homogeneous, transformation_matrix.float().T)

    # Divide by the last element to remove homogeneous coordinates
    transformed = transformed_homogeneous[:, :-1] / transformed_homogeneous[:, -1].reshape((-1, 1)) if not is_tensor else transformed_homogeneous[:, :-1] / transformed_homogeneous[:, -1].view(-1, 1)

    return transformed

def create_gripper_pcd(args, ee_pos, ee_quat, gripper_width, gripper_color=[0, 0, 1], n_particles=300, visualize=False):

    gripper_size = args.gripper_size 

    is_tensor = isinstance(ee_pos, torch.Tensor)
    if is_tensor:
        device = ee_pos.device
        dtype = ee_pos.dtype
        args.ee_fingertip_T_mat = torch.tensor(args.ee_fingertip_T_mat, device=device, dtype=dtype)
        
    else:
        gripper_size = np.asarray(gripper_size)
        gripper_color = np.asarray(gripper_color)

    # Use appropriate functions depending on the input type
    matmul = torch.matmul if is_tensor else np.dot
    concatenate = (lambda x, dim=None: torch.cat(x, dim=dim)) if is_tensor else (lambda x, dim=None: np.concatenate(x, axis=dim))
    stack = torch.stack if is_tensor else np.stack
    tensor = (lambda x: torch.tensor(x, device=device, dtype=dtype)) if is_tensor else np.array

    # transform the tool mesh
    fingermid_pos = matmul(trans.quat2mat(ee_quat),
                           args.ee_fingertip_T_mat[:3, 3]) + ee_pos
    fingertip_mat = matmul(trans.quat2mat(ee_quat),
                           args.ee_fingertip_T_mat[:3, :3])

    fingertip_T_list = []
    
    box_center = np.array([0, 0, 0]) if not is_tensor else torch.tensor([0, 0, 0], device=device)
    box_quat = np.array([0, 0, 0, 1]) if not is_tensor else torch.tensor([0, 0, 0, 1], device=device)

    gripper_pcd = generate_box_point_cloud(args, box_center, box_quat, gripper_size, np_color=gripper_color, n_particles=int(
        n_particles/2), visualize=visualize, crop=False)
    
    gripper_mesh = []
    for k in range(2):
        relative_finger_pos = np.array([0, (2 * k ) * (gripper_width / 2), 0]) if not is_tensor else torch.tensor([0, (2 * k ) * (gripper_width / 2), 0], device=device)
        fingertip_pos = matmul(fingertip_mat, relative_finger_pos.T).T + fingermid_pos
        fingertip_T = concatenate((concatenate(
            (fingertip_mat, fingertip_pos.reshape(3, 1)), dim=1), tensor([0., 0., 0., 1.]).reshape(1, 4)), dim=0)
        fingertip_T_list.append(fingertip_T)
        if args.evenly_spaced:
            gripper_mesh.append(apply_homogeneous_transformation(gripper_pcd, fingertip_T))
        else:
            gripper_mesh.append(copy.deepcopy(gripper_pcd).transform(fingertip_T))

    if visualize:
        visualize_o3d(gripper_mesh, title='gripper mesh')

    # generate the gripper point cloud
    if args.evenly_spaced:
        return concatenate(gripper_mesh, dim=0)
    else:
        return functools.reduce(lambda x, y: x+y, gripper_mesh)


def project_to_yz(point):
    return np.array([point[1], point[2]])


def point_in_projected_box(point_yz, box_center, box_size, box_quat):
    return True
    # box_center_yz = project_to_yz(box_center)
    
    # # Rotate the point to box's local coordinate system
    # rotation_matrix = R.from_quat(box_quat).as_matrix()
    # yz_rotation_matrix = rotation_matrix[1:, 1:]
    # point_local = np.linalg.inv(yz_rotation_matrix).dot(point_yz - box_center_yz)
    
    # # Check if the point is inside the box in YZ plane
    # return np.all(np.abs(point_local) <= (np.array(box_size) / 2)[1:])


