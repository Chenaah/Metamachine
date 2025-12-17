'''
Problem Description:
Given two robot modules, A and B, each with a docking site defined by a position and orientation (quaternion) in their respective local body frames, the goal is to compute the pose of module B in the coordinate frame of module A, assuming the two docking sites are connected such that their relative position and orientation are perfectly aligned (i.e., the docks coincide in both position and orientation).

'''


import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pos, quat):
    T = np.eye(4)
    # Convert from wxyz (MuJoCo) to xyzw (scipy) format
    quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos
    return T

def matrix_to_pose(T):
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    # Convert from xyzw (scipy) to wxyz (MuJoCo) format
    quat = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return pos, quat

def get_B_in_A(pos_A_dock, quat_A_dock, pos_B_dock, quat_B_dock):
    T_A_dock = pose_to_matrix(pos_A_dock, quat_A_dock)
    T_B_dock = pose_to_matrix(pos_B_dock, quat_B_dock)
    T_dockB_B = np.linalg.inv(T_B_dock)
    T_A_B = T_A_dock @ T_dockB_B
    pos_A_B, quat_A_B = matrix_to_pose(T_A_B)
    return pos_A_B, quat_A_B