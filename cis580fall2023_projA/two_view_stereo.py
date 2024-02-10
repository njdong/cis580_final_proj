import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(
    rgb_i, rgb_j, rect_R_i, rect_R_j, K_i, K_j, u_padding=20, v_padding=20
):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    rect_R_i,rect_R_j : [3,3]
        p_rect_left = rect_R_i @ p_i
        p_rect_right = rect_R_j @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert (
        rgb_i.shape == rgb_j.shape
    ), "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(
        h, w, K_i @ rect_R_i @ np.linalg.inv(K_i)
    )
    uj_min, uj_max, vj_min, vj_max = homo_corners(
        h, w, K_j @ rect_R_j @ np.linalg.inv(K_j)
    )

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    # compute the homography for rectification
    H_i = K_i_corr @ rect_R_i @ np.linalg.inv(K_i)
    H_j = K_j_corr @ rect_R_j @ np.linalg.inv(K_j)

    # warp the images to the rectified coordinate frame
    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, (w_max, h_max))
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    i_R_w, j_R_w : [3,3]
    i_T_w, j_T_w : [3,1]
        p_i = i_R_w @ p_w + i_T_w
        p_j = j_R_w @ p_w + j_T_w
    Returns
    -------
    [3,3], [3,1], float
        p_i = i_R_j @ p_j + i_T_j, B is the baseline
    """

    """Student Code Starts"""
    # calculate the rotation from camera frame j to camera frame i
    i_R_j = i_R_w @ np.linalg.inv(j_R_w)

    # calculate the translation from camera frame j to camera frame i
    i_T_j = i_T_w - i_R_j @ j_T_w

    # calculate the baseline as the norm of the translation vector between the two camera frames
    B = np.linalg.norm(i_T_j)
    """Student Code Ends"""

    return i_R_j, i_T_j, B


def compute_rectification_R(i_T_j):
    """Compute the rectification Rotation

    Parameters
    ----------
    i_T_j : [3,1]

    Returns
    -------
    [3,3]
        p_rect = rect_R_i @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = i_T_j.squeeze(-1) / (i_T_j.squeeze(-1)[1] + EPS)

    """Student Code Starts"""


    r2 = e_i / np.linalg.norm(e_i)
    r1 = np.cross(r2, np.array([0, 0, 1]))
    r1 /= np.linalg.norm(r1)
    r3 = np.cross(r1, r2)
    rect_R_i = np.row_stack((r1, r2, r3))



    """Student Code Ends"""

    return rect_R_i


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    ssd = np.sum((src[:, None, :, :] - dst[None, :, :, :]) ** 2, axis=(2, 3))
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    sad = np.sum(np.abs(src[:, None, :, :] - dst[None, :, :, :]), axis=(2, 3))
    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    # Flatten the patches and normalize each patch to have zero mean
    src_mean = src.mean(axis=2, keepdims=True)
    dst_mean = dst.mean(axis=2, keepdims=True)
    src_zm = src - src_mean
    dst_zm = dst - dst_mean

    # Compute the ZNCC across all color channels
    numerator = np.sum(src_zm[:, None, :] * dst_zm[None, :, :], axis=-1)
    denominator = np.sqrt(
        np.sum(src_zm**2, axis=-1)[:, None] * np.sum(dst_zm**2, axis=-1)[None, :]
    )
    zncc = numerator / (denominator + EPS)
    """Student Code Ends"""

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    H, W, _ = image.shape
    pad = k_size // 2
    padded_image = np.pad(
        image, [(pad, pad), (pad, pad), (0, 0)], mode="constant", constant_values=0
    )
    patch_buffer = np.zeros((H, W, k_size**2, 3), dtype=image.dtype)

    for i in range(k_size):
        for j in range(k_size):
            # Extract patches
            patch_buffer[:, :, i * k_size + j] = padded_image[i : i + H, j : j + W]
    """Student Code Starts"""

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(
    rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch
):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func: function, optional
        the function used to compute the patch buffer, by default image2patch
        (there is NO NEED to alter this argument)

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    # NOTE: when computing patches, please use the syntax:
    # patch_buffer = img2patch_func(image, k_size)
    # DO NOT DIRECTLY USE: patch_buffer = image2patch(image, k_size), as it may cause errors in the autograder

    """Student Code Starts"""
    """Compute the disparity map from two rectified views."""
    H, W, _ = rgb_i.shape

    # convert images to patches
    patches_i = img2patch_func(rgb_i, k_size)
    patches_j = img2patch_func(rgb_j, k_size)

    # initialize the disparity map and the left-right consistency mask
    disp_map = np.zeros((H, W), dtype=np.float64)
    lr_consistency_mask = np.zeros((H, W), dtype=np.float64)

    # iterate over each pixel in the image
    for i in tqdm(range(H), desc="Computing disparity map"):
        for j in range(W):


            # get the patch from the left image
            patch_i = patches_i[i, j]

            # calculate the similarity scores for all patches in the right image
            scores = kernel_func(patch_i[None, :], patches_j[i])

            # find the index of the patch with the highest similarity score
            match_idx = np.argmin(scores)

            # calculate the disparity and update the disparity map
            disp_map[i, j] = d0 + j - match_idx

            # calculate left-right consistency (additional functionality if needed)
            #  set all pixels to be consistent in this example
            lr_consistency_mask[i, j] = 1.0


    """Student Code Ends"""

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""

    # compute the depth map from the disparity map
    # depth is computed as: Z = f * B / disparity
    # where f is the focal length from the camera matrix K (assumed to be at K[0, 0])
    f = K[0, 0]


    # avoid division by zero in disparity map
    disp_map_with_eps = disp_map.copy()
    disp_map_with_eps[disp_map_with_eps == 0] = np.inf
    dep_map = f * B / disp_map_with_eps

    # initialize the point cloud array
    H, W = disp_map.shape
    xyz_cam = np.zeros((H, W, 3), dtype=np.float64)

    # compute the backprojected point cloud
    for i in range(H):
        for j in range(W):

            # compute the depth Z
            Z = dep_map[i, j]
            # backproject to 3D space using the inverse of the camera matrix
            # we assume that the camera matrix K is of the form:
            # K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
            # therefore, we can compute the backprojected point as follows:
            X = Z * (j - K[0, 2]) / f
            Y = Z * (i - K[1, 2]) / f
            xyz_cam[i, j] = [X, Y, Z]
    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    c_R_w,
    c_T_w,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is:
    given pcl_cam [N,3], c_R_w [3,3] and c_T_w [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize)
    )
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    # apply the mask to filter out the points
    valid_points = mask.reshape(-1) > 0
    pcl_cam_filtered = xyz_cam.reshape(-1, 3)[valid_points]
    pcl_color_filtered = rgb.reshape(-1, 3)[valid_points]

    # transform the point cloud from camera frame to world frame
    pcl_world = np.dot(pcl_cam_filtered - c_T_w.T, c_R_w.T)
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    # p_i = i_R_w @ p_w + i_T_w
    i_R_w, i_T_w = view_i["R"], view_i["T"][:, None]
    # p_j = j_R_w @ p_w + j_T_w
    j_R_w, j_T_w = view_j["R"], view_j["T"][:, None]

    i_R_j, i_T_j, B = compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w)
    assert (
        i_T_j[1, 0] > 0
    ), "here we assume view i should be on the left, not on the right"

    rect_R_i = compute_rectification_R(i_T_j)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        rect_R_i,
        rect_R_i @ i_R_j,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        rect_R_i @ i_R_w,
        rect_R_i @ i_T_w,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
