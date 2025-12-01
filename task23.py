from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import os



def normalize_points(pts):
    
    mean = np.mean(pts, axis=0)
    centered_pts = pts - mean
    mean_dist = np.mean(np.sqrt(np.sum(centered_pts**2, axis=1)))
    scale = np.sqrt(2) / mean_dist
    
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    
    homogeneous_pts = utils.homogenize(pts)
    normalized_pts = (T @ homogeneous_pts.T).T
    return utils.dehomogenize(normalized_pts), T
   

def find_fundamental_matrix(shape, pts1, pts2):
    """ 
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """
  
    ###########################################################################
    
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    
    u  = pts1_norm[:, 0]
    v  = pts1_norm[:, 1]
    up = pts2_norm[:, 0]  
    vp = pts2_norm[:, 1]  

    total_points = pts1.shape[0]
    matrix_U = np.zeros((total_points, 9))
    matrix_U[:, 0] = up * u
    matrix_U[:, 1] = up * v
    matrix_U[:, 2] = up
    matrix_U[:, 3] = vp * u
    matrix_U[:, 4] = vp * v
    matrix_U[:, 5] = vp
    matrix_U[:, 6] = u
    matrix_U[:, 7] = v
    matrix_U[:, 8] = 1


 
       
    _, _, Vt = np.linalg.svd(matrix_U, full_matrices=False)
    
    F_rank3 = Vt[-1].reshape(3, 3)

    U_F, S_F, Vt_F = np.linalg.svd(F_rank3)
    
    S_F[2] = 0
    
    F_rank2 = U_F @ np.diag(S_F) @ Vt_F

    F = T2.T @ F_rank2 @ T1
    ###########################################################################


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################

    U, S, Vt = np.linalg.svd(F)
    e1 = Vt[-1] 
    e2 = U[:, -1] 
    

                                                #
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera instrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera instrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    pcd = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pcd


if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape

        # you can check against this
        # FCheck, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

        #######################################################################
        # TODO: Your code here                                                #
        #######################################################################
