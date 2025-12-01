import numpy as np
import utils


def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - P: Numpy array of shape (3,4) giving the camera projection matrix P

    """
    M = None
    ###########################################################################
    
    
    total_points = pts2d.shape[0]
    A = np.zeros((2 * total_points, 11))
    b = np.zeros(2 * total_points)

    for i in range(total_points):

        X, Y, Z ,u,v= pts3d[i, 0], pts3d[i, 1], pts3d[i, 2], pts2d[i, 0], pts2d[i, 1]
           
        A[2 * i, :] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z]
        A[2 * i + 1, :] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z]

        b[2 * i] = u
        b[2 * i + 1] = v
    

    At_A = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            At_A[i, j] = np.sum(A[:, i] * A[:, j])
  
    At_b = np.zeros(11)
    for i in range(11):
        At_A[i] = np.sum(A[:, i] * b)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                 #
    
    
    
    
    
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return M


if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """

