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
     M=np.zeros((3,4))
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
        At_b[i] = np.sum(A[:, i] * b)
    
    
    
    augmented_matrix = np.zeros((11, 12))
    augmented_matrix[:,:11]=At_A
    augmented_matrix[:,11]=At_b


    #converting to Echelon form
    for i in range(11):
        
        max_row = i
        for j in range(i + 1, 11):
            if abs(augmented_matrix[j, i]) > abs(augmented_matrix[max_row, i]):
                max_row = j
        
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
    
        for j in range(i + 1, 11):
            if abs(augmented_matrix[i, i]) > 1e-10:
                factor = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j, :] -= factor * augmented_matrix[i, :]


    # converting to Reduced Echelon form    
    for i in range(10, -1, -1):  
        pivot = augmented_matrix[i, i]

        if abs(pivot) >= 1e-10:
            augmented_matrix[i, :] = augmented_matrix[i, :] / pivot

            for j in range(0, i):
                factor = augmented_matrix[j, i]
                augmented_matrix[j, :] -= factor * augmented_matrix[i, :]

    
    
    
    x=augmented_matrix[:,11]
  
    M[0,0]=x[0]
    M[0,1]=x[1]
    M[0,2]=x[2]
    M[0,3]=x[3]
    M[1,0]=x[4]
    M[1,1]=x[5]
    M[1,2]=x[6]
    M[1,3]=x[7]
    M[2,0]=x[8]
    M[2,1]=x[9]
    M[2,2]=x[10]
    M[2,3]=1

    
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

    M = find_projection(pts2d, pts3d)
    print("The Projection Matrix M is:")
    print(M)
    print("\n")


    num_3d_points = pts3d.shape[0]
    pts3d_homogeneous = utils.homogenize(pts3d)

    projected_homogeneous = pts3d_homogeneous @ M.T

    projected_2d = utils.dehomogenize(projected_homogeneous) 

    error = 0.0
    for i in range(num_3d_points):
        error += np.linalg.norm(projected_2d[i, :] - pts2d[i, :])
    average_error = error / num_3d_points
    print(f"Average distance error: {average_error:.6f} pixels")
    print("\n")
   
        



