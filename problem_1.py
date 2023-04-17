import numpy as np

#Function to extract K, R and t matrices
def get_KRT(x_world, x_image):
    #building the measurement Matrix A
    A = []
    for i in range(x_world.shape[0]):
        xw,yw,zw = x_world[i]
        x,y = x_image[i]
        vect_1 = [xw, yw, zw, 1, 0, 0, 0, 0, -x*xw, -x*yw, -x*zw, -x]
        vect_2 = [0, 0, 0, 0, xw, yw, zw, 1, -y*xw, -y*yw, -y*zw, -y]
        A.append(vect_1)
        A.append(vect_2)
        
    A = np.array(A).reshape(x_world.shape[0]*2, 12)
    _, _, V = np.linalg.svd(A)
    #Extracting projection matrix using singular value decompostion
    P = V[-1].reshape((3,4))
    #Perform QR factorization for extracting Rotation and Intrinsic matrices
    R, K = np.linalg.qr(np.linalg.inv(P[:, :3]))
    K = np.linalg.inv(K)
    R = np.linalg.inv(R)
    t = np.dot(np.linalg.inv(K), P[:, 3])
    K = K/K[2,2]

    return R, K , t.reshape(3,1), P

def reprojection_error(x_world, x_image, R, K, t):
    num_points = x_world.shape[0]
    reprojection_errors = []
    for i in range(num_points):
        x = np.append(x_world[i], 1)
        Rt = np.hstack((R,t))
        #Multiplying the 3D world point to get the coordinates in Camera Coordinate system
        proj_X = np.dot(Rt, x)
        #Multiplying Matrix with intrinsic matrix to map the points in image space
        proj_x = np.dot(K, proj_X) / proj_X[2]
        #Calculating the error between projected point and image point
        error = np.linalg.norm(proj_x[:2]-x_image[i])
        reprojection_errors.append(error)
    return reprojection_errors

def main():
    x_world = np.array([[0,0,0],
                   [0,3,0],
                   [0,7,0],
                   [0,11,0],
                   [7,1,0],
                   [0,11,7],
                   [7,9,0],
                   [0,1,7]])

    x_image = np.array([[757,213],
                    [758,415],
                    [758,686],
                    [759,966],
                    [1190,172],
                    [329,1041],
                    [1204,850],
                    [340,159]])
    R, K, t, P = get_KRT(x_world, x_image)
    print("Projection Matrix P: ")
    print(P)
    print("Intrinsic Matrix K: ")
    print(K)
    print("Rotation Matrix R: ")
    print(R)
    print("Translation Vector t: ")
    print(t)
    errors = reprojection_error(x_world, x_image, R, K, t)
    print('The errors of each point are given in the following list: ', errors)
    print('Mean of the reprojection error is: ',sum(errors)/len(errors))
    
if __name__ == '__main__':
    main()