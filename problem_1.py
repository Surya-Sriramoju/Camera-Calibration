import numpy as np

def get_KRT(x_world, x_image):
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
    P = V[-1].reshape((3,4))
    K, R = np.linalg.qr(np.linalg.inv(P[:, :3]))
    t = np.dot(np.linalg.inv(K), P[:, 3])
    return R, K , t, P

def reprojection_error(x_world, x_image, R, K, t,P):
    num_points = x_world.shape[0]
    reprojection_errors = []
    for i in range(num_points):
        X = x_world[i]
        x = np.append(x_image[i], 1)
        proj_X = np.dot(R, X) + t
        proj_x = np.dot(K, proj_X[:3]) / proj_X[2]
        error = np.linalg.norm(x - proj_x) / np.linalg.norm(x[:2])
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
    errors = reprojection_error(x_world, x_image, R, K, t, P)
    print(errors)
    
if __name__ == '__main__':
    main()





