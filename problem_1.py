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
    U, S, V = np.linalg.svd(A)
    P = V[-1].reshape((3,4))
    H,h = P[:,:3], P[:,-1].reshape(3,1)
    R_T, K_inv = np.linalg.qr(np.linalg.inv(H))
    R = np.dot(R_T,H)
    K = np.linalg.inv(K_inv)
    t = np.dot(np.linalg.inv(K), h)
    return R, K , t
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
    R, K, t = get_KRT(x_world, x_image)
    
if __name__ == '__main__':
    main()





