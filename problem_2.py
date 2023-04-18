import cv2
import numpy as np
import glob


#dimensions of the checkerboard
#number of corners per row and column
board_dim = (6, 9)
# Choosing a criteria to stop the iteration after 30
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Array for storing 3D points
points3d = []

#Array for storing 2D points
points2d = []

#initializing the square size
square_size = 21.5
# 3D points real world coordinates
objectp3d = np.zeros((1, board_dim[0]* board_dim[1],3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:board_dim[0],0:board_dim[1]].T.reshape(-1, 2) * square_size
# print(objectp3d)

images = glob.glob('Calibration_Imgs/*.jpg')
for filename in images:
	image = cv2.imread(filename)
	grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(
					grayColor, board_dim,
					cv2.CALIB_CB_ADAPTIVE_THRESH
					+ cv2.CALIB_CB_FAST_CHECK +
					cv2.CALIB_CB_NORMALIZE_IMAGE)

	#refining the pixels
	if ret == True:
		points3d.append(objectp3d)

		# Refining pixel coordinates
		# for given 2d points.
		corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)

		points2d.append(corners2)

		# Draw the corners
		image = cv2.drawChessboardCorners(image,board_dim,corners2, ret)
	# display the corners
	cv2.imshow('img', cv2.resize(image,None, fx = 0.3, fy = 0.3))
	cv2.waitKey(0)

cv2.destroyAllWindows()
#calibrate the camera to extact intrinsic matrix
_, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(points3d, points2d, grayColor.shape[::-1], None, None)

print("Intrinsix Matrix K: ")
print(matrix)

#calculating reprojection error
errors = []
for i in range(len(points3d)):
    imgpoints2, _ = cv2.projectPoints(points3d[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(points2d[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    errors.append(error)
print("The error for each image is given in the following list: ", errors)
print("The mean reprojection error is: ", np.average(errors))