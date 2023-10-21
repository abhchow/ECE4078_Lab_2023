# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:
    def __init__(self, robot, marker_length=0.07):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist
        self.robot = robot

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters() # updated to work with newer OpenCV
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) # updated to work with newer OpenCV
        
    def get_state_vector(self):
        state = self.robot.state
        #print(state)
        return state
    
    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)

        #if (rvecs != None and tvecs != None) or (rvecs.all().all() != None and tvecs.all().all() != None): 
        if type(rvecs) != type(None) and type(tvecs) != type(None):
            for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                # rvec = np.squeeze(rvec)
                # tvec = np.squeeze(tvec)
                #-------------------------------
                # print(f"transposed = {tvec.T}")
                # r_rot_matrix = cv2.Rodrigues(rvec)[0]
                # r_rot_inv= np.linalg.inv(r_rot_matrix)
                # #tranform Tvec to robot coords
                # tvec_rob= r_rot_matrix@(tvec.T-[[0],[0],[-3]]) #z translation applied
                # print(f"tvec in robot coords= {tvec_rob}")
                # #transform tvec back to wold coords
                # tvec_transformed = r_rot_inv@tvec_rob
                # print(tvec_transformed)            
                #-------------------
                #print(f"tvec = {tvec.T}")
                rvec_unit_magnitude = np.linalg.norm(rvec)
                #print(rvec_unit_magnitude)
                if rvec_unit_magnitude != 0: 
                    rvec_unit_vector = (rvec/rvec_unit_magnitude).T
                else:
                    rvec_unit_vector = rvec.T
                #shift tvec 3.5 units in the rvec_unit_vector direction 
                #print(rvec_unit_vector[2])
                #print(f"rvec_unit_vector = {rvec_unit_vector})")
                # temp = rvec_unit_vector[2]
                # rvec_unit_vector[2] = rvec_unit_vector[0]
                # rvec_unit_vector[0] = temp
                rot_z = np.array([[0, -1, 0],
                            [1,  0, 0],
                            [0,  0, 1]])
                rot_y = np.array([[0, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 0]])
                rot_x = np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])
                tvec_translated = tvec.T + rot_y@rot_x@rvec_unit_vector*0.040
                #print(f"tvec_translate = {tvec_translated}\n")
                tvecs[idx] = tvec_translated.T
                #print(f"tvec_translated = {tvec_translated}\n")
                # #transform to world coordinates 
                # world_tvec = np.multiply(rvec, tvec)
                # print(world_tvec)
                # #shift by 3.5 units perpendicular to the plane 
                # world_tvec[2] = world_tvec[2] - 3.5 
                #transform back 
                # rvec_inv = np.linalg.inv(rvec)
                # tvec_transformed = np.dot(rvec_inv, tvec)
                # print(tvec_transformed)
                # print(tvecs[idx])
                # tvecs[idx] = tvec_transformed


        #print(tvecs)


            # print(rvecs)
            # for rvec in rvecs: 
            #     print(rvec)

            # for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            #     print(tvec)
            #     world_tvec = np.dot(rvec,tvec)
            #     #shift by 3.5 units perpendicular to the plane 
            #     world_tvec[2] = world_tvec[2] - 3.5 
            #     #transform back 
            #     rvec_inv = np.linalg.inv(rvec)
            #     tvec_transformed = np.dot(rvec_inv, tvec)
            #     tvecs[idx] = tvec_transformed
            #     print(tvec)
        

        # rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params) # use this instead if you got a value error
            # marker positions in format [x,y,z] with z pointing forward, y pointing down, x pointing right. 
        #print(tvecs)

        robot_state= self.get_state_vector()

        if ids is None:
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            # Some markers appear multiple times but should only be handled once.
            # make sure ID is only 1-10 
            if idi in seen_ids or idi not in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} :
                continue
            else:
                seen_ids.append(idi)

            lm_tvecs = tvecs[ids==idi].T
            lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

            #print(robot_state[0])
            #set covariance to zero of first landmarks
            if robot_state[0] == 0:
                #inital position
                lm_measurement = measure.Marker(lm_bff2d, idi, covariance=0)
            else:                 
                lm_measurement = measure.Marker(lm_bff2d, idi)

            measurements.append(lm_measurement)
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
