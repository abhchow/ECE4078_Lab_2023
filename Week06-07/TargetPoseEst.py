# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2
from YOLO.detector import Detector
from sklearn.cluster import KMeans


# list of target fruits and vegs types
# Make sure the names are the same as the ones used in your YOLO model
TARGET_TYPES = ['orange', 'lemon', 'lime', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']
#TARGET_TYPES = ['orange', 'apple', 'kiwi', 'banana', 'pear', 'melon', 'potato']

def estimate_pose(camera_matrix, obj_info, robot_pose):
    """
    function:
        estimate the pose of a target based on size and location of its bounding box and the corresponding robot pose
    input:
        camera_matrix: list, the intrinsic matrix computed from camera calibration (read from 'param/intrinsic.txt')
            |f_x, s,   c_x|
            |0,   f_y, c_y|
            |0,   0,   1  |
            (f_x, f_y): focal length in pixels
            (c_x, c_y): optical centre in pixels
            s: skew coefficient (should be 0 for PenguinPi)
        obj_info: list, an individual bounding box in an image (generated by get_bounding_box, [label,[x,y,width,height]])
        robot_pose: list, pose of robot corresponding to the image (read from 'lab_output/images.txt', [x,y,theta])
    output:
        target_pose: dict, prediction of target pose
    """
    # read in camera matrix (from camera calibration results)
    focal_length = camera_matrix[0][0]

    # there are 8 possible types of fruits and vegs
    ######### Replace with your codes #########
    # TODO: measure actual sizes of targets [width, depth, height] and update the dictionary of true target dimensions
    target_dimensions_dict = {'orange': [1.0,1.0,0.073], 'lemon': [1.0,1.0,0.041], 
                              'lime': [1.0,1.0,0.052], 'tomato': [1.0,1.0,0.07], 
                              'capsicum': [1.0,1.0,0.097], 'potato': [1.0,1.0,0.062], 
                              'pumpkin': [1.0,1.0,0.08], 'garlic': [1.0,1.0,0.075]}
    # target_dimensions_dict = {'orange': [0.05,0.05,0.05], 'apple': [1.0,1.0,0.05], 
    #                           'kiwi': [1.0,1.0,0.047], 'banana': [1.0,1.0,0.047], 
    #                           'pear': [1.0,1.0,0.075], 'melon': [1.0,1.0,0.055], 
    #                           'potato': [1.0,1.0,0.04]}
    
    #########

    # estimate target pose using bounding box and robot pose
    target_class = obj_info[0]     # get predicted target label of the box
    target_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
    true_height = target_dimensions_dict[target_class][2]   # look up true height of by class label

    # compute pose of the target based on bounding box info, true object height, and robot's pose
    pixel_height = target_box[3]
    pixel_center = target_box[0]
    distance = true_height/pixel_height * focal_length  # estimated distance between the robot and the centre of the image plane based on height
    # training image size 320x240p
    image_width = 320 # change this if your training image is in a different size (check details of pred_0.png taken by your robot)
    x_shift = image_width/2 - pixel_center              # x distance between bounding box centre and centreline in camera view
    theta = np.arctan(x_shift/focal_length)     # angle of object relative to the robot
    ang = theta + robot_pose[2]     # angle of object in the world frame
    
   # relative object location
    distance_obj = distance/np.cos(theta) # relative distance between robot and object
    # if distance_obj <0.7: #object #within certain metres aways
    x_relative = distance_obj * np.cos(theta) # relative x pose
    y_relative = distance_obj * np.sin(theta) # relative y pose
    relative_pose = {'x': x_relative, 'y': y_relative}
    #print(f'relative_pose: {relative_pose}')

    # location of object in the world frame using rotation matrix
    delta_x_world = x_relative * np.cos(robot_pose[2]) - y_relative * np.sin(robot_pose[2])
    delta_y_world = x_relative * np.sin(robot_pose[2]) + y_relative * np.cos(robot_pose[2])
    # add robot pose with delta target pose
    target_pose = {'y': (robot_pose[1]+delta_y_world)[0],
                'x': (robot_pose[0]+delta_x_world)[0]}
        #print(f'delta_x_world: {delta_x_world}, delta_y_world: {delta_y_world}')
    # else: 
        # pass
    # print(f'target_pose: {target_pose}')
    

    return target_pose


def merge_estimations(target_pose_dict):
    """
    function:
        merge estimations of the same target
    input:
        target_pose_dict: dict, generated by estimate_pose
    output:
        target_est: dict, target pose estimations after merging
    """
    target_est = {}

    ######### Replace with your codes #########
    # TODO: replace it with a solution to merge the multiple occurrences of the same class type (e.g., by a distance threshold)

    #target_pose_dict is just #will have {"orange_0": {"y": num,"x": num}, "orange_1":...}
    #replace prev func w k-mean
    data_array=[]
    data_headings=[]

    for index, (key, value) in enumerate(target_pose_dict.items()):
        head, sep, tail = key.partition('_')
        key=head
        data_array.append([value['x'],value['y']])
        data_headings.append(head)

    data_array=np.array(data_array)
    kmeans=KMeans(n_clusters=10,random_state=0,n_init="auto").fit(data_array)
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)
    # plt.figure()
    # plt.scatter(data_array[:,0],data_array[:,1])
    # plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1])

    current_target=[]
    #need to equate cluster index 0 with its position
    for i in range(len(data_headings)):
        temp_heading=data_headings[i]
        cluster_num=kmeans.labels_[i]
        cluster_centre=kmeans.cluster_centers_[cluster_num]
        #target est update will update repeated clusters, need to add new for repeated fruits
        if temp_heading in current_target:#true, string overlap
            #check if not same cluster
            if (cluster_centre[0] != target_est[temp_heading+'_'+str(0)]['x']) and (cluster_centre[1] !=target_est[temp_heading+'_'+str(0)]['y']):
                print(cluster_centre)
                #assumes max 2 of each fruit
                target_est.update({temp_heading+'_'+str(1):{'y':cluster_centre[1],'x':cluster_centre[0]}})
            else:
                pass#same cluster 
        else:
            target_est.update({temp_heading+'_'+str(0):{'y':cluster_centre[1],'x':cluster_centre[0]}})
        #update current_target list at end
        current_target=[]
        for index, (key, value) in enumerate(target_est.items()): #loop target_est to check if exist
            head, sep, tail = key.partition('_')
            current_target.append(head)
    #########
   
    return target_est


# main loop
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))     # get current script directory (TargetPoseEst.py)

    # read in camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # init YOLO model
    model_path = f'{script_dir}/YOLO/model/yolov8_model.pt'
    yolo = Detector(model_path)

    # create a dictionary of all the saved images with their corresponding robot pose
    image_poses = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # estimate pose of targets in each image
    target_pose_dict = {}
    detected_type_list = []
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)
        bounding_boxes, bbox_img, _ = yolo.detect_single_image(input_image)
        # cv2.imshow('bbox', bbox_img)
        # cv2.waitKey(0)
        robot_pose = image_poses[image_path]

        for detection in bounding_boxes:
            # count the occurrence of each target type
            occurrence = detected_type_list.count(detection[0])
            target_pose_dict[f'{detection[0]}_{occurrence}'] = estimate_pose(camera_matrix, detection, robot_pose)

            detected_type_list.append(detection[0])

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = {}
    target_est = merge_estimations(target_pose_dict)
    print(target_est)
    # save target pose estimations
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)

    print('Estimations saved!')
