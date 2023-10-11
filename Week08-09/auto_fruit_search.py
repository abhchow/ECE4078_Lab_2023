        # M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
from copy import deepcopy

# import SLAM components
#section was prev commented out
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
import slam.mapping_utils as mapping_utils #added

# ADDED: import operate components
from operate import Operate

#ADDED: rrt for pathplanning
import rrt
from random import random
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.patches import Circle
import matplotlib.animation as animation 
from collections import deque

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure
import pygame



def clamp_angle(rad_angle=0, min_value=-np.pi, max_value=np.pi):
	"""
	Restrict angle to the range [min, max]
	:param rad_angle: angle in radians
	:param min_value: min angle value
	:param max_value: max angle value
	"""
	# print(f"preclamped: {rad_angle}")
	if min_value > 0:
		min_value *= -1
	
	angle = (rad_angle + max_value) % (2 * np.pi) + min_value
	
	# print(f"clamped: {angle}")
	return angle


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = gt_dict[key]['x'] #REMOVED np.round
            y = gt_dict[key]['y']

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2]) #REMOVED [:-2] in append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
def drive_to_point(waypoint, robot_pose,dt):
    # ####################################################
    # # TODO: replace with your codes to make the robot drive to the waypoint
    # # One simple strategy is to first turn on the spot facing the waypoint,
    # # then drive straight to the way point
    
    #orient to waypoint 
    #calculate angle diff 
   
    angle_diff = robot_pose[2] - np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])
    wheel_vel_lin = 30 # tick/s
    wheel_vel_rot = 15 

    

    # scale in m/tick
    # baseline in m
    # angle_diff in rad 
    # wheel vel in tick /s
    # m * tick/m * s/tick = s
    turn_time = np.abs(angle_diff*operate.ekf.robot.wheels_width/(operate.ekf.robot.wheels_scale*wheel_vel_lin))
    #turn 
    if angle_diff > 0: 
        #turn left 
        command = [0,-1]
    else: 
        #turn right 
        command = [0,1]

    lv, rv = operate.pibot.set_velocity(command, 0, wheel_vel_rot, time=turn_time)
    operate.take_pic()
    drive_meas = measure.Drive(lv, -rv, turn_time)

    operate.update_slam(drive_meas)
    # curr_time = 0
    # while curr_time < turn_time:
    #     operate.pibot.set_velocity(command, wheel_vel_lin, wheel_vel_rot, time=dt)
    #     operate.take_pic()
    #     drive_meas = measure.Drive(wheel_vel_lin, wheel_vel_rot, dt)
    #     operate.update_slam(drive_meas)
    #     curr_time += dt

    # after turning, drive straight to the waypoint
    # wheel_vel_rot = 30 # tick/s
    # # scale in m/tick
    # # dist(waypoint-robot_pose) in m
    # # m  * tick/m * s/tick = s
    drive_time = np.linalg.norm(np.array(waypoint)-np.array(robot_pose[0:2]))/(operate.ekf.robot.wheels_scale*wheel_vel_lin) # replace with your calculation
    #drive straight forard 
    command = [1,0] 
    
    lv, rv = operate.pibot.set_velocity(command, wheel_vel_lin, 0, time=drive_time)    
    operate.take_pic()
    drive_meas = measure.Drive(lv, -rv, drive_time)
    
    operate.update_slam(drive_meas)
    
    # operate.pibot.set_velocity(command, 0, 0, time=drive_time)   
    # ####################################################
    return 

def angle_mav_from_waypoint(waypoint, robot_pose, angle_mov_ave):
     #angle_diff = get_angle_robot_to_goal(robot_pose.T,(np.array(waypoint)).T)
     angle_diff = clamp_angle(np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])-robot_pose[2])
     threshold=0.01
     if abs(angle_diff)<threshold or abs(angle_diff)> np.mean(abs(angle_mov_ave)):
         return True, angle_mov_ave
     else:
         angle_mov_ave = np.append(angle_mov_ave, angle_diff)
         angle_mov_ave = angle_mov_ave[1:]
         return False, angle_mov_ave

def distance_from_waypoint(waypoint, robot_pose, dist_min):
    #dist min= array of floats
    dist_from_waypoint = get_distance_robot_to_goal(robot_pose.T,(np.array(waypoint)).T)
    #need to change min dist to be array of updward trend
    threshold=0.01
    if dist_from_waypoint<threshold or dist_from_waypoint> np.mean(dist_min):
         return True, dist_min #distmin no loner used
    else:
         dist_min = np.append(dist_min, dist_from_waypoint)
         dist_min = dist_min[1:]
         #print(dist_from_waypoint) #moved
         return False, dist_min #dist min updates every iter, but should not on last
    
def update_command(drive_forward=False, drive_backward=False, turn_left=False, turn_right=False, stop=False):
    if drive_forward: 
        operate.command['motion'] = [-1,0]
    elif drive_backward: 
        operate.command['motion'] = [1,0]
    elif turn_left: 
        operate.command['motion'] = [0,1]
    elif turn_right: 
        operate.command['motion'] = [0,-1]
    elif stop:
        operate.command['motion'] = [0, 0]
    return
          
def controller(initial_state, goal_position):
    #states=[x,y,theta]
    ########add control
    K_pv = 0.5
    K_pw =10
    K_theta=1.5

    threshold_distance= 0.0005
    threshold_angle=0.05

    dt= time.time() - operate.control_clock
    
    distance_to_goal=get_distance_robot_to_goal(initial_state,goal_position)
    desired_heading=get_angle_robot_to_goal(initial_state, goal_position)
    angle_goal_diff=initial_state[2]-goal_position[2]

    while not stop_criteria_met: 
        
        wheel_vel_lin = K_pv*distance_to_goal
        wheel_vel_rot = K_pw*desired_heading+K_theta*angle_goal_diff
        
        # Apply control to robot
        #robot.drive(v_k, w_k, delta_time)
        drive_to_point(wheel_vel_lin, wheel_vel_rot,dt)
        new_state = get_robot_pose(wheel_vel_lin,wheel_vel_rot)
                    
        distance_to_goal= get_distance_robot_to_goal(new_state, goal_position)
        desired_heading = get_angle_robot_to_goal(new_state, goal_position)
        angle_goal_diff=new_state[2]-goal_position[2]
        
        #Check for stopping criteria -------------------------------------
        if (distance_to_goal < threshold_distance) and(angle_goal_diff < threshold_angle):
            stop_criteria_met = True

    #error=desired_pt- current_pt
    #control_sig=error*K_gain
    #x-->error-->controller-->plant-->output
    #return nothing, already called drive()

#2x controller helper functions
def get_distance_robot_to_goal(robot_state=np.zeros(3), goal=np.zeros(3)):
	"""
	Compute Euclidean distance between the robot and the goal location
	:param robot_state: 3D vector (x, y, theta) representing the current state of the robot
	:param goal: 3D Cartesian coordinates of goal location
	"""

	if goal.shape[0] < 3:
		goal = np.hstack((goal, np.array([0])))

	x_goal, y_goal,_ = goal
	x, y,_ = robot_state
	x_diff = x_goal - x
	y_diff = y_goal - y

	rho = np.hypot(x_diff, y_diff)

	return rho

def get_angle_robot_to_goal(robot_state=np.zeros(3), goal=np.zeros(3)):
	"""
	Compute angle to the goal relative to the heading of the robot.
	Angle is restricted to the [-pi, pi] interval
	:param robot_state: 3D vector (x, y, theta) representing the current state of the robot
	:param goal: 3D Cartesian coordinates of goal location
	"""
	if goal.shape[0] < 3:
		goal = np.hstack((goal, np.array([0])))

	x_goal, y_goal,_ = goal
	x, y, theta = robot_state
	x_diff = x_goal - x
	y_diff = y_goal - y


	alpha = clamp_angle(np.arctan2(y_diff, x_diff) - theta)


	return alpha

def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    
    #dt = time.time() - operate.control_clock
    #drive_meas = measure.Drive(lv, -rv, dt) #measure
    # lv, rv = operate.ekf.robot.convert_wheels_to_leftright(0, wheel_vel_rot)
    # drive_meas = measure.Drive(lv, rv, turn_time)

    #replace update_slam()
    #operate.take_pic()
    #lms, aruco_img = aruco.aruco_detector.detect_marker_positions(operate.img)
    #previously used EKF.function_name
    #operate.ekf.predict(drive_meas) #predict(raw_drive_meas), add_landmarks,update
    #operate.ekf.update(lms)
    #state= operate.ekf.get_state_vector()
    robot_pose=np.reshape(operate.ekf.robot.state, (3,))
    # print(robot_pose)
    #robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################
    return robot_pose


def update_command(drive_forward=False, drive_backward=False, turn_left=False, turn_right=False, stop=False):
    if drive_forward: 
        operate.command['motion'] = [1,0]
    elif drive_backward: 
        operate.command['motion'] = [-1,0]
    elif turn_left: 
        operate.command['motion'] = [0,1]
    elif turn_right: 
        operate.command['motion'] = [0,-1]
    elif stop:
        operate.command['motion'] = [0, 0]
    return 

def find_path(startpos, endpos, obstacles_current, n_iter, radius, stepSize, bounds, goal_radius):
    rrt_star_graph = rrt.RRT_star(startpos, endpos, obstacles_current, n_iter, radius, stepSize, bounds, goal_radius)
    #if cannot find a path, make obstacle radius iteratively smaller and try again 
    while rrt_star_graph.success == False: 
        radius = radius - 0.05
        if radius < 0: 
            sys.exit("ERROR: Could not find a path from here")
        rrt_star_graph = rrt.RRT_star(startpos, endpos, obstacles_current, n_iter, radius, stepSize, bounds, goal_radius)
    #do not include last element in shortest path as that is the endpos (not offset)
    shortest_path = (rrt.dijkstra(rrt_star_graph))[:-1]
    return rrt_star_graph, shortest_path

def marker_close(operate, distance_threshold=80):
    corners, ids, rejected = cv2.aruco.detectMarkers(operate.img, operate.aruco_det.aruco_dict, parameters=operate.aruco_det.aruco_params)
    if len(corners) != 0: 
        for box in corners[0]:
            corner0 = box[0]
            corner1 = box[1]
            corner2 = box[2]
            corner3 = box[3]
            # diff in y values of each corner
            y_diff1 = abs(corner1[1] - corner0[1])
            y_diff2 = abs(corner2[1] - corner1[1])
            y_diff3 = abs(corner3[1] - corner2[1])
            y_diff4 = abs(corner0[1] - corner3[1])
            if (y_diff1>distance_threshold or 
                y_diff2>distance_threshold or 
                y_diff3>distance_threshold or
                y_diff4>distance_threshold):
                #the markers are too close to the camera 
                return True 
            else: 
                return False

def print_path(rrt_star_graph, shortest_path, fig_name): 
    fig, ax = plt.subplots()
    for edge in rrt_star_graph.edges:
        v1 = rrt_star_graph.vertices[edge[0]]
        v0 = rrt_star_graph.vertices[edge[1]]
        ax.plot((v1[0], v0[0]), (v1[1], v0[1]), 'r-')
    for vertex in rrt_star_graph.vertices:
        if (vertex == startpos):
            ax.plot(vertex[0],vertex[1], 'ko') 
        else: 
            ax.plot(vertex[0],vertex[1], 'ro')
    goal_patch = Circle(endpos, goal_radius, color='g')
    ax.add_patch(goal_patch)
    for obstacle in obstacles_current: 
        obstacle_patch = Circle(obstacle, radius)
        ax.add_patch(obstacle_patch)
    ax.text(startpos[0], startpos[1], "startpos")
    ax.plot(endpos[0], endpos[1], "ko")
    ax.text(endpos[0], endpos[1], "endpos")
    #plot shortest path     
    for (x0,y0), (x1,y1) in zip(shortest_path[:-1], shortest_path[1:]):
        ax.plot((x0,x1), (y0,y1),'b-')
    for (x0,y0) in shortest_path:
        ax.plot(x0,y0,'bo')
        ax.text(x0, y0, f'({x0:.2f}, {y0:.2f})', ha='right', va='bottom', color='blue')
    plt.savefig(f"./graphs/rrt_graph_{str(fig_name)}.png")

def drive_to_target(operate, target): 
    #long winded way to get target values in case more fruits pop up in the detected box labels (even due to background)
    target_idx = [idx for idx, string in enumerate(operate.detected_box_labels) if target == string][0]
    target_x, _, _, target_height = operate.detector_output[target_idx][1]
    print(target_height)
    print(target_x)
    #x and y is relative to the top left corner origin (0,0) of the image 
    #therefore the horizontal center of the image is half of its width (r=240,c=320)
    center_x = operate.img.shape[1]/2
    print(center_x)
    #define a threshold in which the target is close enough to the center 
    tolerance_factor = 5
    #threshold is proportional to target height. if far, small threshod. if close, large threshold
    threshold = target_height/tolerance_factor
    #repeat movement while the target is away from the center. 
    while abs(target_x - center_x)>threshold:
         #if the target x is on the right side of the image, turn right. 
        if target_x > center_x: 
            update_command(turn_right=True)
        #else if the target is on the left side of the image, turn left. 
        else: 
            update_command(turn_left=True)
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.detect_target()
        #recalculate the coords of the targets
        target_idx = [idx for idx, string in enumerate(operate.detected_box_labels) if target == string][0]
        target_x, _, _, target_height = operate.detector_output[target_idx][1]
    #after centering on the target, move towards it until a certain height is met OR an aruco marker becomes too close. 
    # Currently a hardcoded value and does not account for the different heights of the fruit  
    while target_height < 80 and not marker_close(operate): 
        update_command(drive_forward=True)
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas) 
        operate.detect_target()
        if len(operate.lms) == 0: 
            print("lost sight of target")
            break 
        target_idx = [idx for idx, string in enumerate(operate.detected_box_labels) if target == string][0]
        _, _, _, target_height = operate.detector_output[target_idx][1]
        print(f"target_height = {target_height}")
    
    #stop the driving 
    update_command(stop=True)
    drive_meas = operate.control()
    operate.update_slam(drive_meas)
    
    #if the robot made it to the marker 
    if target_height >= 80: 
        return True
    else: 
        print("failed to drive towards target")
        return False


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='party_room_map.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    #copy over from operate, did not copy save/play data
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt') #USING KMART YOLO MODEL 
    args, _ = parser.parse_known_args()

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    #currently no visuals added (see pygame in operate.py)
    fruits_true_pos_tuple = [tuple(array) for array in fruits_true_pos]  
    aruco_true_pos_tuple = [tuple(array) for array in aruco_true_pos]
    obstacles_tuple = fruits_true_pos_tuple+aruco_true_pos_tuple
    n_iter=300 #make sure not too long
    radius=0.30 #for clearance of obsticals. Large radius because penguin Pi itself is large, 
    stepSize= 0.7 #need large stepsize
    bounds = (-1.4, 1.4, -1.4, 1.4)
    goal_radius = 0.20 #marginally less than 0.5m so that robot is fully within the goal. 


    plot_tree = True
    # The following is only a skeleton code for semi-auto navigation


    #start_pos always at origin
    # startpos=(0,0)


    operate = Operate(args) # this is in the wrong place - move it later
    operate.ekf_on = True
    robot_pose = np.array([0,0,0])
    #run YOLO object detector (previously initiated with press of a key button)
    operate.command['inference'] = True 

    measurements = [None]*len(aruco_true_pos)
    for i in range(len(aruco_true_pos)):
        measurements[i] = measure.Marker([[aruco_true_pos[i][0]], [aruco_true_pos[i][1]]], i+1)
    
    operate.ekf.add_landmarks(measurements)

    # operate.ekf.add_landmarks()
    #loop to extract current shopping item
    for shop_item in iter(search_list):
        print("\n ------------------------------------------")
        obstacles_current=deepcopy(obstacles_tuple)
            #search through fruitsmap
        for i in range(len(fruits_list)):
            if fruits_list[i]==shop_item: #have found item we need to shop for
                endpos=fruits_true_pos_tuple[i]
                #remove end_pos from map_copy
                # obstacles_current.pop(i)


        # estimate the robot's pose
        robot_pose = get_robot_pose()

        startpos = (robot_pose[0], robot_pose[1])

        rrt_star_graph, shortest_path = find_path(startpos, endpos, obstacles_current, n_iter, radius, stepSize, bounds, goal_radius)     
        print_path(rrt_star_graph, shortest_path, f"original path to {shop_item}")  #KEEP THIS COMMENTED FOR BEST ACCURACY    

        operate.control_clock = time.time()

        goal_arrived = False
        while goal_arrived == False: 
            
            #step through waypoint in shortest_path 
            for waypoint in shortest_path[1:]:
                angle_arrived = False
                angle_mav=np.ones((1,5))*1e3

                while not angle_arrived:    
                    operate.take_pic()
                    angle_diff = clamp_angle(np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])-robot_pose[2])
                    if angle_diff>0: 
                        #turn left
                        update_command(turn_left=True) #fix mb
                    else:
                        #turn right
                        update_command(turn_right=True)
                    drive_meas = operate.control() 
                    operate.update_slam(drive_meas)
                    robot_pose=get_robot_pose()
                    [angle_arrived, angle_mav]=angle_mav_from_waypoint(waypoint, robot_pose, angle_mav) #returns boolean    
                    #print(f'robot pos is {robot_pose[0],robot_pose[1], clamp_angle(robot_pose[2])*180/np.pi} --- Turning')
                print("Arrived at Angle")


                dist_min=np.ones((1,5))*1e3 #very small number init- check postive upward trend, use for moving average
                waypoint_arrived = False
                target_in_sight_counter = 0 
                while not waypoint_arrived and not marker_close(operate):
                    operate.take_pic()
                    update_command(drive_forward=True)
                    drive_meas = operate.control()
                    operate.update_slam(drive_meas)
                    #scan for fruit while driving forward
                    operate.detect_target() 
                    robot_pose=get_robot_pose()
                    [waypoint_arrived,dist_min]=distance_from_waypoint(waypoint, robot_pose, dist_min)
                    #if a target item is seen in the detected fruitss
                    # if shop_item in operate.detected_box_labels: 
                    #     print(f"Target {shop_item} in sight")
                    #     target_in_sight_counter += 1
                    #     #if the target is reliably in sight
                    #     if target_in_sight_counter > 3:
                    #         #stop and drive to target 
                    #         update_command(stop=True)
                    #         drive_meas = operate.control()
                    #         operate.update_slam(drive_meas)
                    #         print(f"Driving towards target")
                    #         drive_to_target_success = drive_to_target(operate, shop_item)
                    #         if drive_to_target_success: 
                    #             print(f"Made it to target")
                    #             waypoint_arrived = True 
                    #print(f'robot pos is {robot_pose[0],robot_pose[1], clamp_angle(robot_pose[2])*180/np.pi} --- Driving Fwd')

                #if the robot is too close to a marker, stop and find a new shortest path  
                if marker_close(operate): 
                    update_command(stop=True)
                    drive_meas = operate.control()
                    operate.update_slam(drive_meas) 
                    print("Finding a new shortest path...")
                    robot_pose = get_robot_pose()
                    startpos = (robot_pose[0], robot_pose[1])
                    rrt_star_graph, shortest_path  = find_path(startpos, endpos, obstacles_current, n_iter, radius, stepSize, bounds, goal_radius)
                    print_path(rrt_star_graph, shortest_path, f"New path to {shop_item}")
                    #start the for loop again with the new shortest_path
                    break 

                #if arrived at a waypoint, stop.  
                if waypoint_arrived: 
                    update_command(stop=True)
                    drive_meas = operate.control() 
                    operate.update_slam(drive_meas)
                    print(f"Arrived at Waypoint: {waypoint}\n    with current pose {get_robot_pose()}")                    
                    #if the current waypoint is also the final waypoint 
                    if waypoint == shortest_path[-1]: 
                        #setting this to true will cancel the while loop and move onto the next shop_item
                        goal_arrived = True
                        print("--------------------------------------\n")
                        print(f"Arrived at {shop_item}")
                        print("--------------------------------------\n")
                        # #relocalise the robot after arriving at a shop item   
                        # print(f"Relocalising...")
                        # turn_angle = 2*np.pi
                        # original_angle = robot_pose[2]
                        # while angle_diff < turn_angle: 
                        #     operate.take_pic()
                        #     update_command(turn_left=True) 
                        #     drive_meas = operate.control()
                        #     operate.update_slam(drive_meas)
                        #     robot_pose = get_robot_pose()
                        #     current_angle = robot_pose[2] 
                        #     angle_diff = abs(original_angle-current_angle)
                        # print("Finishing localising.")
                        # print("--------------------------------------\n")
    

        #after reaching endpoint should confirm target with YOLO
        #operate.Operate.detect_target() #only used w YOLO
        #self.detector_output= list of lists, box info [label,[x,y,width,height]] for all detected targets in image
        #once detect, no need to append to map, even if low certainty, so that path-planning will avoid it
        #add to fruit list, friut true pos (used in planning)

        # exit
        operate.pibot.set_velocity([0, 0])
        # uInput = input("Add a new waypoint? [Y/N]")
        # if uInput == 'N':
        #     break

    dummy = 1