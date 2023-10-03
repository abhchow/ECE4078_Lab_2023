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

	if min_value > 0:
		min_value *= -1

	angle = (rad_angle + max_value) % (2 * np.pi) + min_value

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
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
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
    with open('M4_Lab1_shopping_list.txt', 'r') as fd:
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
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


    
    # operate.pibot.set_velocity(command, 0, 0, time=drive_time)   
    # ####################################################
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

def get_robot_pose(wheel_vel_lin, wheel_vel_rot, dt):
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


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_Lab1_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    #copy over from operate, did not copy save/play data
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    args, _ = parser.parse_known_args()

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    #currently no visuals added (see pygame in operate.py)
    fruits_true_pos_tuple = [tuple(array) for array in fruits_true_pos]  
    aruco_true_pos_tuple = [tuple(array) for array in aruco_true_pos]
    obstacles_tuple = fruits_true_pos_tuple+aruco_true_pos_tuple
    n_iter=100 #make sure not too long
    radius=0.13 #for clearance of obsticals
    stepSize= 0.5 #need large stepsize
    bounds = (-1.4, 1.4, -1.4, 1.4)
    goal_radius = 0.45

    wheel_vel_lin=30
    wheel_vel_rot=30
    dt=0.1

    plot_tree = False
    # The following is only a skeleton code for semi-auto navigation

    operate = Operate(args)
    operate.ekf_on = True

    #start_pos always at origin
    startpos=(0,0)

    # while True:
    # enter the waypoints
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
        robot_pose = get_robot_pose(wheel_vel_lin, wheel_vel_rot, dt)

        #add pathplanning
        # goalpos = (1.39, 1.39)
        startpos = (robot_pose[0], robot_pose[1])

        rrt_star_graph = rrt.RRT_star(startpos, endpos, obstacles_current, n_iter, radius, stepSize, bounds, goal_radius) #map_copy instead of obstacles_tuple

        if plot_tree:
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

        if rrt_star_graph.success:
            shortest_path= rrt.dijkstra(rrt_star_graph)            
            if plot_tree:
                for (x0,y0), (x1,y1) in zip(shortest_path[:-1], shortest_path[1:]):
                    ax.plot((x0,x1), (y0,y1),'b-')
                    ax.plot(x0,y0,'bo')
        if plot_tree:
            plt.show()

        # robot drives to the waypoint
        #waypoint =                 
        # waypoint = endpos
        for i in range(len(shortest_path)):
            if i==0:
                continue
            waypoint = list(shortest_path[i])
            robot_pose = get_robot_pose(wheel_vel_lin, wheel_vel_rot, dt)
            drive_to_point(waypoint,robot_pose, 0.1)
            print(f"x: {robot_pose[0]}, y: {robot_pose[1]}")
        
        # robot_pose = get_robot_pose(wheel_vel_lin, wheel_vel_rot, dt)
        # print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        # startpos=(robot_pose[0], robot_pose[1])
        #drive to a waypoint test with manual input  
        # initial_state = get_robot_pose(operate)
        # print("Initial robot state: {}", initial_state)
        # drive_to_point((x,y), (0,0,3*np.pi/4))
        # #update robot position 
        # final_state = get_robot_pose(operate)
        # print("Final robot state: {}", final_state)
        # operate.update_slam()


        # #drive to each waypoint 
        # for i in len(shortest_path):
        #     # robot drives to the waypoint
        #     waypoint = shortest_path[i]
        #     drive_to_point(waypoint,robot_pose)
        #     #robot_pose = get_robot_pose()
        #     print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))       


        # #robot_pose = operate.ekf.get_state_vector()
        #     for (x0,y0), (x1,y1), (x2,y2) in zip(shortest_path[:-1], shortest_path[1:]):
        #         #drive to point via controller
        #         theta0=np.arctan(y1-y0,x1-x0)
        #         theta1=np.arctan(y2-y1,x2-x1)
        #         waypoint=[x0,y0,theta0]
        #         goal_position=[x1,y1,theta1]
        #         #controller(way_point,goal_position)
        #         ax.plot((x0,x1), (y0,y1),'b-')
        #         ax.plot(x0,y0,'bo')
        # plt.show()
            
        #     for i in range(len(path)):
        #         # robot drives to the waypoint
        #         waypoint = path[i]
        #         drive_to_point(waypoint,robot_pose)
        #         print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))              
        # else:
        #     plt.plot(rrt_star_graph, obstacles, radius)


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