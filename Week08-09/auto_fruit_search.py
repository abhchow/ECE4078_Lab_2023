        # M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

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
def drive_to_point(waypoint, robot_pose):
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
    turn_time = angle_diff*operate.ekf.robot.wheels_width/(operate.ekf.robot.wheels_scale*wheel_vel_lin)
    #turn 
    if angle_diff > 0: 
        #turn left 
        command = [0,-1]
    else: 
        #turn right 
        command = [0,1]
    print("Turning for {:.2f} seconds".format(turn_time))
    operate.pibot.set_velocity(command, wheel_vel_lin, wheel_vel_rot, time=turn_time)
    # after turning, drive straight to the waypoint
    # wheel_vel_rot = 30 # tick/s
    # # scale in m/tick
    # # dist(waypoint-robot_pose) in m
    # # m  * tick/m * s/tick = s
    drive_time = np.linalg.norm(np.array(waypoint)-np.array(robot_pose[0:2]))/(operate.ekf.robot.wheels_scale*wheel_vel_lin) # replace with your calculation
    #drive straight forard 
    command = [1,0] 
    print("Driving for {:.2f} seconds".format(drive_time))
    operate.pibot.set_velocity(command, wheel_vel_lin, wheel_vel_rot, time=drive_time)    
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    # ####################################################
    return 


def get_robot_pose(operate):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    drive_meas = operate.control()
    operate.update_slam(drive_meas)
    #operate.ekf.predict(drive_meas)
    #operate.ekf.update()
    # update the robot pose [x,y,theta]
    robot_pose = operate.ekf.robot.state
    #robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################
    return robot_pose


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
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
    radius=0.30 #for clearance of obsticals
    stepSize= 0.5 #need large stepsize
    # The following is only a skeleton code for semi-auto navigation

    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        operate = Operate(args)

        #add pathplanning
        endpos = (1.5,1.5)
        startpos = (-1.5,-1.5)

        rrt_star_graph = rrt.RRT_star(startpos, endpos, obstacles_tuple, n_iter, radius, stepSize)

        # fig, ax = plt.subplots()
        # for edge in rrt_star_graph.edges:
        #     v1 = rrt_star_graph.vertices[edge[0]]
        #     v0 = rrt_star_graph.vertices[edge[1]]
        #     ax.plot((v1[0], v0[0]), (v1[1], v0[1]), 'r-')
        # for vertex in rrt_star_graph.vertices:
        #     if (vertex == startpos):
        #         ax.plot(vertex[0],vertex[1], 'ko') 
        #     else: 
        #         ax.plot(vertex[0],vertex[1], 'ro')
        # for obstacle in obstacles_tuple: 
        #     obstacle_patch = Circle(obstacle, radius)
        #     ax.add_patch(obstacle_patch)
        # ax.text(startpos[0], startpos[1], "startpos")
        # ax.plot(endpos[0], endpos[1], "ko")
        # ax.text(endpos[0], endpos[1], "endpos")

        if rrt_star_graph.success:
            shortest_path= rrt.dijkstra(rrt_star_graph)            
        #     for (x0,y0), (x1,y1) in zip(shortest_path[:-1], shortest_path[1:]):
        #         ax.plot((x0,x1), (y0,y1),'b-')
        #         ax.plot(x0,y0,'bo')
        # plt.show()


        #drive to a waypoint test with manual input  
        initial_state = get_robot_pose(operate)
        print("Initial robot state: {}", initial_state)
        drive_to_point((x,y), (0,0,3*np.pi/4))
        #update robot position 
        final_state = get_robot_pose(operate)
        print("Final robot state: {}", final_state)


        # #drive to each waypoint 
        # for i in len(shortest_path):
        #     # robot drives to the waypoint
        #     waypoint = shortest_path[i]
        #     drive_to_point(waypoint,robot_pose)
        #     #robot_pose = get_robot_pose()
        #     print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))       


        #robot_pose = operate.ekf.get_state_vector()
            
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
        # pi.set_velocity([0, 0])
        # uInput = input("Add a new waypoint? [Y/N]")
        # if uInput == 'N':
        #     break
