import json
import numpy as np
from copy import deepcopy
import argparse

def parse_slam_map(fname: str) -> dict:
    with open(fname, 'r') as fd:
        usr_dict = json.load(fd)
    aruco_dict = {}
    for (i, tag) in enumerate(usr_dict['taglist']):
        aruco_dict[tag] = np.reshape([usr_dict['map'][0][i], usr_dict['map'][1][i]], (2, 1))
    
    #print(f'Estimated marker poses: {aruco_dict}')
    
    return aruco_dict

def parse_object_map(fname):
    with open(fname, 'r') as fd:
        usr_dict = json.load(fd)
    target_dict = {}

    for key in usr_dict:
        object_type = key.split('_')[0]
        if object_type not in target_dict:
            target_dict[object_type] = np.array([[usr_dict[key]['x'], usr_dict[key]['y']]])
        else:
            target_dict[object_type] = np.append(target_dict[object_type], [[usr_dict[key]['x'], usr_dict[key]['y']]], axis=0)

    #print(f'Estimated target poses: {target_dict}')
                    
    return target_dict

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Matching the estimated map and the true map')
    parser.add_argument('--true-map', type=str, default='true_map.txt')
    parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt')
    parser.add_argument('--target-est', type=str, default='lab_output/targets.txt')
    parser.add_argument('--slam-only', action='store_true')
    parser.add_argument('--target-only', action='store_true')
    args, _ = parser.parse_known_args()
    
    aruco_est = parse_slam_map(args.slam_est)
    objects_est = parse_object_map(args.target_est)

    output = {}
    for i in range(1,11):
        # print(aruco_est[i])
        
        output[f"aruco{i}_0"] = {}
        output[f"aruco{i}_0"]["y"] = aruco_est[i][1][0]
        output[f"aruco{i}_0"]["x"] = aruco_est[i][0][0]

    # print(objects_est)

    for fruit in objects_est:
        # print(fruit)
        # print(objects_est[fruit])
        # print(len(objects_est[fruit]))
        
        for i in range(len(objects_est[fruit])):
            key = f"{fruit}_{i}"
            output[key] = {}
            output[key]["y"] = objects_est[fruit][i][1]
            output[key]["x"] = objects_est[fruit][i][0]

    # print(output)
    json_object = json.dumps(output, indent=4)

    with open("m3_map.txt", "w") as outfile:
        outfile.write(json_object)
