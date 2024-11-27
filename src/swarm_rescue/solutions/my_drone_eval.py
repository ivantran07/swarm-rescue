import random
# import math
from typing import Optional, List, Type
from enum import Enum
import numpy as np

from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean

import os
import sys

import cv2

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.utils.grid import Grid

from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_abstract import DroneAbstract


class SemanticGrid(Grid):
    def __init__(self,
                 size_area_world,
                 resolution: float,
                 semantic_sensor): #semantic sensor
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.semantic_sensor = semantic_sensor # Semantic sensor instead

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)


class OccupancyGrid(Grid):
    """Simple occupancy grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)


class MyDroneEval(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 0
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    # to calculate map
    OCCUPIED_CERTAINTY_THRESHOLD = 0.99 # probability of being occupied (99% occupied)
    FREE_CERTAINTY_THRESHOLD = 1 - OCCUPIED_CERTAINTY_THRESHOLD # probablity of not being occupied (99% free = 1% occupied)
    GRID_OCCUPIED_THRESHOLD = np.log(OCCUPIED_CERTAINTY_THRESHOLD) - np.log(1 - OCCUPIED_CERTAINTY_THRESHOLD) # using log-odds probability
    GRID_FREE_THRESHOLD = -GRID_OCCUPIED_THRESHOLD

    # to calculate utility
    NUM_REGIONS = 5

    # to calculate distance
    DISTANCE_THRESHOLD = 8
    LIDAR_RANGE = 40

    # to calculate speed
    MAX_SPEED = 1
    
    # Control parameters
    ROTATION_COEFF = 1 / np.pi
    FORWARD_COEFF = 1
    ANGULAR_THRESHOLD = np.pi / 180 * (10) # Angular error threshold (deg) for moving forward

    # to calculate next target point
    NUM_PARTITIONS = 10
    PARTITION_DIST_THRESHOLD =  3

    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)
        # The state is initialized to searching wounded person
        self.state = self.Activity.EXPLORING

        # Those values are used by the random control function
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False

        self.iteration: int = 0

        self.estimated_pose = Pose()

        resolution = 8
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
        
        self.target = None

        # meshgrid to evaluate utility of a possible target
        self.MESH_XX, self.MESH_YY = np.meshgrid(np.arange(self.grid.y_max_grid), np.arange(self.grid.x_max_grid))

    def define_message_for_all(self):
        """
        Sharing the map
        Sharing positions of wounded people
        Sharing if carrying a wounded person

        Test different messaging periods
        """
        pass

    def control(self):
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        found_wounded, found_rescue_center, command_semantic = (
            self.process_semantic_sensor())

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.EXPLORING or self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif (self.state is self.Activity.GRASPING_WOUNDED and
              self.base.grasper.grasped_entities):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        elif (self.state is self.Activity.GRASPING_WOUNDED and
              not found_wounded):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.SEARCHING_RESCUE_CENTER and
              found_rescue_center):
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
              not self.base.grasper.grasped_entities):
            self.state = self.Activity.EXPLORING

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
              not found_rescue_center):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        # print("state: {}, can_grasp: {}, grasped entities: {}"
        #       .format(self.state.name,
        #               self.base.grasper.can_grasp,
        #               self.base.grasper.grasped_entities))

        ##########
        # COMMANDS FOR EACH STATE
        # Searching randomly, but when a rescue center or wounded person is
        # detected, we use a special command
        ##########
        if self.state is self.Activity.EXPLORING:
            command = self.control_explore()
            command["grasper"] = 0

        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.control_explore()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_explore()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        # increment the iteration counter
        self.iteration += 1

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                       self.measured_compass_angle())
        # self.estimated_pose = Pose(np.asarray(self.true_position()),
        #                            self.true_angle())

        self.grid.update_grid(pose=self.estimated_pose)
        if self.iteration % 5 == 0:
            # self.grid.display(self.grid.grid,
            #                       self.estimated_pose,
            #                       title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                                  self.estimated_pose,
                                  title="zoomed occupancy grid")
                # pass
        # Update semantics on the map

        return command

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided

    def control_explore(self):
        """
        The Drone will move forward and turn for a random angle when an
        obstacle is hit
        """
        pos = self.grid._conv_world_to_grid(*self.measured_gps_position())
        # if self.iteration <= 75:
        #     return {"forward": 0,
        #             "rotation": 0.5}

        if self.target is None or np.linalg.norm(self.target - pos) < self.DISTANCE_THRESHOLD:
            occupancy_grid = self.grid.grid.copy()
            # occupancy_grid[occupancy_grid <= self.GRID_FREE_THRESHOLD] = 0
            # occupancy_grid[occupancy_grid >= self.GRID_OCCUPIED_THRESHOLD] = 1
            # np.savetxt('map_complete.txt', self.grid.grid.copy(), fmt='%.2f', delimiter=' ')
            # np.savetxt('map_occupied.txt', bin_grid >= self.GRID_OCCUPIED_THRESHOLD, fmt='%d', delimiter='')
            frontier = np.logical_and(occupancy_grid < 0, occupancy_grid > self.GRID_FREE_THRESHOLD)
            # np.savetxt('map_frontier.txt', frontier, fmt='%d', delimiter='')
            frontier_pos = np.column_stack(np.nonzero(frontier))
            # bin_grid[frontier] = 2
            # np.savetxt('map_frontier.txt', frontier, fmt='%d', delimiter='')
            centroids = self.partition_frontier(frontier_pos, self.NUM_PARTITIONS)
            
            # bin_grid = np.zeros(shape=occupancy_grid.shape)
            # bin_grid[frontier] = 1
            # centroids_indices = np.transpose(np.rint(centroids).astype(int))
            # bin_grid[*centroids_indices] = 2
            # bin_grid[obstacles] = 3
            # np.savetxt('map_centroids.txt', bin_grid, fmt='%d', delimiter='')
            obstacles = occupancy_grid >= self.GRID_OCCUPIED_THRESHOLD
            no_obstacles_in_path_to_target = np.array([self.verify_obstacles(obstacles, pos, centroid, 3) for centroid in centroids])
            # print(no_obstacles_in_path_to_target)
            if not np.any(no_obstacles_in_path_to_target):
                command = {"forward": 0,
                           "rotation": 0} 
                return command 
            filtered_centroids = np.array([centroids[i] for i in range(self.NUM_PARTITIONS) if no_obstacles_in_path_to_target[i]])
            # print(filtered_centroids.shape)
            # print(filtered_centroids)

            unexplored_points = occupancy_grid == 0
            points_in_range = np.array([np.logical_and((self.MESH_XX - point[0])**2 + (self.MESH_YY - point[1])**2 <= (self.LIDAR_RANGE - 5)**2, unexplored_points) for point in filtered_centroids])
            num_points_in_range = np.count_nonzero(points_in_range, axis=(1,2))
            # print()
            # print(num_points_in_range)

            self.target = filtered_centroids[np.argmax(num_points_in_range)]
            # print(f"pos: {pos} target: {self.target}")
            command = {"forward": 0,
                       "rotation": 0}
            return command

        command = self.move_to_target(pos, self.measured_compass_angle(), self.target)
        return command

    # implementation of k-means clustering
    def partition_frontier(self, frontier_pos, num_regions):
        # print(len(frontier_pos))
        centroid_indices = np.random.choice(len(frontier_pos), size=num_regions, replace=False)
        # print(centroid_indices)
        centroids = frontier_pos[centroid_indices]
        # print(centroids)
        go = True
        while go:
            dist_to_centroids = np.array([[np.linalg.norm(centroid - point) for centroid in centroids] for point in frontier_pos])
            closest_centroid = np.argmin(dist_to_centroids, axis=1)
            new_centroids = np.array([np.mean(frontier_pos[closest_centroid == i], axis=0) for i in range(num_regions)])
            # print(new_centroids)
            change_in_pos = np.linalg.norm(new_centroids - centroids, axis=1)
            if np.max(change_in_pos < self.PARTITION_DIST_THRESHOLD):
                go = False
            centroids = new_centroids.copy()
        return centroids

    # implementation of bresenham function (to get list of points in a line)
    def get_points_in_path(self, obstacles_grid, pos, target):
        x1, y1 = pos
        x2, y2 = target
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        # print(pos, target)

        while True:
            if obstacles_grid[x1, y1]:
                return False 
            # print((x1, y1))
            points.append((x1, y1))
            if np.linalg.norm([y2 - y1, x2 - x1]) < 2:
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x1 += sx
            if err2 < dx:
                err += dx
                y1 += sy
        return points
    
    def verify_obstacles(self, obstacles_grid, pos, target, thickness):
        line_points = self.get_points_in_path(obstacles_grid, pos, target)
        if not line_points:
            return line_points
        for (px, py) in line_points:
            for dx in range(-thickness // 2, thickness // 2 + 1):
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    if obstacles_grid[px + dx, py + dy]:
                        return False
        return True

    def determine_target(self, occupancy_grid, pos):
        pass

    def move_to_target(self, pos, orientation, target):
        # Distance to the target
        dist_to_target = np.linalg.norm(target - pos)
        
        # Signed angle (positive = anti-clockwise) of rotation between drone's orientation and target 
        target_vector = target - pos
        drone_orientation = orientation
        target_orientation_rel_to_drone = -np.arctan2(target_vector[1], target_vector[0])
        angle_to_target = target_orientation_rel_to_drone - drone_orientation

        # print(f"pos: {pos} target: {self.target} target_vector: {target_vector}")
        # print(f"theta_1 (drone): {drone_orientation*180/np.pi}, theta_2 (target rel to drone): {target_orientation_rel_to_drone*180/np.pi}, signed angle: {angle_to_target*180/np.pi}")

        # Compute actuator values
        forward = np.clip(self.FORWARD_COEFF * dist_to_target, -1, 1) if abs(angle_to_target) < self.ANGULAR_THRESHOLD else 0
        rotation = np.clip(self.ROTATION_COEFF * angle_to_target, -1, 1)

        command = {"forward": forward,
                   "rotation": rotation}
        return command

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move
        towards a wound person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.EXPLORING
            or self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                        not data.grasped): # And not grasped by another drone
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 30)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0.0
            command["rotation"] = -1.0

        return found_wounded, found_rescue_center, command
