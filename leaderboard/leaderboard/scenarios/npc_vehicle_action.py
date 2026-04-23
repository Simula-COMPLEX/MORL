import math
import random
import sys

import carla

sys.path.insert(0, '/MORL/scenario_runner')

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class NPCVehicle(object):

    def __init__(self, world, traffic_manager, ego_vehicle, npc_location_x, npc_location_y, npc_vehicle_1=None, fixed_flag=False):
        self.world = world
        self.traffic_manager = traffic_manager
        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle = ego_vehicle
        self.npc_vehicle_1 = npc_vehicle_1
        self.npc_location_x = npc_location_x
        self.npc_location_y = npc_location_y
        self.npc_vehicle = None
        self.npc_transform = None
        self.generate_npc(fixed_flag)
        self.control_flag = False

    def generate_npc(self, fixed_flag):
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        npc_vehicle = None
        transform = None
        while npc_vehicle is None:
            if not fixed_flag:
                npc_location_y = random.choice([0.0, -3.5, 3.5, -7.0, 7.0])
                if abs(npc_location_y - 0.0) < 0.05:
                    npc_location_x = random.uniform(10.0, 20.0) * random.choice([1, -1])
                else:
                    npc_location_x = random.uniform(0.0, 20.0) * random.choice([1, -1])
            else:
                npc_location_x = self.npc_location_x
                npc_location_y = self.npc_location_y
            print(f"npc_location_x {npc_location_x} npc_location_y {npc_location_y}")
            waypoint = carla.Transform(ego_location +
                                       npc_location_x * ego_transform.get_forward_vector() +
                                       npc_location_y * ego_transform.get_right_vector(),
                                       ego_rotation)
            npc_waypoint = CarlaDataProvider.get_world().get_map().get_waypoint(waypoint.location, project_to_road=True,
                                                                                lane_type=carla.LaneType.Driving)
            transform = npc_waypoint.transform
            if self.npc_vehicle_1 is not None:
                npc_vehicle_1_location = self.npc_vehicle_1.npc_transform.location
                if abs(transform.location.y - npc_vehicle_1_location.y) < 0.05 and \
                        abs(transform.location.x - npc_vehicle_1_location.x) < 10.0:
                    continue
            npc_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', transform, rolename='npc_vehicle',
                                                              autopilot=True,
                                                              random_location=False, color=None,
                                                              actor_category='car')
        self.npc_transform = transform
        self.npc_vehicle = npc_vehicle

    def exe_action(self, action):
        # 0 driving on the lane
        # 1 change lane right
        # 2 change lane left
        # 3 speed up
        # 4 slow down
        # 5 emergency brake
        if self.control_flag:
            return
        if action == 0:
            self.traffic_manager.auto_lane_change(self.npc_vehicle, False)
            self.control_flag = True
        elif action == 1:
            self.traffic_manager.auto_lane_change(self.npc_vehicle, True)
            npc_waypoint = CarlaDataProvider.get_world().get_map().get_waypoint(
                CarlaDataProvider.get_location(self.npc_vehicle),
                project_to_road=True,
                lane_type=(carla.LaneType.Driving))
            waypoint_right = npc_waypoint.get_right_lane()
            if waypoint_right is not None and waypoint_right.lane_type == carla.LaneType.Driving:
                velocity = self.npc_vehicle.get_velocity()
                speed = self.calculate_speed(velocity)
                # print(f"force_lane_change speed {speed}")
                if speed > 7:
                    control = self.npc_vehicle.get_control()
                    control.throttle = 0.0
                    control.brake = 0.5
                    self.npc_vehicle.apply_control(control)
                else:
                    self.traffic_manager.force_lane_change(self.npc_vehicle, True)
                    self.control_flag = True
        elif action == 2:
            self.traffic_manager.auto_lane_change(self.npc_vehicle, True)
            npc_waypoint = CarlaDataProvider.get_world().get_map().get_waypoint(
                CarlaDataProvider.get_location(self.npc_vehicle),
                project_to_road=True,
                lane_type=(carla.LaneType.Driving))
            waypoint_left = npc_waypoint.get_left_lane()
            if waypoint_left is not None and waypoint_left.lane_type == carla.LaneType.Driving:
                velocity = self.npc_vehicle.get_velocity()
                speed = self.calculate_speed(velocity)
                # print(f"force_lane_change speed {speed}")
                if speed > 7:
                    control = self.npc_vehicle.get_control()
                    control.throttle = 0.0
                    control.brake = 0.5
                    self.npc_vehicle.apply_control(control)
                else:
                    self.traffic_manager.force_lane_change(self.npc_vehicle, False)
                    self.control_flag = True
        else:
            self.traffic_manager.auto_lane_change(self.npc_vehicle, True)
            control = self.npc_vehicle.get_control()
            # print("control.throttle", control.throttle, "control.brake", control.brake)
            if action == 3:
                initial_yaw = self.npc_transform.rotation.yaw % 360
                current_yaw = self.npc_vehicle.get_transform().rotation.yaw % 360
                angle = abs(initial_yaw - current_yaw) % 360
                # print(f"initial_yaw {initial_yaw} current_yaw {current_yaw} angle {angle}")
                if angle < 5 or angle > (360 - 5):
                    control.brake = 0.0
                    if control.throttle <= 0.65:
                        control.throttle += 0.05
                    else:
                        control.throttle = 0.7
                    self.npc_vehicle.apply_control(control)
            elif action == 4:
                control.brake = 0.2
                self.npc_vehicle.apply_control(control)
            elif action == 5:
                control.throttle = 0.0
                control.brake = 1.0
                self.npc_vehicle.apply_control(control)

    def set_auto_lane_change(self):
        self.traffic_manager.auto_lane_change(self.npc_vehicle, True)

    def set_simulate_physics(self, state):
        self.npc_vehicle.set_simulate_physics(state)

    def calculate_speed(self, velocity):
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2)
