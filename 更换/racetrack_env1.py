from itertools import repeat, product
from typing import Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle


class RacetrackEnv1(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 2,
            "other_vehicles": 4,
            "controlled_vehicles": 1
            
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering_reward = 1/(1+self.config["lane_centering_cost"]*lateral**2)
        action_reward = self.config["action_reward"]*np.linalg.norm(action)
        reward = lane_centering_reward \
            + action_reward \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["collision_reward"] * (not(self.vehicle.on_road))
        reward = reward if self.vehicle.on_road else self.config["collision_reward"]
        return utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.steps >= self.config["duration"] or not(self.vehicle.on_road)

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        net.add_lane("a", "b", StraightLane([0, 0], [2000, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=4, speed_limit=speedlimits[2]))
        # smz:absolute axis
        net.add_lane("a", "b", StraightLane([0, 4], [2000, 4], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=4, speed_limit=speedlimits[1]))
        road = Road(network=net, record_history=self.config["show_trajectories"])
        #修改
        self.road = road
        # self.road1 = Road(network=net, record_history=self.config["show_trajectories"])
        #修改
    

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b",1) if i == 0 else \
                self.road.network.random_lane_index(rng)
                #smz: relative longitudinal
            vehicle = self.action_type.vehicle_class(self.road,(0,0),0, speed=11)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            # controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=13,
            #                                                                  longitudinal=0)

            # self.controlled_vehicles.append(controlled_vehicle)
            # self.road.vehicles.append(controlled_vehicle)

        longitudinal = 45
        gap = 30
        speed_other = 10
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 1),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 0),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 1),
                                              longitudinal=longitudinal-20,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 0),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 1),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 0),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 1),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 0),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 1),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 0),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)
        longitudinal += gap
        vehicle = IDMVehicle.make_on_lane(self.road, ("a", "b", 1),
                                              longitudinal=longitudinal,
                                          speed=speed_other)
        self.road.vehicles.append(vehicle)

register(
    id='racetrack-v1',
    entry_point='highway_env.envs:RacetrackEnv1',
)
