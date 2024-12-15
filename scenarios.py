import abc
import traci
import pandas as pd

import constants
import utils

class Scenario(abc.ABC):
    def __init__(self, params : pd.Series):
        """
        The abstract class for the scenario module.
        The scenario takes @params which are generated from a ScenarioManager.
        """
        assert isinstance(params, pd.Series)
        self._params = params
        self._score = pd.Series({"default" : 0})
        return

    @property
    def params(self) -> pd.Series:
        """
        Input configuration for this scenario.
        """
        return self._params

    @property
    def score(self) -> pd.Series:
        """
        Scenario score.
        """
        return self._score


class DrivingScenario(Scenario):
    NO_TRAFFIC = 0
    VEHICLE_ON_LEFT_SIDE = 1
    TWO_VEHICLES = 2
    BICYCLE = 3
    PEDESTRIAN = 4
    PED_XING_COMPLETE = 5

    def __init__(self, params : pd.Series):
        traci.simulation.loadState(constants.sumo.init_state_file)

        self._score = pd.Series({
            "complete" : True
        })
        self._trace = []

        # for i in range(4):
        #     lid = "E0_%d" % i
        #     w = traci.lane.getWidth(lid)
        #     print(lid, w)

        self.add_actors()

        if constants.sumo.gui:
            traci.gui.setZoom(
                constants.sumo.default_view, 
                constants.sumo.dut_zoom
            )
            if constants.sumo.track_dut:
                traci.gui.trackVehicle(
                    constants.sumo.default_view, constants.DUT)

        self._cur_stage = 0
        self.stage_0_start()
        while traci.simulation.getMinExpectedNumber() > 0:
            if not constants.DUT in traci.vehicle.getIDList():
                break

            self.check_for_stage_transition()
            self.trace_actors()

            traci.simulationStep()
            continue
        self.stage_5_end()
        

        return
    
    @property
    def score(self) -> pd.Series:
        return self._score
    
    @property
    def params(self) -> pd.Series:
        return self._params

    @property
    def cur_stage(self) -> int:
        return self._cur_stage
    
    @property
    def trace(self) -> list[pd.Series]:
        return self._trace

    def trace_actors(self):
        """
        Records movement of actors at every timestep.
        """
        time = traci.simulation.getTime()

        # Vehicles
        for vid in traci.vehicle.getIDList():
            x,y = traci.vehicle.getPosition(vid)
            if vid == constants.BIKE:
                wheelbase = constants.bike.wheelbase
                max_steering_angle = constants.bike.max_steering_angle
            else:
                wheelbase = constants.car.wheelbase
                max_steering_angle = constants.car.max_steering_angle

            length = traci.vehicle.getLength(vid)

            s = pd.Series({
                "time" : time,
                "stage" : self.cur_stage,
                "id" : vid,

                "a_min" : -traci.vehicle.getEmergencyDecel(vid),
                "a_max" : traci.vehicle.getAccel(vid),
                "v_max" : traci.vehicle.getMaxSpeed(vid),

                "v0" : traci.vehicle.getSpeed(vid),
                "x0" : x,
                "y0" : y,
                "phi0" : traci.vehicle.getAngle(vid) - 90,
                "width" : traci.vehicle.getWidth(vid),
                "length_fov" : length/2,
                "length_rov" : length/2,

                "lf" : wheelbase / 2,
                "lr" : wheelbase / 2,
                "delta_min" : -max_steering_angle,
                "delta_max" : max_steering_angle
            })
            self.trace.append(s)
            continue

        for pid in traci.person.getIDList():
            x,y = traci.person.getPosition(pid)
            wheelbase = constants.person.wheelbase
            max_steering_angle = constants.person.max_steering_angle
            length = traci.person.getLength(pid)

            s = pd.Series({
                "time" : time,
                "stage" : self.cur_stage,
                "id" : pid,

                "a_min" : -traci.person.getEmergencyDecel(pid),
                "a_max" : traci.person.getAccel(pid),
                "v_max" : traci.person.getMaxSpeed(pid),

                "v0" : traci.person.getSpeed(pid),
                "x0" : x,
                "y0" : y,
                "phi0" : traci.person.getAngle(pid) - 90,
                "width" : traci.person.getWidth(pid),
                "length_fov" : length/2,
                "length_rov" : length/2,

                "lf" : wheelbase / 2,
                "lr" : wheelbase / 2,
                "delta_min" : -max_steering_angle,
                "delta_max" : max_steering_angle
            })
            self.trace.append(s)
            return
        
        return

    def check_for_stage_transition(self):
        if self.cur_stage == 4:
            ped_lid = traci.person.getLaneID(constants.PED)
            if ped_lid == "-E1_0":
                self._cur_stage = 5
                self.stage_4_end()
                self.stage_5_start()

        dut_pos = traci.vehicle.getLanePosition(constants.DUT)
        if dut_pos >= 398:
            stage = 4
        else:
            stage = int(dut_pos/100)
        if stage > self.cur_stage:
            if stage == 1:
                self.stage_0_end()
                self.stage_1_start()
            elif stage == 2:
                self.stage_1_end()
                self.stage_2_start()
            elif stage == 3:
                self.stage_2_end()
                self.stage_3_start()
            elif stage == 4:
                self.stage_3_end()
                self.stage_4_start()
            self._cur_stage = stage
        return
    
    def stage_0_start(self):
        """
        No Traffic
        """
        traci.vehicle.setStop(
            constants.DUT, 
            edgeID="E0", 
            laneIndex=2, 
            pos=100,
            duration=1
        )
        return

    def stage_0_end(self):
        return
    
    def stage_1_start(self):
        """
        Vehicle to the left.
        """
        traci.vehicle.setColor(constants.DUT, constants.RGBA.cyan)
        traci.vehicle.setSpeed(constants.FOE1, constants.ego_speed+1)
        traci.vehicle.setStop(constants.FOE1, "E0", pos=200, laneIndex=2)
        traci.vehicle.setStop(
            constants.DUT, 
            edgeID="E0", 
            laneIndex=2, 
            pos=200,
            duration=1
        )
        return
    
    def stage_1_end(self):
        if constants.FOE1 in traci.vehicle.getIDList():
            traci.vehicle.remove(constants.FOE1)
        return
    
    def stage_2_start(self):
        """
        Vehicle to the left.
        Vehicle in front.
        """
        traci.vehicle.setColor(constants.DUT, constants.RGBA.light_blue)
        traci.vehicle.setSpeed(constants.FOE2, constants.ego_speed+1)
        traci.vehicle.setSpeed(constants.FOE3, constants.ego_speed+0.5)
        traci.vehicle.setStop(
            constants.DUT, 
            edgeID="E0", 
            laneIndex=2, 
            pos=301,
            duration=1
        )
        return
    
    def stage_2_end(self):
        vids = traci.vehicle.getIDList()
        for vid in [constants.FOE2, constants.FOE3]:
            traci.vehicle.remove(vid)
        return
    
    def stage_3_start(self):
        """
        Bicycle
        """
        traci.vehicle.setColor(constants.DUT, constants.RGBA.cyan)
        traci.vehicle.setSpeed(constants.BIKE, constants.ego_speed-.5)
        traci.vehicle.setStop(
            constants.DUT, 
            edgeID="E0", 
            laneIndex=2, 
            pos=398
        )
        traci.vehicle.setStop(
            constants.BIKE, 
            edgeID="E0", 
            laneIndex=1, 
            pos=400
        )
        return
    
    def stage_3_end(self):
        # if constants.BIKE in traci.vehicle.getIDList():
        #     traci.vehicle.remove(constants.BIKE)
        return
    
    def stage_4_start(self):
        """
        Waiting for Pedestrian
        """
        traci.vehicle.setColor(constants.DUT, constants.RGBA.light_blue)
        traci.person.setSpeed(constants.PED,2)
        return
    
    def stage_4_end(self):
        return
    
    def stage_5_start(self):
        traci.vehicle.setColor(constants.DUT, constants.RGBA.cyan)
        traci.vehicle.resume(constants.DUT)
        traci.vehicle.resume(constants.BIKE)
        return
    
    def stage_5_end(self):
        return
    
    def add_actors(self):
        """
        Vehicles
        """
        traci.vehicle.add(constants.DUT, "eastbound", departLane=2)
        traci.vehicle.setMaxSpeed(constants.DUT, constants.ego_speed)
        traci.vehicle.setColor(constants.DUT, constants.RGBA.light_blue)

        traci.vehicle.add(constants.FOE1, "eastbound", 
            departLane=3, departPos=100)
        traci.vehicle.setSpeed(constants.FOE1, 0)

        traci.vehicle.add(constants.FOE2, "eastbound", 
            departLane=3, departPos=200)
        traci.vehicle.setColor(constants.FOE2, constants.RGBA.lime)
        traci.vehicle.setSpeed(constants.FOE2, 0)

        traci.vehicle.add(constants.FOE3, "eastbound", 
            departLane=2, departPos=210)
        traci.vehicle.setColor(constants.FOE3, constants.RGBA.red)
        traci.vehicle.setSpeed(constants.FOE3, 0)

        traci.vehicle.add(constants.BIKE, "eastbound", 
            typeID="bicycle", departLane=1, departPos=300)
        traci.vehicle.setSpeed(constants.BIKE, 0)

        for vid in constants.all_vehicles:
            traci.vehicle.setLaneChangeMode(vid, 0)
            traci.vehicle.setAccel(vid, constants.accel)

        """
        Pedestrian
        """
        traci.person.add(constants.PED, "E0", 390)
        traci.person.appendWalkingStage(
            constants.PED, ["E0", "-E1"], arrivalPos=10)
        traci.person.setSpeed(constants.PED, 0)
        traci.person.setColor(constants.PED, constants.RGBA.yellow)

        traci.simulationStep()
        return
