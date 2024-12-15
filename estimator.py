import warnings
warnings.simplefilter('ignore')

import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString


class DriveableTrajectoryEstimator:
    def __init__(self,
            a_max : float,
            delta_min : float,
            delta_max : float,
            v_max : float,
            lf : float,
            lr : float,
            time_window : float,
            dt : float,
            x0 : float,
            y0 : float,
            v0 : float,
            phi0 : float,
            a_min : float = None,
            n_intervals_a : int = 2,
            n_intervals_delta : int = 5,
            delta_samples : list[float] = [],
            delta_probability : list[float] = None,
            width : float = 0,
            length_fov : float = 0,
            length_rov : float = 0
        ):
        """
        Generates vehicle trajectories using a kinematic bicycle driving model
        with wheel sleep.

        :: REQUIRED PARAMETERS ::
            a_max : Max acceleration (mps^2)
            delta_min : min steering angle of front axle (degrees)
            delta_max : min steering angle of front axle (degrees)
            v_max : Max vehicle speed (mps)
            lf : Distance from front axle to center of mass (m)
            lr : Distance from rear axle to center of mass (m)
            time_window : Simulation Length (s)
            dt : simulation timestep size (s)
            x0 : Initial x position (m)
            y0 : Initial y position (m)
            v0 : Initial speed (mps)
            phi_0 : Initial vehicle heading (degrees).

        :: OPTIONAL PARAMETERS ::
            a_min : Min acceleration (mps^2)
            n_intervals_a : # of samples between a_min and a_max
            n_intervals_delta : # of samples between delta_min and delta_max
            delta_samples : Explicit steering angle values values of front axle
                (degrees)
            delta_probability : Assign given probabilities to each trajectory.
                One probability should be given for each delta sample. If none 
                are given, then the proabilties will default to uniform prob-
                ability.
            width : Vehicle width (m)
            length_rov : Length from center of mass to rear of vehicle (m).
            length_fov : Length from center of mass to front of vehicle (m).
        """
        self._width = width
        self._length_fov = length_fov
        self._length_rov = length_rov

        if a_min is None:
            self._a_min = None
        else:
            self._a_min = float(a_min)
        self._a_max = float(a_max)
        self._n_intervals_a = int(float(n_intervals_a))
        if (a_min is None) or (n_intervals_a == 1):
            self._a_samples = [self.a_max]
        else:
            self._a_samples = utils.n_intervals(
                self.a_min, 
                self.a_max, 
                self.n_intervals_a
            )

        self._delta_min = utils.deg2rad(float(delta_min))
        self._delta_max = utils.deg2rad(float(delta_max))
        self._n_intervals_delta = int(float(n_intervals_delta))
        if not delta_samples:
            self._delta_samples = utils.n_intervals(
                self.delta_min, 
                self.delta_max, 
                self.n_intervals_delta
            )
        else:
            self._delta_samples = \
                [utils.deg2rad(delta) for delta in delta_samples]
            
        if delta_probability is None:
            self._delta_probability = [1/len(self.delta_samples) \
                for i in range(len(self.delta_samples))]
        else:
            if not (len(delta_probability) == len(self.delta_samples)):
                raise ValueError(
                    "Expecting %d probabilities but given %d." % (
                        len(self.delta_samples), len(delta_probability)
                    ))
            self._delta_probability = [float(p) for p in delta_probability]
        
        
        self._delta_probability_map = {}
        for i in range(len(self.delta_samples)):
            self._delta_probability_map[self.delta_samples[i]] \
                = self.delta_probaility[i]

        self._v_max = float(v_max)
        self._lf = float(lf)
        self._lr = float(lr)

        self._time_window = float(time_window)
        self._dt = float(dt)

        self._v0 = float(v0)
        self._x0 = float(x0)
        self._y0 = float(y0)
        self._phi0 = utils.deg2rad(float(phi0))
        
        self._create_vehicle_polygon()
        self._predict_trajectories()
        self._create_concise_trajectories()
        self._boundary, self._shape_df = \
            self._find_exterior_boundary(self.traj_summary)
        self._find_all_trajectory_boundaries()
        # self._plot_summary()
        return
    
    @property
    def a_min(self) -> float:
        """
        Min acceleration in mps^2
        """
        return self._a_min
    
    @property
    def a_max(self) -> float:
        """
        Max acceleration in mps^2
        """
        return self._a_max
    
    @property
    def n_intervals_a(self) -> int:
        """
        Number of intervals sampled between @a_min and @a_max
        """
        return self._n_intervals_a
    
    @property
    def a_samples(self) -> list[float]:
        """
        List of acceleration (mps^2) samples to be simulated.
        """
        return self._a_samples

    @property
    def delta_min(self) -> float:
        """
        Min steering angle of front axle in radians.
        """
        return self._delta_min
    
    @property
    def delta_max(self) -> float:
        """
        Max stering angle of front axle in radians.
        """
        return self._delta_max

    @property
    def delta_probaility(self) -> list[float]:
        """
        Probabilties for each delta value
        """
        return self._delta_probability

    @property
    def delta_probability_map(self) -> dict:
        """
        Mapping of delta samples to delta probabilties.
        """
        return self._delta_probability_map

    @property
    def delta_samples(self) -> list[float]:
        """
        List of steering angle (radians) samples to be simulated.
        """
        return self._delta_samples

    
    @property
    def n_intervals_delta(self) -> int:
        """
        Number of intervals sampled between @delta_min and @delta_max
        """
        return self._n_intervals_delta

    
    @property
    def v_max(self) -> float:
        """
        Upper velocity limit, i.e. max vehicle speed in mps.
        """
        return self._v_max
    
    @property
    def length_fov(self) -> float:
        """
        Length from the center of mass to front of vehicle in meters.
        """
        return self._length_fov
    
    @property
    def length_rov(self) -> float:
        """
        Length from the center of mass to rear of vehicle in meters.
        """
        return self._length_rov

    @property
    def lf(self) -> float:
        """
        Distance from the front axle to center of mass in meters.
        """
        return self._lf
    
    @property
    def lr(self) -> float:
        """
        Distance from the rear axle to center of mass in meters.
        """
        return self._lr
    
    @property
    def time_window(self) -> float:
        """
        Upper limit of simulated time in 0 to @time_window seconds.
        """
        return self._time_window
    
    @property
    def dt(self) -> float:
        """
        Time interval between simulation steps in seconds.
        """
        return self._dt
    
    @property
    def v0(self) -> float:
        """
        Initial velocity/speed in mps.
        """
        return self._v0
    
    @property
    def x0(self) -> float:
        """
        Initial x position in meters.
        """
        return self._x0
    
    @property
    def y0(self) -> float:
        """
        Initial y position in meters
        """
        return self._y0
    
    @property
    def phi0(self) -> float:
        """
        Initial vehicle heading in meters.
        """
        return self._phi0
    
    @property
    def traj_summary(self) -> pd.DataFrame:
        """
        Summary dataframe of all trajectories
        """
        return self._traj_summary
    
    @property
    def traj_summary_concise(self) -> pd.DataFrame:
        """
        A concise summary of all trajectories with 1 row per trajectory.
        """
        return self._traj_summary_concise
    
    @property
    def boundary(self) -> Polygon:
        """
        Boundary of driveable-area as a shapely.geometry.Polygon.
        """
        return self._boundary
    
    @property
    def driveable_area(self) -> float:
        """
        The area of the boudary polygon in m^2
        Area is determined using shoelace formula.
        """
        return self.boundary.area
    
    @property
    def shape_df(self) -> pd.DataFrame:
        """
        Boundary shape as a pandas Dataframe
        """
        return self._shape_df
    
    @property
    def width(self) -> float:
        """
        Vehicle Width in meters
        """
        return self._width
    
    @property
    def trajectory_polygons(self) -> pd.DataFrame:
        """
        Dataframe of polygons for each tracjectoy.
        """
        return self._trajectory_polygons
    
    @property
    def vehicle_linestring(self) -> LineString:
        """
        Linestring of which represents vehicle entity.
        """
        return self._vehicle_linestring
    
    @property
    def vehicle_polygon(self) -> Polygon:
        """
        Polygon which represents the vehicle entity.
        """
        return self._vehicle_polygon

    def predict(self, 
            x : float, 
            y : float, 
            v : float,
            a : float,
            phi : float, 
            delta : float,
            lf : float,
            lr : float,
            dt : float
        )-> list[float, float, float, float]:
        """
        @param x : x position (m)
        @param y : y position (m)
        @param v : Speed (mps)
        @param a : acceleration (mps^2)
        @param phi : heading angle (rad)
        @param delta : steering angle of front axle (rad) 
        @param lf : Distance from front axle to center of gravity (m) 
        @param lr : Distance from rear axle to center of gravity (m) 
        @param dt : Time elapsed

        @return next x, y, phi, and v value.
        """
        beta = np.arctan( (lr/(lf+lr)) * np.tan(delta))
        next_x = x + v * np.cos(phi + beta) * dt
        next_y = y + v * np.sin(phi + beta) * dt
        next_v = v + a * dt
        next_phi = phi + (v/lr) * np.sin(beta) * dt
        return next_x, next_y, next_v, next_phi

    # def 

    def _predict_trajectories(self):
        """
        Gets vehicle trajectories
        """
        traj_hist : list[pd.Series] = []
        i_traj = 0
        for a in self.a_samples:
            for delta in self.delta_samples:
                x,y,v,phi = self.x0, self.y0, self.v0, self.phi0
                time = self.dt

                # Root
                s = pd.Series({
                    "x" : x,
                    "y" : y,
                    "v" : v,
                    "phi" : phi,
                    "a" : a,
                    "delta" : delta,
                    "i_traj" : i_traj
                })
                traj_hist.append(s)


                while time < self.time_window:
                    time += self.dt
                    x,y,v,phi = self.predict(
                        x,y,v,a,phi,delta,self.lf,self.lr,self.dt)
                    if v < 0:
                        v = 0
                    elif v > self.v_max:
                        v = self.v_max
                    s = pd.Series({
                        "x" : x,
                        "y" : y,
                        "v" : v,
                        "phi" : phi,
                        "a" : a,
                        "delta" : delta,
                        "i_traj" : i_traj
                    })
                    traj_hist.append(s)
                    if v == 0:
                        break
                    continue
                i_traj += 1
                continue
            continue
        
        self._traj_summary = pd.DataFrame(traj_hist)
        return
    
    def _create_concise_trajectories(self):
        df = self.traj_summary.copy()
        data : list[pd.Series] = []
        for i_traj in df["i_traj"].unique():
            traj_df = df[df["i_traj"] == i_traj]
            if self.width > 0:
                linestring = LineString(traj_df[["x","y"]].to_numpy())
                polygon = utils.linestring2polygon(linestring, self.width)
            else:
                polygon = None
            delta = traj_df["delta"].iloc[0]
            s = pd.Series({
                "a" : traj_df["a"].iloc[0],
                "delta" : delta,
                "prob" : self.delta_probability_map[delta],
                "path" : linestring,
                "polygon" : polygon
            })
            data.append(s)
            continue
        self._traj_summary_concise = pd.DataFrame(data)
        return
    
    def _widen_trajectory(self, 
            df : pd.DataFrame, angle : float) -> pd.DataFrame:
        """
        Widens a trajectory to half of the vehicles width

        :: PARAMETERS ::
        df : Trajectory dataframe
        angle : Angle to adjust (in degrees)
            Use 90 for left and -90 for right
        """
        if self.width == 0:
            return df
        df = df.copy()
        angle = utils.deg2rad(angle)
        half_veh_width = self.width/2
        for i in range(len(df.index)):
            s = df.iloc[i]
            x,y = utils.project_point(
                s["x"],
                s["y"],
                half_veh_width,
                angle + s["phi"]
            )
            df.iloc[i]["x"] = np.round(x, decimals=6)
            df.iloc[i]["y"] = np.round(y, decimals=6)
        return df

    def _find_exterior_boundary(self, 
            df : pd.DataFrame
        ) -> list[Polygon, pd.DataFrame]:
        """
        Finds the exterior boundary given a dataframe @df of trajectory
        information.

        :: Return ::
        Returns the boundary as a Shapely Polygon and a Dataframe
        """
        df = df.copy()

        # The shape begins at the origin.
        origin = pd.Series({"x" : self.x0, "y" : self.y0})
        points = [origin]

        """
        Start with the leftmost trajectory with the small accel and largest
        steering angle.
        """
        leftmost_traj = df[
            (df["a"] == df["a"].min()) &
            (df["delta"] == df["delta"].max())
        ]
        leftmost_traj = self._widen_trajectory(leftmost_traj, 90)[["x", "y"]]
        for i in range(len(leftmost_traj.index)):
            points.append(leftmost_traj.iloc[i])
        

        """
        Next, follow the trajectory with the same delta increasing in accel
        Starting with the closest points from the previous acceleration
        """
        sorted_accel = np.sort(df["a"].unique()).tolist()
        for i, a in enumerate(sorted_accel[1:], start=1):
            traj_df = df[
                (df["a"] == a) &
                (df["delta"] == df["delta"].max())
            ]
            traj_df = self._widen_trajectory(traj_df, 90)[["x", "y"]]
            closest_pos = self._closest_point(points[-1], traj_df)
            traj_df = traj_df[traj_df.index >= closest_pos.name]
            for i in range(len(traj_df.index)):
                points.append(traj_df.iloc[i])
            continue

        """
        Then get the trajectories with the max accelerations sorting from
        highest to lowest angle.
        """
        sorted_delta = np.sort(df["delta"].unique())[::-1].tolist()
        if self.width == 0:
            sorted_delta = sorted_delta[1:-1]
        for delta in sorted_delta:
            traj_df = df[
                (df["a"] == df["a"].max()) &
                (df["delta"] == delta)
            ][["x","y"]]
            pos = traj_df[traj_df.index == traj_df.index.max()].iloc[0]
            points.append(pos)
            continue

        """
        Get the rightmost trajectory with the smallest acceleration
        """
        rightmost_points : list[pd.Series] = []
        rightmost_traj = df[
            (df["a"] == df["a"].min()) &
            (df["delta"] == df["delta"].min())
        ]
        rightmost_traj = self._widen_trajectory(rightmost_traj, -90)[["x", "y"]]
        for i in range(len(rightmost_traj.index)):
            rightmost_points.append(rightmost_traj.iloc[i])

        """
        Next, follow the trajectory with the same delta increasing in accel
        Starting with the closest points from the previous acceleration
        """
        sorted_accel = np.sort(df["a"].unique()).tolist()
        for i, a in enumerate(sorted_accel[1:], start=1):
            traj_df = df[
                (df["a"] == a) &
                (df["delta"] == df["delta"].min())
            ]
            traj_df = self._widen_trajectory(traj_df, -90)[["x", "y"]]
            closest_pos = self._closest_point(rightmost_points[-1], traj_df)
            traj_df = traj_df[traj_df.index >= closest_pos.name]
            for i in range(len(traj_df.index)):
                rightmost_points.append(traj_df.iloc[i])
            continue

        """
        Add the points in reverse order
        """
        for i in range(len(rightmost_points)):
            points.append(rightmost_points[-i-1])
        
        
        shape_df = pd.DataFrame(points)
        boundary = Polygon(shape_df.to_numpy())
        # self._shape_df = shape_df
        return boundary, shape_df
    
    def _find_all_trajectory_boundaries(self):
        df = self.traj_summary.copy()
        sorted_delta = np.sort(df["delta"].unique())[::-1].tolist()
        polygons = []
        for delta in sorted_delta:
            delta_df = df[df["delta"] == delta]
            polygon, _ = self._find_exterior_boundary(delta_df)
            s = pd.Series({
                "delta" : delta, 
                "polygon" : polygon
            })
            polygons.append(s)
            continue
        self._trajectory_polygons = pd.DataFrame(polygons)
        return
    
    def _closest_point(self, xy : pd.Series, df : pd.DataFrame) -> pd.Series:
        """
        Finds the closest point of @xy to a point in @df.
        """
        assert all([feat in ["x", "y"] for feat in xy.index])
        assert all([feat in ["x", "y"] for feat in df.columns])
        df = df.copy()
        df["dist"] = df.apply(
            lambda s : utils.distance_to(*xy.to_list(), s["x"], s["y"]),
            axis = 1
        )
        return df[df["dist"] == df["dist"].min()].iloc[0][["x", "y"]]

    def plot(self, 
            figsize : tuple[float,float] = (4.5,4.5),
            boundary : bool = False,
            polygons : bool = True
        ) -> plt.Figure:
        """
        Plots drivable area.
        """
        

        # Plot
        plt.clf()
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()


        df = self.traj_summary_concise
        if polygons:
            for polygon in df["polygon"]:
                polygon : Polygon
                if polygon is None:
                    pass
                x,y = polygon.exterior.xy
                ax.fill(x,y, color="lightblue", edgecolor="blue", linewidth=2)
                continue


        df = self.traj_summary
        ax.scatter(
            df["x"], 
            df["y"], 
            marker="+",
            color="black"
        )
        ax.plot(self.x0,self.y0,color="red", marker="+")

        if boundary:
            ax.plot(
                self.shape_df["x"], 
                self.shape_df["y"], 
                marker=".",
                color="blue"
            )

        


        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        return fig
    
    def subject_complexity(self, actors, influence : list[float] = None):
        """
        Calculate influence of each actor in a driving scenario.
        
        :: Parameter ::
        actors : list[DrivableTrajectoryEstimator]
            All actors considered in calculation
        influence : list[float]
            Specify actor influence scale, otherwise actor influence will be 1.
        
        :: Return ::
        Complexity score of the driving scenario
        """
        return subject_complexity(self, actors, influence)
    
    def actor_entropy(
        self,
        actors,
        influence : list[float] = None
    ) -> list[float]:
        """
        Calculate influence of each actor in a driving scenario.
        
        :: Parameter ::
        actors : list[DrivableTrajectoryEstimator]
            All actors considered in calculation
        influence : list[float]
            Specify actor influence scale, otherwise actor influence will be 1.

        :: Return ::
        Ordered influence score of each actor.
        """
        return actor_entropy(self, actors, influence)
    
    def _create_vehicle_polygon(self):
        """
        Creates the polygon and linestring for the vehicle
        """
        
        rear = utils.project_point(
            self.x0, 
            self.y0,
            self.length_rov,
            self.phi0 + np.pi
        )
        front = utils.project_point(
            self.x0, 
            self.y0,
            self.length_fov,
            self.phi0
        )
        linestring = LineString(np.array([rear, front]))
        if self.width > 0:
            polygon = utils.linestring2polygon(linestring, self.width)
        else:
            polygon = None
        
        self._vehicle_linestring = linestring
        self._vehicle_polygon = polygon
        return

def subject_complexity(
        subject : DriveableTrajectoryEstimator, 
        actors : list[DriveableTrajectoryEstimator],
        influence : list[float] = None
    ) -> float:
    """
    Calculate influence of each actor in a driving scenario.
    
    :: Parameter ::
    subject : DriveableTrajectoryEstimator
        Focus of complexity calculation
    actors : list[DrivableTrajectoryEstimator]
        All actors considered in calculation
    influence : list[float]
        Specify actor influence scale, otherwise actor influence will be 1.

    :: Return ::
    Complexity score of the driving scenario
    """
    return sum(actor_entropy(subject, actors, influence))

def actor_entropy(
        subject : DriveableTrajectoryEstimator, 
        actors : list[DriveableTrajectoryEstimator],
        influence : list[float] = None
    ) -> list[float]:
    """
    Calculate influence of each actor in a driving scenario.
    
    :: Parameter ::
    subject : DriveableTrajectoryEstimator
        Focus of complexity calculation
    actors : list[DrivableTrajectoryEstimator]
        All actors considered in calculation
    influence : list[float]
        Specify actor influence scale, otherwise actor influence will be 1.
        
    :: Return ::
    Ordered influence score of each actor.
    """
    if influence is None:
        influence = [1. for actor in actors]

    df_s = subject.traj_summary_concise.copy()

    choose_geometry = \
        lambda s: s["path"] if s["polygon"] is None else s["polygon"]
    
    df_s["geom"] = df_s.apply(choose_geometry, axis=1)

    entropies = []
    for i_actor, actor in enumerate(actors):
        # Subject vehicle
        if actor is subject:
            entropy = actor.traj_summary_concise["prob"]\
                .apply(utils.entropy).sum()
            entropies.append(entropy * influence[i_actor])
            continue
        
        # Other vehicles
        df_a = actor.traj_summary_concise.copy()
        df_a["geom"] = df_a.apply(choose_geometry, axis=1)
        entropy = 0
        all_entropy_a = df_a["prob"].apply(utils.entropy).sum()
        


        """
        Iterate through subject trajorcties to check for overlaps.
        """
        print()

        for geom_s in df_s["geom"]:
            # Does it overlap with the vehicle polygon?
            if actor.vehicle_polygon is None:
                if actor.vehicle_linestring.intersects(geom_s):
                    entropy += all_entropy_a
                    continue
            elif actor.vehicle_polygon.intersects(geom_s):
                entropy += all_entropy_a
                continue
            
            # Otherwise check each individual trajectory.
            for i in range(len(df_a.index)):
                traj_a = df_a.iloc[i]
                geom_a : LineString = traj_a["geom"]
                if geom_a.intersects(geom_s):
                    entropy += utils.entropy(traj_a["prob"])
                continue
            continue
        
        entropies.append(entropy * influence[i_actor])
        continue
    return entropies


def test():
    dae = DriveableTrajectoryEstimator(
        a_min = -6,
        a_max = 4,
        n_intervals_a = 3,
        delta_min = -10,
        delta_max = 10,
        n_intervals_delta = 10,
        v_max = 14,
        lf = 2.5,
        lr = 2.5,
        time_window = 3,
        dt = 0.1,
        x0 = 0,
        y0 = 0,
        v0 = 0,
        phi0 = 0,
        delta_samples = [-10,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,10],
        # delta_samples = [0],
        width = 5
    )
    dae.plot_summary(boundary=True)
    return

    
if __name__ == "__main__":
    test()