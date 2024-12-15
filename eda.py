import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes import Axes
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import time

from estimator import DriveableTrajectoryEstimator, subject_complexity, actor_entropy
import utils
import constants


pd.set_option('display.max_columns', None)

class EDA:
    def __init__(self):
        self.init_trajectories()

        # self.load_gauntlet_data()    
        # self.calc_influence()
        # utils.save(self.df, "data/gaussian.pkl")
        


        # udf = utils.load("data/uniform.pkl")
        gdf = utils.load("data/gaussian.pkl")

        self._df = gdf
        
        # print(self.df)

        ts = 116.5
        df = self.df[self.df["time"]==ts]
        df["scaled"] = df["entropy"]/3.871413
        print(df)
        print("sum(entropy)", df["entropy"].sum(), 
              df["entropy"].sum()/3.871413)
        

        # self.plot_all_stages()
        # self.plot_complexity_all()
        # self.plot_trajectories()
        # self.save_trajectories()
        # self.plot_scenario()
        return
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def traj_ids(self) -> np.ndarray:
        return self._traj_ids

    @property
    def traj_probability(self) -> np.ndarray:
        return self._traj_probability

    def plot_all_stages(self):
        for stage in range(5):
            df = self.df[self.df["stage"] == stage]
            self.plot_by_stage(df)
            # break
        return

    def plot_by_stage(self, df : pd.DataFrame):
        plt.clf()

        fig = plt.figure(figsize=(5,3))
        ax = fig.gca()
        self.plot_complexity_to_time(df, ax)
        ax.set_xlim([df["time"].min(), df["time"].max()])
        ax.legend(loc="center left", bbox_to_anchor=(1,.5))
        stage = df["stage"].iloc[0]
        plt.savefig("out/complexity_%d.png" % stage, bbox_inches="tight")
        return


    def save_trajectories(self):
        self.df[["time","stage","id","entropy"]]\
            .to_csv("out/trace.csv",index=False)
        
        timesteps = self.df["time"].unique()
        for i,ts in enumerate(timesteps):
            plt.clf()
            fig = plt.figure(figsize=(5,4))
            ax = fig.gca()
        
            df = self.df[self.df["time"] == ts].copy()
            self.plot_trajectory(df, ax, legend=False)
            plt.savefig("out/steps/ts_%.2f.pdf" % ts, bbox_inches="tight")
            # break
        return


    def plot_trajectories(self, wait: bool = False):
        timesteps = self.df["time"].unique()

        fig, axes = plt.subplots(2,1, figsize=(8,7))

        complexities = self.plot_complexity_to_time(self.df, axes[0])

        

        plt.draw()
        plt.pause(0.05)
        if wait:
            input("Pause")

        last_vl = None
        last_hl = None
        for i,ts in enumerate(timesteps):
            if last_vl is not None:
                last_vl.remove()
                last_hl.remove()
            
            last_vl = axes[0].vlines(
                ts,
                ymin=0,
                ymax=100,
                zorder=3,
                color="red"   
            )
            last_hl = axes[0].hlines(
                complexities[i],
                xmin=0,
                xmax=300,
                color="red",
                zorder=3
            )


            df = self.df[self.df["time"] == ts].copy()
            self.plot_trajectory(df, axes[1])

            plt.draw()
            plt.pause(constants.graphics.draw_delay)
            continue
        
        if wait:
            input("Simulation End.")
        
        
        # plt.savefig("out/dino.png", bbox_inches="tight")
        
        # plt.show()
        return
    
    def plot_trajectory(self, df : pd.DataFrame, ax : Axes, legend : bool=True):
        df = df.copy()
        ax.cla()

        """
        Draw Road
        """
        ax.hlines(
            [2, 1, -7.4, -8.4], 
            xmin=98, 
            xmax=550, 
            zorder=1,
            color="gray"
        )
        ax.hlines(
            [-6.4, -3.2, 0],
            xmin=98, 
            xmax=550, 
            zorder=1,
            color="gray",
            linestyles= "dashed"
        )
        ax.vlines(
            [500, 504],
            ymin=-7.4,
            ymax=1,
            zorder=1,
            color="gray"   
        )

        """
        Get Position of DUT
        """
        dut_x = df[df["id"] == constants.DUT].iloc[0]["x0"]
        pm = 30
        ax.set_xlim([dut_x-10,dut_x+pm])

        """
        Plot each trajectories
        """
        for i in range(len(df.index)):
            s = df.iloc[i]
            dte : DriveableTrajectoryEstimator = s["dte"]
            traj = dte.traj_summary_concise
            for polygon in traj["polygon"]:
                if not polygon is None:
                    try:
                        patch = patches.Polygon(
                            list(polygon.exterior.coords), 
                            closed=True, 
                            edgecolor='black', 
                            facecolor=(0, 0, 0, 0)
                        )
                        ax.add_patch(patch)
                    except ValueError:
                        pass
                continue
            continue


        """
        Plot actors
        """
        for i in range(len(df.index)):
            s = df.iloc[i]
            x = s["x0"]
            y = s["y0"]
            length = s["width"]
            width = constants.graphics.length_map[s["id"]]
            phi = s["phi0"]
            
            rectangle = patches.Rectangle(
                (x - width / 2, y - length / 2),
                width,
                length,
                angle=phi,
                linewidth=1,
                edgecolor='black',
                facecolor="white",
                label=s["id"]
            )
            hatch = constants.graphics.hatch_map[s["id"]]
            rectangle.set_hatch(hatch)
            
            ax.add_patch(rectangle)
            ax.plot(x,y,marker="+", color="red")
            continue
        
        

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal', adjustable='box')

        if legend:
            ax.legend(loc="upper left")

        return
    

    def plot_scenario(self):
        df = self.df[self.df["time"] == .25].copy()

        df["x0"] = df["x0"]

        plt.clf()
        fig = plt.figure(figsize=(16,5))
        ax = fig.gca()


        """
        Draw Road
        """
        ax.hlines(
            [2, 1, -7.4, -8.4], 
            xmin=98, 
            xmax=550, 
            zorder=1,
            color="gray"
        )
        ax.hlines(
            [-6.4, -3.2, 0],
            xmin=98, 
            xmax=550, 
            zorder=1,
            color="gray",
            linestyles= "dashed"
        )
        ax.vlines(
            [500, 504],
            ymin=-7.4,
            ymax=1,
            zorder=1,
            color="gray"   
        )


        """
        Plot actors
        """
        for i in range(len(df.index)):
            s = df.iloc[i]
            x = s["x0"]
            y = s["y0"]
            length = s["width"]
            width = constants.graphics.length_map[s["id"]]
            phi = s["phi0"]
            
            rectangle = patches.Rectangle(
                (x - width / 2, y - length / 2),
                width,
                length,
                angle=phi,
                linewidth=1,
                edgecolor='black',
                facecolor="white",
                label=s["id"]
            )
            hatch = constants.graphics.hatch_map[s["id"]]
            rectangle.set_hatch(hatch)
            
            ax.add_patch(rectangle)
            if s["id"] in [constants.BIKE, constants.PED]:
                ax.plot(x,y,
                        marker="o", 
                        color="red", 
                        markerfacecolor="none",
                        markersize=15,
                        markeredgewidth=3
                )
            # ax.text(x,y, s["id"], color="black")
            continue

        

        
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal', adjustable='box')

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(MultipleLocator(5))

 
        xlimits = [
            [100,250],
            [250,400],
            [400,550]
        ]
        # xlimits = [
        #     [100,215],
        #     [215,330],
        #     [330,445],
        #     [445,560]
        # ]
        for i,xlimit in enumerate(xlimits):
            ax.set_xlim(xlimit)
            if i == (len(xlimits) - 1):
                ax.set_xlabel("x (m)")
            plt.savefig(
                "out/scenario/scenario_%d.pdf" % i, 
                bbox_inches="tight"
            )

      
        
        return

    
    def plot_complexity_to_time(self, 
            df : pd.DataFrame, 
            ax : Axes
        ) -> Axes:
        df = df.copy()

        # Complexity per time step
        timesteps = df["time"].unique()
        complexities = [df[df["time"] == ts]["entropy"].sum() \
            for ts in timesteps]
        
        
        
        end_time = df["time"].max()
        
        """
        Prepare data
        """
        actor_order = constants.graphics.actor_order

        data = {"time" : []}
        for actor in actor_order:
            data[actor] = []

        for ts in timesteps:
            df = self.df[self.df["time"] == ts]
            data["time"].append(ts)
            for actor in actor_order:
                try:
                    entropy = df[df["id"] == actor]["entropy"].iloc[0]
                except IndexError:
                    entropy = 0
                data[actor].append(entropy)
                continue
            continue

        df = pd.DataFrame(data)
        
        x = df["time"]
        y = df.drop("time", axis=1).values.T
        labels = df.columns[1:]

        colors = constants.graphics.white
        """
        Create the graph
        """
        ax.cla()

        stack = ax.stackplot(
            x,y,
            labels = labels,
            colors = colors
        )

        hatches = constants.graphics.hatches
        for patch, hatch in zip(stack, hatches):
            patch.set_hatch(hatch)
            continue

        ax.plot(x, complexities, c="black", lw=1)

        ax.set_xlabel("Time (s)")
        ax.set_xticks(range(0,131, 10))


        """
        Y labels
        """
        # Scale to the first check
        base = complexities[0]
        complexities_scaled : np.ndarray = np.array(complexities)/base

        upper_limit = complexities_scaled.max()
        
        y_ticks = [base * x for x in range(1,int(upper_limit+1))]
        ax.set_yticks(y_ticks)

        y_ticklabels = [x for x in range(1,len(y_ticks)+1)]
        ax.set_yticklabels(y_ticklabels)
        
        ax.set_ylabel("Complexity Scale")
        ax.set_ylim([0,max(complexities)])
        # quit()

        """
        Complexity Data
        """
        data = {
            "ts" : timesteps,
            "complexity" : complexities,
            "scale" : complexities_scaled
        }
        complex_df = pd.DataFrame(data)
        complex_df.to_csv("out/complexities.csv", index=False)

        """
        Other
        """
        ax.legend(loc="upper left")
        ax.set_xlim([0,end_time])

        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        return complexities

    def plot_complexity_all(self):
        fig = plt.figure(figsize=(16,3))
        ax = fig.gca()
        self.plot_complexity_to_time(self.df, ax)
        ax.set_aspect(.6)
        plt.savefig("out/complexity_all.pdf", bbox_inches="tight")
        return

    def calc_influence(self):
        timesteps = self.df["time"].unique()
        end = max(timesteps)
        data = []
        for ts in timesteps:
            print("%.2f of %.2f" % (ts, end), end="\r")
            df = self.df[self.df["time"] == ts].copy()
        
            subject = df[df["id"] == constants.DUT].iloc[0]["dte"]
            subject : DriveableTrajectoryEstimator
            actors = df["dte"].tolist()
            influence = [constants.influence_map[_id] for _id in df["id"]]

            df["entropy"] = subject.actor_entropy(actors, influence)    
            data.append(df)

        self._df = pd.concat(data)
        print()
        return

    def init_trajectories(self):
        """
        Convert Maja's ids to steering angles
        """
        traj_ids = np.array([5,4,3,2,1.5,1,.5,0,-.5,-1.,-1.5,-2,-3,-4,-5])
        self._traj_ids = traj_ids

        if constants.kinematics_model.distribution == constants.distribution.gaussian:
            traj_probability = utils.gaussian_pdf(traj_ids, mu = 0, sigma = 1)
        else:
            traj_probability = None
        self._traj_probability = traj_probability
        return

    def load_gauntlet_data(self):
        fn = "data/guantlet.feather"
        df = pd.read_feather(fn)

        # Smaller Dataset
        # df = df[df["time"] == 80]
        
        # Correct max speed data.
        speed_map = {
            constants.DUT : 5,
            constants.FOE1 : 6,
            constants.FOE2 : 6,
            constants.FOE3 : 5.5,
            constants.BIKE : 4.5,
            constants.PED : 2
        }
        df["v_max"] = [speed_map[_id] for _id in df["id"]]

    

        df["delta_probability"] = \
            [self.traj_probability for _ in range(len(df.index))]
        
        """
        Construct discrete delta samples for all actors
        """
        tids = (self.traj_ids + 5) / 10
        delta_samples_car = [
            utils.project(
                - constants.car.max_steering_angle, 
                constants.car.max_steering_angle, 
                tid
            ) for tid in tids
        ] 
        delta_samples_bike = [
            utils.project(
                - constants.bike.max_steering_angle, 
                constants.bike.max_steering_angle, 
                tid
            ) for tid in tids
        ] 
        delta_samples_ped = [
            utils.project(
                - constants.person.max_steering_angle, 
                constants.person.max_steering_angle, 
                tid
            ) for tid in tids
        ] 
        delta_samples_map = {
            constants.DUT : delta_samples_car,
            constants.FOE1 : delta_samples_car,
            constants.FOE2 : delta_samples_car,
            constants.FOE3 : delta_samples_car,
            constants.BIKE : delta_samples_bike,
            constants.PED : delta_samples_ped
        }
        
        df["delta_samples"] = df["id"].apply(lambda x: delta_samples_map[x])
        df["n_intervals_a"] = constants.kinematics_model.n_intervals_a
        df["time_window"] = constants.kinematics_model.time_window
        df["dt"] = constants.kinematics_model.dt
        

        """
        Estimate Driveable Area
        """
        feats = ["a_max", "a_max", "n_intervals_a", "delta_min", "delta_max", 
            "delta_samples", "v_max", "lf", "lr", "time_window", "dt", "x0",
            "y0", "v0", "phi0", "width", "length_fov", "length_rov", "delta_probability"]
        
        # for 
        # kwargs = df.iloc[0][feats].to_dict()
        # dte = DriveableTrajectoryEstimator(**kwargs)

        df["dte"] = df.apply(
            lambda s : DriveableTrajectoryEstimator(**s[feats].to_dict()),
            axis = 1
        )

        self._df = df
        return
    
if __name__ == "__main__":
    EDA()