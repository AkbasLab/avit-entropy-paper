from typing import Callable
import constants
import traci_clients
import scenarios

import pandas as pd
import numpy as np

class Runner:
    def __init__(self):
        self._rng = np.random.RandomState(seed=constants.seed)
        
        self._traci_client = traci_clients.GenericClient(
            constants.sumo.config)
        
        self._scenario = scenarios.DrivingScenario

        params = pd.Series({
            "flag" : 1
        })

        scenario = scenarios.DrivingScenario(params)

        self.traci_client.close()

        df = pd.DataFrame(scenario.trace)
        df.to_feather("out/guantlet.feather")
        print(df)
        return
    

    @property
    def traci_client(self) -> traci_clients.GenericClient:
        return self._traci_client

    @property
    def scenario(self) -> scenarios.DrivingScenario:
        return self._scenario

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    def random_seed(self):
        return self.rng.randint(2**32-1)
    
if __name__ == "__main__":
    Runner()