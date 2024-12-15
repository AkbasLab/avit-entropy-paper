import shutil
import warnings
if shutil.which("sumo") is None:
    warnings.warn("Cannot find sumo/tools in the system path. Please verify that the lastest SUMO is installed from https://www.eclipse.org/sumo/")

import traci

import constants

class TraCIClient:
    def __init__(self, config : dict, priority : int = 1):
        """
        Barebones TraCI client.

        --- Parameters ---
        priority : int
            Priority of clients. MUST BE UNIQUE
        config : dict
            SUMO arguments stored as a python dictionary.
        """
        
        self._config = config
        self._priority = priority
        

        self.connect()
        return

    @property
    def priority(self) -> int:
        """
        Priority of TraCI client.
        """
        return self._priority

    @property
    def config(self) -> dict:
        """
        SUMO arguments stored as a python dictionary.
        """
        return self._config

    def run_to_end(self):
        """
        Runs the client until the end.
        """
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # more traci commands
        return

    def close(self):
        """
        Closes the client.
        """
        traci.close()
        return


    def connect(self):
        """
        Start or initialize the TraCI connection.
        """
        warnings.simplefilter("ignore", ResourceWarning)
        # Start the traci server with the first client
        if self.priority == 1:
            cmd = []

            for key, val in self.config.items():
                if key == "gui":
                    sumo = "sumo"
                    if val: sumo +="-gui"
                    cmd.append(sumo)
                    continue
                
                if key == "--remote-port":
                    continue

                cmd.append(key)
                if val != "":
                    cmd.append(str(val))
                continue

            traci.start(cmd,port=self.config["--remote-port"])
            traci.setOrder(self.priority)
            return
        
        # Initialize every client after the first.
        traci.init(port=self.config["--remote-port"])
        traci.setOrder(self.priority)
        return    

class GenericClient(TraCIClient):
    def __init__(self, new_config : dict):
        config = {
            "gui" : constants.sumo.gui,
            "--error-log" : constants.sumo.error_log_file,
            "--num-clients" : 1,
            "--remote-port" : 5522,
            "--delay" : constants.sumo.delay_ms,
            "--gui-settings-file" : constants.sumo.gui_setting_file,
            "--seed" : constants.seed,
            "--default.action-step-length" : constants.sumo.action_step_length,
            "--step-length" : constants.sumo.step_length,
            "--lanechange.duration" : constants.sumo.lane_change_duration
        }
        if constants.sumo.start:
            config["--start"] = ""
        if constants.sumo.quit_on_end:
            config["--quit-on-end"] = ""
        if constants.sumo.quiet_mode:
            config["--no-warnings"] = ""
            config["--no-step-log"] = ""

        for key, val in new_config.items():
            config[key] = val

        self._init_state_fn = constants.sumo.init_state_file
        super().__init__(config)
        traci.simulation.saveState(self._init_state_fn)
        return

    @property
    def init_state_fn(self) -> str:
        return self._init_state_fn