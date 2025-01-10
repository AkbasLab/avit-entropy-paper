import constants
import eda
import estimator
import run_sumo_scenario
import scenarios
import traci_clients
import utils

import os

def main():
    print("All required python modules are installed")

    print("Checking for sumo installation...\n")
    os.system("sumo --version")

    return

if __name__ == "__main__":
    main()