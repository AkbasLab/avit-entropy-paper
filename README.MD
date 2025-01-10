# AV IT Entropy
This repository cotains the source code used for our IEEE MOST 2025 submission "<u>Information Theory Based Quantitative Complexity Analysis for Autonomous Vehicle Safety Testing</u>".

This paper features a scenario example of an autonomous vehicle (DUT) interacting with other cars (FOE1, FOE2, and FOE3), a bicyclist (BIKE), and a pedestrian (PED).

![SUMO simulation](graphics/sumo.gif)

Afterwards, we recreate the scenario, and perform a step-by-step analysis of scenario complexity at every moment using our proposed complexity calculation methodology.

![Driving Complexity](graphics/driving-complexity.gif)

## Installation

This software is developed and tested using Python v3.9. The simulations are performed using [SUMO Traffic Simulator v1.21.0](https://eclipse.dev/sumo/) for linux. If you do not want to run the SUMO simulation and just use the [data](data/guantlet.feather), it is not required. 

To install the required packages use:

```python
pip install -r requirements.txt
```

To verify all required packages are installed run `verify_requirements.py`.

## Notes

This repo uses an unpacked version of the `DriveableTrajectoryEstmator`, however a packaged version is available on PyPi, and may be installed via pip using:

```python
pip install avit-entropy
```