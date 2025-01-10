#!/bin/bash

python3.9 -m pip uninstall -y -r <(python3.9 -m pip freeze)
