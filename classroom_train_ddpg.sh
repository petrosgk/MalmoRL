#!/usr/bin/env bash

python run_classroom.py --ms-per-tick 10 --clients clients.txt --steps 1000000 --action-space continuous --agent ddpg --mode training
