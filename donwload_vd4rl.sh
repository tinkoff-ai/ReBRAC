#!/bin/bash

python3 -m pip install gdown
gdown -O vd4rl.tar.gz 1F4LIH_khOFw1asVvXo82OMa2tZ0Ax5Op # walker_walk
tar xf vd4rl.tar.gz
gdown -O vd4rl.tar.gz 1WR2LfK0y94C_1r2e1ps1dg6zSMHlVY_e # cheetah_run
tar xf vd4rl.tar.gz
gdown -O vd4rl.tar.gz 1zTBL8KWR3o07BQ62jJR7CeatN7vb-vjd # humanoid_walk
tar xf vd4rl.tar.gz
rm vd4rl.tar.gz
