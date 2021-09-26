import glob
import os
import random
import sys
import time
import numpy as np
import cv2
from Queue import Queue
from ast import literal_eval

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla 

IM_HEIGHT = 1200
IM_WIDTH = 1600

first_image = True
real_cords = []

outF3 = open("calculated_coordinates_high_q.txt", "r")
outF4 = open("real_coordinates_high_q.txt", "r")
ctr = 0
real_cors_high = np.array(literal_eval(outF4.read()))
calc_cors_high = np.array(literal_eval(outF3.read()))


def plot_trajectory():
    global ctr, curr_loc_real, prev_loc_real, curr_loc_calc, prev_loc_calc

    if (ctr > 0):
        prev_loc_real = curr_loc_real
        prev_loc_calc = curr_loc_calc

    curr_loc_real = carla.Location(x=real_cors_high[ctr][0], y=real_cors_high[ctr][1], z=0) 
    curr_loc_calc = carla.Location(x=calc_cors_high[ctr][0], y=calc_cors_high[ctr][1], z=0) 

    if (ctr > 0):
        debug.draw_line(prev_loc_real, curr_loc_real, thickness=.5, life_time=0)
        debug.draw_line(prev_loc_calc, curr_loc_calc, thickness=.5, color=carla.Color(0,255,0), life_time=0)
    
    ctr = ctr + 1
      

actor_list = []

try:
    # connect
    client = carla.Client("localhost", 2000)
    # client.set_timeout(5)

    recording_name = 'recording01.log'
    client.replay_file(recording_name, 0, 25, 0)
    client.set_replayer_ignore_hero(False)
    
    world = client.get_world()

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.09
    settings.synchronous_mode = True # Enables synchronous mode
    world.apply_settings(settings)

    debug = world.debug

    # blueprint_library = world.get_blueprint_library()

    # vehicle_bp = blueprint_library.filter("model3")[0]
    # print(vehicle_bp)

    # vehicle = world.get_actor(86)
    # actor_list.append(vehicle)
    
    while True:
        world.tick()
        plot_trajectory()        

    
finally:    
    for actor in actor_list:
        actor.destroy()
    print("Cleaned up!")
