import glob
import os
import random
import sys
import time
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla 

actor_list = []
ROUTE = 'straight_route'
QUALITY = 'high'
WEATHER = 'sunny'

try:
    # connect
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    recording_name = 'recording_%s.log' %ROUTE
    print("Recording on file: %s" % client.start_recorder(recording_name))

    world = client.get_world()

    # settings = world.get_settings()
    # settings.fixed_delta_seconds = 0.05
    # world.apply_settings(settings)

    debug = world.debug

    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter("model3")[0]
    print(vehicle_bp)


    # ROUNDABOUT LOCATION
    # spawn_point = carla.Transform(carla.Location(x=-2.948311, y=-61.594730, z=0.275307), carla.Rotation(pitch=0.000000, yaw=92.004189, roll=0.000000))

    # STRAIGHT LOCATION
    spawn_point = carla.Transform(carla.Location(x=-6.862468242645264, y=82.956146240234375, z=0.275307), carla.Rotation(pitch=0.000000, yaw=92.004189, roll=0.000000))

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)
    print(vehicle.id)

    time.sleep(25)
    
finally:    
    for actor in actor_list:
        actor.destroy()
    print("Cleaned up!")
    print("Stop recording")
    client.stop_recorder()