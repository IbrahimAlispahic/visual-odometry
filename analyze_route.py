import glob
import os
import random
import sys
import time
import numpy as np
import cv2
from Queue import Queue

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
ROUTE = 'roundabout_route'
QUALITY = 'high'
WEATHER = 'sunny'

def save_route(image):
    global first_image, prev_loc, curr_loc, real_cords

    if not first_image:
        prev_loc = curr_loc
        debug.draw_line(prev_loc, curr_loc, thickness=.5, life_time=0)
    else:
        first_image = False

    curr_loc = image.transform.location
    real_cords.append([curr_loc.x, curr_loc.y])

    cc = carla.ColorConverter.LogarithmicDepth
    # image.save_to_disk('%s/%s/_out_high_q/%d.png' %(ROUTE, WEATHER, image.frame))
         

actor_list = []

try:
    # connect
    client = carla.Client("localhost", 2000)
    # client.set_timeout(5)

    recording_name = 'recording_%s.log' %ROUTE
    client.replay_file(recording_name, 3, 24, 0)
    # client.set_replayer_time_factor(0.7)
    # client.set_replayer_ignore_hero(True)
    
    world = client.get_world()

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.09
    settings.synchronous_mode = True # Enables synchronous mode
    world.apply_settings(settings)

    debug = world.debug

    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter("model3")[0]
    print(vehicle_bp)

    # ROUNDABOUT
    vehicle_id = 86
    # STRAIGHT
    # vehicle_id = 87

    vehicle = world.get_actor(vehicle_id)
    actor_list.append(vehicle)

    # SENSORS
    cc = carla.ColorConverter.LogarithmicDepth

    # DEFINE CAMERAS
    # SET PROPERTIES
    cam_rgb_bp = blueprint_library.find('sensor.camera.rgb')
    cam_rgb_bp.set_attribute("image_size_x", str(IM_WIDTH))
    cam_rgb_bp.set_attribute("image_size_y", str(IM_HEIGHT))
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    # MONO CAMERA
    cam_rgb = world.spawn_actor(cam_rgb_bp, spawn_point, attach_to=vehicle)
    actor_list.append(cam_rgb)
    # STEREO #1
    spawn_point_1 = carla.Transform(carla.Location(x=2.5, y=0.25, z=0.7))
    cam_stereo_right = world.spawn_actor(cam_rgb_bp, spawn_point_1, attach_to=vehicle)
    actor_list.append(cam_stereo_right)
    # STEREO #2
    spawn_point_2 = carla.Transform(carla.Location(x=2.5, y=-0.25, z=0.7))
    cam_stereo_left = world.spawn_actor(cam_rgb_bp, spawn_point_2, attach_to=vehicle)
    actor_list.append(cam_stereo_left)

    # cam_rgb.listen(lambda image: save_route(image))
    # cam_stereo_1.listen(lambda image: image.save_to_disk('_out_first/%d.png' % image.frame, cc))
    # cam_stereo_2.listen(lambda image: image.save_to_disk('_out_second/%d.png' % image.frame, cc))

    image_queue = Queue()
    cam_rgb.listen(image_queue.put)

    image_queue_right = Queue()
    cam_stereo_right.listen(image_queue_right.put)

    image_queue_left = Queue()
    cam_stereo_left.listen(image_queue_left.put)
    
    while True:
        world.tick()

        image = image_queue.get()
        save_route(image)

        # image_right = image_queue_right.get()
        # image_right.save_to_disk('%s/%s/_out_right_high_q/%d.png' %(ROUTE, WEATHER, image_right.frame))

        # image_left = image_queue_left.get()
        # image_left.save_to_disk('%s/%s/_out_left_high_q/%d.png' %(ROUTE, WEATHER, image_left.frame))

    
finally:
    # print("save coordinates to file...")
    # outF = open("%s/%s/real_coordinates_high_q.txt"  %(ROUTE, WEATHER), "w")
    # outF.write(str(real_cords))
    
    for actor in actor_list:
        actor.destroy()
    print("Cleaned up!")
