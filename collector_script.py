import glob
import os
import sys


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla 
import math 
import random 
import cv2
import numpy as np
import pandas as pd
from queue import Queue
from datetime import datetime

# Connect the client
client = carla.Client('localhost', 2000) 
# client.set_timeout(10.0)

# Show available maps
for map_carla in client.get_available_maps():
    print(map_carla)

# Setup world
# world = client.load_world('Town01'))
world = client.get_world()

# Customize weather
# weather = carla.WeatherParameters(cloudiness=10.0, precipitation=10.0, fog_density=10.0)
# world.set_weather(weather)

# Run sync mode to prevent server to continue the simulation
# before the client finish processing the current frame
original_settings = world.get_settings()
settings = world.get_settings()
settings.fixed_delta_seconds = 0.05
settings.synchronous_mode = True
world.apply_settings(settings)
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Init blueprints
bp = world.get_blueprint_library()  

depth_cam_bp = bp.find('sensor.camera.depth')
depth_cam_bp.set_attribute('image_size_x', '2560')
depth_cam_bp.set_attribute('image_size_y', '720')
depth_cam_bp.set_attribute('fov', '120.0')

dash_cam_bp = bp.find('sensor.camera.rgb')
dash_cam_bp.set_attribute('image_size_x', '3840')
dash_cam_bp.set_attribute('image_size_y', '2160')
dash_cam_bp.set_attribute('fov', '140.0')

radar_bp = bp.find('sensor.other.radar')
radar_bp.set_attribute('horizontal_fov', '120')
radar_bp.set_attribute('vertical_fov', '20')
radar_bp.set_attribute('points_per_second', '10000')
radar_bp.set_attribute('range', '250')

actor_list = []

# Add the ego vehicle
spawn_points = world.get_map().get_spawn_points() 
vehicle_bp = bp.find('vehicle.tesla.model3') 
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
actor_list.append(vehicle)

# Move the spectator behind the vehicle to view it
spectator = world.get_spectator() 
vehicle_location, vehicle_rotation = vehicle.get_transform().location, vehicle.get_transform().rotation
camera_location = vehicle_location + carla.Location(x=2, z=2)
camera_rotation = vehicle_rotation
camera_rotation.yaw += 0
transform = carla.Transform(camera_location, camera_rotation)
spectator.set_transform(transform)
world.tick()

# Add traffic
for i in range(10): 
    vehicle_bp = random.choice(bp.filter('vehicle')) 
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points)) 
    if npc is not None:
        actor_list.append(npc)

# Set it in motion
for v in world.get_actors().filter('*vehicle*'): 
    v.set_autopilot(True)
# vehicle.set_autopilot(False)

# Define sensors queue
sensor_queue = Queue()


# Define dash-cam feed callbacks
def dash_cam_feed_handler(image, sensor_queue):
    timestamp = datetime.now().timestamp()
    image.save_to_disk(os.path.join(f'outputs/dash/frame-{image.frame}-ts-{timestamp}.png'))
    sensor_data = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    sensor_queue.put((image.frame, 'dash_cam', timestamp, sensor_data))
    
    
# Define depth-cam feed callbacks
def depth_cam_feed_handler(image, sensor_queue):
    timestamp = datetime.now().timestamp()
    image.convert(carla.ColorConverter.LogarithmicDepth)
    image.save_to_disk(os.path.join(f'outputs/depth/frame-{image.frame}-ts-{timestamp}.png'))
    sensor_data = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    sensor_queue.put((image.frame, 'depth_cam', timestamp, sensor_data))
    
    
# Define radar feed callbacks
def radar_feed_handler(sensor_data, sensor_queue):
    timestamp = datetime.now().timestamp()
    radar_data = np.zeros((len(sensor_data), 7))
    for i, detection in enumerate(sensor_data):
        x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
        y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
        z = detection.depth * math.sin(detection.altitude)
        radar_data[i, :] = [x, y, z, detection.velocity, detection.depth, detection.altitude, detection.azimuth]
    df = pd.DataFrame(radar_data, columns=['x', 'y', 'z', 'velocity', 'depth', 'altitude', 'azimuth'])
    df.to_csv(os.path.join(f'outputs/radar/frame-{sensor_data.frame}-ts-{timestamp}.csv'), index=False)
    sensor_queue.put((sensor_data.frame, 'radar', timestamp, radar_data))

# Set depth-camera
depth_cam = world.spawn_actor(depth_cam_bp, carla.Transform(carla.Location(x=2, z=2)), attach_to=vehicle)
actor_list.append(depth_cam)

# Set dash-camera
dash_cam = world.spawn_actor(dash_cam_bp, carla.Transform(carla.Location(x=-2, z=2), carla.Rotation(yaw=180)), attach_to=vehicle)
actor_list.append(dash_cam)

# Set radar
radar = world.spawn_actor(radar_bp, carla.Transform(carla.Location(x=2, z=2)), attach_to=vehicle)
actor_list.append(radar)

depth_cam.listen(lambda data: depth_cam_feed_handler(data, sensor_queue))
dash_cam.listen(lambda data: dash_cam_feed_handler(data, sensor_queue))
radar.listen(lambda data: radar_feed_handler(data, sensor_queue))

# vehicle.set_autopilot(True)
try:
    while True:
        world.tick()
        frame_id, sensor_name, timestamp, sensor_data = sensor_queue.get(block=True)
        # if sensor_name == 'dash_cam':
        #     cv2.imshow('Dash Cam', sensor_data)
        # elif sensor_name == 'depth_cam':
        #     cv2.imshow('Depth Cam', sensor_data)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        print(frame_id, sensor_name, timestamp)
except Exception as e:
    print(e)
finally:
    world.apply_settings(original_settings)
    cv2.destroyAllWindows()
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])




