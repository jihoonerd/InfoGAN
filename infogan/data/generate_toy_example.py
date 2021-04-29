import numpy as np


def generate_circle_toy_data():
    x1 = np.arange(-5, 5, 0.1)
    y1 = np.sqrt(5**2 - x1**2)

    x2 = np.arange(-5, 5, 0.1)
    y2 = -np.sqrt(5**2 - x2**2)

    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)



def generate_circle_toy_data_by_angle():
        
    start_theta = 180
    target_theta = 0

    if start_theta > target_theta:
        cw_path = np.linspace(start_theta, target_theta, 30)
        ccw_path = np.linspace(start_theta, target_theta + 360, 30)
    else:
        cw_path = np.linspace(start_theta + 360, target_theta, 30)
        ccw_path = np.linspace(start_theta, target_theta, 30)
    
    cw_x = np.array([np.cos(np.radians(degree % 360)) for degree in cw_path]).reshape(-1, 1)
    cw_y = np.array([np.sin(np.radians(degree % 360)) for degree in cw_path]).reshape(-1, 1)

    ccw_x = np.array([np.cos(np.radians(degree % 360)) for degree in ccw_path]).reshape(-1, 1)
    ccw_y = np.array([np.sin(np.radians(degree % 360)) for degree in ccw_path]).reshape(-1, 1)

    cw_coord = np.concatenate([cw_x, cw_y], axis=1)
    ccw_coord = np.concatenate([ccw_x, ccw_y], axis=1)
    
    return cw_coord, ccw_coord
    