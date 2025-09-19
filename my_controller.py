import math
from controller import Robot
import matplotlib.pyplot as plt
import numpy as np

# === Initialization ===
robot = Robot()
TIME_STEP = 32             
MAX_SPEED = 6.28         
WHEEL_RADIUS = 0.0205     
LINEAR_SPEED = WHEEL_RADIUS * MAX_SPEED  
SCAN_INTERVAL = 20   
TOTAL_SIMULATION_TIME = 100 
Lines = []

# === Utility ===
def distance_to_line(pt, a, b):
    x0,y0 = pt; x1,y1 = a; x2,y2 = b
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = math.hypot(x2-x1, y2-y1)
    return num/den if den>0 else 0

def split_and_merge(P):
    if len(P) < 2: return
    a, b = P[0], P[-1]
    dists = [distance_to_line(pt, a, b) for pt in P]
    idx = np.argmax(dists)
    if dists[idx] > 2:
        split_and_merge(P[:idx+1])
        split_and_merge(P[idx:])
    else:
        Lines.append((a, b))

def get_heading_deg(compass_vals):
    rad = math.atan2(compass_vals[1], compass_vals[0])
    bearing = (rad - 1.5708) / math.pi * 180.0
    if bearing < 0: bearing += 360
    return (360 - bearing) % 360

def perform_lidar_scan(pos, heading_deg):
    raw = lidar.getRangeImage()
    max_r = lidar.getMaxRange()
    fov   = lidar.getFov()
    res   = lidar.getHorizontalResolution()
    step  = fov / res
    h_rad = math.radians(heading_deg)

    print(f"[DEBUG] first 10 ranges: {raw[:10]}")
    pts = []
    for i, d in enumerate(raw):
        if d == float('inf'):
            continue
        beam_ang = -fov/2 + i*step
        g_ang = h_rad + beam_ang
        xg = pos[0] + d * math.cos(g_ang)
        yg = pos[1] + d * math.sin(g_ang)
        pts.append((xg, yg))

    print(f"[DEBUG] got {len(pts)} valid points this scan (max_r={max_r:.2f} m)")
    return pts

# === Devices ===
left  = robot.getDevice('left wheel motor')
right = robot.getDevice('right wheel motor')
left.setPosition(float('inf')); right.setPosition(float('inf'))

compass = robot.getDevice('compass');   compass.enable(TIME_STEP)
lidar   = robot.getDevice('LDS-01');   lidar.enable(TIME_STEP); lidar.enablePointCloud()

# === Run ===
samples = []
pos = [0.0, 0.0]
total_steps = int((TOTAL_SIMULATION_TIME*1000)/TIME_STEP)
scan_every  = int((SCAN_INTERVAL*1000)/TIME_STEP)

for step in range(total_steps):
    robot.step(TIME_STEP)

    hdg = get_heading_deg(compass.getValues())
    dt = TIME_STEP/1000.0
    pos[0] += LINEAR_SPEED * dt * math.cos(math.radians(hdg))
    pos[1] += LINEAR_SPEED * dt * math.sin(math.radians(hdg))

    if step % scan_every == 0 and step != 0:
        pts = perform_lidar_scan(pos, hdg)
        samples.extend(pts)

    left.setVelocity(MAX_SPEED)
    right.setVelocity(MAX_SPEED)

# post-process & plot
split_and_merge(samples)

xs = [p[0] for p in samples]; ys = [p[1] for p in samples]
plt.figure(figsize=(6,6))
plt.scatter(xs, ys, s=2)
plt.title("Raw LIDAR Points"); plt.axis('equal'); plt.show()

plt.figure(figsize=(6,6))
for a,b in Lines:
    plt.plot([a[0],b[0]], [a[1],b[1]], '-o', markersize=2)
plt.title("Split & Merged Lines"); plt.axis('equal'); plt.show()

