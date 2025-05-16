import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from itertools import combinations
import matplotlib.widgets as widgets
import time
import uuid

# تعریف ثابت‌ها
GRID_SIZE = 10
RADIUS = 2  # شعاع پوشش حسگرها
COMM_RADIUS = 2  # شعاع ارتباط حسگرها
K = 3  # حداقل تعداد حسگرها برای شناسایی
C = 2  # تعداد گروه‌های مختلف برای توقف

# لیست‌ها برای ذخیره ورودی‌ها
sensors = []
thieves = []
walls = []
exit_pos = None
input_mode = None  # حالت ورودی: 'sensors', 'thieves', 'walls', 'exit'
grid_colors = np.zeros((GRID_SIZE, GRID_SIZE))  # آرایه برای ردیابی رنگ‌ها
groups_cache = {}  # کش برای تابع count_groups

# تابع فاصله منهتن
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# ماتریس ارتباطی
def create_communication_matrix(sensors, comm_radius):
    n = len(sensors)
    comm_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and np.sqrt(sum((sensors[i][k] - sensors[j][k])**2 for k in range(2))) <= comm_radius:
                comm_matrix[i][j] = 1
    return comm_matrix

# بررسی اینکه آیا موقعیت در محدوده حسگر است
def in_sensor_range(sensor_pos, pos, radius):
    return np.sqrt(sum((sensor_pos[k] - pos[k])**2 for k in range(2))) <= radius

# محاسبه حسگرهایی که یک موقعیت را می‌بینند
def visible_sensors(pos, sensors, radius):
    return [i for i, s in enumerate(sensors) if in_sensor_range(s, pos, radius)]

# بررسی معتبر بودن حرکت
def valid_move(pos, new_pos, walls):
    if new_pos in walls or not (0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE):
        return False
    return abs(pos[0] - new_pos[0]) + abs(pos[1] - new_pos[1]) == 1

# محاسبه تعداد گروه‌های شناسایی‌کننده با کش
def count_groups(pos, sensors, comm_matrix, k):
    if pos in groups_cache:
        return groups_cache[pos]
    visible = visible_sensors(pos, sensors, RADIUS)
    if len(visible) < k:
        groups_cache[pos] = 0
        return 0
    groups = 0
    for comb in combinations(visible, k):
        connected = True
        for i in range(len(comb)):
            for j in range(i + 1, len(comb)):
                if comm_matrix[comb[i]][comb[j]] == 0:
                    connected = False
                    break
            if not connected:
                break
        if connected:
            groups += 1
    groups_cache[pos] = groups
    return groups

# پیش‌پردازش گرید برای افزودن دیوارهای مجازی
def preprocess_grid(sensors, walls, comm_matrix, k, c):
    virtual_walls = set(walls)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pos = (i, j)
            if pos not in sensors + thieves + [exit_pos] and count_groups(pos, sensors, comm_matrix, k) >= c:
                virtual_walls.add(pos)
    return list(virtual_walls)

# CSP برای تخصیص حسگرها به دزدان
def assign_sensors(thieves, sensors, k, comm_matrix):
    n_sensors = len(sensors)
    n_thieves = len(thieves)
    assignment = [[] for _ in range(n_thieves)]
    used_sensors = set()

    def is_valid_assignment(thief_idx, sensor_idx):
        if sensor_idx in used_sensors:
            return False
        visible = visible_sensors(thieves[thief_idx], sensors, RADIUS)
        if sensor_idx not in visible:
            return False
        if len(assignment[thief_idx]) >= k:
            return False
        current_sensors = assignment[thief_idx] + [sensor_idx]
        for i in range(len(current_sensors)):
            for j in range(i + 1, len(current_sensors)):
                if comm_matrix[current_sensors[i]][current_sensors[j]] == 0:
                    return False
        return True

    def backtrack(thief_idx):
        if thief_idx == n_thieves:
            return all(len(assignment[i]) >= k for i in range(n_thieves))
        for sensor_idx in range(n_sensors):
            if is_valid_assignment(thief_idx, sensor_idx):
                assignment[thief_idx].append(sensor_idx)
                used_sensors.add(sensor_idx)
                if backtrack(thief_idx + 1 if len(assignment[thief_idx]) >= k else thief_idx):
                    return True
                assignment[thief_idx].pop()
                used_sensors.remove(sensor_idx)
        return False

    if backtrack(0):
        return assignment
    return None

# CSP برای مسیریابی دزد با بهینه‌سازی
def find_path(thief_start, sensors, walls, exit_pos, comm_matrix, k, c):
    walls = preprocess_grid(sensors, walls, comm_matrix, k, c)
    max_steps = 2 * GRID_SIZE
    path = [thief_start]
    visited = {thief_start}

    def is_valid_step(current_path, next_pos):
        if next_pos in visited:
            return False
        if not valid_move(current_path[-1], next_pos, walls):
            return False
        groups = count_groups(next_pos, sensors, comm_matrix, k)
        if groups >= c:
            return False
        if len(current_path) >= 2:
            last_move = (current_path[-1][0] - current_path[-2][0], current_path[-1][1] - current_path[-2][1])
            new_move = (next_pos[0] - current_path[-1][0], next_pos[1] - current_path[-1][1])
            if last_move == (-new_move[0], -new_move[1]):
                return False
        return True

    def backtrack(current_path):
        if len(current_path) > max_steps:
            return False
        if current_path[-1] == exit_pos:
            return True
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        directions.sort(key=lambda d: manhattan_distance(
            (current_path[-1][0] + d[0], current_path[-1][1] + d[1]), exit_pos))
        for dx, dy in directions:
            next_pos = (current_path[-1][0] + dx, current_path[-1][1] + dy)
            if is_valid_step(current_path, next_pos):
                current_path.append(next_pos)
                visited.add(next_pos)
                if backtrack(current_path):
                    return True
                current_path.pop()
                visited.remove(next_pos)
        return False

    if backtrack(path):
        print(f"Path found for thief starting at {thief_start}: {path}")
        return path, False
    else:
        final_pos = path[-1] if path else thief_start
        groups = count_groups(final_pos, sensors, comm_matrix, k)
        is_stopped = groups >= c
        if is_stopped:
            print(f"Thief stopped at {final_pos} with {groups} detection groups")
        else:
            print(f"No valid path found for thief starting at {thief_start}")
        return path, is_stopped

# تابع برای دریافت ورودی گرافیکی با نمایش رنگی و اشکال
def setup_input_gui():
    global input_mode, grid_colors, sensors, thieves, walls, exit_pos
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(True)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    title = ax.set_title("Select Mode to Add Elements (Done to Finish)")

    # نمایش گرید رنگی
    grid = ax.imshow(grid_colors, cmap='jet', interpolation='nearest', alpha=0.5)

    # اشکال برای نمایش عناصر
    sensor_plots = []
    thief_plots = []
    wall_plots = []
    exit_plot = None

    def update_plots():
        nonlocal sensor_plots, thief_plots, wall_plots, exit_plot
        for plot in sensor_plots + thief_plots + wall_plots:
            plot.remove()
        if exit_plot is not None:
            exit_plot.remove()

        sensor_plots = []
        thief_plots = []
        wall_plots = []
        exit_plot = None

        for pos in sensors:
            plot = ax.scatter(pos[1], pos[0], color='blue', marker='o', s=100)
            sensor_plots.append(plot)

        for pos in thieves:
            plot = ax.scatter(pos[1], pos[0], color='green', marker='*', s=150)
            thief_plots.append(plot)

        for pos in walls:
            plot = ax.scatter(pos[1], pos[0], color='black', marker='s', s=200)
            wall_plots.append(plot)

        if exit_pos is not None:
            exit_plot = ax.scatter(exit_pos[1], exit_pos[0], color='red', marker='D', s=150)

    def onclick(event):
        global sensors, thieves, walls, exit_pos, input_mode, grid_colors
        if event.xdata is None or event.ydata is None or event.inaxes != ax:
            return
        x, y = int(round(event.ydata)), int(round(event.xdata))
        pos = (x, y)

        if event.button == 3:
            if pos in sensors:
                sensors.remove(pos)
                grid_colors[pos[0], pos[1]] = 0
                print(f"Sensor removed at: {pos}")
            elif pos in thieves:
                thieves.remove(pos)
                grid_colors[pos[0], pos[1]] = 0
                print(f"Thief removed at: {pos}")
            elif pos in walls:
                walls.remove(pos)
                grid_colors[pos[0], pos[1]] = 0
                print(f"Wall removed at: {pos}")
            elif pos == exit_pos:
                exit_pos = None
                grid_colors[pos[0], pos[1]] = 0
                print(f"Exit removed at: {pos}")
        else:
            if input_mode == 'sensors' and pos not in sensors + thieves + walls + [exit_pos]:
                sensors.append(pos)
                grid_colors[pos[0], pos[1]] = 1
                print(f"Sensor added at: {pos}")
            elif input_mode == 'thieves' and pos not in sensors + thieves + walls + [exit_pos]:
                thieves.append(pos)
                grid_colors[pos[0], pos[1]] = 2
                print(f"Thief added at: {pos}")
            elif input_mode == 'walls' and pos not in sensors + thieves + walls + [exit_pos]:
                walls.append(pos)
                grid_colors[pos[0], pos[1]] = 3
                print(f"Wall added at: {pos}")
            elif input_mode == 'exit' and exit_pos is None and pos not in sensors + thieves + walls:
                exit_pos = pos
                grid_colors[pos[0], pos[1]] = 4
                print(f"Exit set at: {pos}")

        grid.set_data(grid_colors)
        update_plots()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    ax_sensor = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_thief = plt.axes([0.81, 0.05, 0.1, 0.075])
    ax_wall = plt.axes([0.7, 0.15, 0.1, 0.075])
    ax_exit = plt.axes([0.81, 0.15, 0.1, 0.075])
    ax_done = plt.axes([0.75, 0.25, 0.1, 0.075])

    btn_sensor = widgets.Button(ax_sensor, 'Sensors')
    btn_thief = widgets.Button(ax_thief, 'Thieves')
    btn_wall = widgets.Button(ax_wall, 'Walls')
    btn_exit = widgets.Button(ax_exit, 'Exit')
    btn_done = widgets.Button(ax_done, 'Done')

    def set_mode_sensor(event):
        global input_mode
        input_mode = 'sensors'
        title.set_text("Adding Sensors (Right-click to Remove)")
        print("Mode set to: Adding Sensors")
        fig.canvas.flush_events()
        time.sleep(0.1)
        fig.canvas.draw_idle()

    def set_mode_thief(event):
        global input_mode
        input_mode = 'thieves'
        title.set_text("Adding Thieves (Right-click to Remove)")
        print("Mode set to: Adding Thieves")
        fig.canvas.flush_events()
        time.sleep(0.1)
        fig.canvas.draw_idle()

    def set_mode_wall(event):
        global input_mode
        input_mode = 'walls'
        title.set_text("Adding Walls (Right-click to Remove)")
        print("Mode set to: Adding Walls")
        fig.canvas.flush_events()
        time.sleep(0.1)
        fig.canvas.draw_idle()

    def set_mode_exit(event):
        global input_mode
        input_mode = 'exit'
        title.set_text("Setting Exit (Right-click to Remove)")
        print("Mode set to: Setting Exit")
        fig.canvas.flush_events()
        time.sleep(0.1)
        fig.canvas.draw_idle()

    def finish_input(event):
        plt.close()
        run_algorithm()

    btn_sensor.on_clicked(set_mode_sensor)
    btn_thief.on_clicked(set_mode_thief)
    btn_wall.on_clicked(set_mode_wall)
    btn_exit.on_clicked(set_mode_exit)
    btn_done.on_clicked(finish_input)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

# اجرای الگوریتم و ایجاد انیمیشن
def run_algorithm():
    global sensors, thieves, walls, exit_pos
    if not sensors or not thieves or exit_pos is None:
        print("Error: Sensors, thieves, and exit must be defined.")
        return

    comm_matrix = create_communication_matrix(sensors, COMM_RADIUS)
    assignments = assign_sensors(thieves, sensors, K, comm_matrix)
    if assignments:
        print("Sensor assignments to thieves:", assignments)
    else:
        print("Assignment not possible.")
        assignments = [[] for _ in thieves]

    paths = []
    frozen = set()
    for i, thief in enumerate(thieves):
        path, is_frozen = find_path(thief, sensors, walls, exit_pos, comm_matrix, K, C)
        paths.append(path)
        if is_frozen:
            frozen.add(i)
        print(f"Full path of Thief {i+1}:", path)
        if is_frozen:
            print(f"Thief {i+1} stopped!")

    create_animation(sensors, thieves, walls, exit_pos, paths, frozen)

# تابع برای ایجاد انیمیشن
def create_animation(sensors, thieves, walls, exit_pos, paths, frozen):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(True)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))

    for wall in walls:
        ax.scatter(wall[1], GRID_SIZE - 1 - wall[0], color='black', marker='s', s=200)

    for i, sensor in enumerate(sensors):
        ax.scatter(sensor[1], GRID_SIZE - 1 - sensor[0], color='blue', marker='o', s=100, label='Sensors' if i == 0 else "")
        circle = plt.Circle((sensor[1], GRID_SIZE - 1 - sensor[0]), RADIUS, color='blue', fill=False, linestyle='--')
        ax.add_patch(circle)

    ax.scatter(exit_pos[1], GRID_SIZE - 1 - exit_pos[0], color='green', marker='D', s=150, label='Exit')

    thief_plots = []
    path_plots = []
    detection_plots = []
    for i, thief in enumerate(thieves):
        color = 'red' if i in frozen else 'green'
        plot, = ax.plot([], [], color=color, marker='*', markersize=15, label=f'Thief {i+1}')
        path_plot, = ax.plot([], [], color=color, linestyle='-', linewidth=2)
        detection_plot, = ax.plot([], [], color='red', linestyle='--', label='Detection' if i == 0 else "")
        thief_plots.append(plot)
        path_plots.append(path_plot)
        detection_plots.append(detection_plot)

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Thief Tracking System - 07:09 PM CEST, May 16, 2025")

    def update(frame):
        for i, (thief, path) in enumerate(zip(thieves, paths)):
            step = min(frame, len(path) - 1)
            current_pos = path[step]
            color = 'red' if i in frozen else 'green'

            thief_plots[i].set_data([current_pos[1]], [GRID_SIZE - 1 - current_pos[0]])

            path_x = [p[1] for p in path[:step + 1]]
            path_y = [GRID_SIZE - 1 - p[0] for p in path[:step + 1]]
            path_plots[i].set_data(path_x, path_y)

            visible = visible_sensors(current_pos, sensors, RADIUS)
            if visible:
                det_x = []
                det_y = []
                for sensor_idx in visible:
                    sensor_pos = sensors[sensor_idx]
                    det_x.extend([current_pos[1], sensor_pos[1]])
                    det_y.extend([GRID_SIZE - 1 - current_pos[0], GRID_SIZE - 1 - sensor_pos[0]])
                detection_plots[i].set_data(det_x, det_y)
            else:
                detection_plots[i].set_data([], [])

        return thief_plots + path_plots + detection_plots

    frames = max(len(path) for path in paths) if paths else 1
    ani = FuncAnimation(fig, update, frames=frames, interval=1000, blit=True)

    writer = PillowWriter(fps=1)
    ani.save('thief_tracking_animation.gif', writer=writer)
    plt.close()

if __name__ == "__main__":
    setup_input_gui()