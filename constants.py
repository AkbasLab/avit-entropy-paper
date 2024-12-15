
seed = 4827
n_tests = 10_000

class RGBA:
    light_blue = (12,158,236,255)
    rosey_red = (244,52,84,255)
    cyan = (0,255,255,255)
    lime = (0,255,0,255)
    black = (0,0,0,255)
    dark_gray = (20,20,20,255)
    red = (255,0,0,255)
    yellow = (255,255,0,255)
    white = (255,255,255,255)

class sumo:
    gui = True
    start = True
    quit_on_end = True
    pause_after_initialze = False
    track_dut = True
    delay_ms = 100
    action_step_length = 0.25
    step_length = 0.25
    quiet_mode = True
    dut_zoom = 400
    lane_change_duration = 0.5
    show_polygons = True
    override_polygon_color = False
    polygon_color = RGBA.lime
    error_log_file = "temp/error.txt"
    gui_setting_file = "map/gui.xml"
    init_state_file = "temp/init-state.xml"
    default_view = 'View #0'
    config = {
        "--net-file" : "map/highway.net.xml",
        "--route-files" : "map/routes.rou.xml"
    }


ego_speed = 5
# approaching_lane_length = 500

DUT = "dut"
FOE1 = "foe1"
FOE2 = "foe2"
FOE3 = "foe3"
BIKE = "bike"
PED = "ped"

all_vehicles = [DUT, FOE1, FOE2, FOE3]
accel = 1

class car:
    wheelbase = 2.63
    max_steering_angle = 10
    length = 5
    influence = 1

class bike:
    wheelbase = 1.05
    max_steering_angle = 40
    length = 1.6
    influence = .9

class person:
    wheelbase = 0.5
    max_steering_angle = 40
    length = 0.214
    influence = .8

influence_map = {
    DUT : car.influence,
    FOE1 : car.influence,
    FOE2 : car.influence,
    FOE3 : car.influence,
    BIKE : bike.influence,
    PED : person.influence
}

"""
Distribution names
"""
class distribution:
    gaussian = "gaussian standard normal"
    uniform = "uniform random"

"""
Misc Kinematics Model configuration
"""
class kinematics_model:
    n_intervals_a = 1
    time_window = 3.    # seconds
    dt = 0.25           # seconds
    distribution = distribution.gaussian

"""
Graphics
"""
class graphics:
    draw_delay = 0.05
    length_map = {
        DUT : car.length,
        FOE1 : car.length,
        FOE2 : car.length,
        FOE3 : car.length,
        BIKE : bike.length,
        PED : person.length
    }
    actor_order = [DUT, FOE1, FOE2, FOE3, BIKE, PED]
    hatches = ["..", "//", "--", "**", "\\\\", "oo"]
    white = ["#ffffff","#ffffff","#ffffff","#ffffff","#ffffff","#ffffff"]
    hatch_map = {
        DUT : hatches[0],
        FOE1 : hatches[1],
        FOE2 : hatches[2],
        FOE3 : hatches[3],
        BIKE : hatches[4],
        PED : hatches[5]
    }