"""Bounded RAVEN integration smoke test, run with Isaac Sim's python.sh.

Uses reference geometry or RAVEN_SMOKE_SCENE, the actual Pegasus PX4 node and
RAVEN's ZED/Ouster graph builders. No training, navigation, or scene download.
An outer container timeout must also cover Kit startup and blocked native calls.
"""
import json
from functools import partial
import os
from pathlib import Path
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "disable_viewport_updates": True})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("pegasus.simulator")

import pegasus.simulator.extension as pegasus_extension
import omni.graph.core as og
print("PEGASUS_SOURCE", pegasus_extension.__file__, flush=True)
print("PEGASUS_NODE_VERSION", og.GraphRegistry().get_node_type_version(
    "pegasus.simulator.PegasusMultirotorPX4Node"), flush=True)

import numpy as np
import omni.timeline
import omni.usd
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from pxr import UsdLux, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.ogn.api.spawn_multirotor import spawn_px4_multirotor_node
from pegasus.simulator.ogn.api.spawn_zed_camera import add_zed_stereo_camera_subgraph
from pegasus.simulator.ogn.api.spawn_ouster_lidar import add_ouster_lidar_subgraph
import importlib
drone_nodes = importlib.import_module("pegasus.simulator.ogn.python.nodes.OgnPegasusMultirotorNodeBase")
from pegasus.simulator.params import ROBOTS

report = {"passed": False, "received": {}, "px4_heartbeat": False, "steps": 0,
          "isaac_version": Path("/isaac-sim/VERSION").read_text().strip(),
          "scene": os.environ.get("RAVEN_SMOKE_SCENE", "") or "smoke_geometry"}
subscriptions = []
node = None
output = Path(os.environ.get("RAVEN_SMOKE_REPORT", "/tmp/raven-isaac6-smoke.json"))


def record(label, message):
    if isinstance(message, (Image, PointCloud2)):
        if not message.width or not message.height or not len(message.data):
            return
    if isinstance(message, CameraInfo) and (not message.width or not message.k[0]):
        return
    if isinstance(message, Clock):
        report["clock_seconds"] = message.clock.sec + message.clock.nanosec / 1e9
    report["received"][label] = report["received"].get(label, 0) + 1


try:
    assert report["isaac_version"].startswith("6.0.0"), report["isaac_version"]
    domain = int(os.environ.get("ROS_DOMAIN_ID", "213"))
    rclpy.init(domain_id=domain)
    node = rclpy.create_node("raven_isaac6_smoke")
    topics = {
        "clock": (Clock, "/clock"),
        "left_rgb": (Image, "/robot_1/sensors/front_stereo/left/image_rect"),
        "right_rgb": (Image, "/robot_1/sensors/front_stereo/right/image_rect"),
        "depth": (Image, "/robot_1/sensors/front_stereo/left/depth_ground_truth"),
        "camera_info": (CameraInfo, "/robot_1/sensors/front_stereo/left/camera_info"),
        "lidar": (PointCloud2, "/robot_1/sensors/lidar/point_cloud"),
    }
    for label, (message_type, topic) in topics.items():
        subscriptions.append(node.create_subscription(
            message_type, topic, partial(record, label), qos_profile_sensor_data
        ))

    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world
    scene = os.environ.get("RAVEN_SMOKE_SCENE", "")
    if scene:
        scene_path = Path("/isaac-sim/AirStack/scenes") / scene / f"{scene}_Export.usd"
        assert scene_path.is_file(), scene_path
        pg.load_environment(str(scene_path))
    else:
        world.scene.add_default_ground_plane()
        world.scene.add(FixedCuboid(
            prim_path="/World/obstacle", position=np.array([5.0, 0.0, 2.0]),
            scale=np.array([1.0, 6.0, 4.0]), color=np.array([0.8, 0.25, 0.1]),
        ))
    stage = omni.usd.get_context().get_stage()
    if not scene:
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)
    graph = spawn_px4_multirotor_node(
        pegasus_node_name="PX4Multirotor", drone_prim="/World/base_link",
        usd_file=ROBOTS["Iris"], vehicle_id=1, domain_id=domain,
        init_pos=[0.0, 0.0, 1.0], robot_name="robot_1",
    )
    add_zed_stereo_camera_subgraph(
        graph, "/World/base_link", robot_name="robot_1", frame_width=320, frame_height=200,
    )
    add_ouster_lidar_subgraph(
        graph, "/World/base_link", robot_name="robot_1", lidar_min_range=0.3,
    )
    for _ in range(30):
        app.update()
    world.reset()
    omni.timeline.get_timeline_interface().play()
    # First-use MDL/shader compilation can take minutes. The outer container
    # timeout covers it; measure sensor readiness after this warm-up frame.
    print("SMOKE_WARMUP", report["scene"], flush=True)
    world.step(render=True)
    report["physics_scenes"] = [str(prim.GetPath()) for prim in stage.Traverse()
                                if prim.IsA(UsdPhysics.Scene)]
    assert report["physics_scenes"] == [world.get_physics_context().prim_path], report["physics_scenes"]
    clock_attribute = og.Controller.attribute(
        "/World/base_link/PX4MultirotorGraph/robot_1_IsaacReadSimTime.outputs:simulationTime")
    clock_start = clock_attribute.get()
    physics_start = world.current_time
    deadline = time.monotonic() + float(os.environ.get("RAVEN_SMOKE_SECONDS", "180"))
    last_progress = time.monotonic()
    while app.is_running() and time.monotonic() < deadline:
        world.step(render=True)
        report["steps"] += 1
        rclpy.spin_once(node, timeout_sec=0.0)
        for data in drone_nodes.drone_sim_dict.values():
            backend = data["backend"]
            report["px4_heartbeat"] |= bool(getattr(backend, "_received_first_hearbeat", False))
            position = data["multirotor"].state.position
            assert np.isfinite(position).all(), f"Invalid vehicle position: {position}"
            report["vehicle_position"] = position.tolist()
        if time.monotonic() - last_progress > 5:
            print("SMOKE_PROGRESS " + json.dumps(report), flush=True)
            last_progress = time.monotonic()
        if (report["px4_heartbeat"] and report["steps"] >= 100
                and report.get("clock_seconds", 0) > 0
                and all(report["received"].get(label, 0) >= 3 for label in topics)):
            report["physics_elapsed"] = world.current_time - physics_start
            report["ros_clock_elapsed"] = clock_attribute.get() - clock_start
            assert abs(report["physics_elapsed"] - report["ros_clock_elapsed"]) < 0.05, (
                "ROS clock and Pegasus physics diverged", report)
            report["passed"] = True
            break
    assert report["passed"], "Missing PX4/physics/ROS sensor data before timeout: " + json.dumps(report)
except BaseException as exc:
    report["error"] = repr(exc)
    raise
finally:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("SMOKE_RESULT " + json.dumps(report), flush=True)
    for data in list(drone_nodes.drone_sim_dict.values()):
        try:
            data["backend"].stop()
        except Exception as exc:
            print(f"Backend cleanup: {exc}", flush=True)
    omni.timeline.get_timeline_interface().stop()
    if node is not None:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    app.close()
