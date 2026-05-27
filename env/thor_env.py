"""
thor_env.py
PyBullet-based 3D environment — replaces AI2-THOR/Minigrid.
Renders real 3D first-person RGB frames for the VLM.
Works on Windows, macOS, Linux. No binary downloads.

pip install pybullet
"""

import math
import base64
import io
from random import random
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

# ── Camera settings ────────────────────────────────────────────────────
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FOV    = 60
CAM_NEAR   = 0.1
CAM_FAR    = 50.0
EYE_HEIGHT = 0.6    # agent eye height in meters

# ── Movement settings ──────────────────────────────────────────────────
MOVE_STEP   = 0.5           # meters per MoveAhead
ROTATE_STEP = math.pi / 4   # 45° per RotateLeft / RotateRight

# ── Room geometry ──────────────────────────────────────────────────────
ROOM_HALF = 5.0
WALL_H    = 2.5
WALL_T    = 0.1

# ── Scene definitions ──────────────────────────────────────────────────
# Each entry: (label, [x,y,z_center], [r,g,b,a], [half_x, half_y, half_z])
SCENES = {
    "FloorPlan1": [
        ("television",  [ 4.0,  0.0, 0.6], [0.1, 0.1, 0.7, 1], [0.40, 0.08, 0.55]),
        ("sofa",        [ 0.0,  4.0, 0.4], [0.5, 0.2, 0.1, 1], [1.00, 0.40, 0.40]),
        ("chair",       [ 2.5,  2.5, 0.4], [0.7, 0.5, 0.1, 1], [0.35, 0.35, 0.40]),
        ("table",       [ 1.5, -2.0, 0.3], [0.4, 0.2, 0.0, 1], [0.70, 0.70, 0.30]),
        ("lamp",        [-3.0,  1.5, 0.7], [0.9, 0.9, 0.0, 1], [0.15, 0.15, 0.70]),
    ],
    "FloorPlan2": [
        ("laptop",      [ 3.0,  1.0, 0.5], [0.2, 0.6, 0.2, 1], [0.25, 0.18, 0.08]),
        ("desk",        [ 3.0,  0.0, 0.4], [0.4, 0.2, 0.0, 1], [0.80, 0.40, 0.40]),
        ("bookshelf",   [-3.5,  0.0, 0.9], [0.3, 0.1, 0.0, 1], [0.45, 0.20, 0.90]),
        ("plant",       [ 0.0, -3.5, 0.5], [0.0, 0.6, 0.0, 1], [0.25, 0.25, 0.50]),
    ],
    "FloorPlan201": [
        ("refrigerator",[ 4.0,  0.0, 0.9], [0.8, 0.8, 0.9, 1], [0.40, 0.35, 0.90]),
        ("sink",        [ 0.0,  4.0, 0.5], [0.7, 0.7, 0.9, 1], [0.35, 0.25, 0.35]),
        ("stove",       [ 2.0,  2.0, 0.5], [0.3, 0.3, 0.3, 1], [0.50, 0.40, 0.45]),
        ("microwave",   [-2.0,  3.0, 0.8], [0.5, 0.5, 0.5, 1], [0.28, 0.20, 0.20]),
    ],
    "FloorPlan301": [
        ("toilet",      [ 3.0,  1.5, 0.5], [0.9, 0.9, 0.9, 1], [0.28, 0.25, 0.50]),
        ("bathtub",     [-2.0,  2.0, 0.3], [0.8, 0.8, 1.0, 1], [0.80, 0.38, 0.32]),
        ("sink",        [ 0.0,  3.5, 0.5], [0.7, 0.7, 0.9, 1], [0.28, 0.20, 0.30]),
        ("mirror",      [ 0.0, -4.5, 0.9], [0.6, 0.8, 0.9, 1], [0.50, 0.05, 0.70]),
    ],
    "FloorPlan401": [
        ("bed",         [ 3.5,  0.0, 0.4], [0.6, 0.4, 0.8, 1], [1.00, 0.70, 0.35]),
        ("dresser",     [-3.5,  0.0, 0.6], [0.4, 0.2, 0.0, 1], [0.50, 0.28, 0.60]),
        ("nightstand",  [ 3.5,  1.8, 0.4], [0.3, 0.2, 0.0, 1], [0.28, 0.28, 0.38]),
        ("pillow",      [ 3.3, -0.3, 0.8], [0.9, 0.8, 0.7, 1], [0.28, 0.18, 0.10]),
    ],
}


class ThorEnv:
    ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Stop"]

    def __init__(self, scene: str = "FloorPlan1", max_steps: int = 50):
        self.scene     = scene
        self.max_steps = max_steps
        self.step_count = 0 
        self.agent_x     = 0.0 
        self.agent_y     = 0.0 
        self.agent_yaw   = math.pi
        self.client    = p.connect(p.DIRECT)   # headless — no display required
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._object_labels = {}
        self.reset()

    # ── Scene loading ──────────────────────────────────────────────────

    def _load_scene(self) -> None:
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Floor
        p.loadURDF("plane.urdf")

        # Ceiling (light grey)
        self._add_box(
            [0, 0, WALL_H], [ROOM_HALF, ROOM_HALF, WALL_T],
            [0.95, 0.95, 0.95, 1], label=None
        )

        # Walls
        wall_color = [0.88, 0.86, 0.82, 1]
        half_w = ROOM_HALF + WALL_T
        for pos, half in [
            ([ ROOM_HALF, 0, WALL_H/2], [WALL_T,  half_w,  WALL_H/2]),
            ([-ROOM_HALF, 0, WALL_H/2], [WALL_T,  half_w,  WALL_H/2]),
            ([0,  ROOM_HALF, WALL_H/2], [half_w,  WALL_T,  WALL_H/2]),
            ([0, -ROOM_HALF, WALL_H/2], [half_w,  WALL_T,  WALL_H/2]),
        ]:
            self._add_box(pos, half, wall_color, label=None)

        # Furniture objects
        self._object_labels = {}
        for entry in SCENES.get(self.scene, SCENES["FloorPlan1"]):
            label, pos, color, half_ext = entry
            body_id = self._add_box(pos, half_ext, color, label=label)
            self._object_labels[body_id] = label
        # add at the very end of _load_scene():
        print(f"[Env] Loaded {len(self._object_labels)} objects: {list(self._object_labels.values())}")

    def _add_box(self, pos, half_extents, color, label=None) -> int:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=pos)

    # ── Camera helpers ─────────────────────────────────────────────────

    def _view_matrix(self) -> list:
        yaw = self.agent_yaw
        fx  = self.agent_x + math.cos(yaw) * 0.1
        fy  = self.agent_y + math.sin(yaw) * 0.1
        return p.computeViewMatrix(
            cameraEyePosition   =[self.agent_x, self.agent_y, EYE_HEIGHT],
            cameraTargetPosition=[fx, fy, EYE_HEIGHT],
            cameraUpVector      =[0, 0, 1],
        )

    def _proj_matrix(self) -> list:
        return p.computeProjectionMatrixFOV(
            CAM_FOV, CAM_WIDTH / CAM_HEIGHT, CAM_NEAR, CAM_FAR
        )

    # ── Public API ─────────────────────────────────────────────────────

    def reset(self, scene: str = None) -> None:
        if scene:
            self.scene = scene
        self.agent_x    = 0.0
        self.agent_y    = 0.0
        self.agent_yaw  = math.pi   # face -x (away from TV at [4,0])
        self.step_count = 0
        self._load_scene()

    def step(self, action: str) -> None:
        assert action in self.ACTIONS, f"Unknown action: {action}"
        if action == "MoveAhead":
            nx = self.agent_x + math.cos(self.agent_yaw) * MOVE_STEP
            ny = self.agent_y + math.sin(self.agent_yaw) * MOVE_STEP
            # Clamp within room bounds
            self.agent_x = max(-ROOM_HALF + 0.6, min(ROOM_HALF - 0.6, nx))
            self.agent_y = max(-ROOM_HALF + 0.6, min(ROOM_HALF - 0.6, ny))
        elif action == "RotateLeft":
            self.agent_yaw += ROTATE_STEP
        elif action == "RotateRight":
            self.agent_yaw -= ROTATE_STEP
        # LookUp / LookDown / Stop are no-ops (handled in main.py)
        p.stepSimulation()
        self.step_count += 1

    def get_frame_b64(self) -> str:
        """Render first-person 3D frame and return as base64 JPEG."""
        _, _, rgb_px, _, _ = p.getCameraImage(
            CAM_WIDTH, CAM_HEIGHT,
            self._view_matrix(),
            self._proj_matrix(),
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.array(rgb_px, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH, 4)[:, :, :3]
        img = Image.fromarray(rgb)
        img = img.resize((320, 240), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # def get_visible_objects(self) -> list[str]:
    #     visible = []
    #     for body_id, label in self._object_labels.items():
    #         pos, _ = p.getBasePositionAndOrientation(body_id)
    #         dx = pos[0] - self.agent_x
    #         dy = pos[1] - self.agent_y
    #         dist = math.hypot(dx, dy)

    #         # Distance threshold: scales with room size
    #         if dist > 6.0:
    #             continue

    #         # Angle from agent's forward direction to the object
    #         angle_to_obj = math.atan2(dy, dx)
    #         angle_diff = (angle_to_obj - self.agent_yaw + math.pi) % (2 * math.pi) - math.pi
    #         angle_diff = abs(angle_diff)

    #         # FOV widens as agent gets closer (near=120°, far=60°)
    #         fov = math.radians(120 - 60 * (dist / 6.0))

    #         if angle_diff < fov / 2:
    #             visible.append(label)

    #     return visible
    def get_visible_objects(self) -> list[str]:
        visible = []
        for body_id, label in self._object_labels.items():
            pos, _ = p.getBasePositionAndOrientation(body_id)
            dist = math.hypot(pos[0] - self.agent_x, pos[1] - self.agent_y)
            print(f"[Vis] {label}: dist={dist:.2f}m agent=({self.agent_x:.1f},{self.agent_y:.1f})")
            if dist < 6.0:
                visible.append(label)
        return visible

    def is_done(self) -> bool:
        return self.step_count >= self.max_steps

    def close(self) -> None:
        p.disconnect(self.client)
