"""
main.py
NavAgent 3D — PyBullet + LLaVA + Llama3

Usage:
    python main.py --scene FloorPlan1 --goal "find the television"
"""

import argparse
import time
import math
import pybullet as p

from env.thor_env import ThorEnv
from perception.vlm_describer import describe_scene
from planner.planner import NavPlanner

REPLAN_INTERVAL = 5


def check_success(env: ThorEnv, goal: str) -> bool:
    """Success only when agent is physically within 2m of the target object."""
    goal_lower = goal.lower()
    for body_id, label in env._object_labels.items():
        if label.lower() in goal_lower or any(
            w in label.lower() for w in goal_lower.split() if len(w) > 3
        ):
            pos, _ = p.getBasePositionAndOrientation(body_id)
            dist = math.hypot(pos[0] - env.agent_x, pos[1] - env.agent_y)
            if dist < 2.0:
                return True
    return False


def get_geometric_actions(env: ThorEnv, goal: str) -> list[str]:
    """Compute actions directly from coordinates — geometric fallback controller."""
    goal_lower = goal.lower()
    target_pos = None

    for body_id, label in env._object_labels.items():
        if label.lower() in goal_lower or any(
            w in label.lower() for w in goal_lower.split() if len(w) > 3
        ):
            pos, _ = p.getBasePositionAndOrientation(body_id)
            target_pos = pos
            break

    if target_pos is None:
        return ["RotateLeft"]

    dx = target_pos[0] - env.agent_x
    dy = target_pos[1] - env.agent_y
    dist = math.hypot(dx, dy)

    if dist < 2.0:
        return ["Stop"]

    # compute angle difference between agent facing and target direction
    angle_to_target = math.atan2(dy, dx)
    angle_diff = (angle_to_target - env.agent_yaw + math.pi) % (2 * math.pi) - math.pi

    if angle_diff > math.radians(30):
        return ["RotateLeft", "MoveAhead"]
    elif angle_diff < -math.radians(30):
        return ["RotateRight", "MoveAhead"]
    else:
        return ["MoveAhead", "MoveAhead"]


def run_episode(scene: str, goal: str, verbose: bool = True) -> dict:
    env     = ThorEnv(scene=scene)
    planner = NavPlanner()

    try:
        env.reset()
        start_time  = time.time()
        total_steps = 0
        success     = False

        print(f"\n{'='*50}")
        print(f"Scene : {scene}")
        print(f"Goal  : {goal}")
        print(f"{'='*50}")

        while not env.is_done() and not success:

            # ── Perceive ──────────────────────────────────────
            frame_b64       = env.get_frame_b64()
            scene_desc      = describe_scene(frame_b64, goal)
            visible_objects = env.get_visible_objects()

            if verbose:
                print(f"\n[Step {total_steps:02d}] Scene  : {scene_desc[:120]}...")
                print(f"          Visible: {visible_objects}")

            # coordinate-based success check
            if check_success(env, goal):
                success = True
                break

            # ── Plan ──────────────────────────────────────────
            plan    = planner.plan(scene_desc, goal, visible_objects)
            actions = plan.get("actions", [])

            # fallback: if parse error or planner only rotates, use geometric controller
            only_rotations = all(
                a in ["RotateLeft", "RotateRight", "LookUp", "LookDown"]
                for a in actions
            )
            if not actions or only_rotations:
                actions = get_geometric_actions(env, goal)
                if verbose:
                    print(f"          [Geometric fallback] Actions: {actions}")
            else:
                if verbose:
                    print(f"          Reason : {plan.get('reasoning', '')}")
                    print(f"          Actions: {actions}")

            # ── Execute ───────────────────────────────────────
            for action in actions[:REPLAN_INTERVAL]:
                if action == "Stop":
                    if check_success(env, goal):
                        success = True
                    break
                env.step(action)
                total_steps += 1
                if check_success(env, goal):
                    success = True
                    break

        elapsed = round(time.time() - start_time, 2)
        result  = {
            "scene":     scene,
            "goal":      goal,
            "success":   success,
            "steps":     total_steps,
            "latency_s": elapsed,
        }

        print(f"\n{'─'*50}")
        print(f"Result : {'✓ SUCCESS' if success else '✗ FAILED'}")
        print(f"Steps  : {total_steps}  |  Latency: {elapsed}s")
        return result

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--goal",  default="find the television")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_episode(scene=args.scene, goal=args.goal, verbose=not args.quiet)