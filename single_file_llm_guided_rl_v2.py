
from __future__ import annotations
import argparse
import copy
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gymnasium as gym
import highway_env  # noqa: F401  # registers envs


# ============================================================
# configs
# ============================================================

ALL_MODES = [
    "baseline_sac",
    "shaping_sac",
    "rule_hier",
    "llm_hier",
    "real_llm_hier",
    "constrained_rule_hier",
    "constrained_llm_hier",
    "constrained_real_llm_hier",
]


@dataclass
class EnvConfig:
    env_id: str = "highway-v0"
    lanes_count: int = 4
    vehicles_count: int = 20
    duration: int = 40
    simulation_frequency: int = 15
    policy_frequency: int = 5
    lane_width: float = 4.0
    vehicles_obs_count: int = 8
    max_speed: float = 32.0
    min_speed: float = 0.0
    offscreen_rendering: bool = True
    normalize_obs: bool = True
    absolute_obs: bool = False
    order: str = "sorted"

    def to_gym_config(self) -> Dict[str, Any]:
        return {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": self.vehicles_obs_count,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "normalize": self.normalize_obs,
                "absolute": self.absolute_obs,
                "order": self.order,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
            },
            "lanes_count": self.lanes_count,
            "vehicles_count": self.vehicles_count,
            "duration": self.duration,
            "simulation_frequency": self.simulation_frequency,
            "policy_frequency": self.policy_frequency,
            "offscreen_rendering": self.offscreen_rendering,
        }


@dataclass
class PlannerConfig:
    planner_interval: int = 5
    waypoint_horizon: int = 5
    waypoint_gap: float = 8.0
    replan_ttc_threshold: float = 2.5
    replan_front_distance: float = 12.0
    target_speed_free: float = 28.0
    target_speed_cautious: float = 22.0
    target_speed_brake: float = 16.0
    conservative_headway: float = 18.0
    normal_headway: float = 12.0
    speed_max: float = 30.0
    acc_max: float = 3.0
    steer_max: float = 0.30
    comfort_acc_delta: float = 1.5
    print_prompt_on_replan: bool = False
    print_real_llm_error: bool = True


@dataclass
class RewardConfig:
    use_env_reward: bool = False
    env_reward_scale: float = 0.1
    w_progress: float = 1.0
    w_waypoint_pos: float = 0.6
    w_waypoint_speed: float = 0.25
    w_collision: float = 5.0
    w_headway: float = 1.2
    w_overspeed: float = 0.8
    w_action: float = 0.05
    w_action_smooth: float = 0.03
    w_lane_change: float = 0.2


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    lambda_lr: float = 1e-3
    hidden_dim: int = 256
    batch_size: int = 128
    buffer_size: int = 100000
    start_steps: int = 2000
    update_after: int = 1000
    update_every: int = 1
    target_entropy: float = -2.0

    cost_limit_collision: float = 0.02
    cost_limit_headway: float = 0.10
    cost_limit_overspeed: float = 0.10
    cost_limit_comfort: float = 0.15


@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 10
    max_steps_per_episode: int = 200
    eval_every: int = 1
    device: str = "cpu"
    mode: str = "rule_hier"
    render: bool = False
    print_every_step: bool = True
    use_shield_during_train: bool = False


@dataclass
class EvalConfig:
    enabled: bool = True
    episodes: int = 3
    seed_offset: int = 1000
    print_step: bool = False


@dataclass
class SafetyConfig:
    verifier_enabled: bool = True
    shield_enabled_eval: bool = True
    shield_enabled_train: bool = False

    ttc_safe: float = 2.0
    min_front_distance_hard: float = 8.0
    lane_offset_hard: float = 0.9

    emergency_brake_acc: float = -0.8
    overspeed_brake_acc: float = -0.4
    shield_steer_scale: float = 0.5

    cost_collision_weight: float = 1.0
    cost_headway_weight: float = 1.0
    cost_overspeed_weight: float = 0.5
    cost_comfort_weight: float = 0.2


@dataclass
class LLMConfig:
    backend: str = "mock"  # mock | real
    model_name: str = "gpt-4o-mini"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1/chat/completions"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_sec: int = 20
    system_prompt: str = (
        "You are a highway autonomous driving planner. "
        "Return JSON only, no markdown, no extra commentary."
    )
    fallback_to_mock_on_error: bool = True
    fallback_to_mock_on_parse_error: bool = True


@dataclass
class CompareConfig:
    enabled: bool = False
    modes: List[str] = field(default_factory=lambda: [
        "baseline_sac",
        "shaping_sac",
        "rule_hier",
        "llm_hier",
        "constrained_rule_hier",
        "constrained_llm_hier",
    ])
    disable_step_print_during_compare: bool = True
    save_json_path: str = ""


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)


def get_config() -> Config:
    return Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# mode helpers
# ============================================================

RULE_MODES = {"rule_hier", "constrained_rule_hier"}
MOCK_LLM_MODES = {"llm_hier", "constrained_llm_hier"}
REAL_LLM_MODES = {"real_llm_hier", "constrained_real_llm_hier"}
LLM_MODES = MOCK_LLM_MODES | REAL_LLM_MODES
PLANNER_MODES = RULE_MODES | LLM_MODES
CONSTRAINED_MODES = {"constrained_rule_hier", "constrained_llm_hier", "constrained_real_llm_hier"}


def mode_uses_planner(mode: str) -> bool:
    return mode in PLANNER_MODES


def mode_uses_rule_planner(mode: str) -> bool:
    return mode in RULE_MODES


def mode_uses_llm_planner(mode: str) -> bool:
    return mode in LLM_MODES


def mode_uses_real_llm(mode: str, cfg: Config) -> bool:
    if mode in REAL_LLM_MODES:
        return True
    if mode in MOCK_LLM_MODES:
        return False
    return cfg.llm.backend == "real"


def mode_is_constrained(mode: str) -> bool:
    return mode in CONSTRAINED_MODES


# ============================================================
# env wrapper
# ============================================================

class HighwayEnvWrapper:
    def __init__(self, cfg, render: bool = False):
        self.cfg = cfg
        render_mode = "human" if render else None
        self.env = gym.make(
            cfg.env.env_id,
            config=cfg.env.to_gym_config(),
            render_mode=render_mode,
        )
        self.current_waypoints = None
        self.current_constraints = None
        self.last_obs = None
        self.last_action = np.zeros(2, dtype=np.float32)

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        self.last_obs = obs
        self.last_action = np.zeros(2, dtype=np.float32)

        scene = self.get_scene_dict()
        self.current_waypoints = self._default_waypoints(scene)
        self.current_constraints = self._default_constraints()

        low_state = self.build_low_state(obs)
        return low_state, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = next_obs
        self.last_action = action.copy()
        low_state = self.build_low_state(next_obs)
        return low_state, float(env_reward), bool(terminated), bool(truncated), info

    def apply_plan(self, waypoints, constraints):
        self.current_waypoints = np.asarray(waypoints, dtype=np.float32)
        self.current_constraints = dict(constraints)

    def build_low_state(self, obs) -> np.ndarray:
        scene = self.get_scene_dict()
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        wp_feat = self._waypoint_features(scene, self.current_waypoints)
        c_feat = self._constraint_features(self.current_constraints)

        scene_feat = np.array([
            scene["ego_speed"] / max(self.cfg.env.max_speed, 1e-6),
            np.clip(scene["front_distance"] / 100.0, 0.0, 1.0),
            np.clip(scene["front_rel_speed"] / 20.0, -1.0, 1.0),
            np.clip(scene["lane_id"] / max(self.cfg.env.lanes_count - 1, 1), 0.0, 1.0),
            np.clip(scene["ttc"] / 10.0, 0.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([obs_flat, wp_feat, c_feat, scene_feat], axis=0)

    def get_scene_dict(self) -> dict:
        base_env = self.env.unwrapped
        ego = base_env.vehicle

        ego_x = float(ego.position[0])
        ego_y = float(ego.position[1])
        ego_speed = float(getattr(ego, "speed", 0.0))
        heading = float(getattr(ego, "heading", 0.0))
        lane_id = self._lane_id(ego)

        front_same = self._nearest_vehicle(ego, lane_id)
        left_lane = lane_id - 1 if lane_id > 0 else None
        right_lane = lane_id + 1 if lane_id < self.cfg.env.lanes_count - 1 else None
        front_left = self._nearest_vehicle(ego, left_lane) if left_lane is not None else None
        front_right = self._nearest_vehicle(ego, right_lane) if right_lane is not None else None

        front_distance, front_rel_speed = self._vehicle_relation(ego, front_same)
        left_front_distance, _ = self._vehicle_relation(ego, front_left)
        right_front_distance, _ = self._vehicle_relation(ego, front_right)

        closing_speed = max(0.0, -front_rel_speed)
        ttc = front_distance / max(1e-3, closing_speed) if front_distance < 1e5 and closing_speed > 0 else 99.0

        return {
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_speed": ego_speed,
            "heading": heading,
            "lane_id": lane_id,
            "lane_center_y": lane_id * self.cfg.env.lane_width,
            "left_lane_id": left_lane,
            "right_lane_id": right_lane,
            "front_distance": front_distance,
            "front_rel_speed": front_rel_speed,
            "left_front_distance": left_front_distance,
            "right_front_distance": right_front_distance,
            "left_clear": left_front_distance > 20.0 if left_lane is not None else False,
            "right_clear": right_front_distance > 20.0 if right_lane is not None else False,
            "ttc": ttc,
            "collision": bool(getattr(ego, "crashed", False)),
            "last_action": self.last_action.copy(),
        }

    def action_dim(self) -> int:
        return int(np.prod(self.env.action_space.shape))

    def close(self):
        self.env.close()

    def _lane_id(self, vehicle) -> int:
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None:
            return 0
        if isinstance(lane_index, tuple):
            return int(lane_index[-1])
        return int(lane_index)

    def _nearest_vehicle(self, ego, target_lane_id: int | None):
        if target_lane_id is None:
            return None

        candidates = []
        for other in self.env.unwrapped.road.vehicles:
            if other is ego:
                continue
            if self._lane_id(other) != target_lane_id:
                continue
            dx = float(other.position[0] - ego.position[0])
            if dx > 0:
                candidates.append((dx, other))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _vehicle_relation(self, ego, other):
        if other is None:
            return 1e6, 0.0
        distance = float(other.position[0] - ego.position[0])
        rel_speed = float(getattr(other, "speed", 0.0) - getattr(ego, "speed", 0.0))
        return max(distance, 0.0), rel_speed

    def _default_waypoints(self, scene: dict) -> np.ndarray:
        waypoints = []
        for i in range(1, self.cfg.planner.waypoint_horizon + 1):
            waypoints.append([
                scene["ego_x"] + i * self.cfg.planner.waypoint_gap,
                scene["ego_y"],
                self.cfg.planner.target_speed_free,
            ])
        return np.asarray(waypoints, dtype=np.float32)

    def _default_constraints(self) -> dict:
        return {
            "min_headway": self.cfg.planner.normal_headway,
            "speed_max": self.cfg.planner.speed_max,
            "acc_max": self.cfg.planner.acc_max,
            "steer_max": self.cfg.planner.steer_max,
            "comfort_acc_delta": self.cfg.planner.comfort_acc_delta,
            "risk_mode": 0,
        }

    def _waypoint_features(self, scene: dict, waypoints: np.ndarray | None) -> np.ndarray:
        if waypoints is None or len(waypoints) == 0:
            return np.zeros(6, dtype=np.float32)

        wp = np.asarray(waypoints[0], dtype=np.float32)
        dx = (float(wp[0]) - scene["ego_x"]) / 50.0
        dy = (float(wp[1]) - scene["ego_y"]) / max(self.cfg.env.lane_width, 1e-6)
        dv = (float(wp[2]) - scene["ego_speed"]) / max(self.cfg.env.max_speed, 1e-6)
        dist = np.sqrt((float(wp[0]) - scene["ego_x"]) ** 2 + (float(wp[1]) - scene["ego_y"]) ** 2) / 50.0

        if len(waypoints) >= 2:
            wp2 = np.asarray(waypoints[1], dtype=np.float32)
            curvature_hint = (float(wp2[1]) - float(wp[1])) / max(self.cfg.env.lane_width, 1e-6)
        else:
            curvature_hint = 0.0

        lane_offset = (scene["ego_y"] - scene["lane_center_y"]) / max(self.cfg.env.lane_width, 1e-6)
        return np.array([dx, dy, dv, dist, curvature_hint, lane_offset], dtype=np.float32)

    def _constraint_features(self, constraints: dict | None) -> np.ndarray:
        if constraints is None:
            constraints = self._default_constraints()

        return np.array([
            float(constraints["min_headway"]) / 30.0,
            float(constraints["speed_max"]) / 40.0,
            float(constraints["acc_max"]) / 5.0,
            float(constraints["steer_max"]),
            float(constraints["comfort_acc_delta"]) / 5.0,
            float(constraints.get("risk_mode", 0)),
        ], dtype=np.float32)


# ============================================================
# high-level planners
# ============================================================

def heuristic_llm_decision(cfg, scene: dict, style: str = "mock") -> dict:
    lane_id = scene["lane_id"]
    target_lane = lane_id
    reason = "keep lane and maintain cruising speed"

    lane_scores = {lane_id: scene["front_distance"]}

    if scene["left_lane_id"] is not None:
        bonus = 8.0 if scene["left_clear"] else -10.0
        if style == "real":
            bonus += 1.0
        lane_scores[scene["left_lane_id"]] = scene["left_front_distance"] + bonus

    if scene["right_lane_id"] is not None:
        bonus = 6.0 if scene["right_clear"] else -10.0
        if style == "real":
            bonus += 0.5
        lane_scores[scene["right_lane_id"]] = scene["right_front_distance"] + bonus

    if scene["front_distance"] < 20.0 or scene["ttc"] < cfg.planner.replan_ttc_threshold:
        best_lane = max(lane_scores, key=lane_scores.get)
        if best_lane != lane_id:
            target_lane = best_lane
            if best_lane < lane_id:
                reason = "change lane left for better clearance"
            else:
                reason = "change lane right for better clearance"
        else:
            reason = "keep lane because adjacent lanes are not beneficial"

    if scene["front_distance"] < 10.0:
        target_speed = cfg.planner.target_speed_brake
        headway = cfg.planner.conservative_headway
        risk_mode = 1
        reason += "; strong caution due to close front vehicle"
    elif scene["front_distance"] < 22.0 or scene["ttc"] < cfg.planner.replan_ttc_threshold:
        target_speed = cfg.planner.target_speed_cautious
        headway = cfg.planner.conservative_headway
        risk_mode = 1
        reason += "; cautious mode due to medium risk"
    else:
        target_speed = cfg.planner.target_speed_free
        headway = cfg.planner.normal_headway
        risk_mode = 0
        reason += "; free-flow cruising"

    target_y = target_lane * cfg.env.lane_width
    waypoints = []
    for i in range(1, cfg.planner.waypoint_horizon + 1):
        alpha = i / cfg.planner.waypoint_horizon
        x = scene["ego_x"] + i * cfg.planner.waypoint_gap
        y = (1.0 - alpha) * scene["ego_y"] + alpha * target_y
        v = target_speed
        waypoints.append([x, y, v])

    constraints = {
        "min_headway": headway,
        "speed_max": cfg.planner.speed_max,
        "acc_max": cfg.planner.acc_max,
        "steer_max": cfg.planner.steer_max,
        "comfort_acc_delta": cfg.planner.comfort_acc_delta,
        "risk_mode": risk_mode,
    }

    return {
        "decision": {
            "target_lane": int(target_lane),
            "target_speed": float(target_speed),
            "reason": reason,
        },
        "constraints": constraints,
        "waypoints": waypoints,
    }


class RulePlanner:
    def __init__(self, cfg):
        self.cfg = cfg

    def plan(self, scene: dict) -> dict:
        lane_id = scene["lane_id"]
        target_lane = lane_id
        trigger = "periodic"

        if scene["front_distance"] < 18.0:
            left_gain = scene["left_front_distance"] - scene["front_distance"] if scene["left_lane_id"] is not None else -1e9
            right_gain = scene["right_front_distance"] - scene["front_distance"] if scene["right_lane_id"] is not None else -1e9

            if scene["left_clear"] and left_gain > right_gain and scene["left_lane_id"] is not None:
                target_lane = scene["left_lane_id"]
                trigger = "avoid_front_vehicle_left"
            elif scene["right_clear"] and scene["right_lane_id"] is not None:
                target_lane = scene["right_lane_id"]
                trigger = "avoid_front_vehicle_right"

        if scene["front_distance"] < 10.0:
            target_speed = self.cfg.planner.target_speed_brake
            headway = self.cfg.planner.conservative_headway
            risk_mode = 1
        elif scene["front_distance"] < 22.0 or scene["ttc"] < self.cfg.planner.replan_ttc_threshold:
            target_speed = self.cfg.planner.target_speed_cautious
            headway = self.cfg.planner.conservative_headway
            risk_mode = 1
        else:
            target_speed = self.cfg.planner.target_speed_free
            headway = self.cfg.planner.normal_headway
            risk_mode = 0

        target_y = target_lane * self.cfg.env.lane_width
        waypoints = []
        for i in range(1, self.cfg.planner.waypoint_horizon + 1):
            alpha = i / self.cfg.planner.waypoint_horizon
            x = scene["ego_x"] + i * self.cfg.planner.waypoint_gap
            y = (1.0 - alpha) * scene["ego_y"] + alpha * target_y
            v = target_speed
            waypoints.append([x, y, v])

        constraints = {
            "min_headway": headway,
            "speed_max": self.cfg.planner.speed_max,
            "acc_max": self.cfg.planner.acc_max,
            "steer_max": self.cfg.planner.steer_max,
            "comfort_acc_delta": self.cfg.planner.comfort_acc_delta,
            "risk_mode": risk_mode,
        }

        return {
            "waypoints": np.asarray(waypoints, dtype=np.float32),
            "constraints": constraints,
            "planner_info": {
                "planner_name": "rule",
                "target_lane": int(target_lane),
                "target_speed": float(target_speed),
                "trigger": trigger,
            },
        }


class PromptBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, scene: dict) -> str:
        safety_pref = "conservative" if (
            scene["front_distance"] < 22.0 or scene["ttc"] < self.cfg.planner.replan_ttc_threshold
        ) else "normal"

        lane_options = {
            "current_lane_id": scene["lane_id"],
            "left_lane_id": scene["left_lane_id"],
            "right_lane_id": scene["right_lane_id"],
            "left_clear": scene["left_clear"],
            "right_clear": scene["right_clear"],
        }

        prompt = f"""
You are a high-level planner for highway autonomous driving.

Task:
Given the structured scene, generate:
1. local waypoints: a list of [x, y, v]
2. rule constraints in JSON

Planning objectives:
- avoid collision
- maintain safe headway
- follow traffic flow
- keep control smooth and comfortable

Scene summary:
- ego_x: {scene['ego_x']:.2f}
- ego_y: {scene['ego_y']:.2f}
- ego_speed: {scene['ego_speed']:.2f}
- heading: {scene['heading']:.3f}
- lane_id: {scene['lane_id']}
- front_distance: {scene['front_distance']:.2f}
- front_rel_speed: {scene['front_rel_speed']:.2f}
- left_front_distance: {scene['left_front_distance']:.2f}
- right_front_distance: {scene['right_front_distance']:.2f}
- ttc: {scene['ttc']:.2f}
- lane_options: {lane_options}
- preferred_safety_mode: {safety_pref}

Output format:
{{
  "decision": {{
    "target_lane": int,
    "target_speed": float,
    "reason": "..."
  }},
  "constraints": {{
    "min_headway": float,
    "speed_max": float,
    "acc_max": float,
    "steer_max": float,
    "comfort_acc_delta": float,
    "risk_mode": int
  }},
  "waypoints": [[x, y, v], ...]
}}
Return JSON only.
""".strip()
        return prompt


class BaseLLMBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return "base"

    def generate(self, prompt: str, scene: dict) -> str:
        raise NotImplementedError


class MockLLMBackend(BaseLLMBackend):
    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str, scene: dict) -> str:
        response_obj = heuristic_llm_decision(self.cfg, scene, style="mock")
        return json.dumps(response_obj, ensure_ascii=False, indent=2)


class RealLLMBackend(BaseLLMBackend):
    """
    OpenAI-compatible REST placeholder:
    - no extra dependency
    - uses urllib from standard library
    - if call fails, optionally falls back to mock backend
    """
    @property
    def name(self) -> str:
        return "real"

    def _resolve_api_key(self) -> str:
        return self.cfg.llm.api_key or os.getenv("OPENAI_API_KEY", "")

    def _resolve_api_base(self) -> str:
        return self.cfg.llm.api_base or os.getenv("OPENAI_API_BASE", "")

    def _post_json(self, url: str, headers: dict, payload: dict) -> dict:
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.llm.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _call_real_api(self, prompt: str) -> str:
        api_key = self._resolve_api_key()
        api_base = self._resolve_api_base()
        if not api_key:
            raise RuntimeError(
                "RealLLMBackend requires cfg.llm.api_key or OPENAI_API_KEY."
            )
        if not api_base:
            raise RuntimeError(
                "RealLLMBackend requires cfg.llm.api_base or OPENAI_API_BASE."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": self.cfg.llm.model_name,
            "messages": [
                {"role": "system", "content": self.cfg.llm.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.cfg.llm.temperature,
            "max_tokens": self.cfg.llm.max_tokens,
        }
        response = self._post_json(api_base, headers, payload)
        return response["choices"][0]["message"]["content"]

    def generate(self, prompt: str, scene: dict) -> str:
        try:
            return self._call_real_api(prompt)
        except Exception as e:
            if self.cfg.planner.print_real_llm_error:
                print(f"[REAL-LLM][WARN] {type(e).__name__}: {e}")
            if self.cfg.llm.fallback_to_mock_on_error:
                print("[REAL-LLM][WARN] fallback_to_mock_on_error=True, use mock result.")
                return MockLLMBackend(self.cfg).generate(prompt, scene)
            raise


class LLMPlanner:
    def __init__(self, cfg, backend: BaseLLMBackend):
        self.cfg = cfg
        self.prompt_builder = PromptBuilder(cfg)
        self.backend = backend

    def _extract_json_string(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()

        left = text.find("{")
        right = text.rfind("}")
        if left == -1 or right == -1 or right <= left:
            raise ValueError("Cannot find JSON object in LLM response.")
        return text[left:right + 1]

    def _normalize_constraints(self, constraints: dict) -> dict:
        defaults = {
            "min_headway": self.cfg.planner.normal_headway,
            "speed_max": self.cfg.planner.speed_max,
            "acc_max": self.cfg.planner.acc_max,
            "steer_max": self.cfg.planner.steer_max,
            "comfort_acc_delta": self.cfg.planner.comfort_acc_delta,
            "risk_mode": 0,
        }
        out = dict(defaults)
        out.update(constraints or {})
        out["min_headway"] = float(out["min_headway"])
        out["speed_max"] = float(np.clip(out["speed_max"], 0.0, self.cfg.env.max_speed + 5.0))
        out["acc_max"] = float(np.clip(out["acc_max"], 0.5, 5.0))
        out["steer_max"] = float(np.clip(out["steer_max"], 0.05, 1.0))
        out["comfort_acc_delta"] = float(np.clip(out["comfort_acc_delta"], 0.1, 5.0))
        out["risk_mode"] = int(out.get("risk_mode", 0))
        return out

    def _normalize_waypoints(self, scene: dict, waypoints: list, decision: dict) -> np.ndarray:
        normalized = []
        target_lane = int(decision.get("target_lane", scene["lane_id"]))
        target_y_default = target_lane * self.cfg.env.lane_width
        target_speed = float(decision.get("target_speed", self.cfg.planner.target_speed_free))

        for i in range(self.cfg.planner.waypoint_horizon):
            if i < len(waypoints):
                wp = waypoints[i]
                x = float(wp[0])
                y = float(wp[1])
                v = float(wp[2])
            else:
                alpha = (i + 1) / self.cfg.planner.waypoint_horizon
                x = scene["ego_x"] + (i + 1) * self.cfg.planner.waypoint_gap
                y = (1.0 - alpha) * scene["ego_y"] + alpha * target_y_default
                v = target_speed
            normalized.append([x, y, v])

        return np.asarray(normalized, dtype=np.float32)

    def _parse_or_fallback(self, raw_response: str, scene: dict) -> dict:
        try:
            json_str = self._extract_json_string(raw_response)
            obj = json.loads(json_str)
            decision = obj.get("decision", {})
            constraints = self._normalize_constraints(obj.get("constraints", {}))
            waypoints = self._normalize_waypoints(scene, obj.get("waypoints", []), decision)
            return {
                "decision": {
                    "target_lane": int(decision.get("target_lane", scene["lane_id"])),
                    "target_speed": float(decision.get("target_speed", self.cfg.planner.target_speed_free)),
                    "reason": str(decision.get("reason", "llm decision")),
                },
                "constraints": constraints,
                "waypoints": waypoints,
            }
        except Exception as e:
            if self.cfg.planner.print_real_llm_error:
                print(f"[LLM-PARSE][WARN] {type(e).__name__}: {e}")
            if self.cfg.llm.fallback_to_mock_on_parse_error:
                fallback = heuristic_llm_decision(
                    self.cfg,
                    scene,
                    style="real" if self.backend.name == "real" else "mock",
                )
                return {
                    "decision": fallback["decision"],
                    "constraints": self._normalize_constraints(fallback["constraints"]),
                    "waypoints": np.asarray(fallback["waypoints"], dtype=np.float32),
                }
            raise

    def plan(self, scene: dict) -> dict:
        prompt = self.prompt_builder.build(scene)
        raw_response = self.backend.generate(prompt, scene)
        parsed = self._parse_or_fallback(raw_response, scene)
        decision = parsed["decision"]

        return {
            "waypoints": parsed["waypoints"],
            "constraints": parsed["constraints"],
            "planner_info": {
                "planner_name": f"{self.backend.name}_llm",
                "target_lane": int(decision["target_lane"]),
                "target_speed": float(decision["target_speed"]),
                "reason": decision["reason"],
                "trigger": f"{self.backend.name}_llm_planning",
            },
            "prompt": prompt,
            "raw_response": raw_response,
        }


def build_planner(cfg):
    mode = cfg.train.mode
    if mode_uses_rule_planner(mode):
        return RulePlanner(cfg)
    if mode_uses_llm_planner(mode):
        backend = RealLLMBackend(cfg) if mode_uses_real_llm(mode, cfg) else MockLLMBackend(cfg)
        return LLMPlanner(cfg, backend)
    return None


def print_plan_debug(cfg, plan: dict, ep: int, t: int | None = None, prefix: str = "PLAN"):
    wp0 = plan["waypoints"][0]
    planner_info = plan.get("planner_info", {})
    tag = f"[{prefix}][ep={ep}]" if t is None else f"[{prefix}][ep={ep}][t={t}]"

    print(
        f"{tag} planner={planner_info.get('planner_name', 'rule')} "
        f"trigger={planner_info.get('trigger', 'unknown')} "
        f"target_lane={planner_info.get('target_lane', -1)} "
        f"target_speed={planner_info.get('target_speed', -1):.2f} "
        f"wp0=({wp0[0]:.1f},{wp0[1]:.1f},{wp0[2]:.1f}) "
        f"constraints={plan['constraints']}"
    )

    if "reason" in planner_info:
        print(f"{tag} reason={planner_info['reason']}")

    if cfg.planner.print_prompt_on_replan and "prompt" in plan:
        print(f"{tag} PROMPT:\n{plan['prompt']}")
        print(f"{tag} RESPONSE:\n{plan.get('raw_response', '')}")


# ============================================================
# verifier + shield + costs
# ============================================================

class SafetyVerifier:
    def __init__(self, cfg):
        self.cfg = cfg

    def check(self, scene: dict, action, constraints: dict) -> dict:
        action = np.asarray(action, dtype=np.float32)
        acc = float(action[0])
        steer = float(action[1])

        front_distance = float(scene["front_distance"])
        speed = float(scene["ego_speed"])
        ttc = float(scene["ttc"])
        lane_center_y = float(scene["lane_center_y"])
        ego_y = float(scene["ego_y"])
        lane_offset = abs(ego_y - lane_center_y) / max(self.cfg.env.lane_width, 1e-6)

        unsafe_headway = front_distance < float(constraints["min_headway"])
        overspeed = speed > float(constraints["speed_max"])
        hard_brake_zone = front_distance < self.cfg.safety.min_front_distance_hard
        low_ttc = ttc < self.cfg.safety.ttc_safe
        large_lane_offset = lane_offset > self.cfg.safety.lane_offset_hard
        over_acc = abs(acc) > float(constraints["acc_max"])
        over_steer = abs(steer) > float(constraints["steer_max"])

        emergency = hard_brake_zone or (low_ttc and unsafe_headway)

        return {
            "unsafe_headway": bool(unsafe_headway),
            "overspeed": bool(overspeed),
            "hard_brake_zone": bool(hard_brake_zone),
            "low_ttc": bool(low_ttc),
            "large_lane_offset": bool(large_lane_offset),
            "over_acc": bool(over_acc),
            "over_steer": bool(over_steer),
            "emergency": bool(emergency),
            "lane_offset": float(lane_offset),
            "needs_intervention": bool(
                emergency or overspeed or large_lane_offset or over_acc or over_steer
            ),
        }


class SafetyShield:
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, scene: dict, action, constraints: dict, verify_info: dict):
        action = np.asarray(action, dtype=np.float32).copy()
        modified = False
        reasons = []

        acc, steer = float(action[0]), float(action[1])

        if verify_info["emergency"] and acc > self.cfg.safety.emergency_brake_acc:
            acc = self.cfg.safety.emergency_brake_acc
            modified = True
            reasons.append("emergency_brake")

        if verify_info["overspeed"] and acc > self.cfg.safety.overspeed_brake_acc:
            acc = self.cfg.safety.overspeed_brake_acc
            modified = True
            reasons.append("overspeed_brake")

        ego_y = float(scene["ego_y"])
        lane_center_y = float(scene["lane_center_y"])
        lateral_err = ego_y - lane_center_y
        if verify_info["large_lane_offset"] and np.sign(steer) == np.sign(lateral_err) and abs(steer) > 1e-6:
            steer = steer * self.cfg.safety.shield_steer_scale
            modified = True
            reasons.append("steer_suppression")

        acc = float(np.clip(acc, -float(constraints["acc_max"]), float(constraints["acc_max"])))
        steer = float(np.clip(steer, -float(constraints["steer_max"]), float(constraints["steer_max"])))

        safe_action = np.array([acc, steer], dtype=np.float32)
        return safe_action, {
            "shield_modified": bool(modified),
            "shield_reasons": reasons,
        }


def compute_costs(cfg, scene: dict, next_scene: dict, action, constraints: dict) -> dict:
    action = np.asarray(action, dtype=np.float32)
    last_action = np.asarray(scene.get("last_action", np.zeros_like(action)), dtype=np.float32)

    collision_cost = 1.0 if next_scene["collision"] else 0.0

    headway_gap = max(0.0, float(constraints["min_headway"]) - float(next_scene["front_distance"]))
    headway_cost = headway_gap / max(float(constraints["min_headway"]), 1e-6)

    overspeed_gap = max(0.0, float(next_scene["ego_speed"]) - float(constraints["speed_max"]))
    overspeed_cost = overspeed_gap / max(float(constraints["speed_max"]), 1e-6)

    comfort_cost = (
        abs(float(action[0] - last_action[0])) +
        abs(float(action[1] - last_action[1]))
    ) / 2.0

    total_cost = (
        cfg.safety.cost_collision_weight * collision_cost
        + cfg.safety.cost_headway_weight * headway_cost
        + cfg.safety.cost_overspeed_weight * overspeed_cost
        + cfg.safety.cost_comfort_weight * comfort_cost
    )

    return {
        "collision_cost": float(collision_cost),
        "headway_cost": float(headway_cost),
        "overspeed_cost": float(overspeed_cost),
        "comfort_cost": float(comfort_cost),
        "total_cost": float(total_cost),
    }


# ============================================================
# reward shaping
# ============================================================

def compute_reward(
    cfg,
    scene: dict,
    next_scene: dict,
    action,
    env_reward: float,
    constraints: dict,
    planner_used: bool,
    current_waypoints=None,
):
    action = np.asarray(action, dtype=np.float32)
    last_action = np.asarray(scene.get("last_action", np.zeros_like(action)), dtype=np.float32)

    waypoint_pos_error = 0.0
    waypoint_speed_error = 0.0
    if planner_used and current_waypoints is not None and len(current_waypoints) > 0:
        target_wp = np.asarray(current_waypoints[0], dtype=np.float32)
        target_y = float(target_wp[1])
        target_v = float(target_wp[2])
        waypoint_pos_error = abs(next_scene["ego_y"] - target_y) / max(cfg.env.lane_width, 1e-6)
        waypoint_speed_error = abs(target_v - next_scene["ego_speed"]) / max(cfg.env.max_speed, 1e-6)
    else:
        target_speed = constraints.get("speed_max", cfg.planner.speed_max)
        waypoint_speed_error = abs(target_speed - next_scene["ego_speed"]) / max(cfg.env.max_speed, 1e-6)

    progress_reward = np.clip(next_scene["ego_speed"] / max(cfg.env.max_speed, 1e-6), 0.0, 1.2)

    collision_penalty = 1.0 if next_scene["collision"] else 0.0
    headway_penalty = max(0.0, constraints["min_headway"] - next_scene["front_distance"]) / max(
        constraints["min_headway"], 1e-6
    )
    overspeed_penalty = max(0.0, next_scene["ego_speed"] - constraints["speed_max"]) / max(
        constraints["speed_max"], 1e-6
    )
    action_penalty = abs(float(action[0])) + abs(float(action[1]))
    smooth_penalty = abs(float(action[0] - last_action[0])) + abs(float(action[1] - last_action[1]))
    lane_change_penalty = abs(next_scene["lane_id"] - scene["lane_id"])

    reward = (
        cfg.reward.w_progress * progress_reward
        - cfg.reward.w_waypoint_pos * waypoint_pos_error
        - cfg.reward.w_waypoint_speed * waypoint_speed_error
        - cfg.reward.w_collision * collision_penalty
        - cfg.reward.w_headway * headway_penalty
        - cfg.reward.w_overspeed * overspeed_penalty
        - cfg.reward.w_action * action_penalty
        - cfg.reward.w_action_smooth * smooth_penalty
        - cfg.reward.w_lane_change * lane_change_penalty
    )

    if cfg.reward.use_env_reward:
        reward += cfg.reward.env_reward_scale * float(env_reward)

    terms = {
        "progress_reward": float(progress_reward),
        "waypoint_pos_error": float(waypoint_pos_error),
        "waypoint_speed_error": float(waypoint_speed_error),
        "collision_penalty": float(collision_penalty),
        "headway_penalty": float(headway_penalty),
        "overspeed_penalty": float(overspeed_penalty),
        "action_penalty": float(action_penalty),
        "smooth_penalty": float(smooth_penalty),
        "lane_change_penalty": float(lane_change_penalty),
        "reward": float(reward),
    }
    return float(reward), terms


def choose_reward(
    cfg,
    mode: str,
    scene: dict,
    next_scene: dict,
    action,
    env_reward: float,
    constraints: dict,
    current_waypoints=None,
):
    if mode == "baseline_sac":
        return env_reward, {"reward": float(env_reward), "source": "env"}

    planner_used = mode_uses_planner(mode)
    reward, terms = compute_reward(
        cfg=cfg,
        scene=scene,
        next_scene=next_scene,
        action=action,
        env_reward=env_reward,
        constraints=constraints,
        planner_used=planner_used,
        current_waypoints=current_waypoints,
    )
    terms["source"] = "shaped"
    return reward, terms


# ============================================================
# SAC
# ============================================================

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        self.cost_collision = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_headway = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_overspeed = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_comfort = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_total = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, cost_dict=None):
        if cost_dict is None:
            cost_dict = {
                "collision_cost": 0.0,
                "headway_cost": 0.0,
                "overspeed_cost": 0.0,
                "comfort_cost": 0.0,
                "total_cost": 0.0,
            }

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.cost_collision[self.ptr] = cost_dict["collision_cost"]
        self.cost_headway[self.ptr] = cost_dict["headway_cost"]
        self.cost_overspeed[self.ptr] = cost_dict["overspeed_cost"]
        self.cost_comfort[self.ptr] = cost_dict["comfort_cost"]
        self.cost_total[self.ptr] = cost_dict["total_cost"]

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=torch.as_tensor(self.state[idx], device=self.device),
            action=torch.as_tensor(self.action[idx], device=self.device),
            reward=torch.as_tensor(self.reward[idx], device=self.device),
            next_state=torch.as_tensor(self.next_state[idx], device=self.device),
            done=torch.as_tensor(self.done[idx], device=self.device),
            cost_collision=torch.as_tensor(self.cost_collision[idx], device=self.device),
            cost_headway=torch.as_tensor(self.cost_headway[idx], device=self.device),
            cost_overspeed=torch.as_tensor(self.cost_overspeed[idx], device=self.device),
            cost_comfort=torch.as_tensor(self.cost_comfort[idx], device=self.device),
            cost_total=torch.as_tensor(self.cost_total[idx], device=self.device),
        )
        return batch


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.backbone(state)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mu, std)

        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mu_action = torch.tanh(mu)
        return action, log_prob, mu_action


class SACAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.gamma = cfg.sac.gamma
        self.tau = cfg.sac.tau
        self.batch_size = cfg.sac.batch_size
        self.target_entropy = cfg.sac.target_entropy

        self.actor = Actor(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.q1 = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.q2 = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.q1_target = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.q2_target = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.sac.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.sac.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.sac.critic_lr)

        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.sac.alpha_lr)

        self.replay = ReplayBuffer(state_dim, action_dim, cfg.sac.buffer_size, self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state_t)
            else:
                action, _, _ = self.actor.sample(state_t)
        return action.cpu().numpy()[0]

    def update(self):
        if self.replay.size < self.batch_size:
            return {}

        batch = self.replay.sample(self.batch_size)
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q1_next = self.q1_target(next_state, next_action)
            q2_next = self.q2_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            target_q = reward + (1.0 - done) * self.gamma * q_next

        q1 = self.q1(state, action)
        q2 = self.q2(state, action)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob, _ = self.actor.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

    def _soft_update(self, src, dst):
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


class ConstrainedSACAgent(SACAgent):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__(state_dim, action_dim, cfg)

        self.log_lambda_collision = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.log_lambda_headway = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.log_lambda_overspeed = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.log_lambda_comfort = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)

        self.lambda_opt = torch.optim.Adam(
            [
                self.log_lambda_collision,
                self.log_lambda_headway,
                self.log_lambda_overspeed,
                self.log_lambda_comfort,
            ],
            lr=self.cfg.sac.lambda_lr,
        )

    @property
    def lambda_collision(self):
        return self.log_lambda_collision.exp()

    @property
    def lambda_headway(self):
        return self.log_lambda_headway.exp()

    @property
    def lambda_overspeed(self):
        return self.log_lambda_overspeed.exp()

    @property
    def lambda_comfort(self):
        return self.log_lambda_comfort.exp()

    def update(self):
        if self.replay.size < self.batch_size:
            return {}

        batch = self.replay.sample(self.batch_size)
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        cost_collision = batch["cost_collision"]
        cost_headway = batch["cost_headway"]
        cost_overspeed = batch["cost_overspeed"]
        cost_comfort = batch["cost_comfort"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q1_next = self.q1_target(next_state, next_action)
            q2_next = self.q2_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            target_q = reward + (1.0 - done) * self.gamma * q_next

        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob, _ = self.actor.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_penalty = (
            self.lambda_collision.detach() * cost_collision
            + self.lambda_headway.detach() * cost_headway
            + self.lambda_overspeed.detach() * cost_overspeed
            + self.lambda_comfort.detach() * cost_comfort
        )

        actor_loss = (self.alpha.detach() * log_prob - q_new + actor_penalty).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        lambda_loss = -(
            self.log_lambda_collision * (cost_collision.mean().detach() - self.cfg.sac.cost_limit_collision)
            + self.log_lambda_headway * (cost_headway.mean().detach() - self.cfg.sac.cost_limit_headway)
            + self.log_lambda_overspeed * (cost_overspeed.mean().detach() - self.cfg.sac.cost_limit_overspeed)
            + self.log_lambda_comfort * (cost_comfort.mean().detach() - self.cfg.sac.cost_limit_comfort)
        )

        self.lambda_opt.zero_grad()
        lambda_loss.backward()
        self.lambda_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
            "lambda_collision": float(self.lambda_collision.item()),
            "lambda_headway": float(self.lambda_headway.item()),
            "lambda_overspeed": float(self.lambda_overspeed.item()),
            "lambda_comfort": float(self.lambda_comfort.item()),
            "mean_cost_collision": float(cost_collision.mean().item()),
            "mean_cost_headway": float(cost_headway.mean().item()),
            "mean_cost_overspeed": float(cost_overspeed.mean().item()),
            "mean_cost_comfort": float(cost_comfort.mean().item()),
        }


def build_agent(state_dim: int, action_dim: int, cfg):
    if mode_is_constrained(cfg.train.mode):
        return ConstrainedSACAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)
    return SACAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)


# ============================================================
# evaluation
# ============================================================

class EpisodeMetrics:
    def __init__(self):
        self.rewards = []
        self.speeds = []
        self.front_distances = []
        self.ttcs = []
        self.abs_acc = []
        self.abs_steer = []
        self.smooth_acc = []
        self.smooth_steer = []
        self.lane_changes = 0
        self.replans = 0
        self.collided = False
        self.steps = 0

        self.interventions = 0
        self.unsafe_headway_count = 0
        self.overspeed_count = 0
        self.low_ttc_count = 0

        self.cost_collision = []
        self.cost_headway = []
        self.cost_overspeed = []
        self.cost_comfort = []
        self.cost_total = []

    def update(self, scene: dict, next_scene: dict, action, reward: float, replanned: bool,
               verify_info=None, shield_info=None, cost_dict=None):
        action = np.asarray(action, dtype=np.float32)
        last_action = np.asarray(scene.get("last_action", np.zeros_like(action)), dtype=np.float32)

        self.rewards.append(float(reward))
        self.speeds.append(float(next_scene["ego_speed"]))
        self.front_distances.append(float(min(next_scene["front_distance"], 1000.0)))
        self.ttcs.append(float(min(next_scene["ttc"], 100.0)))
        self.abs_acc.append(float(abs(action[0])))
        self.abs_steer.append(float(abs(action[1])))
        self.smooth_acc.append(float(abs(action[0] - last_action[0])))
        self.smooth_steer.append(float(abs(action[1] - last_action[1])))

        if next_scene["lane_id"] != scene["lane_id"]:
            self.lane_changes += 1
        if replanned:
            self.replans += 1
        if next_scene["collision"]:
            self.collided = True

        if verify_info is not None:
            self.unsafe_headway_count += int(verify_info.get("unsafe_headway", False))
            self.overspeed_count += int(verify_info.get("overspeed", False))
            self.low_ttc_count += int(verify_info.get("low_ttc", False))

        if shield_info is not None:
            self.interventions += int(shield_info.get("shield_modified", False))

        if cost_dict is not None:
            self.cost_collision.append(float(cost_dict["collision_cost"]))
            self.cost_headway.append(float(cost_dict["headway_cost"]))
            self.cost_overspeed.append(float(cost_dict["overspeed_cost"]))
            self.cost_comfort.append(float(cost_dict["comfort_cost"]))
            self.cost_total.append(float(cost_dict["total_cost"]))

        self.steps += 1

    def summary(self) -> dict:
        def safe_mean(x):
            return float(np.mean(x)) if len(x) > 0 else 0.0

        def safe_min(x):
            return float(np.min(x)) if len(x) > 0 else 0.0

        return {
            "episode_return": float(np.sum(self.rewards)),
            "steps": int(self.steps),
            "collision": int(self.collided),
            "success": int(not self.collided),
            "mean_speed": safe_mean(self.speeds),
            "mean_front_distance": safe_mean(self.front_distances),
            "min_front_distance": safe_min(self.front_distances),
            "mean_ttc": safe_mean(self.ttcs),
            "min_ttc": safe_min(self.ttcs),
            "mean_abs_acc": safe_mean(self.abs_acc),
            "mean_abs_steer": safe_mean(self.abs_steer),
            "mean_smooth_acc": safe_mean(self.smooth_acc),
            "mean_smooth_steer": safe_mean(self.smooth_steer),
            "lane_changes": int(self.lane_changes),
            "replans": int(self.replans),
            "interventions": int(self.interventions),
            "unsafe_headway_rate": float(self.unsafe_headway_count / max(self.steps, 1)),
            "overspeed_rate": float(self.overspeed_count / max(self.steps, 1)),
            "low_ttc_rate": float(self.low_ttc_count / max(self.steps, 1)),
            "mean_cost_collision": safe_mean(self.cost_collision),
            "mean_cost_headway": safe_mean(self.cost_headway),
            "mean_cost_overspeed": safe_mean(self.cost_overspeed),
            "mean_cost_comfort": safe_mean(self.cost_comfort),
            "mean_cost_total": safe_mean(self.cost_total),
        }


def need_replan(cfg, step_idx: int, scene: dict) -> bool:
    return (
        step_idx % cfg.planner.planner_interval == 0
        or scene["front_distance"] < cfg.planner.replan_front_distance
        or scene["ttc"] < cfg.planner.replan_ttc_threshold
    )


def summarize_eval_results(results: list[dict]) -> dict:
    if len(results) == 0:
        return {}

    keys = [
        "episode_return",
        "steps",
        "collision",
        "success",
        "mean_speed",
        "mean_front_distance",
        "min_front_distance",
        "mean_ttc",
        "min_ttc",
        "mean_abs_acc",
        "mean_abs_steer",
        "mean_smooth_acc",
        "mean_smooth_steer",
        "lane_changes",
        "replans",
        "interventions",
        "unsafe_headway_rate",
        "overspeed_rate",
        "low_ttc_rate",
        "mean_cost_collision",
        "mean_cost_headway",
        "mean_cost_overspeed",
        "mean_cost_comfort",
        "mean_cost_total",
    ]
    summary = {}
    for k in keys:
        summary[k] = float(np.mean([r[k] for r in results]))
    return summary


def evaluate_policy(cfg, agent, mode: str):
    print("\n================ EVALUATION START ================\n")

    eval_env = HighwayEnvWrapper(cfg, render=False)
    verifier = SafetyVerifier(cfg) if cfg.safety.verifier_enabled else None
    shield = SafetyShield(cfg) if cfg.safety.shield_enabled_eval else None

    eval_results = []

    for ep in range(cfg.eval.episodes):
        state, _ = eval_env.reset(seed=cfg.train.seed + cfg.eval.seed_offset + ep)
        planner = build_planner(cfg)

        metrics = EpisodeMetrics()

        if planner is not None:
            init_scene = eval_env.get_scene_dict()
            plan = planner.plan(init_scene)
            eval_env.apply_plan(plan["waypoints"], plan["constraints"])
            print_plan_debug(cfg, plan, ep=ep, t=None, prefix="EVAL-PLAN")

        for t in range(cfg.train.max_steps_per_episode):
            scene = eval_env.get_scene_dict()
            replanned = False

            if planner is not None and need_replan(cfg, t, scene):
                plan = planner.plan(scene)
                eval_env.apply_plan(plan["waypoints"], plan["constraints"])
                replanned = True
                print_plan_debug(cfg, plan, ep=ep, t=t, prefix="EVAL-PLAN")

            constraints = eval_env.current_constraints
            raw_action = agent.select_action(state, evaluate=True).astype(np.float32)

            verify_info = verifier.check(scene, raw_action, constraints) if verifier is not None else {}
            shield_info = {"shield_modified": False, "shield_reasons": []}
            action = raw_action

            if shield is not None:
                action, shield_info = shield.apply(scene, raw_action, constraints, verify_info)

            next_state, env_reward, terminated, truncated, _ = eval_env.step(action)
            next_scene = eval_env.get_scene_dict()

            reward, reward_terms = choose_reward(
                cfg=cfg,
                mode=mode,
                scene=scene,
                next_scene=next_scene,
                action=action,
                env_reward=env_reward,
                constraints=constraints,
                current_waypoints=eval_env.current_waypoints,
            )

            cost_dict = compute_costs(cfg, scene, next_scene, action, constraints)

            metrics.update(
                scene=scene,
                next_scene=next_scene,
                action=action,
                reward=reward,
                replanned=replanned,
                verify_info=verify_info,
                shield_info=shield_info,
                cost_dict=cost_dict,
            )

            if cfg.eval.print_step:
                print(
                    f"[EVAL-STEP][ep={ep}][t={t}] mode={mode} "
                    f"speed={next_scene['ego_speed']:.2f} "
                    f"front_d={next_scene['front_distance']:.2f} "
                    f"ttc={next_scene['ttc']:.2f} "
                    f"raw_action=[{raw_action[0]:.2f},{raw_action[1]:.2f}] "
                    f"action=[{action[0]:.2f},{action[1]:.2f}] "
                    f"reward={reward:.3f} source={reward_terms['source']} "
                    f"collision={int(next_scene['collision'])} "
                    f"intervene={int(shield_info.get('shield_modified', False))}"
                )

            state = next_state
            if terminated or truncated:
                break

        ep_summary = metrics.summary()
        eval_results.append(ep_summary)

        print(
            f"[EVAL][ep={ep}] "
            f"return={ep_summary['episode_return']:.3f} "
            f"steps={ep_summary['steps']} "
            f"success={ep_summary['success']:.0f} "
            f"collision={ep_summary['collision']:.0f} "
            f"mean_speed={ep_summary['mean_speed']:.3f} "
            f"min_front_d={ep_summary['min_front_distance']:.3f} "
            f"interventions={ep_summary['interventions']:.0f} "
            f"unsafe_headway_rate={ep_summary['unsafe_headway_rate']:.3f} "
            f"mean_cost_total={ep_summary['mean_cost_total']:.3f}"
        )

    summary = summarize_eval_results(eval_results)
    print("\n================ EVALUATION SUMMARY ================\n")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    eval_env.close()
    return {"episodes": eval_results, "summary": summary}


# ============================================================
# train loop
# ============================================================

def maybe_print_update(mode: str, update_info: dict):
    if not update_info:
        return
    msg = (
        f"[UPDATE] q1={update_info['q1_loss']:.4f} "
        f"q2={update_info['q2_loss']:.4f} "
        f"actor={update_info['actor_loss']:.4f} "
        f"alpha={update_info['alpha']:.4f}"
    )
    if mode_is_constrained(mode):
        msg += (
            f" lambda_col={update_info['lambda_collision']:.4f}"
            f" lambda_head={update_info['lambda_headway']:.4f}"
            f" lambda_over={update_info['lambda_overspeed']:.4f}"
            f" lambda_comf={update_info['lambda_comfort']:.4f}"
        )
    print(msg)


def run(cfg):
    set_seed(cfg.train.seed)

    env = HighwayEnvWrapper(cfg, render=cfg.train.render)
    verifier = SafetyVerifier(cfg) if cfg.safety.verifier_enabled else None
    shield = SafetyShield(cfg) if cfg.safety.shield_enabled_train else None

    state, _ = env.reset(seed=cfg.train.seed)
    state_dim = state.shape[0]
    action_dim = env.action_dim()
    agent = build_agent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)

    total_steps = 0
    train_episode_results = []

    for ep in range(cfg.train.episodes):
        state, _ = env.reset(seed=cfg.train.seed + ep)
        planner = build_planner(cfg)

        metrics = EpisodeMetrics()

        if planner is not None:
            init_scene = env.get_scene_dict()
            plan = planner.plan(init_scene)
            env.apply_plan(plan["waypoints"], plan["constraints"])
            print_plan_debug(cfg, plan, ep=ep, t=None, prefix="PLAN")

        for t in range(cfg.train.max_steps_per_episode):
            scene = env.get_scene_dict()
            replanned = False

            if planner is not None and need_replan(cfg, t, scene):
                plan = planner.plan(scene)
                env.apply_plan(plan["waypoints"], plan["constraints"])
                replanned = True
                print_plan_debug(cfg, plan, ep=ep, t=t, prefix="PLAN")

            constraints = env.current_constraints

            if total_steps < cfg.sac.start_steps:
                raw_action = env.env.action_space.sample().astype(np.float32)
            else:
                raw_action = agent.select_action(state, evaluate=False).astype(np.float32)

            verify_info = verifier.check(scene, raw_action, constraints) if verifier is not None else {}
            shield_info = {"shield_modified": False, "shield_reasons": []}
            action = raw_action

            if shield is not None:
                action, shield_info = shield.apply(scene, raw_action, constraints, verify_info)

            next_state, env_reward, terminated, truncated, _ = env.step(action)
            next_scene = env.get_scene_dict()

            reward, reward_terms = choose_reward(
                cfg=cfg,
                mode=cfg.train.mode,
                scene=scene,
                next_scene=next_scene,
                action=action,
                env_reward=env_reward,
                constraints=constraints,
                current_waypoints=env.current_waypoints,
            )

            done = terminated or truncated
            cost_dict = compute_costs(cfg, scene, next_scene, action, constraints)
            agent.replay.add(state, action, reward, next_state, float(done), cost_dict=cost_dict)

            metrics.update(
                scene=scene,
                next_scene=next_scene,
                action=action,
                reward=reward,
                replanned=replanned,
                verify_info=verify_info,
                shield_info=shield_info,
                cost_dict=cost_dict,
            )

            update_info = {}
            if total_steps >= cfg.sac.update_after and total_steps % cfg.sac.update_every == 0:
                update_info = agent.update()

            if cfg.train.print_every_step:
                print(
                    f"[STEP][ep={ep}][t={t}] mode={cfg.train.mode} "
                    f"speed={next_scene['ego_speed']:.2f} "
                    f"front_d={next_scene['front_distance']:.2f} "
                    f"ttc={next_scene['ttc']:.2f} "
                    f"raw_action=[{raw_action[0]:.2f},{raw_action[1]:.2f}] "
                    f"action=[{action[0]:.2f},{action[1]:.2f}] "
                    f"reward={reward:.3f} source={reward_terms['source']} "
                    f"collision={int(next_scene['collision'])} "
                    f"intervene={int(shield_info.get('shield_modified', False))} "
                    f"cost={cost_dict['total_cost']:.3f}"
                )
                maybe_print_update(cfg.train.mode, update_info)

            state = next_state
            total_steps += 1

            if done:
                break

        ep_summary = metrics.summary()
        train_episode_results.append(ep_summary)
        print(
            f"[EPISODE END] ep={ep} return={ep_summary['episode_return']:.3f} "
            f"steps={ep_summary['steps']} collision={ep_summary['collision']} "
            f"success={ep_summary['success']} interventions={ep_summary['interventions']} "
            f"mean_cost_total={ep_summary['mean_cost_total']:.3f} "
            f"replay_size={agent.replay.size}"
        )

    env.close()
    train_summary = summarize_eval_results(train_episode_results)

    result = {
        "mode": cfg.train.mode,
        "train_episodes": train_episode_results,
        "train_summary": train_summary,
        "eval": None,
    }

    if cfg.eval.enabled:
        result["eval"] = evaluate_policy(cfg, agent, mode=cfg.train.mode)

    return result


# ============================================================
# compare module
# ============================================================

def _metric_from_result(exp_result: dict, key: str) -> float:
    if exp_result.get("eval") is not None and exp_result["eval"].get("summary"):
        return float(exp_result["eval"]["summary"].get(key, 0.0))
    return float(exp_result["train_summary"].get(key, 0.0))


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.3f}"


def format_compare_table(results: list[dict]) -> str:
    headers = [
        "mode",
        "return",
        "success",
        "collision",
        "mean_speed",
        "min_front_d",
        "interventions",
        "unsafe_headway",
        "cost_total",
        "replans",
        "time_sec",
    ]
    rows = []
    for r in results:
        rows.append([
            r["mode"],
            _fmt(_metric_from_result(r, "episode_return")),
            _fmt(_metric_from_result(r, "success")),
            _fmt(_metric_from_result(r, "collision")),
            _fmt(_metric_from_result(r, "mean_speed")),
            _fmt(_metric_from_result(r, "min_front_distance")),
            _fmt(_metric_from_result(r, "interventions")),
            _fmt(_metric_from_result(r, "unsafe_headway_rate")),
            _fmt(_metric_from_result(r, "mean_cost_total")),
            _fmt(_metric_from_result(r, "replans")),
            _fmt(r.get("elapsed_sec", 0.0)),
        ])

    widths = []
    for col_idx, header in enumerate(headers):
        max_len = len(header)
        for row in rows:
            max_len = max(max_len, len(str(row[col_idx])))
        widths.append(max_len)

    line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    out = [line, sep]
    for row in rows:
        out.append(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def run_compare(cfg, modes: Optional[List[str]] = None) -> dict:
    modes = modes or cfg.compare.modes
    results = []

    print("\n================ COMPARE START ================\n")
    for idx, mode in enumerate(modes):
        if mode not in ALL_MODES:
            print(f"[COMPARE][SKIP] unknown mode={mode}")
            continue

        exp_cfg = copy.deepcopy(cfg)
        exp_cfg.train.mode = mode
        if exp_cfg.compare.disable_step_print_during_compare:
            exp_cfg.train.print_every_step = False

        print(f"[COMPARE][{idx + 1}/{len(modes)}] run mode={mode}")
        start = time.time()
        result = run(exp_cfg)
        elapsed = time.time() - start
        result["elapsed_sec"] = float(elapsed)
        results.append(result)

        print(
            f"[COMPARE][DONE] mode={mode} "
            f"return={_metric_from_result(result, 'episode_return'):.3f} "
            f"success={_metric_from_result(result, 'success'):.3f} "
            f"collision={_metric_from_result(result, 'collision'):.3f} "
            f"time_sec={elapsed:.2f}"
        )
        print()

    table = format_compare_table(results)
    print("\n================ COMPARE TABLE ================\n")
    print(table)

    payload = {"results": results, "table": table}
    if cfg.compare.save_json_path:
        with open(cfg.compare.save_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[COMPARE] saved json to {cfg.compare.save_json_path}")

    return payload


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="rule_hier",
        choices=ALL_MODES + ["compare"],
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--eval-print-step", action="store_true")
    parser.add_argument("--print-prompt-on-replan", action="store_true")

    parser.add_argument("--shield-train", action="store_true")
    parser.add_argument("--disable-shield-eval", action="store_true")

    parser.add_argument("--llm-backend", type=str, default="mock", choices=["mock", "real"])
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--llm-api-base", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--llm-timeout", type=int, default=20)
    parser.add_argument("--disable-llm-fallback", action="store_true")

    parser.add_argument("--compare-modes", type=str, default="")
    parser.add_argument("--compare-save-json", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = get_config()
    cfg.train.mode = args.mode if args.mode != "compare" else cfg.train.mode
    cfg.train.episodes = args.episodes
    cfg.train.max_steps_per_episode = args.max_steps
    cfg.train.render = args.render
    cfg.train.seed = args.seed
    cfg.train.use_shield_during_train = args.shield_train

    cfg.eval.episodes = args.eval_episodes
    cfg.eval.enabled = not args.disable_eval
    cfg.eval.print_step = args.eval_print_step
    cfg.planner.print_prompt_on_replan = args.print_prompt_on_replan

    cfg.safety.shield_enabled_eval = not args.disable_shield_eval
    cfg.safety.shield_enabled_train = args.shield_train

    cfg.llm.backend = args.llm_backend
    cfg.llm.model_name = args.llm_model
    cfg.llm.api_key = args.llm_api_key
    cfg.llm.api_base = args.llm_api_base
    cfg.llm.timeout_sec = args.llm_timeout
    cfg.llm.fallback_to_mock_on_error = not args.disable_llm_fallback
    cfg.llm.fallback_to_mock_on_parse_error = not args.disable_llm_fallback

    if args.compare_modes.strip():
        cfg.compare.modes = [m.strip() for m in args.compare_modes.split(",") if m.strip()]
    cfg.compare.save_json_path = args.compare_save_json

    if args.mode == "compare":
        run_compare(cfg, modes=cfg.compare.modes)
    else:
        run(cfg)

# 单跑：
# python single_file_llm_guided_rl.py --mode baseline_sac --episodes 5
# python single_file_llm_guided_rl.py --mode shaping_sac --episodes 5
# python single_file_llm_guided_rl.py --mode rule_hier --episodes 5
# python single_file_llm_guided_rl.py --mode llm_hier --episodes 5
# python single_file_llm_guided_rl.py --mode constrained_rule_hier --episodes 5
# python single_file_llm_guided_rl.py --mode constrained_llm_hier --episodes 5
# python single_file_llm_guided_rl.py --mode real_llm_hier --episodes 5 --llm-backend real --llm-api-key YOUR_KEY
# python single_file_llm_guided_rl.py --mode constrained_real_llm_hier --episodes 5 --llm-backend real --llm-api-key YOUR_KEY
#
# 比较：
# python single_file_llm_guided_rl.py --mode compare --episodes 5 --compare-modes baseline_sac,rule_hier,llm_hier,constrained_rule_hier,constrained_llm_hier
# python single_file_llm_guided_rl.py --mode compare --episodes 5 --compare-save-json compare_results.json

# # Mock LLM
# python single_file_llm_guided_rl_v2.py --mode llm_hier --episodes 5

# # Real LLM（OpenAI-compatible REST 占位接口）
# python single_file_llm_guided_rl_v2.py \
#   --mode real_llm_hier \
#   --episodes 5 \
#   --llm-backend real \
#   --llm-api-key YOUR_KEY \
#   --llm-api-base https://api.openai.com/v1/chat/completions

# # Constrained + Mock LLM
# python single_file_llm_guided_rl_v2.py --mode constrained_llm_hier --episodes 5

# # 一键对比
# python single_file_llm_guided_rl_v2.py \
#   --mode compare \
#   --episodes 5 \
#   --compare-modes baseline_sac,rule_hier,llm_hier,constrained_rule_hier,constrained_llm_hier