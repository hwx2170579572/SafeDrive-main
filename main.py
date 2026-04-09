import os
import copy
import queue
import time
import json
import re
import random
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import gymnasium as gym
import highway_env  
import highway_env.envs 
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter  

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# =====================================================================
# Part 0: (Configuration & Setup)
# =====================================================================
@dataclass
class SystemConfig:
    # --- --- translated note
    EXP_NAME: str = f"HybridDrive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    LOG_DIR: str = "./runs/"
    CHECKPOINT_DIR: str = "./checkpoints/"
    SAVE_INTERVAL: int = 50          # Episode translated note
    MAX_EPISODES: int = 500
    STABILITY_PROFILE: str = "C"  # A= , B= , C=
    BATCH_SIZE: int = 128            # translated note
    REPLAY_BUFFER_SIZE: int = 20000

    # --- --- translated note
    FAST_FREQ_HZ: int = 10    
    LLM_COOLDOWN_STEPS: int = 30     # LLM (30 = 3 )
    LLM_MIN_TRIGGER_INTERVAL_SEC: float = 5.0  # ， translated note
    LLM_LOG_EVERY_N: int = 4                  # N （ ） translated note
    LLM_VERBOSE_LOG: bool = False
    ENV_NAME: str = "highway-v0"
    OBSERVATION_TYPE: str = "Kinematics" 
    OBS_NORMALIZE: bool = True
    OBS_X_RANGE_M: float = 100.0
    OBS_Y_RANGE_M: float = 10.0
    OBS_V_RANGE_MPS: float = 30.0
    VEHICLES_COUNT: int = 10          
    FEATURES_COUNT: int = 5          
    ASYNC_BUFFER_MAXSIZE: int = 5    
    LANE_WIDTH_M: float = 4.0
    DEFAULT_LANE_COUNT: int = 4
    DEFAULT_EGO_LANE_ID: int = 1
    TRAJ_MIN_DX_M: float = 4.0
    TRAJ_MIN_DX_GAP_M: float = 4.0
    TRAJ_SMOOTH_ALPHA: float = 0.80

    # --- （ 1/6/7）---
    MEMORY_RETRIEVAL_TOP_K: int = 3
    MEMORY_RETRIEVE_FAILURE_K: int = 2
    MEMORY_RETRIEVE_SUCCESS_K: int = 1
    MEMORY_TEXT_SIM_WEIGHT: float = 0.25
    MEMORY_UPSERT_DIST: float = 0.35
    MEMORY_SUCCESS_ADD_INTERVAL: int = 20
    MEMORY_MIN_SAFE_TTC: float = 2.5
    MEMORY_MIN_SUCCESS_SPEED: float = 10.0
    FEWSHOT_SAMPLE_PAIRS: int = 2
    FEWSHOT_HARD_RATIO: float = 0.5
    TEMPLATE_SPEED_GAIN: float = 0.30
    TEMPLATE_GAP_REF: float = 35.0
    TEMPLATE_GAP_CLAMP_MIN: float = 0.65        # ↓ 0.75→0.65（ ）
    TEMPLATE_GAP_CLAMP_MAX: float = 1.80        # ， translated note

    # --- （ 2/4/10）---
    UNCERTAINTY_TRIGGER_STD: float = 0.45
    UNCERTAINTY_SCORE_REF: float = 0.90
    GATE_RISK_THRESHOLD: float = 0.65
    GATE_RISK_WEIGHT: float = 0.60
    GATE_UNCERT_WEIGHT: float = 0.40
    GATE_MIN_BLEND: float = 0.25
    REVIEW_MIN_FRONT_GAP: float = 13.0
    REVIEW_MIN_TTC: float = 2.4
    REVIEW_STEER_TRIGGER: float = 0.25
    REVIEW_BOUNDARY_STEER_EPS: float = 0.02
    REVIEW_MIN_LANE_GAP: float = 12.0
    REVIEW_LANE_MIN_TTC: float = 2.2
    REVIEW_ACCEL_LIMIT: float = -0.35
    REVIEW_HARD_BRAKE: float = -0.9
    NEAR_MISS_TTC: float = 1.2                  # ← 1.2 translated note

    # --- ： A / ( v1.1)---
    TRAJ_QUERY_COUNT: int = 4                   # ← 4 ( ) translated note
    TRAJ_HORIZON_STEPS: int = 3
    TRAJ_QUERY_TEMPERATURE: float = 0.8         # ← 0.8 ( !)
    TRAJ_GUIDE_ONLINE_BLEND: float = 0.18       # ← 0.18 ( )
    TRAJ_GUIDE_BLEND_MIN: float = 0.08
    TRAJ_GUIDE_DECAY_EPISODES: int = 220
    TRAJ_GUIDE_MIN_PHASE: float = 0.45
    TRAJ_GUIDE_LOSS_WEIGHT: float = 2.0         # ← 2.0 ( )
    TRAJ_GUIDE_JS_WEIGHT: float = 0.35
    TRAJ_GUIDE_EXPERT_STD: float = 0.35
    TRAJ_GUIDE_CONF_FLOOR: float = 0.20
    TRAJ_GUIDE_ACCEL_REF: float = 12.0          # ← 12.0 ( )
    TRAJ_GUIDE_ACCEL_NORM: float = 16.0         # ← 16.0 (dx/16 → [-1,1])

    # --- (R = R_ms + R_lc + R_e + R_s) ---
    PAPER_REWARD_BLEND: float = 0.35
    REWARD_SPEED_MIN: float = 10.0
    REWARD_SPEED_MAX: float = 30.0
    REWARD_TTC_THRESHOLD: float = 3.0
    REWARD_ACTION_DIFF_WEIGHT: float = 1.2
    REWARD_LOW_SPEED_PENALTY: float = 0.25

    # --- --- translated note
    REPLAY_HARD_RATIO: float = 0.30
    REPLAY_HARD_CAPACITY_RATIO: float = 0.5
    
    # --- --- translated note
    CBF_FAILURE_THRESHOLD: float = 0.5  # translated note
    
    # LLM_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    # LLM_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-4jWXenyMvfiLIMLNiTy1lLoVrR4FC1Y2KoUModv8NTq4Plg5")  
    # LLM_MODEL: str = "gpt-4.1-nano"

    # --- LLM API （ / ）---
    LLM_API_TYPE: str = "openai"  # openai（ ）
    LLM_API_BASE: str = "https://api.chatanywhere.tech"  # translated note
    LLM_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-4jWXenyMvfiLIMLNiTy1lLoVrR4FC1Y2KoUModv8NTq4Plg5")  # Key， translated note
    LLM_CHAT_MODEL: str = "gpt-4.1-nano"  # LLM_CHAT_MODEL，
  
    
    # --- SAC ---
    ACTOR_LR: float = 1e-4
    CRITIC_LR: float = 3e-4
    GAMMA: float = 0.99              
    TAU: float = 0.005               
    GRAD_CLIP_NORM: float = 5.0      
    CROSS_ATTN_DIM: int = 256        
    MIN_LOG_STD: float = -20.0       
    MAX_LOG_STD: float = 2.0         

    def __post_init__(self):
        profile_raw = os.getenv("SAFE_STABILITY_PROFILE", self.STABILITY_PROFILE)
        profile = str(profile_raw).strip().upper()

        profile_a = {
            "REVIEW_MIN_FRONT_GAP": 14.0,
            "REVIEW_MIN_TTC": 2.6,
            "REVIEW_LANE_MIN_TTC": 2.4,
            "REVIEW_ACCEL_LIMIT": -0.38,
            "REVIEW_HARD_BRAKE": -0.95,
            "NEAR_MISS_TTC": 1.3,
            "GATE_RISK_THRESHOLD": 0.60,
            "GATE_MIN_BLEND": 0.30,
            "REPLAY_HARD_RATIO": 0.18,
            "TRAJ_GUIDE_ONLINE_BLEND": 0.14,
            "TRAJ_GUIDE_LOSS_WEIGHT": 1.8,
            "TRAJ_GUIDE_JS_WEIGHT": 0.22,
            "REWARD_ACTION_DIFF_WEIGHT": 1.35,
            "REWARD_LOW_SPEED_PENALTY": 0.24,
            "LLM_COOLDOWN_STEPS": 40,
            "LLM_MIN_TRIGGER_INTERVAL_SEC": 6.0,
            "MEMORY_RETRIEVE_FAILURE_K": 3,
            "MEMORY_RETRIEVE_SUCCESS_K": 1,
        }

        profile_b = {
            "REVIEW_MIN_FRONT_GAP": 13.2,
            "REVIEW_MIN_TTC": 2.35,
            "REVIEW_LANE_MIN_TTC": 2.20,
            "REVIEW_ACCEL_LIMIT": -0.34,
            "REVIEW_HARD_BRAKE": -0.90,
            "NEAR_MISS_TTC": 1.2,
            "GATE_RISK_THRESHOLD": 0.64,
            "GATE_MIN_BLEND": 0.24,
            "REPLAY_HARD_RATIO": 0.20,
            "TRAJ_GUIDE_ONLINE_BLEND": 0.17,
            "TRAJ_GUIDE_LOSS_WEIGHT": 1.6,
            "TRAJ_GUIDE_JS_WEIGHT": 0.24,
            "REWARD_ACTION_DIFF_WEIGHT": 1.10,
            "REWARD_LOW_SPEED_PENALTY": 0.32,
            "LLM_COOLDOWN_STEPS": 30,
            "LLM_MIN_TRIGGER_INTERVAL_SEC": 5.0,
            "MEMORY_RETRIEVE_FAILURE_K": 2,
            "MEMORY_RETRIEVE_SUCCESS_K": 2,
        }
        profile_c = {
            # === 1. ： ===
            "REVIEW_MIN_LANE_GAP": 8.0,        # ( 12.0) ， 7
            "REVIEW_LANE_MIN_TTC": 1.5,        # ( 2.2) translated note
            "REVIEW_STEER_TRIGGER": 0.35,      # ( 0.25) RL ，

            # === 2. ： ， ===
            "REVIEW_MIN_FRONT_GAP": 9.0,       # ( 13.2) translated note
            "REVIEW_MIN_TTC": 1.6,             # ( 2.35) translated note
            "REWARD_LOW_SPEED_PENALTY": 1.5,   # ( 0.32) [ ] ！

            # === 3. ： ===
            "REWARD_ACTION_DIFF_WEIGHT": 0.4,  # ( 1.10) ， RL CBF

            # === 4. “ ” ===
            "TRAJ_GUIDE_ONLINE_BLEND": 0.35,   # ( 0.17) ， ， RL
            "TRAJ_GUIDE_LOSS_WEIGHT": 3.0,     # ( 1.6) Loss， Actor
            
            # === 5. ===
            "PAPER_REWARD_BLEND": 0.60,        # ( 0.35) ( )
        }

        if profile in ("A", "CONSERVATIVE"):
            for k, v in profile_a.items():
                setattr(self, k, v)
            self.STABILITY_PROFILE = "A"
        elif profile in ("B", "BALANCE", "BALANCED"):
            for k, v in profile_b.items():
                setattr(self, k, v)
            self.STABILITY_PROFILE = "B"
        elif profile in ("C", "AGGRESSIVE"):
            for k, v in profile_c.items():
                setattr(self, k, v)
            self.STABILITY_PROFILE = "C"
        else:
            # profile ， 。
            for k, v in profile_b.items():
                setattr(self, k, v)
            self.STABILITY_PROFILE = "B"


# --- --- translated note
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()] 
)
logger = logging.getLogger("HybridDriving")


# =====================================================================
# Part 1: translated note
# =====================================================================
class DualFreqEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: SystemConfig):
        super().__init__(env)
        self.config = config
        self.step_counter = 0
        self.last_llm_trigger_step = -self.config.LLM_COOLDOWN_STEPS
        self.last_llm_trigger_time = 0.0
        # self.text_description_queue = queue.Queue(maxsize=self.config.ASYNC_BUFFER_MAXSIZE)
        self.text_description_queue = LatestTextSceneContainer()
        # self.state_history_buffer = deque(maxlen=self.config.LLM_COOLDOWN_STEPS)

    def reset(self, **kwargs):
        self.step_counter = 0
        self.last_llm_trigger_step = -self.config.LLM_COOLDOWN_STEPS
        self.last_llm_trigger_time = 0.0
        # self.state_history_buffer.clear()
        
        obs, info = self.env.reset(**kwargs)
        vector_state = self._extract_vector_state(obs)
        # self.state_history_buffer.append(vector_state)
        
        return vector_state, info

    def step(self, action):
        # 💥 [ ： (Absolute Physical Wall)] 💥
        # LKA， translated note
        try:
            vehicle = self.env.unwrapped.vehicle
            y = vehicle.position[1]
            lane_width = self.config.LANE_WIDTH_M
            lane_count = self.config.DEFAULT_LANE_COUNT
            try:
                lane_idx = vehicle.lane_index
                lane_count = len(self.env.unwrapped.road.network.graph[lane_idx[0]][lane_idx[1]])
            except Exception:
                pass
                
            # ( ， ) translated note
            top_red_line = -lane_width / 2.0 + 0.6
            bottom_red_line = (lane_count - 1) * lane_width + lane_width / 2.0 - 0.6

            # ， (action[1] < 0)，
            if y < top_red_line and action[1] < 0:
                action[1] = 0.2
            # ， (action[1] > 0)，
            elif y > bottom_red_line and action[1] > 0:
                action[1] = -0.2
                
        except Exception:
            pass # translated note

        # ---------------- ----------------
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1

        vector_state = self._extract_vector_state(obs)

        now = time.time()
        step_ok = self.step_counter - self.last_llm_trigger_step >= self.config.LLM_COOLDOWN_STEPS
        time_ok = (now - self.last_llm_trigger_time) >= self.config.LLM_MIN_TRIGGER_INTERVAL_SEC
        if step_ok and time_ok:
            text_scene = self._generate_text_description(vector_state)
            self.last_llm_trigger_step = self.step_counter
            self.last_llm_trigger_time = now
            self.text_description_queue.put(text_scene)

        return vector_state, reward, terminated, truncated, info

    def _extract_vector_state(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32).flatten()

    def _generate_text_description(self, current_state: np.ndarray) -> str:
        # --- ： ---
        matrix = current_state.reshape(self.config.VEHICLES_COUNT, self.config.FEATURES_COUNT).copy()
        matrix[:, 1] *= 100.0  # dx translated note
        matrix[:, 2] *= 10.0   # dy translated note
        matrix[:, 3] *= 30.0   # dvx m/s translated note
        matrix[:, 4] *= 30.0   # dvy 
        # --------------------------------
        
        # ego_vx = matrix[0][3] # ： * 30.0，
        try:
        # (m/s) translated note
            ego_vx = self.env.unwrapped.vehicle.speed
        except Exception:
            ego_vx = 20.0 # translated note
        lane_count = self.config.DEFAULT_LANE_COUNT
        ego_lane_id = self.config.DEFAULT_EGO_LANE_ID
        try:
            lane_index = self.env.unwrapped.vehicle.lane_index
            if isinstance(lane_index, tuple) and len(lane_index) >= 3:
                lane_count = len(self.env.unwrapped.road.network.graph[lane_index[0]][lane_index[1]])
                ego_lane_id = int(lane_index[2])
        except Exception:
            pass
        
        # desc = f"- ： ， {ego_vx:.1f}。\n ：\n"
        # --- ： ---
        desc = (
            f"- Ego status: driving on highway, current speed {ego_vx:.1f} (desired cruise speed: 25.0+).\n"
            f"- Lane context: current lane ID={ego_lane_id}, total lane count={lane_count}, lane width={self.config.LANE_WIDTH_M:.2f}.\n"
            "Nearby interacting vehicles:\n"
        )
        # ------------------------------------
        has_other_vehicles = False
        for i in range(1, self.config.VEHICLES_COUNT):
            if matrix[i][0] > 0.5: 
                has_other_vehicles = True
                dx, dy, dvx = matrix[i][1], matrix[i][2], matrix[i][3]
                desc += f"  - Vehicle {i}: longitudinal distance {dx:.2f}, lateral distance {dy:.2f}, relative speed {dvx:.2f}.\n"
        if not has_other_vehicles:
            desc += "  - No close interacting vehicles in view; road is clear."
        return desc


# =====================================================================
# Part 2: ( ) - CoT
# =====================================================================
# translated note
# ==========================================
class LatestTextSceneContainer:
    """A non-blocking single-value container for fast-system to slow-system communication."""
    def __init__(self):
        self.lock = threading.Lock()
        self.text_scene = None
        self.event = threading.Event()

    def put(self, text: str):
        with self.lock:
            self.text_scene = text
        self.event.set()

    def get_blocking(self) -> str:
        self.event.wait() # translated note
        with self.lock:
            text = self.text_scene
        self.event.clear()
        return text
class LatentSkillEncoder(nn.Module):
    def __init__(self, config: SystemConfig):
        super().__init__()
        self.config = config
        # translated note
        self.action_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, self.config.CROSS_ATTN_DIM))
        self.traj_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, self.config.CROSS_ATTN_DIM))
        self.action_map = {"lane_keep": 0, "lane_change_left": 1, "lane_change_right": 2, "accelerate": 3, "decelerate": 4}

    def forward(self, macro_decision: dict) -> torch.Tensor:
        # 1. ( Token 0)
        action_idx = self.action_map.get(macro_decision.get("action", "lane_keep"), 0)
        action_one_hot = torch.zeros(5, device=next(self.parameters()).device)
        action_one_hot[action_idx] = 1.0
        # --- --- translated note
        raw_speed = macro_decision.get("target_speed", 20.0)
        try:
            # translated note
            speed_val = float(raw_speed)
        except (ValueError, TypeError):
            # ， translated note
            logger.warning(f"Invalid speed format from LLM output: {raw_speed}, automatically replaced with safe default value 10.0")
            speed_val = 10.0
            
        norm_speed = torch.tensor([speed_val / 40.0], dtype=torch.float32, device=next(self.parameters()).device)
        # ----------------------------
        action_feat = torch.cat([action_one_hot, norm_speed], dim=0).unsqueeze(0) # (1, 6)
        action_token = self.action_encoder(action_feat).unsqueeze(1) # (1, 1, DIM)

        # 2. ( Token)
        # ： ( 3 ) translated note
        default_traj = [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
        traj = macro_decision.get("reference_trajectory", default_traj)
        
        # ( 3 ) translated note
        traj = traj[:3]
        while len(traj) < 3: traj.append(traj[-1])
        
        traj_tensor = torch.tensor(traj, dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0) # (1, 3, 2)
        # : x/100, y/10
        traj_tensor[..., 0] /= 100.0
        traj_tensor[..., 1] /= 10.0
        
        traj_tokens = self.traj_encoder(traj_tensor) # (1, 3, DIM)

        # 3. translated note
        # : (Batch=1, SeqLen=4, Dim=256)
        return torch.cat([action_token, traj_tokens], dim=1)

class AsyncKVBuffer:
    def __init__(self, config: SystemConfig):
        self.lock = threading.Lock()
        # 4 (1 Token + 3 Token)
        self.latest_kv_tensor = torch.zeros((1, 4, config.CROSS_ATTN_DIM))
        self.latest_decision_info = {}
        self.latest_scene_text = ""

    def update(self, kv_tensor: torch.Tensor, decision_info: dict, scene_text: str):
        with self.lock:
            self.latest_kv_tensor = kv_tensor.detach().clone()
            self.latest_decision_info = decision_info
            self.latest_scene_text = scene_text

    def get_latest(self):
        with self.lock: return self.latest_kv_tensor.clone(), self.latest_decision_info, self.latest_scene_text


class DiLuMemoryStore:
    def __init__(self, capacity=100, text_sim_weight=0.25, upsert_dist=0.35):
        self.lock = threading.Lock()
        self.memories = deque(maxlen=capacity)
        self.text_sim_weight = float(np.clip(text_sim_weight, 0.0, 0.7))
        self.upsert_dist = float(max(upsert_dist, 0.05))

    def _scene_tokens(self, scene: str) -> set:
        scene = scene.lower()
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", scene)
        # token， translated note
        return {t for t in tokens if len(t) >= 2}

    def _text_distance(self, a_scene: str, b_scene: str) -> float:
        a_tokens = self._scene_tokens(a_scene)
        b_tokens = self._scene_tokens(b_scene)
        if not a_tokens and not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        sim = inter / max(union, 1)
        return 1.0 - sim

    def _extract_scene_signature(self, scene: str) -> dict:
        signature = {
            "min_front_dx": 80.0,
            "min_front_ttc": 99.0,
            "density": 0.0,
            "left_gap": 80.0,
            "right_gap": 80.0,
        }
        vehicle_pattern = r"longitudinal distance\s*([-+]?\d+(?:\.\d+)?),\s*lateral distance\s*([-+]?\d+(?:\.\d+)?),\s*relative speed\s*([-+]?\d+(?:\.\d+)?)"
        vehicles = re.findall(vehicle_pattern, scene)
        if not vehicles:
            return signature

        parsed = []
        for item in vehicles:
            dx, dy, dvx = float(item[0]), float(item[1]), float(item[2])
            parsed.append((dx, dy, dvx))

        signature["density"] = min(len(parsed) / 4.0, 1.0)
        front = [v for v in parsed if v[0] > 0 and abs(v[1]) < 2.2]
        if front:
            signature["min_front_dx"] = min(v[0] for v in front)
            closing = [v for v in front if v[2] < 0]
            if closing:
                signature["min_front_ttc"] = min(v[0] / (-v[2] + 1e-3) for v in closing)

        left = [v for v in parsed if v[1] > 2.2]
        right = [v for v in parsed if v[1] < -2.2]
        if left:
            signature["left_gap"] = min(abs(v[0]) for v in left)
        if right:
            signature["right_gap"] = min(abs(v[0]) for v in right)
        return signature

    def _signature_distance(self, a: dict, b: dict) -> float:
        keys = ["min_front_dx", "min_front_ttc", "density", "left_gap", "right_gap"]
        scales = {
            "min_front_dx": 40.0,
            "min_front_ttc": 8.0,
            "density": 1.0,
            "left_gap": 30.0,
            "right_gap": 30.0,
        }
        return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) / scales[k] for k in keys)

    def _composite_distance(self, query_scene: str, query_sig: dict, mem: dict, text_weight: float = 0.25) -> float:
        mem_sig = mem.get("scene_signature") or self._extract_scene_signature(mem.get("scene", ""))
        sig_dist = self._signature_distance(query_sig, mem_sig)
        text_dist = self._text_distance(query_scene, mem.get("scene", ""))
        w = float(np.clip(text_weight, 0.0, 0.7))
        return (1.0 - w) * sig_dist + w * text_dist

    def _upsert_memory(self, entry: dict, upsert_dist: float = None):
        if upsert_dist is None:
            upsert_dist = self.upsert_dist
        query_scene = entry.get("scene", "")
        query_sig = entry.get("scene_signature") or self._extract_scene_signature(query_scene)
        mem_type = entry.get("memory_type", "failure")

        best_idx = -1
        best_dist = 1e9
        for idx, old in enumerate(self.memories):
            if old.get("memory_type", "failure") != mem_type:
                continue
            dist = self._composite_distance(query_scene, query_sig, old, text_weight=self.text_sim_weight)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx >= 0 and best_dist <= upsert_dist:
            old = self.memories[best_idx]
            old["scene"] = query_scene
            old["decision"] = entry.get("decision", old.get("decision", {}))
            old["scene_signature"] = query_sig
            old["ts"] = entry.get("ts", time.time())
            old["hit_count"] = int(old.get("hit_count", 1)) + 1
            old_score = float(old.get("quality_score", 0.0))
            new_score = float(entry.get("quality_score", old_score))
            old["quality_score"] = 0.7 * old_score + 0.3 * new_score
            if mem_type == "failure":
                old["reflection"] = entry.get("reflection", old.get("reflection", ""))
            else:
                old["summary"] = entry.get("summary", old.get("summary", ""))
            self.memories[best_idx] = old
            return False

        self.memories.append(entry)
        return True

    def add_failure_memory(self, scene: str, decision: dict, reflection: str, metrics: dict = None):
        with self.lock:
            metrics = metrics or {}
            entry = {
                "memory_type": "failure",
                "scene": scene,
                "decision": decision,
                "reflection": reflection,
                "scene_signature": self._extract_scene_signature(scene),
                "quality_score": -abs(float(metrics.get("risk", 1.0))),
                "metrics": metrics,
                "hit_count": 1,
                "ts": time.time(),
            }
            is_new = self._upsert_memory(entry)
            logger.warning(f"[Memory Archive] {'added' if is_new else 'updated'} high-risk lesson: {reflection}")

    def add_success_memory(self, scene: str, decision: dict, summary: str, metrics: dict = None):
        with self.lock:
            metrics = metrics or {}
            entry = {
                "memory_type": "success",
                "scene": scene,
                "decision": decision,
                "summary": summary,
                "scene_signature": self._extract_scene_signature(scene),
                "quality_score": float(metrics.get("score", 1.0)),
                "metrics": metrics,
                "hit_count": 1,
                "ts": time.time(),
            }
            self._upsert_memory(entry)

    def retrieve_relevant_memories(self, scene: str, k: int = 2, memory_type: str = None) -> list:
        with self.lock:
            if not self.memories:
                return []
            query_sig = self._extract_scene_signature(scene)
            scored = []
            for mem in self.memories:
                if memory_type and mem.get("memory_type", "failure") != memory_type:
                    continue
                dist = self._composite_distance(scene, query_sig, mem, text_weight=self.text_sim_weight)
                # ， DiLu translated note
                age_penalty = min(max((time.time() - float(mem.get("ts", time.time()))) / 3600.0, 0.0), 48.0) / 48.0
                quality_bonus = float(mem.get("quality_score", 0.0)) * 0.05
                final_score = dist + 0.08 * age_penalty - quality_bonus
                scored.append((final_score, mem))
            if not scored:
                return []
            scored.sort(key=lambda x: x[0])
            return [item[1] for item in scored[:k]]

    def format_relevant_memories(self, scene: str, k: int = 2) -> str:
        relevant = self.retrieve_relevant_memories(scene, k, memory_type="failure")
        if not relevant:
            return "No retrievable similar high-risk memories."
        lines = []
        for i, mem in enumerate(relevant):
            decision = mem.get("decision", {})
            metrics = mem.get("metrics", {})
            lines.append(
                f"[Similar High-Risk Memory {i+1}]\n"
                f"- Wrong decision: {decision.get('action', 'unknown')} (speed={decision.get('target_speed', 'NA')})\n"
                f"- Key metrics: risk={metrics.get('risk', 'NA')}, ttc={metrics.get('ttc', 'NA')}\n"
                f"- Reflection takeaway: {mem.get('reflection', '')}"
            )
        return "\n".join(lines)

    def format_success_memories(self, scene: str, k: int = 1) -> str:
        relevant = self.retrieve_relevant_memories(scene, k, memory_type="success")
        if not relevant:
            return "No retrievable successful experiences."
        lines = []
        for i, mem in enumerate(relevant):
            decision = mem.get("decision", {})
            lines.append(
                f"[Similar Successful Experience {i+1}]\n"
                f"- Reusable action: {decision.get('action', 'lane_keep')} (speed={decision.get('target_speed', 'NA')})\n"
                f"- Success takeaway: {mem.get('summary', '')}"
            )
        return "\n".join(lines)

    def build_counterfactual_guidance(self, scene: str, k: int = 2) -> str:
        relevant = self.retrieve_relevant_memories(scene, k, memory_type="failure")
        if not relevant:
            return "No counterfactual suggestions."
        alt_map = {
            "accelerate": "lane_keep or decelerate",
            "lane_change_left": "lane_keep or lane_change_right",
            "lane_change_right": "lane_keep or lane_change_left",
            "lane_keep": "decelerate or safe lane change",
            "decelerate": "lane_keep (when front is safe)",
        }
        lines = []
        for i, mem in enumerate(relevant):
            failed_action = mem.get("decision", {}).get("action", "lane_keep")
            lines.append(
                f"[Counterfactual {i+1}] If a similar state appears again, avoid prioritizing {failed_action}; "
                f"consider {alt_map.get(failed_action, 'lane_keep')}."
            )
        return "\n".join(lines)

    def get_memory_stats(self) -> dict:
        with self.lock:
            total = len(self.memories)
            failures = sum(1 for m in self.memories if m.get("memory_type", "failure") == "failure")
            successes = total - failures
            return {
                "total": total,
                "failure": failures,
                "success": successes,
            }
            
    def retrieve_negative_memories(self, k: int = 2) -> str:
        with self.lock:
            if not self.memories: return "No high-risk historical lessons."
            failures = [m for m in self.memories if m.get("memory_type", "failure") == "failure"]
            if not failures:
                return "No high-risk historical lessons."
            retrieved = failures[-k:]
            return "\n".join([f"[High-Risk Lesson {i+1}]\n- Dangerous scene: {m['scene']}\n- Wrong decision: {m['decision']['action']} (set speed: {m['decision']['target_speed']})\n- Accident reflection: {m['reflection']}\n" for i, m in enumerate(retrieved)])


class SlowSystemThread(threading.Thread):
    def __init__(self, config: SystemConfig, input_queue: queue.Queue, output_buffer: AsyncKVBuffer, memory_store: DiLuMemoryStore):
        super().__init__(daemon=True)
        self.config, self.input_queue, self.output_buffer, self.memory_store = config, input_queue, output_buffer, memory_store
        self.encoder = LatentSkillEncoder(config)
        self.delimiter = "####"
        self.llm_decision_counter = 0
        self.last_logged_action_id = None
        # ： LLM 。 translated note
        self.action_id_to_name = {
            0: "lane_change_left",
            1: "lane_keep",
            2: "lane_change_right",
            3: "accelerate",
            4: "decelerate",
        }
        self.speed_level_to_target_speed = {
            0: 10.0,
            1: 15.0,
            2: 20.0,
            3: 25.0,
            4: 30.0,
        }
        # ： 1 ID， 2
        self.trajectory_templates = {
            0: {"name": "keep", "dx": [10.0, 20.0, 30.0], "dy": [0.0, 0.0, 0.0]},
            1: {"name": "left_change", "dx": [8.0, 18.0, 30.0], "dy": [0.8, 2.8, 4.0]},
            2: {"name": "right_change", "dx": [8.0, 18.0, 30.0], "dy": [-0.8, -2.8, -4.0]},
            3: {"name": "accelerate", "dx": [12.0, 26.0, 42.0], "dy": [0.0, 0.0, 0.0]},
            4: {"name": "decelerate", "dx": [6.0, 12.0, 18.0], "dy": [0.0, 0.0, 0.0]},
        }
        self.curriculum_few_shot_pool = self._init_curriculum_few_shot_pool()
        # self.client = OpenAI(api_key=self.config.LLM_API_KEY)
        # base_url
        self.client = OpenAI(
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_API_BASE 
        )

    def _init_curriculum_few_shot_pool(self) -> dict:
        return {
            "easy": [
                {
                    "user": f"""{self.delimiter} Driving scenario description:
- Lane context: current lane ID=2, total lane count=5, lane width=4.00.
- Ego status: driving on highway, current speed 21.0 (desired cruise speed: 25.0+).
Nearby vehicles:
  - Vehicle 1: longitudinal distance 40.00, lateral distance 0.00, relative speed +2.00.
{self.delimiter} Available actions:
0 Turn-left, 1 IDLE, 2 Turn-right, 3 Acceleration, 4 Deceleration
{self.delimiter} Requirement:
Prefer efficient but safe cruising.
""",
                    "assistant": {
                        "scene_analysis": "Currently in the middle lane of a 5-lane road, with ample front space and no lateral interference.",
                        "trajectory_prediction": "No short-horizon collision risk; cruise efficiency can be improved.",
                        "reasoning": "High-efficiency cruising conditions are satisfied, so choose acceleration with an accelerating trajectory.",
                        "action_id": 3,
                        "speed_level": 3,
                        "trajectory_id": 3,
                    },
                    "tag": "efficiency",
                },
                {
                    "user": f"""{self.delimiter} Driving scenario description:
- Lane context: current lane ID=1, total lane count=3, lane width=4.00.
- Ego status: driving on highway, current speed 24.0 (desired cruise speed: 25.0+).
Nearby vehicles:
  - Vehicle 1: longitudinal distance 18.00, lateral distance 0.00, relative speed -4.50.
{self.delimiter} Available actions:
0 Turn-left, 1 IDLE, 2 Turn-right, 3 Acceleration, 4 Deceleration
{self.delimiter} Requirement:
Keep high speed, lane change to overtake if blocked.
""",
                    "assistant": {
                        "scene_analysis": "Middle lane of a 3-lane road. A slower front vehicle is approaching, and the left lane (ID=0) is clear.",
                        "trajectory_prediction": "Keeping lane increases rear-end risk; a left lane change can bypass the obstacle.",
                        "reasoning": "Left side is safe and not beyond boundary, so decisively execute a left lane change to overtake.",
                        "action_id": 0,
                        "speed_level": 3,
                        "trajectory_id": 1,
                    },
                    "tag": "lane_change_overtake",
                },
            ],
            "hard": [
                {
                    "user": f"""{self.delimiter} Driving scenario description:
- Lane context: current lane ID=0, total lane count=4, lane width=4.00.
- Ego status: driving on highway, current speed 25.0 (desired cruise speed: 25.0+).
Nearby vehicles:
  - Vehicle 1: longitudinal distance 19.20, lateral distance 0.00, relative speed -1.70.
  - Vehicle 2: longitudinal distance 10.60, lateral distance -4.00, relative speed -3.70.
{self.delimiter} Available actions:
0 Turn-left, 1 IDLE, 2 Turn-right, 3 Acceleration, 4 Deceleration
{self.delimiter} Requirement:
Output discrete decision only.
""",
                    "assistant": {
                        "scene_analysis": "Currently at the leftmost lane (ID=0), so left lane change is forbidden. Slow vehicles block both front and right lane (ID=1).",
                        "trajectory_prediction": "No lane-change space; forcing speed or lane change may cause collision or road departure.",
                        "reasoning": "With boundary constraints and both sides blocked, deceleration is the highest-priority safe action.",
                        "action_id": 4,
                        "speed_level": 1,
                        "trajectory_id": 4,
                    },
                    "tag": "boundary_and_close_front",
                },
                {
                    "user": f"""{self.delimiter} Driving scenario description:
- Lane context: current lane ID=2, total lane count=3, lane width=4.00.
- Ego status: driving on highway, current speed 22.0 (desired cruise speed: 25.0+).
Nearby vehicles:
  - Vehicle 1: longitudinal distance 12.00, lateral distance 0.00, relative speed -3.00.
{self.delimiter} Available actions:
0 Turn-left, 1 IDLE, 2 Turn-right, 3 Acceleration, 4 Deceleration
{self.delimiter} Requirement:
Avoid risky lane changes under limited side gaps.
""",
                    "assistant": {
                        "scene_analysis": "Currently at the rightmost side of a 3-lane road (ID=2), so right lane change (action=2) is forbidden. Lead vehicle is rapidly closing.",
                        "trajectory_prediction": "At the right boundary, only deceleration or left lane change is feasible.",
                        "reasoning": "Right side is boundary-limited; only left borrow lane or emergency deceleration is valid. Assuming left is safe, execute left lane change.",
                        "action_id": 0,
                        "speed_level": 2,
                        "trajectory_id": 1,
                    },
                    "tag": "right_boundary_evasion",
                },
            ],
        }

    def _extract_scene_features(self, text_scene: str) -> dict:
        features = {
            "traffic_density": 0.0,
            "min_front_dx": 80.0,
            "min_front_ttc": 99.0,
            "left_gap": 80.0,
            "right_gap": 80.0,
        }
        vehicle_pattern = r"longitudinal distance\s*([-+]?\d+(?:\.\d+)?),\s*lateral distance\s*([-+]?\d+(?:\.\d+)?),\s*relative speed\s*([-+]?\d+(?:\.\d+)?)"
        vehicles = re.findall(vehicle_pattern, text_scene)
        if not vehicles:
            return features

        parsed = [(float(v[0]), float(v[1]), float(v[2])) for v in vehicles]
        features["traffic_density"] = min(len(parsed) / 4.0, 1.0)
        front = [v for v in parsed if v[0] > 0 and abs(v[1]) < 2.2]
        if front:
            features["min_front_dx"] = min(v[0] for v in front)
            closing = [v for v in front if v[2] < 0]
            if closing:
                features["min_front_ttc"] = min(v[0] / (-v[2] + 1e-3) for v in closing)
        left = [v for v in parsed if v[1] > 2.2]
        right = [v for v in parsed if v[1] < -2.2]
        if left:
            features["left_gap"] = min(abs(v[0]) for v in left)
        if right:
            features["right_gap"] = min(abs(v[0]) for v in right)
        return features

    def _infer_curriculum_level(self, text_scene: str) -> str:
        f = self._extract_scene_features(text_scene)
        if f["min_front_dx"] < 18.0 or f["min_front_ttc"] < 2.0 or f["traffic_density"] > 0.65:
            return "hard"
        return "easy"

    def _sample_few_shot_messages(self, text_scene: str) -> list:
        level = self._infer_curriculum_level(text_scene)
        easy_pool = self.curriculum_few_shot_pool["easy"]
        hard_pool = self.curriculum_few_shot_pool["hard"]
        sample_pairs = max(1, self.config.FEWSHOT_SAMPLE_PAIRS)

        chosen = []
        if level == "hard":
            hard_num = max(1, int(round(sample_pairs * self.config.FEWSHOT_HARD_RATIO)))
            easy_num = max(0, sample_pairs - hard_num)
            chosen.extend(random.sample(hard_pool, k=min(hard_num, len(hard_pool))))
            if easy_num > 0:
                chosen.extend(random.sample(easy_pool, k=min(easy_num, len(easy_pool))))
        else:
            chosen.extend(random.sample(easy_pool, k=min(sample_pairs, len(easy_pool))))

        # user/assistant
        msg_list = []
        for item in chosen:
            msg_list.append({"role": "user", "content": item["user"]})
            msg_list.append({"role": "assistant", "content": json.dumps(item["assistant"], ensure_ascii=False)})
        return msg_list

    def _extract_lane_context(self, text_scene: str) -> tuple:
        lane_count = self.config.DEFAULT_LANE_COUNT
        ego_lane_id = self.config.DEFAULT_EGO_LANE_ID
        lane_width = self.config.LANE_WIDTH_M
        pattern = r"Lane context: current lane ID=(\d+),\s*total lane count=(\d+),\s*lane width=([0-9.]+)"
        matched = re.search(pattern, text_scene)
        if matched:
            ego_lane_id = int(matched.group(1))
            lane_count = max(1, int(matched.group(2)))
            lane_width = max(1.0, float(matched.group(3)))
        ego_lane_id = min(max(ego_lane_id, 0), lane_count - 1)
        return lane_count, ego_lane_id, lane_width

    def _post_process_trajectory(self, traj: list, action_id: int, text_scene: str) -> list:
        lane_count, ego_lane_id, lane_width = self._extract_lane_context(text_scene)
        left_limit = ego_lane_id * lane_width
        right_limit = -((lane_count - 1) - ego_lane_id) * lane_width

        processed = []
        prev_dx = 0.0
        prev_dy = 0.0
        for idx, point in enumerate(traj):
            try:
                dx = float(point[0])
                dy = float(point[1])
            except (TypeError, ValueError, IndexError):
                dx = max(self.config.TRAJ_MIN_DX_M, prev_dx + self.config.TRAJ_MIN_DX_GAP_M)
                dy = prev_dy

            dx = max(dx, self.config.TRAJ_MIN_DX_M)
            dx = max(dx, prev_dx + self.config.TRAJ_MIN_DX_GAP_M)
            dy = float(np.clip(dy, right_limit, left_limit))

            if idx > 0:
                dy = self.config.TRAJ_SMOOTH_ALPHA * dy + (1.0 - self.config.TRAJ_SMOOTH_ALPHA) * prev_dy
                dy = float(np.clip(dy, right_limit, left_limit))

            processed.append([round(dx, 3), round(dy, 3)])
            prev_dx, prev_dy = dx, dy

        if action_id in (1, 3, 4):
            for point in processed:
                point[1] = round(float(np.clip(point[1], -0.35 * lane_width, 0.35 * lane_width)), 3)

        if action_id == 0:
            for point in processed:
                point[1] = round(max(point[1], min(lane_width, left_limit)), 3)
        elif action_id == 2:
            for point in processed:
                point[1] = round(min(point[1], max(-lane_width, right_limit)), 3)

        while len(processed) < 3:
            processed.append(processed[-1])
        return processed[:3]

    def _build_parametric_trajectory(self, action_id: int, trajectory_id: int, speed_level: int, text_scene: str) -> tuple:
        lane_count, ego_lane_id, lane_width = self._extract_lane_context(text_scene)
        scene_f = self._extract_scene_features(text_scene)
        profile = self.trajectory_templates[trajectory_id]

        base_dx = profile["dx"]
        base_dy = profile["dy"]
        speed_scale = 1.0 + self.config.TEMPLATE_SPEED_GAIN * (speed_level - 2)
        gap_raw = scene_f["min_front_dx"] / max(self.config.TEMPLATE_GAP_REF, 1e-6)
        ttc = float(scene_f.get("min_front_ttc", 99.0))
        # TTC ， ， 。
        ttc_factor = float(np.clip(ttc / 4.0, 0.70, 1.25))
        gap_scale = float(np.clip(gap_raw * ttc_factor, self.config.TEMPLATE_GAP_CLAMP_MIN, self.config.TEMPLATE_GAP_CLAMP_MAX))

        dx_scale = speed_scale * gap_scale
        
        dy_scale = lane_width / 4.0
        if action_id in (1, 3, 4):
            dy_scale *= 0.35

        param_traj = [[base_dx[i] * dx_scale, base_dy[i] * dy_scale] for i in range(3)]
        decoder_trace = {
            "template": profile["name"],
            "speed_scale": round(speed_scale, 3),
            "gap_scale": round(gap_scale, 3),
            "dx_scale": round(dx_scale, 3),
            "lane_count": lane_count,
            "ego_lane_id": ego_lane_id,
            "lane_width": lane_width,
        }
        return param_traj, decoder_trace

    def _safe_int(self, value, default_value: int, min_val: int, max_val: int, field_name: str) -> int:
        try:
            parsed = int(value)
            if parsed < min_val or parsed > max_val:
                raise ValueError
            return parsed
        except (TypeError, ValueError):
            logger.warning(f"Discrete field {field_name} invalid: {value}, fallback to {default_value}")
            return default_value

    def _decode_macro_decision(self, llm_decision: dict, text_scene: str) -> dict:
        action_id = self._safe_int(llm_decision.get("action_id", 1), default_value=1, min_val=0, max_val=4, field_name="action_id")
        speed_level = self._safe_int(llm_decision.get("speed_level", 2), default_value=2, min_val=0, max_val=4, field_name="speed_level")
        trajectory_id = self._safe_int(llm_decision.get("trajectory_id", action_id), default_value=action_id, min_val=0, max_val=4, field_name="trajectory_id")

        lane_count, ego_lane_id, lane_width = self._extract_lane_context(text_scene)

        # 💥 [ ： ] 💥
        # ， ， translated note
        if action_id == 0 and ego_lane_id <= 0:
            logger.warning("Intercepted: vehicle is already at the leftmost lane, but LLM requested left lane change; forced overwrite to lane_keep.")
            action_id, trajectory_id = 1, 0
        elif action_id == 2 and ego_lane_id >= lane_count - 1:
            logger.warning("Intercepted: vehicle is already at the rightmost lane, but LLM requested right lane change; forced overwrite to lane_keep.")
            action_id, trajectory_id = 1, 0

        # ， “ + ” 。
        if action_id == 1:
            if trajectory_id in (1, 2):
                trajectory_id = 0
        elif action_id == 0:
            trajectory_id = 1
        elif action_id == 2:
            trajectory_id = 2
        elif action_id == 3:
            trajectory_id = 3
        elif action_id == 4:
            trajectory_id = 4

        reasoning = llm_decision.get("reasoning", "Model did not provide reasoning; using default conservative decoding.")
        scene_analysis = llm_decision.get("scene_analysis", "")
        trajectory_prediction = llm_decision.get("trajectory_prediction", "")
        param_traj, decoder_trace = self._build_parametric_trajectory(action_id, trajectory_id, speed_level, text_scene)
        decoded_traj = self._post_process_trajectory(copy.deepcopy(param_traj), action_id, text_scene)

        base_speed = self.speed_level_to_target_speed[speed_level]
        front_dx = self._extract_scene_features(text_scene)["min_front_dx"]
        speed_adjust = float(np.clip((front_dx - 20.0) / 20.0, -0.8, 0.8))
        target_speed = float(np.clip(base_speed * (1.0 + 0.08 * speed_adjust), 10.0, 32.0))

        return {
            "scene_analysis": scene_analysis,
            "trajectory_prediction": trajectory_prediction,
            "reasoning": reasoning,
            "action_id": action_id,
            "speed_level": speed_level,
            "trajectory_id": trajectory_id,
            "action": self.action_id_to_name[action_id],
            "target_speed": target_speed,
            "reference_trajectory": decoded_traj,
            "decoder_trace": decoder_trace,
        }

    def run(self):
        logger.info("LLM reasoning thread with CoT motion prediction and reflection has started...")
        while True:
            # text_scene = self.input_queue.get()

            # # --- ： ， ---
            # while not self.input_queue.empty():
            #     try:
            #         text_scene = self.input_queue.get_nowait()
            #         self.input_queue.task_done()
            #     except queue.Empty:
            #         break
            # 【 ： 】 translated note
            text_scene = self.input_queue.get_blocking()

            try: macro_decision = self._call_llm_with_reflection(text_scene)
            except Exception as e:
                logger.error(f"LLM call failed, triggering conservative fallback policy: {e}")
                macro_decision = {
                    "action": "decelerate", "target_speed": 10.0, 
                    "reference_trajectory": [[5.0, 0.0], [10.0, 0.0], [15.0, 0.0]], 
                    "reasoning": "System exception fallback."
                }
            
            kv_tensor = self.encoder(macro_decision)
            self.output_buffer.update(kv_tensor, macro_decision, text_scene)
            # self.input_queue.task_done()

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _call_llm_with_reflection(self, text_scene: str) -> dict:
        system_prompt = """
        You are an advanced autonomous-driving decision core. Your mission is to drive on a multi-lane highway with both safety and high efficiency.
        Perform chain-of-thought style internal reasoning, but output only discrete decision indicators in the final JSON.

        [Driving Rules]
        1. Efficiency first (highest priority): target highway cruise speed is 25.0+ m/s.
           If the front 30 meters is clear, prioritize throttle (action_id=3, speed_level=3 or 4).
        2. Prefer lane change over braking when blocked by a slower lead vehicle (action_id=0 or 2 when safe).
        3. Decelerate only in critical situations: allow action_id=4 mainly when front distance < 15m and rapidly closing.
        4. Dynamic boundary awareness (CRITICAL): always use [total lane count] and [current lane ID] from the scene.
           - Leftmost lane ID is always 0. If in lane 0, do NOT choose left lane change (action_id=0).
           - Rightmost lane ID is (total lane count - 1). If at rightmost lane, do NOT choose right lane change (action_id=2).
           - Any decision causing road-edge departure is fatal.

        [Reasoning Outputs]
        1. scene_analysis: analyze surrounding vehicles and lane boundary constraints.
        2. trajectory_prediction: predict nearby vehicle motion and short-horizon risk.
        3. macro decision: output action_id, speed_level, trajectory_id.

        [Discrete Dictionary]
        - action_id: 0=lane_change_left, 1=lane_keep, 2=lane_change_right, 3=accelerate, 4=decelerate
        - speed_level: 0=10.0m/s, 1=15.0m/s, 2=20.0m/s, 3=25.0m/s, 4=30.0m/s
        - trajectory_id: 0=straight, 1=left-change, 2=right-change, 3=accelerating, 4=decelerating

        [Output Format]
        Return JSON only, containing exactly these keys:
        "scene_analysis", "trajectory_prediction", "reasoning", "action_id", "speed_level", "trajectory_id"
        action_id/speed_level/trajectory_id must be integers.
        """
        neg_memories = self.memory_store.format_relevant_memories(text_scene, self.config.MEMORY_RETRIEVE_FAILURE_K)
        pos_memories = self.memory_store.format_success_memories(text_scene, self.config.MEMORY_RETRIEVE_SUCCESS_K)
        counterfactual_guidance = self.memory_store.build_counterfactual_guidance(text_scene, self.config.MEMORY_RETRIEVE_FAILURE_K)
        curriculum_examples = self._sample_few_shot_messages(text_scene)
        user_prompt = (
                        f"{self.delimiter} High-risk reflections:\n{neg_memories}\n\n"
                f"{self.delimiter} Reusable successful experiences:\n{pos_memories}\n\n"
                f"{self.delimiter} Counterfactual guidance:\n{counterfactual_guidance}\n\n"
                        f"{self.delimiter} Current scene:\n{text_scene}\n\n"
                        f"{self.delimiter} Task:\nPlease output CoT decision JSON, and action_id/speed_level/trajectory_id must be integers."
                )
        messages = [{"role": "system", "content": system_prompt}] + curriculum_examples + [{"role": "user", "content": user_prompt}]
        
        start_t = time.time()
        res = self.client.chat.completions.create(
            model=self.config.LLM_CHAT_MODEL,
                        messages=messages,
            response_format={"type": "json_object"}, temperature=0.1, max_tokens=512
        )
        # decision = json.loads(res.choices[0].message.content)
        # logger.info(f"LLM [{time.time()-start_t:.2f}s] : {decision.get('action')} | : {decision.get('reference_trajectory')}")
        # return decision
        # : translated note
        raw_content = res.choices[0].message.content
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:-3].strip()
        llm_decision = json.loads(raw_content)
        decision = self._decode_macro_decision(llm_decision, text_scene)
        action_val = decision.get('action', 'lane_keep')
        speed_val = decision.get('target_speed', 20.0)
        traj_val = decision.get('reference_trajectory', 'no_trajectory')
        action_id = decision.get('action_id', 1)
        speed_level = decision.get('speed_level', 2)
        trajectory_id = decision.get('trajectory_id', 0)
        decoder_trace = decision.get('decoder_trace', {})

        self.llm_decision_counter += 1
        log_every = max(1, int(self.config.LLM_LOG_EVERY_N))
        action_changed = (self.last_logged_action_id is None) or (action_id != self.last_logged_action_id)
        should_log = self.config.LLM_VERBOSE_LOG or action_changed or (self.llm_decision_counter % log_every == 0)
        if should_log:
            logger.info(
                f"LLM decision [{time.time()-start_t:.2f}s] "
                f"discrete(action_id={action_id}, speed_level={speed_level}, trajectory_id={trajectory_id}) "
                f"=> action: {action_val} | target speed: {speed_val} | planned waypoints: {traj_val} | decoder trace: {decoder_trace}"
            )
            self.last_logged_action_id = action_id
        
        return decision
        


# =====================================================================
# Part 3: ： AC CBF ( )
# =====================================================================
class RiskAssessmentModule:
    @staticmethod
    def evaluate(state: torch.Tensor, config: SystemConfig) -> torch.Tensor:
        batch_size = state.shape[0]
        risk_metrics = torch.zeros((batch_size, 3), device=state.device)
        # --- ： ---
        state_3d = state.view(batch_size, config.VEHICLES_COUNT, config.FEATURES_COUNT).clone()
        state_3d[:, :, 1] *= 100.0
        state_3d[:, :, 3] *= 30.0
        # ----------------------------
        
        presence = state_3d[:, 1:, 0]
        dx = state_3d[:, 1:, 1] + 1e-3
        dvx = state_3d[:, 1:, 3]
        
        danger_mask = (presence > 0.5) & (dx > 0) & (dvx < 0)
        inv_ttc = torch.zeros_like(dx)
        inv_ttc[danger_mask] = -dvx[danger_mask] / dx[danger_mask]
        max_inv_ttc, _ = torch.max(inv_ttc, dim=1)
        
        risk_metrics[:, 1] = max_inv_ttc
        risk_metrics[:, 0] = torch.clamp(max_inv_ttc * 0.5, 0.0, 1.0) 
        return risk_metrics

class SafetyShieldCBF:
    @staticmethod
    def project_action(raw_action: torch.Tensor, risk_metrics: torch.Tensor) -> torch.Tensor:
        safe_action = raw_action.clone()
        danger_mask = (risk_metrics[:, 0] > 0.8) | (risk_metrics[:, 1] > 0.5)
        acceleration = raw_action[:, 0]
        violation_idx = danger_mask & (acceleration > 0)
        if violation_idx.any():
            safe_action[violation_idx, 0] = -1.0 
            safe_action[violation_idx, 1] *= 0.5 
        return safe_action

class UncertaintyAwareGate:
    @staticmethod
    def _extract_lane_ctx(decision_info: dict, config: SystemConfig):
        if not decision_info:
            return config.DEFAULT_LANE_COUNT, config.DEFAULT_EGO_LANE_ID
        trace = decision_info.get("decoder_trace", {}) if isinstance(decision_info, dict) else {}
        lane_count = int(trace.get("lane_count", config.DEFAULT_LANE_COUNT))
        ego_lane_id = int(trace.get("ego_lane_id", config.DEFAULT_EGO_LANE_ID))
        lane_count = max(1, lane_count)
        ego_lane_id = max(0, min(ego_lane_id, lane_count - 1))
        return lane_count, ego_lane_id

    @staticmethod
    def _intent_prior(decision_info: dict, device: torch.device, config: SystemConfig) -> torch.Tensor:
        action_id = decision_info.get("action_id", 1) if decision_info else 1
        prior_map = {
            0: torch.tensor([0.0, 0.35], device=device),   # translated note
            1: torch.tensor([0.0, 0.0], device=device),    # translated note
            2: torch.tensor([0.0, -0.35], device=device),  # translated note
            3: torch.tensor([0.60, 0.0], device=device),   # translated note
            4: torch.tensor([-0.80, 0.0], device=device),  # translated note
        }
        prior = prior_map.get(action_id, prior_map[1]).clone()

        lane_count, ego_lane_id = UncertaintyAwareGate._extract_lane_ctx(decision_info, config)
        at_left_boundary = ego_lane_id == 0
        at_right_boundary = ego_lane_id == lane_count - 1

        if at_left_boundary and prior[1].item() > 0:
            prior[1] = 0.0
        if at_right_boundary and prior[1].item() < 0:
            prior[1] = 0.0
        return prior

    @staticmethod
    def apply(
        proposed_action: torch.Tensor,
        risk_metrics: torch.Tensor,
        uncertainty_score: torch.Tensor,
        decision_info: dict,
        config: SystemConfig,
    ):
        gated_action = proposed_action.clone()
        batch_size = proposed_action.shape[0]
        blend_strength = torch.zeros((batch_size, 1), dtype=proposed_action.dtype, device=proposed_action.device)
        trigger_mask = (risk_metrics[:, 0] >= config.GATE_RISK_THRESHOLD) | (uncertainty_score[:, 0] >= config.UNCERTAINTY_TRIGGER_STD)

        for b in range(batch_size):
            if not bool(trigger_mask[b].item()):
                continue
            risk_val = float(risk_metrics[b, 0].item())
            unc_norm = float(min(max(uncertainty_score[b, 0].item() / max(config.UNCERTAINTY_SCORE_REF, 1e-6), 0.0), 1.0))
            lam = config.GATE_RISK_WEIGHT * risk_val + config.GATE_UNCERT_WEIGHT * unc_norm
            lam = float(min(max(lam, config.GATE_MIN_BLEND), 1.0))
            prior = UncertaintyAwareGate._intent_prior(decision_info, proposed_action.device, config)
            gated_action[b] = (1.0 - lam) * proposed_action[b] + lam * prior
            blend_strength[b, 0] = lam

        return torch.clamp(gated_action, -1.0, 1.0), blend_strength, trigger_mask

class ShortHorizonSafetyReviewer:
    @staticmethod
    def review(state: torch.Tensor, action: torch.Tensor, config: SystemConfig, decision_info: dict = None):
        reviewed = action.clone()
        intervention_mask = torch.zeros((action.shape[0],), dtype=torch.bool, device=action.device)
        min_ttc_tensor = torch.full((action.shape[0], 1), 99.0, dtype=action.dtype, device=action.device)

        trace = decision_info.get("decoder_trace", {}) if isinstance(decision_info, dict) else {}
        lane_count = int(trace.get("lane_count", config.DEFAULT_LANE_COUNT))
        ego_lane_id = int(trace.get("ego_lane_id", config.DEFAULT_EGO_LANE_ID))
        lane_count = max(1, lane_count)
        ego_lane_id = max(0, min(ego_lane_id, lane_count - 1))
        at_left_boundary = ego_lane_id == 0
        at_right_boundary = ego_lane_id == lane_count - 1

        # --- ： ---
        state_3d = state.view(state.shape[0], config.VEHICLES_COUNT, config.FEATURES_COUNT).clone()
        state_3d[:, :, 1] *= 100.0
        state_3d[:, :, 2] *= 10.0
        state_3d[:, :, 3] *= 30.0
        # ----------------------------

        for b in range(action.shape[0]):
            others = state_3d[b, 1:, :] # state_3d
            presence = others[:, 0] > 0.5
            dx = others[:, 1]
            dy = others[:, 2]
            dvx = others[:, 3]

            front_mask = presence & (dx > 0) & (torch.abs(dy) < config.LANE_WIDTH_M * 0.55)
            min_front_dx = 1e6
            min_ttc = 99.0
            if front_mask.any():
                front_dx = dx[front_mask]
                min_front_dx = float(front_dx.min().item())
                front_dvx = dvx[front_mask]
                close_mask = front_dvx < -1e-3
                if close_mask.any():
                    ttc = front_dx[close_mask] / (-front_dvx[close_mask] + 1e-3)
                    min_ttc = float(ttc.min().item())
            min_ttc_tensor[b, 0] = min_ttc

            accel = float(reviewed[b, 0].item())
            steer = float(reviewed[b, 1].item())
            touched = False

            # ： 。 translated note
            if at_left_boundary and steer > config.REVIEW_BOUNDARY_STEER_EPS:
                reviewed[b, 1] = min(0.0, reviewed[b, 1].item())
                touched = True
            if at_right_boundary and steer < -config.REVIEW_BOUNDARY_STEER_EPS:
                reviewed[b, 1] = max(0.0, reviewed[b, 1].item())
                touched = True

            if accel > 0 and (min_front_dx < config.REVIEW_MIN_FRONT_GAP or min_ttc < config.REVIEW_MIN_TTC):
                reviewed[b, 0] = min(config.REVIEW_ACCEL_LIMIT, reviewed[b, 0].item())
                touched = True
            if min_ttc < 0.8:
                reviewed[b, 0] = min(config.REVIEW_HARD_BRAKE, reviewed[b, 0].item())
                touched = True

            if steer > config.REVIEW_STEER_TRIGGER:
                left_mask = presence & (dy > config.LANE_WIDTH_M * 0.35)
                if left_mask.any():
                    lane_dx = torch.abs(dx[left_mask])
                    lane_dvx = torch.abs(dvx[left_mask])
                    lane_ttc = lane_dx / (lane_dvx + 1e-3)
                    if float(lane_dx.min().item()) < config.REVIEW_MIN_LANE_GAP or float(lane_ttc.min().item()) < config.REVIEW_LANE_MIN_TTC:
                        reviewed[b, 1] = 0.0
                        touched = True
            elif steer < -config.REVIEW_STEER_TRIGGER:
                right_mask = presence & (dy < -config.LANE_WIDTH_M * 0.35)
                if right_mask.any():
                    lane_dx = torch.abs(dx[right_mask])
                    lane_dvx = torch.abs(dvx[right_mask])
                    lane_ttc = lane_dx / (lane_dvx + 1e-3)
                    if float(lane_dx.min().item()) < config.REVIEW_MIN_LANE_GAP or float(lane_ttc.min().item()) < config.REVIEW_LANE_MIN_TTC:
                        reviewed[b, 1] = 0.0
                        touched = True

            intervention_mask[b] = touched

        return torch.clamp(reviewed, -1.0, 1.0), intervention_mask, min_ttc_tensor

class QueryAnchorTrajectoryPredictor:
    def __init__(self, config: SystemConfig):
        self.config = config

    def _extract_state_features(self, state: torch.Tensor) -> dict:
        # --- ： ---
        state_3d = state.view(state.shape[0], self.config.VEHICLES_COUNT, self.config.FEATURES_COUNT).clone()
        state_3d[:, :, 1] *= 100.0
        state_3d[:, :, 2] *= 10.0
        state_3d[:, :, 3] *= 30.0
        # ----------------------------
        others = state_3d[:, 1:, :]
        presence = others[:, :, 0] > 0.5
        dx = others[:, :, 1]
        dy = others[:, :, 2]
        dvx = others[:, :, 3]

        features = {
            "front_gap": torch.full((state.shape[0],), 80.0, device=state.device),
            "front_ttc": torch.full((state.shape[0],), 99.0, device=state.device),
            "left_gap": torch.full((state.shape[0],), 80.0, device=state.device),
            "right_gap": torch.full((state.shape[0],), 80.0, device=state.device),
            "ego_speed": state_3d[:, 0, 3] ,
        }

        for b in range(state.shape[0]):
            b_presence = presence[b]
            b_dx = dx[b]
            b_dy = dy[b]
            b_dvx = dvx[b]

            front_mask = b_presence & (b_dx > 0) & (torch.abs(b_dy) < self.config.LANE_WIDTH_M * 0.55)
            if front_mask.any():
                front_dx = b_dx[front_mask]
                features["front_gap"][b] = torch.min(front_dx)
                front_dvx = b_dvx[front_mask]
                closing = front_dvx < -1e-3
                if closing.any():
                    ttc = front_dx[closing] / (-front_dvx[closing] + 1e-3)
                    features["front_ttc"][b] = torch.min(ttc)

            left_mask = b_presence & (b_dy > self.config.LANE_WIDTH_M * 0.35)
            right_mask = b_presence & (b_dy < -self.config.LANE_WIDTH_M * 0.35)
            if left_mask.any():
                features["left_gap"][b] = torch.min(torch.abs(b_dx[left_mask]))
            if right_mask.any():
                features["right_gap"][b] = torch.min(torch.abs(b_dx[right_mask]))

        return features

    def _build_anchor_bank(self, action_id: int, lane_width: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        keep = [
            [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
            [[11.0, 0.2], [22.0, 0.1], [33.0, 0.0]],
            [[9.0, -0.2], [18.0, -0.1], [27.0, 0.0]],
            [[8.0, 0.0], [16.0, 0.0], [24.0, 0.0]],
        ]
        left = [
            [[8.0, 0.7 * lane_width], [18.0, 0.95 * lane_width], [30.0, 1.0 * lane_width]],
            [[9.0, 0.45 * lane_width], [20.0, 0.85 * lane_width], [32.0, 1.0 * lane_width]],
            [[7.0, 0.35 * lane_width], [15.0, 0.7 * lane_width], [25.0, 0.9 * lane_width]],
            [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        ]
        right = [
            [[8.0, -0.7 * lane_width], [18.0, -0.95 * lane_width], [30.0, -1.0 * lane_width]],
            [[9.0, -0.45 * lane_width], [20.0, -0.85 * lane_width], [32.0, -1.0 * lane_width]],
            [[7.0, -0.35 * lane_width], [15.0, -0.7 * lane_width], [25.0, -0.9 * lane_width]],
            [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        ]
        accel = [
            [[12.0, 0.0], [26.0, 0.0], [42.0, 0.0]],
            [[13.0, 0.15], [28.0, 0.08], [45.0, 0.0]],
            [[11.0, -0.15], [24.0, -0.08], [39.0, 0.0]],
            [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        ]
        decel = [
            [[6.0, 0.0], [12.0, 0.0], [18.0, 0.0]],
            [[7.0, 0.1], [14.0, 0.05], [20.0, 0.0]],
            [[5.0, -0.1], [10.0, -0.05], [15.0, 0.0]],
            [[8.0, 0.0], [16.0, 0.0], [24.0, 0.0]],
        ]
        table = {0: left, 1: keep, 2: right, 3: accel, 4: decel}
        anchors = table.get(action_id, keep)
        return torch.tensor(anchors[: self.config.TRAJ_QUERY_COUNT], device=device, dtype=dtype)

    def _mode_scores(self, action_id: int, feat: dict, batch_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scores = torch.zeros((self.config.TRAJ_QUERY_COUNT,), device=device, dtype=dtype)
        front_gap = float(feat["front_gap"][batch_idx].item())
        front_ttc = float(feat["front_ttc"][batch_idx].item())
        left_gap = float(feat["left_gap"][batch_idx].item())
        right_gap = float(feat["right_gap"][batch_idx].item())

        if action_id == 0:
            scores[0] = left_gap / 18.0 - 0.2
            scores[1] = left_gap / 22.0
            scores[2] = left_gap / 25.0 - 0.15
            scores[3] = front_gap / 28.0 - 0.45
        elif action_id == 2:
            scores[0] = right_gap / 18.0 - 0.2
            scores[1] = right_gap / 22.0
            scores[2] = right_gap / 25.0 - 0.15
            scores[3] = front_gap / 28.0 - 0.45
        elif action_id == 3:
            scores[0] = front_gap / 20.0 + max(front_ttc - 2.0, 0.0) * 0.1
            scores[1] = front_gap / 24.0
            scores[2] = front_gap / 24.0
            scores[3] = 0.25
        elif action_id == 4:
            scores[0] = (2.5 - min(front_ttc, 2.5)) * 0.8 + (12.0 - min(front_gap, 12.0)) * 0.06
            scores[1] = scores[0] * 0.85
            scores[2] = scores[0] * 0.75
            scores[3] = 0.2
        else:
            scores[0] = front_gap / 24.0
            scores[1] = front_gap / 28.0
            scores[2] = front_gap / 28.0
            scores[3] = 0.4
        return scores

    def predict(self, state: torch.Tensor, decision_info: dict = None) -> dict:
        device = state.device
        dtype = state.dtype
        batch_size = state.shape[0]
        feat = self._extract_state_features(state)

        if decision_info is None:
            action_id = 1
            speed_level = 2
        else:
            action_id = int(decision_info.get("action_id", 1))
            speed_level = int(decision_info.get("speed_level", 2))

        traj_batch, prob_batch, unc_batch = [], [], []
        lane_width = self.config.LANE_WIDTH_M
        for b in range(batch_size):
            anchors = self._build_anchor_bank(action_id, lane_width, device, dtype)
            speed_scale = 1.0 + 0.08 * (speed_level - 2)
            gap_scale = float(np.clip(float(feat["front_gap"][b].item()) / 25.0, 0.75, 1.25))
            anchors[:, :, 0] = anchors[:, :, 0] * speed_scale * gap_scale

            scores = self._mode_scores(action_id, feat, b, device, dtype)
            probs = torch.softmax(scores / max(self.config.TRAJ_QUERY_TEMPERATURE, 1e-3), dim=-1)
            unc = 1.0 - probs

            traj_batch.append(anchors)
            prob_batch.append(probs)
            unc_batch.append(unc)

        trajectories = torch.stack(traj_batch, dim=0)
        probs = torch.stack(prob_batch, dim=0)
        uncertainty = torch.stack(unc_batch, dim=0)

        expected_traj = (probs.unsqueeze(-1).unsqueeze(-1) * trajectories).sum(dim=1)
        first_wp = expected_traj[:, 0, :]
        
        # guide_acc: dx ， [-1,1]
        # dx [6,14] ， 12 ( )
        guide_acc = torch.clamp((first_wp[:, 0] - self.config.TRAJ_GUIDE_ACCEL_REF) / self.config.TRAJ_GUIDE_ACCEL_NORM, -1.0, 1.0)
        guide_steer = torch.clamp(first_wp[:, 1] / max(self.config.LANE_WIDTH_M, 1e-6), -1.0, 1.0)
        guide_action = torch.stack([guide_acc, guide_steer], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1, keepdim=True)

        return {
            "trajectories": trajectories,
            "probs": probs,
            "uncertainty": uncertainty,
            "expected_trajectory": expected_traj,
            "guide_action": guide_action,
            "traj_entropy": entropy,
        }

class RiskSensitiveGaussianAC(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, config: SystemConfig):
        super().__init__()
        self.config = config
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, config.CROSS_ATTN_DIM))
        self.seq_pos_embed = nn.Parameter(torch.randn(1, 4, config.CROSS_ATTN_DIM) * 0.02)
        # CrossAttention KV
        self.cross_attention = nn.MultiheadAttention(embed_dim=config.CROSS_ATTN_DIM, num_heads=4, batch_first=True)
        
        self.actor_net = nn.Sequential(nn.Linear(config.CROSS_ATTN_DIM, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mu_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        
        critic_in = config.CROSS_ATTN_DIM + action_dim + 3 
        def create_critic(): return nn.Sequential(nn.Linear(critic_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.critic_net1, self.critic_net2 = create_critic(), create_critic()

    def get_shared_latent(self, state: torch.Tensor, key_value_tensor: torch.Tensor):
        # state query, : (B, 1, Dim)
        q = self.state_encoder(state).unsqueeze(1)
        # KV ， Attention
        # kv_with_pos = key_value_tensor + self.seq_pos_embed 
        # 💥 ： (GPU/CPU)
        kv_with_pos = key_value_tensor + self.seq_pos_embed.to(key_value_tensor.device)
        latent_skill_vector, _ = self.cross_attention(query=q, key=kv_with_pos, value=kv_with_pos)
        return latent_skill_vector.squeeze(1)
        # key_value_tensor (B, SeqLen=4, Dim)
        # 4 Token（ +3 ）
        latent_skill_vector, _ = self.cross_attention(query=q, key=key_value_tensor, value=key_value_tensor)
        return latent_skill_vector.squeeze(1) # : (B, Dim)

    def actor_distribution_params(self, latent: torch.Tensor):
        actor_features = self.actor_net(latent)
        mu = self.mu_layer(actor_features)
        log_std = torch.clamp(self.log_std_layer(actor_features), self.config.MIN_LOG_STD, self.config.MAX_LOG_STD)
        return mu, log_std
    
    def forward_actor(self, latent: torch.Tensor, deterministic: bool = False):
        mu, log_std = self.actor_distribution_params(latent)
        dist = Normal(mu, torch.exp(log_std))
        
        if deterministic:
            raw_action = mu; log_prob = None
        else:
            raw_action = dist.rsample()
            log_prob = dist.log_prob(raw_action).sum(axis=-1, keepdim=True)
            log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(axis=-1, keepdim=True)
        return torch.tanh(raw_action), raw_action, log_prob

    def estimate_uncertainty(self, latent: torch.Tensor) -> torch.Tensor:
        actor_features = self.actor_net(latent)
        log_std = torch.clamp(self.log_std_layer(actor_features), self.config.MIN_LOG_STD, self.config.MAX_LOG_STD)
        std = torch.exp(log_std)
        return std.mean(dim=-1, keepdim=True)

    def forward_critic(self, latent, action, risk_metrics):
        x = torch.cat([latent, action, risk_metrics], dim=-1)
        return self.critic_net1(x), self.critic_net2(x)


# =====================================================================
# Part 4: 、
# =====================================================================
class SimpleReplayBuffer:
    def __init__(self, capacity=10000, hard_ratio=0.45, hard_capacity_ratio=0.5):
        self.buffer = deque(maxlen=capacity)
        hard_capacity = max(1, int(capacity * hard_capacity_ratio))
        self.hard_buffer = deque(maxlen=hard_capacity)
        self.hard_ratio = float(np.clip(hard_ratio, 0.0, 0.9))

    def push(self, *args, hard=False):
        self.buffer.append(args)
        if hard:
            self.hard_buffer.append(args)

    def sample(self, batch_size):
        hard_n = min(int(round(batch_size * self.hard_ratio)), len(self.hard_buffer))
        normal_n = batch_size - hard_n

        hard_batch = random.sample(self.hard_buffer, hard_n) if hard_n > 0 else []
        normal_source = self.buffer
        if len(normal_source) < normal_n:
            normal_n = len(normal_source)
        normal_batch = random.sample(normal_source, normal_n) if normal_n > 0 else []

        batch = hard_batch + normal_batch
        while len(batch) < batch_size and len(self.buffer) > 0:
            batch.append(random.choice(self.buffer))
        random.shuffle(batch)
        return map(lambda x: torch.stack(x) if isinstance(x[0], torch.Tensor) else torch.tensor(x, dtype=torch.float32), zip(*batch))

    def __len__(self): return len(self.buffer)

def save_checkpoint(episode, ac_net, log_alpha, optim_actor, optim_critic, optim_alpha, config):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(config.CHECKPOINT_DIR, f"hybrid_sac_ep{episode}.pth")
    torch.save({
        'episode': episode,
        'ac_net_state_dict': ac_net.state_dict(),
        'log_alpha': log_alpha,
        'optim_actor_state_dict': optim_actor.state_dict(),
        'optim_critic_state_dict': optim_critic.state_dict(),
        'optim_alpha_state_dict': optim_alpha.state_dict(),
    }, path)
    logger.info(f"Checkpoint saved successfully to: {path}")

def _extract_min_ttc_and_speed_from_state(state: torch.Tensor, config: SystemConfig):
    state_3d = state.view(1, config.VEHICLES_COUNT, config.FEATURES_COUNT).clone()
    state_3d[:, :, 1] *= 100.0
    state_3d[:, :, 2] *= 10.0
    state_3d[:, :, 3] *= 30.0

    others = state_3d[0, 1:, :]
    presence = others[:, 0] > 0.5
    dx = others[:, 1]
    dy = others[:, 2]
    dvx = others[:, 3]

    front_mask = presence & (dx > 0) & (torch.abs(dy) < config.LANE_WIDTH_M * 0.55)
    min_ttc = 99.0
    if front_mask.any():
        front_dx = dx[front_mask]
        front_dvx = dvx[front_mask]
        closing = front_dvx < -1e-3
        if closing.any():
            ttc = front_dx[closing] / (-front_dvx[closing] + 1e-3)
            min_ttc = float(ttc.min().item())

    ego_speed = float(state_3d[0, 0, 3].item())
    return min_ttc, ego_speed


def compute_risk_aware_reward(
    base_reward: float,
    proposed_action: torch.Tensor,
    safe_action: torch.Tensor,
    risk_metrics: torch.Tensor,
    state: torch.Tensor,
    info: dict,
    decision_info: dict,
    config: SystemConfig,
):
    min_ttc, ego_speed = _extract_min_ttc_and_speed_from_state(state, config)

    # R_ms: / translated note
    r_ms = 0.0
    if bool(info.get("arrived", False)):
        r_ms = 10.0
    elif bool(info.get("crashed", False)):
        r_ms = -10.0

    # R_lc: （ LLM ）
    action_id = int((decision_info or {}).get("action_id", 1))
    steer = float(safe_action[0, 1].item())
    r_lc = 0.0
    if action_id in (0, 2):
        lane_change_ok = (action_id == 0 and steer > 0.05) or (action_id == 2 and steer < -0.05)
        r_lc = 2.0 if lane_change_ok else -0.2

    # R_e: translated note
    if config.REWARD_SPEED_MIN <= ego_speed <= config.REWARD_SPEED_MAX:
        r_e = 0.3 * (ego_speed - config.REWARD_SPEED_MIN) / max(config.REWARD_SPEED_MAX - config.REWARD_SPEED_MIN, 1e-6)
    elif ego_speed < config.REWARD_SPEED_MIN:
        # “ ” ， translated note
        r_e = -config.REWARD_LOW_SPEED_PENALTY * (config.REWARD_SPEED_MIN - ego_speed) / max(config.REWARD_SPEED_MIN, 1e-6)
    else:
        r_e = 0.0

    # R_s: TTC
    r_s = -1.0 / (min_ttc + 0.1) if min_ttc < config.REWARD_TTC_THRESHOLD else 0.0

    paper_reward = r_ms + r_lc + r_e + r_s
    action_diff = torch.norm(proposed_action - safe_action).item()
    reward = (1.0 - config.PAPER_REWARD_BLEND) * base_reward + config.PAPER_REWARD_BLEND * paper_reward
    reward -= config.REWARD_ACTION_DIFF_WEIGHT * action_diff
    reward -= 4.0 * risk_metrics[0, 0].item()

    reward_terms = {
        "r_ms": r_ms,
        "r_lc": r_lc,
        "r_e": r_e,
        "r_s": r_s,
        "paper_reward": paper_reward,
    }
    return reward, action_diff, reward_terms


def build_structured_reflection(
    decision_info: dict,
    risk_val: float,
    step_ttc: float,
    gate_intervened: int,
    review_intervened: int,
    action_diff: float,
):
    action = (decision_info or {}).get("action", "lane_keep")
    action_id = int((decision_info or {}).get("action_id", 1))
    speed = float((decision_info or {}).get("target_speed", 20.0))

    causes = []
    if step_ttc < 1.2:
        causes.append("front TTC too low")
    if risk_val > 0.75:
        causes.append("risk score too high")
    if gate_intervened:
        causes.append("uncertainty gate triggered")
    if review_intervened:
        causes.append("short-horizon safety review triggered")
    if not causes:
        causes.append("large deviation between action and safety projection")

    action_hint = {
        0: "Prioritize checking left-side available gap; if not satisfied, keep lane/decelerate",
        1: "When lane keeping, dynamically reduce speed based on front TTC",
        2: "Prioritize checking right-side available gap; if not satisfied, keep lane/decelerate",
        3: "Avoid further acceleration when front space is insufficient",
        4: "Deceleration can be kept, but avoid unnecessary long forced slowdown",
    }.get(action_id, "Use a more conservative and interpretable discrete action")

    return (
        f"action={action} (target_speed={speed:.1f}) has a safety conflict; "
        f"trigger reasons: {', '.join(causes)}; "
        f"metrics: risk={risk_val:.2f}, ttc={step_ttc:.2f}, diff={action_diff:.2f}; "
        f"fix suggestion: {action_hint}."
    )


def _symmetric_kl_diag_gaussian(mu_p, log_std_p, mu_q, log_std_q):
    var_p = torch.exp(2.0 * log_std_p)
    var_q = torch.exp(2.0 * log_std_q)

    kl_pq = (log_std_q - log_std_p) + (var_p + (mu_p - mu_q).pow(2)) / (2.0 * var_q + 1e-8) - 0.5
    kl_qp = (log_std_p - log_std_q) + (var_q + (mu_q - mu_p).pow(2)) / (2.0 * var_p + 1e-8) - 0.5
    return 0.5 * (kl_pq.sum(dim=-1, keepdim=True) + kl_qp.sum(dim=-1, keepdim=True))

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def train_hybrid_system(resume_path: str = None):
    """
    Unified training function, supporting training from scratch or resuming from a checkpoint.
    :param resume_path: Checkpoint file path; if provided, training resumes from this path.
    """
    config = SystemConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing hybrid dual-frequency system... (device: {device})")
    logger.info(
        "Stability profile: %s | REVIEW_MIN_TTC=%.2f | REVIEW_LANE_MIN_TTC=%.2f | REPLAY_HARD_RATIO=%.2f | TRAJ_BLEND=%.2f",
        config.STABILITY_PROFILE,
        config.REVIEW_MIN_TTC,
        config.REVIEW_LANE_MIN_TTC,
        config.REPLAY_HARD_RATIO,
        config.TRAJ_GUIDE_ONLINE_BLEND,
    )
    
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, config.EXP_NAME))
    
    env = gym.make(config.ENV_NAME)
    env.unwrapped.config.update({
        "observation": {
            "type": config.OBSERVATION_TYPE,
            "vehicles_count": config.VEHICLES_COUNT,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": config.OBS_NORMALIZE,
            "features_range": {
                "x": [-config.OBS_X_RANGE_M, config.OBS_X_RANGE_M],
                "y": [-config.OBS_Y_RANGE_M, config.OBS_Y_RANGE_M],
                "vx": [-config.OBS_V_RANGE_MPS, config.OBS_V_RANGE_MPS],
                "vy": [-config.OBS_V_RANGE_MPS, config.OBS_V_RANGE_MPS],
            },
        },
        "action": {"type": "ContinuousAction"}, "policy_frequency": config.FAST_FREQ_HZ,
    })
    env = DualFreqEnvWrapper(env, config)
    
    # memory_store, kv_buffer, text_queue = DiLuMemoryStore(capacity=100), AsyncKVBuffer(config), queue.Queue(maxsize=config.ASYNC_BUFFER_MAXSIZE)
    # SlowSystemThread(config, text_queue, kv_buffer, memory_store).start() 

    # ： translated note
    memory_store = DiLuMemoryStore(
        capacity=100,
        text_sim_weight=config.MEMORY_TEXT_SIM_WEIGHT,
        upsert_dist=config.MEMORY_UPSERT_DIST,
    )
    kv_buffer = AsyncKVBuffer(config)
    SlowSystemThread(config, env.text_description_queue, kv_buffer, memory_store).start()
    
    
    state_dim, action_dim = config.VEHICLES_COUNT * config.FEATURES_COUNT, 2  
    ac_net = RiskSensitiveGaussianAC(state_dim, action_dim, config).to(device)
    traj_predictor = QueryAnchorTrajectoryPredictor(config)
        
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=config.ACTOR_LR)
    
    optimizer_actor = optim.Adam(
        list(ac_net.actor_net.parameters()) + list(ac_net.mu_layer.parameters()) + list(ac_net.log_std_layer.parameters()), lr=config.ACTOR_LR)
    # ac_net.seq_pos_embed
    optimizer_critic = optim.Adam(
        list(ac_net.critic_net1.parameters()) + 
        list(ac_net.critic_net2.parameters()) + 
        list(ac_net.state_encoder.parameters()) + 
        list(ac_net.cross_attention.parameters()) + 
        [ac_net.seq_pos_embed],  # <== translated note
        lr=config.CRITIC_LR
    )
    
    start_episode = 1
    
    # 【 ： ， 】 translated note
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resume path detected, loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        ac_net.load_state_dict(checkpoint['ac_net_state_dict'])
        
        # Alpha translated note
        log_alpha.data = checkpoint['log_alpha'].to(device).data
        log_alpha.requires_grad = True
        
        # translated note
        optimizer_actor.load_state_dict(checkpoint['optim_actor_state_dict'])
        optimizer_critic.load_state_dict(checkpoint['optim_critic_state_dict'])
        if 'optim_alpha_state_dict' in checkpoint:
            alpha_optim.load_state_dict(checkpoint['optim_alpha_state_dict'])
            
        start_episode = checkpoint['episode'] + 1
        logger.info(f"State restored successfully! Continuing from episode {start_episode}.")
    elif resume_path:
        logger.warning(f"Specified resume path does not exist ({resume_path}); training will start from scratch.")

    # Target Network ( )
    # ac_net_target = copy.deepcopy(ac_net).to(device)
    # Target Network ( PyTorch ， deepcopy )
    ac_net_target = RiskSensitiveGaussianAC(state_dim, action_dim, config).to(device)
    ac_net_target.load_state_dict(ac_net.state_dict())
    for param in ac_net_target.parameters(): param.requires_grad = False
    
    # if device.type == "cuda": ac_net = torch.compile(ac_net)

    scaler = GradScaler('cuda') if device.type == "cuda" else None
    replay_buffer = SimpleReplayBuffer(
        config.REPLAY_BUFFER_SIZE,
        hard_ratio=config.REPLAY_HARD_RATIO,
        hard_capacity_ratio=config.REPLAY_HARD_CAPACITY_RATIO,
    )
    last_failed_scene = "" 
    global_step = 0
    
    for episode in range(start_episode, config.MAX_EPISODES + 1):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        episode_reward, episode_steps, episode_cbf_interventions = 0, 0, 0
        episode_gate_interventions = 0
        episode_review_interventions = 0
        episode_near_miss_steps = 0
        episode_uncertainty_sum = 0.0
        episode_ttc_min = 99.0
        episode_speed_sum = 0.0
        episode_action_smoothness_sum = 0.0
        episode_traj_entropy_sum = 0.0
        episode_reward_ms = 0.0
        episode_reward_lc = 0.0
        episode_reward_e = 0.0
        episode_reward_s = 0.0
        episode_guide_conf_sum = 0.0
        # === ： ===
        episode_guide_acc_sum = 0.0
        prev_final_action = None
        done = False

        guide_phase = max(
            config.TRAJ_GUIDE_MIN_PHASE,
            1.0 - float(episode - 1) / max(float(config.TRAJ_GUIDE_DECAY_EPISODES), 1.0),
        )
        effective_blend = max(config.TRAJ_GUIDE_BLEND_MIN, config.TRAJ_GUIDE_ONLINE_BLEND * guide_phase)
        effective_guide_loss_w = config.TRAJ_GUIDE_LOSS_WEIGHT * guide_phase
        
        while not done:
            global_step += 1
            kv_tensor, decision_info, current_scene_text = kv_buffer.get_latest()
            kv_tensor = kv_tensor.to(device)
            
            with torch.no_grad():
                with autocast('cuda', dtype=torch.bfloat16) if scaler else torch.autocast("cpu", enabled=False): 
                    latent = ac_net.get_shared_latent(state, kv_tensor)
                    policy_action, raw_action, _ = ac_net.forward_actor(latent)
                    risk_metrics = RiskAssessmentModule.evaluate(state, config)
                    uncertainty_score = ac_net.estimate_uncertainty(latent)
                    traj_pred = traj_predictor.predict(state, decision_info)
                    guide_action = traj_pred["guide_action"]
                    traj_entropy = traj_pred["traj_entropy"]
                    guide_conf = traj_pred["probs"].max(dim=-1).values.unsqueeze(-1)

                    # === ： batch acc ( [0, 0]) ===
                    step_guide_acc = float(guide_action[0, 0].item())
                    episode_guide_acc_sum += step_guide_acc
                    episode_guide_conf_sum += float(guide_conf.mean().item())
                    # =========================================================

                    proposed_action = torch.clamp(
                        (1.0 - effective_blend) * policy_action +
                        effective_blend * guide_action,
                        -1.0,
                        1.0,
                    )

                    gated_action, gate_blend, gate_trigger_mask = UncertaintyAwareGate.apply(
                        proposed_action, risk_metrics, uncertainty_score, decision_info, config
                    )
                    cbf_action = SafetyShieldCBF.project_action(gated_action, risk_metrics)
                    safe_action, review_trigger_mask, min_ttc = ShortHorizonSafetyReviewer.review(state, cbf_action, config, decision_info)

            uncertainty_val = float(uncertainty_score.mean().item())
            episode_uncertainty_sum += uncertainty_val
            episode_traj_entropy_sum += float(traj_entropy.mean().item())
            gate_intervened = int(gate_trigger_mask.any().item())
            review_intervened = int(review_trigger_mask.any().item())
            episode_gate_interventions += gate_intervened
            episode_review_interventions += review_intervened
            step_ttc = float(min_ttc.min().item())
            episode_ttc_min = min(episode_ttc_min, step_ttc)
            if step_ttc < config.NEAR_MISS_TTC:
                episode_near_miss_steps += 1

            ego_speed = float((state.view(1, config.VEHICLES_COUNT, config.FEATURES_COUNT)[0, 0, 3] * 30.0).item())
            episode_speed_sum += max(ego_speed, 0.0)
            if prev_final_action is not None:
                episode_action_smoothness_sum += torch.norm(safe_action - prev_final_action).item()
            prev_final_action = safe_action.clone()
            
            next_state, base_reward, terminated, truncated, info = env.step(safe_action.cpu().numpy().flatten())
            done = terminated or truncated
            episode_steps += 1
            
            reward, action_diff, reward_terms = compute_risk_aware_reward(
                base_reward,
                proposed_action,
                safe_action,
                risk_metrics,
                state,
                info,
                decision_info,
                config,
            )
            episode_reward += reward
            episode_cbf_interventions += action_diff
            episode_reward_ms += reward_terms["r_ms"]
            episode_reward_lc += reward_terms["r_lc"]
            episode_reward_e += reward_terms["r_e"]
            episode_reward_s += reward_terms["r_s"]
            
            risk_val = float(risk_metrics[0, 0].item())
            failure_trigger = (
                action_diff > config.CBF_FAILURE_THRESHOLD
                or step_ttc < config.NEAR_MISS_TTC
                or (risk_val > 0.85 and review_intervened > 0)
            )
            if failure_trigger and current_scene_text != last_failed_scene and decision_info:
                reflection_msg = build_structured_reflection(
                    decision_info,
                    risk_val=risk_val,
                    step_ttc=step_ttc,
                    gate_intervened=gate_intervened,
                    review_intervened=review_intervened,
                    action_diff=action_diff,
                )
                memory_store.add_failure_memory(
                    current_scene_text,
                    decision_info,
                    reflection_msg,
                    metrics={
                        "risk": round(risk_val, 4),
                        "ttc": round(step_ttc, 4),
                        "gate": gate_intervened,
                        "review": review_intervened,
                        "action_diff": round(action_diff, 4),
                    },
                )
                last_failed_scene = current_scene_text

            if (
                decision_info
                and episode_steps % config.MEMORY_SUCCESS_ADD_INTERVAL == 0
                and gate_intervened == 0
                and review_intervened == 0
                and step_ttc >= config.MEMORY_MIN_SAFE_TTC
                and risk_val < 0.45
                and ego_speed >= config.MEMORY_MIN_SUCCESS_SPEED
            ):
                success_summary = (
                    f"Low-risk stable pass; keep safe headway and avoid unnecessary intervention; "
                    f"risk={risk_val:.2f}, ttc={step_ttc:.2f}."
                )
                memory_store.add_success_memory(
                    current_scene_text,
                    decision_info,
                    success_summary,
                    metrics={"score": 1.0 + 0.2 * float(ego_speed > 20.0), "risk": round(risk_val, 4), "ttc": round(step_ttc, 4)},
                )
            
            next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            next_kv_tensor, _, _ = kv_buffer.get_latest()
            next_kv_tensor = next_kv_tensor.to(device)
            # ReplayBuffer， Batch ， kv_tensor (SeqLen=4, Dim)
            hard_transition = (
                step_ttc < config.NEAR_MISS_TTC
                or gate_intervened > 0
                or review_intervened > 0
                or action_diff > config.CBF_FAILURE_THRESHOLD
            )
            replay_buffer.push(
                state.squeeze(0),
                kv_tensor.squeeze(0),
                raw_action.squeeze(0),
                safe_action.squeeze(0),
                guide_action.squeeze(0),
                guide_conf.squeeze(0),
                reward,
                next_state_t.squeeze(0),
                next_kv_tensor.squeeze(0), # <== ！ translated note
                done,
                risk_metrics.squeeze(0),
                hard=hard_transition,
            )
            state = next_state_t
            
            if len(replay_buffer) > config.BATCH_SIZE:
                # b_states, b_kvs, b_raw_actions, b_safe_actions, b_guide_actions, b_guide_confs, b_rewards, b_next_states, b_dones, b_risks = replay_buffer.sample(config.BATCH_SIZE)
                b_states, b_kvs, b_raw_actions, b_safe_actions, b_guide_actions, b_guide_confs, b_rewards, b_next_states, b_next_kvs, b_dones, b_risks = replay_buffer.sample(config.BATCH_SIZE)
                # b_kvs (Batch, SeqLen, Dim)， to(device)， unsqueeze
                # ( ): translated note
                b_states, b_kvs, b_raw_actions, b_safe_actions, b_guide_actions, b_guide_confs, b_rewards, b_next_states, b_dones, b_risks = \
                    b_states.to(device), b_kvs.to(device), b_raw_actions.to(device), b_safe_actions.to(device), b_guide_actions.to(device), \
                    b_guide_confs.to(device), b_rewards.unsqueeze(1).to(device), b_next_states.to(device), b_dones.unsqueeze(1).to(device), b_risks.to(device)
                # ... ( b_XXX .to(device) ) ...
                b_next_kvs = b_next_kvs.to(device)
                alpha = log_alpha.exp().item() 
                
                # translated note
                with autocast('cuda', dtype=torch.bfloat16) if scaler else torch.autocast("cpu", enabled=False):
                    with torch.no_grad():
                        # 💥 [ ] Critic S_{t+1} ， Next KV
                        next_latent = ac_net.get_shared_latent(b_next_states, b_next_kvs)
                        next_action, _, next_log_prob = ac_net.forward_actor(next_latent)
                        b_next_risks = RiskAssessmentModule.evaluate(b_next_states, config)
                        t_q1, t_q2 = ac_net_target.forward_critic(next_latent, next_action, b_next_risks)
                        td_target = b_rewards + config.GAMMA * (1 - b_dones) * (torch.min(t_q1, t_q2) - alpha * next_log_prob)
                        
                    curr_latent = ac_net.get_shared_latent(b_states, b_kvs)
                    # Critic （safe_action），
                    q1_pred, q2_pred = ac_net.forward_critic(curr_latent, b_safe_actions, b_risks)
                    critic_loss = 0.5 * F.mse_loss(q1_pred, td_target) + 0.5 * F.mse_loss(q2_pred, td_target)
                
                optimizer_critic.zero_grad()
                if scaler:
                    scaler.scale(critic_loss).backward()
                    scaler.unscale_(optimizer_critic)
                    critic_params = (
                        list(ac_net.critic_net1.parameters())
                        + list(ac_net.critic_net2.parameters())
                        + list(ac_net.state_encoder.parameters())
                        + list(ac_net.cross_attention.parameters())
                    )
                    torch.nn.utils.clip_grad_norm_(critic_params, config.GRAD_CLIP_NORM)
                    scaler.step(optimizer_critic)
                else:
                    critic_loss.backward()
                    critic_params = (
                        list(ac_net.critic_net1.parameters())
                        + list(ac_net.critic_net2.parameters())
                        + list(ac_net.state_encoder.parameters())
                        + list(ac_net.cross_attention.parameters())
                    )
                    torch.nn.utils.clip_grad_norm_(critic_params, config.GRAD_CLIP_NORM)
                    optimizer_critic.step()
                
                with autocast('cuda', dtype=torch.bfloat16) if scaler else torch.autocast("cpu", enabled=False):
                    detached_latent = curr_latent.detach()
                    new_action, new_raw, log_prob = ac_net.forward_actor(detached_latent)
                    mu_new, log_std_new = ac_net.actor_distribution_params(detached_latent)
                    q1_new, q2_new = ac_net.forward_critic(detached_latent, new_action, b_risks)
                    
                    policy_loss = (alpha * log_prob - torch.min(q1_new, q2_new)).mean()
                    cbf_imitation_loss = F.mse_loss(new_action, b_safe_actions)
                    guide_conf_weight = torch.clamp(
                        b_guide_confs.view(-1, 1),
                        min=config.TRAJ_GUIDE_CONF_FLOOR,
                        max=1.0,
                    )
                    traj_consistency_loss = ((new_action - b_guide_actions).pow(2).sum(dim=-1, keepdim=True) * guide_conf_weight).mean()

                    # 💥 [ ] clamp， atanh
                    clamped_guide = torch.clamp(b_guide_actions, -0.999, 0.999)
                    expert_mu = torch.clamp(torch.atanh(clamped_guide), -5.0, 5.0)

                    # expert_mu = torch.atanh(torch.clamp(b_guide_actions, -0.999, 0.999))
                    expert_log_std = torch.full_like(log_std_new, np.log(config.TRAJ_GUIDE_EXPERT_STD))
                    
                    traj_js_loss = (_symmetric_kl_diag_gaussian(mu_new, log_std_new, expert_mu, expert_log_std) * guide_conf_weight).mean()

                    traj_guide_loss = traj_consistency_loss + config.TRAJ_GUIDE_JS_WEIGHT * traj_js_loss
                    actor_loss = policy_loss + 10.0 * cbf_imitation_loss + effective_guide_loss_w * traj_guide_loss
                    
                optimizer_actor.zero_grad()
                if scaler:
                    scaler.scale(actor_loss).backward()
                    scaler.unscale_(optimizer_actor)
                    actor_params = (
                        list(ac_net.actor_net.parameters())
                        + list(ac_net.mu_layer.parameters())
                        + list(ac_net.log_std_layer.parameters())
                    )
                    torch.nn.utils.clip_grad_norm_(actor_params, config.GRAD_CLIP_NORM)
                    scaler.step(optimizer_actor)
                else:
                    actor_loss.backward()
                    actor_params = (
                        list(ac_net.actor_net.parameters())
                        + list(ac_net.mu_layer.parameters())
                        + list(ac_net.log_std_layer.parameters())
                    )
                    torch.nn.utils.clip_grad_norm_(actor_params, config.GRAD_CLIP_NORM)
                    optimizer_actor.step()
                
                alpha_loss = -(log_alpha * (log_prob + (-action_dim)).detach()).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()
                
                if scaler: scaler.update()
                soft_update(ac_net_target, ac_net, config.TAU)
                
                if global_step % 100 == 0:
                    writer.add_scalar('Loss/Actor', actor_loss.item(), global_step)
                    writer.add_scalar('Loss/Critic', critic_loss.item(), global_step)
                    writer.add_scalar('Loss/Alpha', alpha_loss.item(), global_step)
                    writer.add_scalar('Loss/TrajectoryConsistency', traj_consistency_loss.item(), global_step)
                    writer.add_scalar('Loss/TrajectoryJSLite', traj_js_loss.item(), global_step)
                    writer.add_scalar('Metrics/Alpha', alpha, global_step)
                    writer.add_scalar('Safety/Uncertainty', uncertainty_val, global_step)
                    writer.add_scalar('Safety/TrajEntropy', float(traj_entropy.mean().item()), global_step)
                    writer.add_scalar('Safety/GuideBlendEffective', effective_blend, global_step)
                    writer.add_scalar('Safety/GuideLossWeightEffective', effective_guide_loss_w, global_step)
                    writer.add_scalar('Safety/GateBlend', float(gate_blend.mean().item()), global_step)
                    writer.add_scalar('Safety/MinTTC', step_ttc, global_step)
                    writer.add_scalar('Safety/GateIntervened', gate_intervened, global_step)
                    writer.add_scalar('Safety/ReviewerIntervened', review_intervened, global_step)

        is_crash = info.get('crashed', False)
        near_miss_rate = (episode_near_miss_steps / max(episode_steps, 1))
        mean_uncertainty = (episode_uncertainty_sum / max(episode_steps, 1))
        mean_traj_entropy = (episode_traj_entropy_sum / max(episode_steps, 1))
        mean_speed = (episode_speed_sum / max(episode_steps, 1))
        # === ： ===
        mean_guide_acc = episode_guide_acc_sum / max(episode_steps, 1)
        mean_guide_conf = episode_guide_conf_sum / max(episode_steps, 1)
        mean_action_smoothness = (episode_action_smoothness_sum / max(episode_steps - 1, 1))
        mem_stats = memory_store.get_memory_stats()
        logger.info(
            f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {episode_steps} | "
            f"Crash: {is_crash} | NearMissRate: {near_miss_rate:.3f} | MinTTC: {episode_ttc_min:.2f} | "
            f"CBF Acc. Diff: {episode_cbf_interventions:.2f} | TrajEntropy: {mean_traj_entropy:.3f} | "
            f"GuideConf: {mean_guide_conf:.3f} | Gate/Review: {episode_gate_interventions}/{episode_review_interventions} | "
            f"Mem(F/S/T): {mem_stats['failure']}/{mem_stats['success']}/{mem_stats['total']}"
        )
        
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Steps', episode_steps, episode)
        writer.add_scalar('Episode/CBF_Interventions', episode_cbf_interventions, episode)
        writer.add_scalar('Episode/Crash_Rate', 1.0 if is_crash else 0.0, episode)
        writer.add_scalar('Episode/Near_Miss_Rate', near_miss_rate, episode)
        writer.add_scalar('Episode/Min_TTC', episode_ttc_min, episode)
        writer.add_scalar('Episode/Uncertainty_Mean', mean_uncertainty, episode)
        writer.add_scalar('Episode/Trajectory_Entropy_Mean', mean_traj_entropy, episode)
        writer.add_scalar('Episode/Gate_Interventions', episode_gate_interventions, episode)
        writer.add_scalar('Episode/Reviewer_Interventions', episode_review_interventions, episode)
        writer.add_scalar('Episode/Mean_Speed', mean_speed, episode)
        # === ： TensorBoard ===
        writer.add_scalar('Episode/Mean_Guide_Accel', mean_guide_acc, episode)
        writer.add_scalar('Episode/Mean_Guide_Confidence', mean_guide_conf, episode)
        writer.add_scalar('Episode/Memory_Total', mem_stats['total'], episode)
        writer.add_scalar('Episode/Memory_Failure', mem_stats['failure'], episode)
        writer.add_scalar('Episode/Memory_Success', mem_stats['success'], episode)
        writer.add_scalar('Episode/Reward_Rms', episode_reward_ms / max(episode_steps, 1), episode)
        writer.add_scalar('Episode/Reward_Rlc', episode_reward_lc / max(episode_steps, 1), episode)
        writer.add_scalar('Episode/Reward_Re', episode_reward_e / max(episode_steps, 1), episode)
        writer.add_scalar('Episode/Reward_Rs', episode_reward_s / max(episode_steps, 1), episode)
        writer.add_scalar('Episode/Action_Smoothness', mean_action_smoothness, episode)
        
        if episode % config.SAVE_INTERVAL == 0:
            save_checkpoint(episode, ac_net, log_alpha, optimizer_actor, optimizer_critic, alpha_optim, config)

    writer.close()
    logger.info("Training finished!")


# =====================================================================
# Part 5: / (Evaluation & Inference)
# =====================================================================
# =====================================================================
# Part 5: / (Evaluation & Inference) -
# =====================================================================
def evaluate_model(checkpoint_path: str, num_episodes: int = 5, save_video: bool = True):
    """Load a trained model for evaluation and optionally record the run as MP4 videos."""
    from gymnasium.wrappers import RecordVideo  # translated note

    config = SystemConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model for evaluation: {checkpoint_path} (device: {device})")
    
    # 1. 【 】 rgb_array，
    env = gym.make(config.ENV_NAME, render_mode="rgb_array")
    
    # 2. translated note
    env.unwrapped.config.update({
        "observation": {
            "type": config.OBSERVATION_TYPE,
            "vehicles_count": config.VEHICLES_COUNT,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": config.OBS_NORMALIZE,
            "features_range": {
                "x": [-config.OBS_X_RANGE_M, config.OBS_X_RANGE_M],
                "y": [-config.OBS_Y_RANGE_M, config.OBS_Y_RANGE_M],
                "vx": [-config.OBS_V_RANGE_MPS, config.OBS_V_RANGE_MPS],
                "vy": [-config.OBS_V_RANGE_MPS, config.OBS_V_RANGE_MPS],
            },
        },
        "action": {"type": "ContinuousAction"},
        "policy_frequency": config.FAST_FREQ_HZ,
        "duration": 40,  # translated note
    })
    
    # 3. （ ， ）
    # ./runs/videos/ /
    video_dir = os.path.join(config.LOG_DIR, "videos", config.EXP_NAME)
    os.makedirs(video_dir, exist_ok=True)
    if save_video:
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: True,  # Episode translated note
            name_prefix="HybridDrive_Demo"           # translated note
        )
        try:
            # run_dilu.py ， gym
            env.unwrapped.set_record_video_wrapper(env)
        except Exception:
            pass
    
    # 4. translated note
    env = DualFreqEnvWrapper(env, config)
    
    state_dim, action_dim = config.VEHICLES_COUNT * config.FEATURES_COUNT, 2
    ac_net = RiskSensitiveGaussianAC(state_dim, action_dim, config).to(device)
    traj_predictor = QueryAnchorTrajectoryPredictor(config)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ac_net.load_state_dict(checkpoint['ac_net_state_dict'], strict=False)
    
    # eval() ，
    ac_net.eval()
    
    memory_store = DiLuMemoryStore(
        capacity=100,
        text_sim_weight=config.MEMORY_TEXT_SIM_WEIGHT,
        upsert_dist=config.MEMORY_UPSERT_DIST,
    )
    kv_buffer = AsyncKVBuffer(config)
    SlowSystemThread(config, env.text_description_queue, kv_buffer, memory_store).start()
    
    # 💥 ================= ： (Cold-Start Warm-up) ================= 💥
    # LLM RL 0
    dummy_encoder = LatentSkillEncoder(config).to(device)
    default_safe_decision = {
        "action": "lane_keep",
        "action_id": 1,
        "target_speed": 20.0,
        "speed_level": 2,
        "trajectory_id": 0,
        "reference_trajectory": [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        "decoder_trace": {"template": "keep"}
    }
    with torch.no_grad():
        initial_kv = dummy_encoder(default_safe_decision).cpu()
    kv_buffer.update(initial_kv, default_safe_decision, "System initial default safety prior (Waiting for LLM...)")
    # ====================================================================================

    if save_video:
        logger.info(f"Model loaded. Video will be saved to: {video_dir}")
        logger.info("Background recording in progress, please wait...")
    else:
        logger.info("Model loaded. Current mode is evaluation (video saving disabled).")
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        episode_reward = 0
        episode_steps = 0
        episode_gate_interventions = 0
        episode_review_interventions = 0
        episode_ttc_min = 99.0
        episode_near_miss_steps = 0
        done = False
        
        while not done:
            kv_tensor, decision_info, _ = kv_buffer.get_latest()
            kv_tensor = kv_tensor.to(device)
            
            with torch.no_grad():
                latent = ac_net.get_shared_latent(state, kv_tensor)
                policy_action, _, _ = ac_net.forward_actor(latent, deterministic=True)
                risk_metrics = RiskAssessmentModule.evaluate(state, config)
                uncertainty_score = ac_net.estimate_uncertainty(latent)
                traj_pred = traj_predictor.predict(state, decision_info)
                guide_action = traj_pred["guide_action"]
                proposed_action = torch.clamp(
                    (1.0 - config.TRAJ_GUIDE_ONLINE_BLEND) * policy_action +
                    config.TRAJ_GUIDE_ONLINE_BLEND * guide_action,
                    -1.0,
                    1.0,
                )
                gated_action, _, gate_trigger_mask = UncertaintyAwareGate.apply(
                    proposed_action, risk_metrics, uncertainty_score, decision_info, config
                )
                cbf_action = SafetyShieldCBF.project_action(gated_action, risk_metrics)
                safe_action, review_trigger_mask, min_ttc = ShortHorizonSafetyReviewer.review(state, cbf_action, config, decision_info)
                episode_gate_interventions += int(gate_trigger_mask.any().item())
                episode_review_interventions += int(review_trigger_mask.any().item())
                step_ttc = float(min_ttc.min().item())
                episode_ttc_min = min(episode_ttc_min, step_ttc)
                if step_ttc < config.NEAR_MISS_TTC:
                    episode_near_miss_steps += 1
                
            next_state, reward, terminated, truncated, info = env.step(safe_action.cpu().numpy().flatten())
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

            if save_video:
                try:
                    env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
                except Exception:
                    pass
            
        near_miss_rate = episode_near_miss_steps / max(episode_steps, 1)
        logger.info(
            f"Recorded Episode {episode} finished | total reward: {episode_reward:.2f} | crashed: {info.get('crashed', False)} | "
            f"NearMissRate: {near_miss_rate:.3f} | MinTTC: {episode_ttc_min:.2f} | Gate/Review: {episode_gate_interventions}/{episode_review_interventions}"
        )
        
    env.close()
    if save_video:
        logger.info(f"All video recordings completed. Please check {video_dir} for .mp4 files.")


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # ( ) translated note
    # -----------------------------------------------------------------
    
    # A： translated note
    train_hybrid_system()

    # B： Checkpoint
    # train_hybrid_system(resume_path="./checkpoints/hybrid_sac_ep50.pth")

    # C： translated note
    # evaluate_model(checkpoint_path="./checkpoints/hybrid_sac_ep500.pth", num_episodes=10)




    # : python hybrid_driving_system.py

    # : ./runs 。 ， ：tensorboard --logdir=./runs
    # http://localhost:6006， ：
    # Actor Critic Loss (Loss/Actor, Loss/Critic)
    # Episode (Episode/Reward)
    # (Episode/Crash_Rate)
    # (Episode/CBF_Interventions)， ， 。

    # : （ ） ./checkpoints ， 。



    
    #     运行方式：
#     python single_file_llm_guided_rl.py --mode baseline_sac --episodes 3
# python single_file_llm_guided_rl.py --mode shaping_sac --episodes 3
# python single_file_llm_guided_rl.py --mode rule_hier --episodes 3
