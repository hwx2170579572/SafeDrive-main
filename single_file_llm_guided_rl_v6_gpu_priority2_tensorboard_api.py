
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Force this script to see/use GPU 0 only.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import gymnasium as gym
import highway_env  # noqa: F401  # registers envs


# ============================================================
# user-editable LLM forwarding API config
# Edit only this block when switching forwarding service / key / model.
# ============================================================

USER_EDITABLE_LLM = {
    "backend": "real",  # "real" uses forwarding API, "mock" uses local heuristic mock planner
    "model_name": "gpt-4.1-nano",
    "api_base": "https://api.chatanywhere.tech",
    # Replace the placeholder below with your real ChatAnywhere key.
    "api_key": "sk-4jWXenyMvfiLIMLNiTy1lLoVrR4FC1Y2KoUModv8NTq4Plg5",
}




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

DEFAULT_FORMAL_MAIN_MODES = [
    'baseline_sac',
    'shaping_sac',
    'constrained_rule_hier',
    'constrained_llm_hier',
    'constrained_real_llm_hier',
]

FORMAL_REPORT_METRICS = [
    'episode_return',
    'success',
    'collision',
    'mean_speed',
    'unsafe_headway_rate',
    'mean_cost_total',
]

PROTOCOL_VERSION = 'v6'
PROMPT_TEMPLATE_VERSION = 'prompt_v2'
NORMALIZATION_VERSION = 'llm_norm_v2'


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
    planner_interval: int = 10
    waypoint_horizon: int = 5
    waypoint_gap: float = 8.0
    replan_ttc_threshold: float = 2.0
    replan_front_distance: float = 12.0

    target_speed_free: float = 26.0
    target_speed_cautious: float = 20.0
    target_speed_brake: float = 15.0

    conservative_headway: float = 18.0
    normal_headway: float = 12.0

    speed_max: float = 29.5
    acc_max: float = 2.5
    steer_max: float = 0.20
    comfort_acc_delta: float = 1.8

    lane_change_cooldown_steps: int = 14
    lane_change_min_improvement: float = 8.0

    print_prompt_on_replan: bool = False
    print_real_llm_error: bool = True


@dataclass
class RewardConfig:
    use_env_reward: bool = True
    env_reward_scale: float = 0.15

    w_progress = 1.15
    w_waypoint_pos = 0.35
    w_waypoint_speed = 0.22

    w_collision = 6.0
    w_headway = 2.4
    w_overspeed = 0.9

    w_action = 0.02
    w_action_smooth = 0.15
    w_lane_change = 0.40


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr = 2e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    lambda_lr = 5e-4

    hidden_dim: int = 256
    batch_size: int = 128
    buffer_size: int = 100000
    start_steps: int = 2000
    update_after: int = 1000
    update_every: int = 1
    target_entropy: float = -2.0

    cost_limit_collision = 0.02
    cost_limit_headway = 0.08
    cost_limit_overspeed = 0.08
    cost_limit_comfort = 0.35

    # constrained SAC: learn state-action cost critics so constraints can backprop to actor
    initial_lambda: float = 0.05
    lambda_max: float = 50.0

    # second-priority stabilization for collision/headway dual updates
    lambda_ema_beta: float = 0.97
    min_lambda_collision: float = 0.010
    min_lambda_headway: float = 0.010


@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 10
    max_steps_per_episode: int = 200
    eval_every: int = 1
    device: str = "cuda:0"
    mode: str = "rule_hier"
    render: bool = False
    print_every_step: bool = False
    use_shield_during_train: bool = False


@dataclass
class DiagnosticsConfig:
    export_train_log: bool = False
    train_log_path: str = ""
    train_plot_path: str = ""
    save_train_json: bool = False
    train_json_path: str = ""
    include_plot_grid: bool = True
    tensorboard_log_dir: str = ""
    tensorboard_flush_secs: int = 10


@dataclass
class EvalConfig:
    enabled: bool = True
    episodes: int = 3
    seed_offset: int = 1000
    print_step: bool = False
    report_raw_and_shielded: bool = True
    primary_report: str = "raw"


@dataclass
class SafetyConfig:
    verifier_enabled: bool = True
    shield_enabled_eval: bool = True
    shield_enabled_train: bool = False

    ttc_safe = 2.5
    min_front_distance_hard = 8.0
    lane_offset_hard = 0.9

    emergency_brake_acc = -0.9
    overspeed_brake_acc = -0.55
    shield_steer_scale = 0.75

    # action output is normalized to [-1, 1]; these scales convert physical constraints
    # to the normalized action domain used by actor / shield / verifier
    action_acc_scale: float = 5.0
    action_steer_scale: float = 0.7853981633974483  # ~pi/4

    cost_collision_weight = 1.0
    cost_headway_weight = 1.3
    cost_overspeed_weight = 0.6
    cost_comfort_weight = 0.10

    # second-priority dense collision-risk shaping for constrained learning
    collision_risk_ttc_threshold: float = 4.0
    collision_risk_front_distance: float = 10.0



@dataclass
class LLMConfig:
    backend: str = str(USER_EDITABLE_LLM["backend"])  # mock | real
    model_name: str = str(USER_EDITABLE_LLM["model_name"])
    api_key: str = str(USER_EDITABLE_LLM["api_key"])
    # ChatAnywhere uses an OpenAI-compatible chat completions endpoint.
    # You may pass either:
    #   1) https://api.chatanywhere.tech
    #   2) https://api.chatanywhere.tech/v1
    #   3) https://api.chatanywhere.tech/v1/chat/completions
    # The runtime will normalize all of them to the final chat-completions URL.
    api_base: str = str(USER_EDITABLE_LLM["api_base"])
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_sec: int = 20
    system_prompt: str = (
        "You are a highway autonomous driving planner. "
        "Return JSON only, no markdown, no extra commentary."
    )
    fallback_to_mock_on_error: bool = True
    fallback_to_mock_on_parse_error: bool = True
    retry_times: int = 3
    retry_backoff_sec: List[float] = field(default_factory=lambda: [1.0, 3.0, 5.0])
    log_prompts_and_responses: bool = True
    deterministic_in_formal: bool = True
    call_log_path: str = ""


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
    ablation_modes: List[str] = field(default_factory=list)
    disable_step_print_during_compare: bool = True
    use_shield_in_eval: bool = False
    primary_report: str = "raw"
    emit_dual_tables: bool = True
    num_seeds: int = 1
    seed_stride: int = 100
    explicit_seeds: List[int] = field(default_factory=list)
    save_json_path: str = ""
    save_csv_path: str = ""
    save_latex_path: str = ""
    formal_modes: List[str] = field(default_factory=lambda: list(DEFAULT_FORMAL_MAIN_MODES))
    require_all_modes_success: bool = True
    min_successful_runs_per_mode: int = 1


@dataclass
class AblationConfig:
    disable_waypoint_features: bool = False
    disable_constraint_features: bool = False
    disable_waypoint_reward: bool = False
    disable_constraint_costs: bool = False
    disable_lane_stabilization: bool = False


@dataclass
class WorkflowConfig:
    stage: str = "dev"  # dev | freeze | formal
    freeze_save_path: str = ""
    freeze_load_path: str = ""
    formal_strict: bool = True
    allow_real_llm_smoke: bool = False
    smoke_test: bool = False
    paper_eligible: bool = False
    protocol_hash: str = ""
    frozen_protocol_path: str = ""
    code_hash: str = ""
    dependency_versions: Dict[str, str] = field(default_factory=dict)


@dataclass
class FreezeProtocolConfig:
    protocol_version: str = PROTOCOL_VERSION
    protocol_name: str = "paper_main_protocol"
    main_table_modes: List[str] = field(default_factory=lambda: list(DEFAULT_FORMAL_MAIN_MODES))
    ablation_modes: List[str] = field(default_factory=list)
    report_metrics: List[str] = field(default_factory=lambda: list(FORMAL_REPORT_METRICS))
    primary_report: str = "raw"
    report_raw_and_shielded: bool = True
    dev_seeds: List[int] = field(default_factory=lambda: [42, 52, 62])
    formal_seeds: List[int] = field(default_factory=lambda: [142, 242, 342])
    llm_retry_times: int = 3
    llm_retry_backoff_sec: List[float] = field(default_factory=lambda: [1.0, 3.0, 5.0])
    prompt_template_hash: str = ""
    normalization_version: str = NORMALIZATION_VERSION
    min_successful_runs_per_mode: int = 1


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    freeze_protocol: FreezeProtocolConfig = field(default_factory=FreezeProtocolConfig)


def get_config() -> Config:
    cfg = Config()
    cfg.eval.primary_report = str(cfg.eval.primary_report).lower()
    cfg.compare.primary_report = str(cfg.compare.primary_report).lower()
    cfg.workflow.stage = str(cfg.workflow.stage).lower()
    return cfg


def resolve_runtime_device(requested_device: str) -> str:
    requested = str(requested_device or "cuda:0").strip()
    requested_lower = requested.lower()

    # This script is intentionally pinned to GPU 0 only.
    allowed_aliases = {"auto", "gpu", "cuda", "cuda:0"}
    if requested_lower not in allowed_aliases:
        raise RuntimeError(
            f"Unsupported device '{requested_device}'. "
            "This script is pinned to GPU0 only; use --device cuda:0."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU0 is required but CUDA is unavailable. "
            "Please install CUDA-enabled PyTorch and ensure NVIDIA driver is ready."
        )

    return "cuda:0"


def describe_runtime_device(device_str: str) -> str:
    device = torch.device(device_str)
    if device.type == "cuda" and torch.cuda.is_available():
        if device.index is None:
            index = torch.cuda.current_device()
        else:
            index = int(device.index)
        name = torch.cuda.get_device_name(index)
        return f"{device_str} ({name})"
    return str(device)


def configure_torch_runtime(device_str: str) -> None:
    device = torch.device(device_str)
    if device.type != "cuda":
        raise RuntimeError("This script is pinned to GPU0 and does not support non-CUDA runtime.")
    if device.index not in (None, 0):
        raise RuntimeError(f"Only cuda:0 is allowed, got '{device_str}'.")
    torch.cuda.set_device(0)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ============================================================
# workflow + protocol helpers
# ============================================================

WORKFLOW_STAGES = {"dev", "freeze", "formal"}


def _canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_seed_list(seeds: List[int], fallback: List[int]) -> List[int]:
    out: List[int] = []
    for s in seeds:
        try:
            sv = int(s)
        except Exception:
            continue
        if sv not in out:
            out.append(sv)
    if len(out) == 0:
        out = [int(s) for s in fallback]
    return out


def dataclass_to_plain_dict(obj) -> dict:
    return asdict(obj)


def stable_json_dumps(obj: dict) -> str:
    return _canonical_json_dumps(obj)


def compute_sha256_text(text: str) -> str:
    return _sha256_text(text)


def compute_file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return _sha256_bytes(f.read())


def compute_code_hash() -> str:
    script_path = os.path.abspath(__file__) if "__file__" in globals() else ""
    if script_path and os.path.isfile(script_path):
        return compute_file_sha256(script_path)
    return ""


def collect_dependency_versions() -> Dict[str, str]:
    return {
        "python": str(sys.version.split()[0]),
        "platform": str(platform.platform()),
        "numpy": str(getattr(np, "__version__", "unknown")),
        "torch": str(getattr(torch, "__version__", "unknown")),
        "gymnasium": str(getattr(gym, "__version__", "unknown")),
        "highway_env": str(getattr(highway_env, "__version__", "unknown")),
        "cuda_available": str(bool(torch.cuda.is_available())),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "cudnn_version": str(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
    }


def get_prompt_template_hash(cfg) -> str:
    prompt_builder = PromptBuilder(cfg)
    return _sha256_text(prompt_builder.template_signature())


def canonical_protocol_dict(cfg: Config) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": dataclass_to_plain_dict(cfg.env),
        "planner": dataclass_to_plain_dict(cfg.planner),
        "reward": dataclass_to_plain_dict(cfg.reward),
        "sac": dataclass_to_plain_dict(cfg.sac),
        "safety": dataclass_to_plain_dict(cfg.safety),
        "llm": {
            "backend": str(cfg.llm.backend),
            "model_name": str(cfg.llm.model_name),
            "api_base": str(cfg.llm.api_base),
            "temperature": float(cfg.llm.temperature),
            "max_tokens": int(cfg.llm.max_tokens),
            "timeout_sec": int(cfg.llm.timeout_sec),
            "system_prompt": str(cfg.llm.system_prompt),
            "retry_times": int(cfg.llm.retry_times),
            "retry_backoff_sec": [float(x) for x in cfg.llm.retry_backoff_sec],
            "deterministic_in_formal": bool(cfg.llm.deterministic_in_formal),
            "normalization_version": str(cfg.freeze_protocol.normalization_version),
            "prompt_template_hash": str(cfg.freeze_protocol.prompt_template_hash),
        },
        "eval": {
            "enabled": bool(cfg.eval.enabled),
            "episodes": int(cfg.eval.episodes),
            "seed_offset": int(cfg.eval.seed_offset),
            "report_raw_and_shielded": bool(cfg.eval.report_raw_and_shielded),
            "primary_report": str(cfg.eval.primary_report).lower(),
        },
        "compare": {
            "primary_report": str(cfg.compare.primary_report).lower(),
            "formal_modes": [str(m) for m in cfg.compare.formal_modes],
            "require_all_modes_success": bool(cfg.compare.require_all_modes_success),
            "min_successful_runs_per_mode": int(cfg.compare.min_successful_runs_per_mode),
        },
        "workflow": {
            "stage": str(cfg.workflow.stage).lower(),
        },
        "freeze_protocol": {
            "protocol_name": str(cfg.freeze_protocol.protocol_name),
            "main_table_modes": [str(m) for m in cfg.freeze_protocol.main_table_modes],
            "ablation_modes": [str(m) for m in cfg.freeze_protocol.ablation_modes],
            "report_metrics": [str(m) for m in cfg.freeze_protocol.report_metrics],
            "primary_report": str(cfg.freeze_protocol.primary_report).lower(),
            "report_raw_and_shielded": bool(cfg.freeze_protocol.report_raw_and_shielded),
            "dev_seeds": [int(s) for s in cfg.freeze_protocol.dev_seeds],
            "formal_seeds": [int(s) for s in cfg.freeze_protocol.formal_seeds],
            "llm_retry_times": int(cfg.freeze_protocol.llm_retry_times),
            "llm_retry_backoff_sec": [float(x) for x in cfg.freeze_protocol.llm_retry_backoff_sec],
            "prompt_template_hash": str(cfg.freeze_protocol.prompt_template_hash),
            "normalization_version": str(cfg.freeze_protocol.normalization_version),
            "min_successful_runs_per_mode": int(cfg.freeze_protocol.min_successful_runs_per_mode),
        },
        "train_budget": {
            "episodes": int(cfg.train.episodes),
            "max_steps_per_episode": int(cfg.train.max_steps_per_episode),
        },
        "ablation": dataclass_to_plain_dict(cfg.ablation),
    }


def build_frozen_protocol_payload(cfg: Config, source_file: str) -> dict:
    canonical = canonical_protocol_dict(cfg)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": _utc_now_iso(),
        "protocol_name": str(cfg.freeze_protocol.protocol_name),
        "code_sha256": str(cfg.workflow.code_hash),
        "dependency_versions": dict(cfg.workflow.dependency_versions),
        "canonical_config": canonical,
        "manifest": {
            "main_table_modes": [str(m) for m in cfg.freeze_protocol.main_table_modes],
            "ablation_modes": [str(m) for m in cfg.freeze_protocol.ablation_modes],
            "report_metrics": [str(m) for m in cfg.freeze_protocol.report_metrics],
            "primary_report": str(cfg.freeze_protocol.primary_report).lower(),
            "report_raw_and_shielded": bool(cfg.freeze_protocol.report_raw_and_shielded),
            "dev_seeds": [int(s) for s in cfg.freeze_protocol.dev_seeds],
            "formal_seeds": [int(s) for s in cfg.freeze_protocol.formal_seeds],
            "train_budget": {
                "episodes": int(cfg.train.episodes),
                "max_steps_per_episode": int(cfg.train.max_steps_per_episode),
            },
            "eval_budget": {
                "episodes": int(cfg.eval.episodes),
                "seed_offset": int(cfg.eval.seed_offset),
            },
            "llm_retry_policy": {
                "retry_times": int(cfg.freeze_protocol.llm_retry_times),
                "retry_backoff_sec": [float(x) for x in cfg.freeze_protocol.llm_retry_backoff_sec],
            },
            "prompt_template_hash": str(cfg.freeze_protocol.prompt_template_hash),
            "normalization_version": str(cfg.freeze_protocol.normalization_version),
            "min_successful_runs_per_mode": int(cfg.freeze_protocol.min_successful_runs_per_mode),
            "source_file": os.path.abspath(source_file),
        },
    }
    protocol_sha256 = _sha256_text(_canonical_json_dumps(payload))
    payload["protocol_sha256"] = protocol_sha256
    return payload


def save_frozen_protocol(cfg: Config, path: str, source_file: str) -> dict:
    payload = build_frozen_protocol_payload(cfg, source_file)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_frozen_protocol(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    expected = str(payload.get("protocol_sha256", ""))
    verify = dict(payload)
    verify.pop("protocol_sha256", None)
    actual = _sha256_text(_canonical_json_dumps(verify))
    if expected != actual:
        raise RuntimeError(f"Protocol hash mismatch: expected={expected} actual={actual}")
    return payload


def apply_frozen_protocol_to_cfg(cfg: Config, protocol: dict) -> Config:
    canonical = dict(protocol.get("canonical_config", {}))
    for section_name, section_obj in [
        ("env", cfg.env),
        ("planner", cfg.planner),
        ("reward", cfg.reward),
        ("sac", cfg.sac),
        ("safety", cfg.safety),
        ("ablation", cfg.ablation),
    ]:
        section_values = canonical.get(section_name, {})
        for k, v in section_values.items():
            if hasattr(section_obj, k):
                setattr(section_obj, k, v)

    llm_values = canonical.get("llm", {})
    for k in [
        "backend", "model_name", "api_base", "temperature", "max_tokens", "timeout_sec",
        "system_prompt", "retry_times", "retry_backoff_sec", "deterministic_in_formal"
    ]:
        if k in llm_values and hasattr(cfg.llm, k):
            setattr(cfg.llm, k, llm_values[k])

    eval_values = canonical.get("eval", {})
    for k in ["enabled", "episodes", "seed_offset", "report_raw_and_shielded", "primary_report"]:
        if k in eval_values and hasattr(cfg.eval, k):
            setattr(cfg.eval, k, eval_values[k])

    compare_values = canonical.get("compare", {})
    if "primary_report" in compare_values:
        cfg.compare.primary_report = str(compare_values["primary_report"]).lower()
    if "formal_modes" in compare_values:
        cfg.compare.formal_modes = [str(m) for m in compare_values["formal_modes"]]
    if "require_all_modes_success" in compare_values:
        cfg.compare.require_all_modes_success = bool(compare_values["require_all_modes_success"])
    if "min_successful_runs_per_mode" in compare_values:
        cfg.compare.min_successful_runs_per_mode = int(compare_values["min_successful_runs_per_mode"])

    fp_values = canonical.get("freeze_protocol", {})
    for k in [
        "protocol_name", "main_table_modes", "ablation_modes", "report_metrics", "primary_report",
        "report_raw_and_shielded", "dev_seeds", "formal_seeds", "llm_retry_times",
        "llm_retry_backoff_sec", "prompt_template_hash", "normalization_version", "min_successful_runs_per_mode"
    ]:
        if k in fp_values and hasattr(cfg.freeze_protocol, k):
            setattr(cfg.freeze_protocol, k, fp_values[k])

    budget = canonical.get("train_budget", {})
    if budget:
        cfg.train.episodes = int(budget.get("episodes", cfg.train.episodes))
        cfg.train.max_steps_per_episode = int(budget.get("max_steps_per_episode", cfg.train.max_steps_per_episode))
    return cfg


def validate_cfg_against_protocol(cfg: Config, protocol: dict, strict: bool = True) -> tuple[bool, List[str]]:
    mismatches: List[str] = []
    expected = protocol.get("canonical_config", {})
    current = canonical_protocol_dict(cfg)
    if expected != current:
        expected_text = stable_json_dumps(expected)
        current_text = stable_json_dumps(current)
        mismatches.append(
            "canonical_config_mismatch: expected_hash="
            + _sha256_text(expected_text)
            + " current_hash="
            + _sha256_text(current_text)
        )
    manifest_code_hash = str(protocol.get("code_sha256", ""))
    if manifest_code_hash and manifest_code_hash != cfg.workflow.code_hash:
        mismatches.append(f"code_sha256 mismatch: manifest={manifest_code_hash} current={cfg.workflow.code_hash}")
    manifest_deps = protocol.get("dependency_versions", {})
    current_deps = cfg.workflow.dependency_versions
    for k, v in manifest_deps.items():
        cv = str(current_deps.get(k, ""))
        if str(v) != cv:
            mismatches.append(f"dependency mismatch {k}: manifest={v} current={cv}")
    ok = len(mismatches) == 0
    if strict and not ok:
        raise RuntimeError("Formal strict validation failed: " + "; ".join(mismatches))
    return ok, mismatches


def resolve_paper_eligibility(cfg: Config, mode: str, is_compare: bool) -> bool:
    stage = str(cfg.workflow.stage).lower()
    if stage != "formal":
        return False
    if bool(cfg.workflow.smoke_test):
        return False
    return True


def enforce_workflow_stage_policy(cfg: Config, requested_mode: str, is_compare: bool) -> None:
    stage = str(cfg.workflow.stage).lower()
    if stage == "freeze":
        return
    if stage == "dev":
        if is_compare:
            candidate_modes = [str(m) for m in cfg.compare.modes]
            if any(mode_uses_real_llm(m, cfg) for m in candidate_modes if mode_uses_llm_planner(m)) and not cfg.workflow.allow_real_llm_smoke:
                raise RuntimeError(
                    "Dev stage forbids real LLM planner runs in compare. Use --allow-real-llm-smoke for smoke tests only."
                )
        else:
            if mode_uses_llm_planner(requested_mode) and mode_uses_real_llm(requested_mode, cfg) and not (cfg.workflow.allow_real_llm_smoke or cfg.workflow.smoke_test):
                raise RuntimeError(
                    "Dev stage forbids real LLM single runs unless --allow-real-llm-smoke or --smoke-test is set."
                )
    elif stage == "formal":
        if not cfg.workflow.freeze_load_path:
            raise RuntimeError("Formal stage requires --freeze-load.")
        if is_compare:
            cfg.eval.report_raw_and_shielded = True
            cfg.eval.primary_report = "raw"
        if requested_mode != "compare" and requested_mode not in cfg.freeze_protocol.main_table_modes:
            raise RuntimeError(
                f"Formal single run mode must be in frozen main table modes: {cfg.freeze_protocol.main_table_modes}"
            )


def configure_llm_for_stage(cfg: Config) -> None:
    if str(cfg.workflow.stage).lower() == "formal" and mode_uses_real_llm(cfg.train.mode, cfg):
        if cfg.llm.deterministic_in_formal:
            cfg.llm.temperature = 0.0
        cfg.llm.fallback_to_mock_on_error = False
        cfg.llm.fallback_to_mock_on_parse_error = False


def _ensure_runtime_llm_log(cfg) -> List[dict]:
    if not hasattr(cfg, "_runtime_llm_call_log"):
        setattr(cfg, "_runtime_llm_call_log", [])
    return getattr(cfg, "_runtime_llm_call_log")


def reset_runtime_llm_log(cfg) -> None:
    setattr(cfg, "_runtime_llm_call_log", [])


def append_llm_call_log(cfg, entry: dict) -> None:
    log = _ensure_runtime_llm_log(cfg)
    log.append(entry)


def get_llm_call_log(cfg) -> List[dict]:
    return list(_ensure_runtime_llm_log(cfg))


def export_llm_call_log(cfg, save_path: str, run_tag: str) -> str:
    if not save_path:
        return ""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    payload = {
        "run_tag": run_tag,
        "stage": cfg.workflow.stage,
        "protocol_hash": cfg.workflow.protocol_hash,
        "llm_call_log": get_llm_call_log(cfg),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return os.path.abspath(save_path)


def record_llm_interaction(cfg: Config, scene: dict, prompt: str, raw_response: str, mode: str, seed: int, step_idx: Optional[int]) -> dict:
    record = {
        "timestamp_utc": _utc_now_iso(),
        "stage": str(cfg.workflow.stage),
        "mode": str(mode),
        "seed": int(seed),
        "step_idx": None if step_idx is None else int(step_idx),
        "prompt_hash": _sha256_text(prompt),
        "response_hash": _sha256_text(raw_response),
        "prompt_len": len(prompt),
        "response_len": len(raw_response),
    }
    append_llm_call_log(cfg, record)
    return record


def build_run_metadata(
    cfg: Config,
    mode: str,
    seed: int,
    protocol: Optional[dict],
    source_file: str,
    run_status: str = "success",
    failure_reason: str = "",
) -> dict:
    return {
        "workflow_stage": str(cfg.workflow.stage),
        "paper_eligible": bool(cfg.workflow.paper_eligible),
        "mode": str(mode),
        "seed": int(seed),
        "train_budget": {
            "episodes": int(cfg.train.episodes),
            "max_steps_per_episode": int(cfg.train.max_steps_per_episode),
        },
        "eval_budget": {
            "episodes": int(cfg.eval.episodes),
            "seed_offset": int(cfg.eval.seed_offset),
        },
        "primary_report": str(cfg.eval.primary_report),
        "report_raw_and_shielded": bool(cfg.eval.report_raw_and_shielded),
        "protocol_sha256": str(cfg.workflow.protocol_hash),
        "frozen_protocol_path": str(cfg.workflow.frozen_protocol_path),
        "code_sha256": str(cfg.workflow.code_hash),
        "llm_backend": str(cfg.llm.backend),
        "llm_model_name": str(cfg.llm.model_name),
        "run_status": str(run_status),
        "failure_reason": str(failure_reason),
        "timestamp": _utc_now_iso(),
        "source_file": os.path.abspath(source_file),
    }


def initialize_workflow(cfg, run_mode: str) -> Optional[dict]:
    cfg.workflow.stage = str(cfg.workflow.stage).lower()
    if cfg.workflow.stage not in WORKFLOW_STAGES:
        raise RuntimeError(f"Unknown workflow stage: {cfg.workflow.stage}")

    cfg.workflow.code_hash = compute_code_hash()
    cfg.workflow.dependency_versions = collect_dependency_versions()
    cfg.freeze_protocol.dev_seeds = _normalize_seed_list(cfg.freeze_protocol.dev_seeds, [42, 52, 62])
    cfg.freeze_protocol.formal_seeds = _normalize_seed_list(cfg.freeze_protocol.formal_seeds, [142, 242, 342])
    cfg.freeze_protocol.main_table_modes = [str(m) for m in cfg.freeze_protocol.main_table_modes]
    cfg.freeze_protocol.ablation_modes = [str(m) for m in cfg.freeze_protocol.ablation_modes]
    cfg.compare.formal_modes = list(cfg.freeze_protocol.main_table_modes)
    cfg.compare.min_successful_runs_per_mode = int(cfg.freeze_protocol.min_successful_runs_per_mode)
    cfg.workflow.paper_eligible = resolve_paper_eligibility(cfg, cfg.train.mode, is_compare=(run_mode == "compare"))
    cfg.freeze_protocol.prompt_template_hash = get_prompt_template_hash(cfg)

    loaded_protocol = None
    if cfg.workflow.stage == "freeze":
        payload = save_frozen_protocol(cfg, cfg.workflow.freeze_save_path or "frozen_protocol.json", __file__)
        cfg.workflow.protocol_hash = str(payload.get("protocol_sha256", ""))
        cfg.workflow.frozen_protocol_path = os.path.abspath(cfg.workflow.freeze_save_path or "frozen_protocol.json")
        print(f"[WORKFLOW] stage=freeze saved protocol to {cfg.workflow.frozen_protocol_path} protocol_hash={cfg.workflow.protocol_hash}")
        return payload

    if cfg.workflow.stage == "formal":
        loaded_protocol = load_frozen_protocol(cfg.workflow.freeze_load_path)
        cfg.workflow.protocol_hash = str(loaded_protocol.get("protocol_sha256", ""))
        cfg.workflow.frozen_protocol_path = os.path.abspath(cfg.workflow.freeze_load_path)
        apply_frozen_protocol_to_cfg(cfg, loaded_protocol)
        cfg.workflow.paper_eligible = resolve_paper_eligibility(cfg, cfg.train.mode, is_compare=(run_mode == "compare"))
        validate_cfg_against_protocol(cfg, loaded_protocol, strict=bool(cfg.workflow.formal_strict))
        configure_llm_for_stage(cfg)
        print(f"[WORKFLOW] stage=formal loaded protocol from {cfg.workflow.frozen_protocol_path} protocol_hash={cfg.workflow.protocol_hash}")
    return loaded_protocol


def build_result_metadata(cfg, run_type: str, seeds: Optional[List[int]] = None) -> dict:
    return {
        "run_type": str(run_type),
        "workflow_stage": str(cfg.workflow.stage),
        "protocol_hash": str(cfg.workflow.protocol_hash),
        "frozen_protocol_path": str(cfg.workflow.frozen_protocol_path),
        "paper_eligible": bool(cfg.workflow.paper_eligible),
        "code_sha256": str(cfg.workflow.code_hash),
        "dependency_versions": dict(cfg.workflow.dependency_versions),
        "seeds_used": [int(s) for s in (seeds or [])],
    }


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
        return str(cfg.llm.backend).lower() == "real"
    return str(cfg.llm.backend).lower() == "real"


def mode_is_constrained(mode: str) -> bool:
    return mode in CONSTRAINED_MODES


def physical_to_normalized_action_bounds(cfg, constraints: dict) -> tuple[float, float]:
    acc_bound = float(constraints.get("acc_max", cfg.planner.acc_max)) / max(cfg.safety.action_acc_scale, 1e-6)
    steer_bound = float(constraints.get("steer_max", cfg.planner.steer_max)) / max(cfg.safety.action_steer_scale, 1e-6)
    acc_bound = float(np.clip(acc_bound, 0.05, 1.0))
    steer_bound = float(np.clip(steer_bound, 0.05, 1.0))
    return acc_bound, steer_bound


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
        if self.cfg.ablation.disable_waypoint_features:
            wp_feat = np.zeros(6, dtype=np.float32)
        else:
            wp_feat = self._waypoint_features(scene, self.current_waypoints)
        if self.cfg.ablation.disable_constraint_features:
            c_feat = np.zeros(6, dtype=np.float32)
        else:
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


def build_waypoints_from_target(cfg, scene: dict, target_lane: int, target_speed: float) -> np.ndarray:
    target_y = float(target_lane) * cfg.env.lane_width
    waypoints = []
    for i in range(1, cfg.planner.waypoint_horizon + 1):
        alpha = i / cfg.planner.waypoint_horizon
        x = scene["ego_x"] + i * cfg.planner.waypoint_gap
        y = (1.0 - alpha) * scene["ego_y"] + alpha * target_y
        v = target_speed
        waypoints.append([x, y, v])
    return np.asarray(waypoints, dtype=np.float32)


def lane_front_distance(scene: dict, lane_id: int) -> float:
    if lane_id == scene["lane_id"]:
        return float(scene["front_distance"])
    if scene.get("left_lane_id") is not None and lane_id == scene["left_lane_id"]:
        return float(scene["left_front_distance"])
    if scene.get("right_lane_id") is not None and lane_id == scene["right_lane_id"]:
        return float(scene["right_front_distance"])
    return 1e6


def stabilize_lane_decision(cfg, scene: dict, proposed_lane: int, planner_state: dict, step_idx: Optional[int]) -> tuple[int, str]:
    current_lane = int(scene["lane_id"])
    proposed_lane = int(np.clip(proposed_lane, 0, max(cfg.env.lanes_count - 1, 0)))

    if cfg.ablation.disable_lane_stabilization:
        return proposed_lane, "ablation_disabled"

    if proposed_lane == current_lane:
        planner_state["active_target_lane"] = current_lane
        return proposed_lane, "keep_lane"

    emergency = (
        float(scene["front_distance"]) < float(cfg.safety.min_front_distance_hard)
        or float(scene["ttc"]) < float(cfg.safety.ttc_safe)
    )
    improvement = lane_front_distance(scene, proposed_lane) - lane_front_distance(scene, current_lane)

    if (not emergency) and float(scene["ego_speed"]) < 5.0 and float(scene["front_distance"]) > cfg.planner.normal_headway:
        planner_state["active_target_lane"] = current_lane
        return current_lane, "hold_low_speed"

    if (not emergency) and improvement < cfg.planner.lane_change_min_improvement:
        planner_state["active_target_lane"] = current_lane
        return current_lane, "insufficient_gain"

    last_change_step = int(planner_state.get("last_change_step", -10**9))
    active_target_lane_raw = planner_state.get("active_target_lane", current_lane)
    active_target_lane = current_lane if active_target_lane_raw is None else int(active_target_lane_raw)
    if step_idx is not None and (step_idx - last_change_step) < cfg.planner.lane_change_cooldown_steps:
        return active_target_lane, "cooldown_hold"

    planner_state["active_target_lane"] = proposed_lane
    planner_state["last_change_step"] = int(step_idx if step_idx is not None else last_change_step)
    return proposed_lane, "accepted"


class RulePlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = {"last_change_step": -10**9}

    def plan(self, scene: dict, step_idx: Optional[int] = None) -> dict:
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

        target_lane, stabilization_tag = stabilize_lane_decision(self.cfg, scene, target_lane, self.state, step_idx)

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

        waypoints = build_waypoints_from_target(self.cfg, scene, int(target_lane), float(target_speed))

        constraints = {
            "min_headway": headway,
            "speed_max": self.cfg.planner.speed_max,
            "acc_max": self.cfg.planner.acc_max,
            "steer_max": self.cfg.planner.steer_max,
            "comfort_acc_delta": self.cfg.planner.comfort_acc_delta,
            "risk_mode": risk_mode,
        }

        return {
            "waypoints": waypoints,
            "constraints": constraints,
            "planner_info": {
                "planner_name": "rule",
                "target_lane": int(target_lane),
                "target_speed": float(target_speed),
                "trigger": trigger,
                "stabilization": stabilization_tag,
            },
        }


class PromptBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def template_signature(self) -> str:
        return PROMPT_TEMPLATE_VERSION + "\n" + str(self.cfg.llm.system_prompt) + "\n" + self.build({
            "ego_x": 0.0, "ego_y": 0.0, "ego_speed": 0.0, "heading": 0.0, "lane_id": 0,
            "front_distance": 30.0, "front_rel_speed": 0.0, "left_front_distance": 30.0,
            "right_front_distance": 30.0, "ttc": 99.0, "left_lane_id": None, "right_lane_id": 1,
            "left_clear": False, "right_clear": True,
        })

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

    def _log_call(self, **kwargs) -> None:
        payload = {
            "timestamp_utc": _utc_now_iso(),
            "stage": str(self.cfg.workflow.stage),
            "mode": str(self.cfg.train.mode),
            "backend": str(self.name),
        }
        payload.update(kwargs)
        append_llm_call_log(self.cfg, payload)

    def generate(self, prompt: str, scene: dict) -> str:
        raise NotImplementedError


class MockLLMBackend(BaseLLMBackend):
    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str, scene: dict) -> str:
        response_obj = heuristic_llm_decision(self.cfg, scene, style="mock")
        response = json.dumps(response_obj, ensure_ascii=False, indent=2)
        self._log_call(
            status="success",
            prompt_hash=_sha256_text(prompt),
            response_hash=_sha256_text(response),
            attempt=1,
            fallback_used=False,
        )
        return response


class RealLLMBackend(BaseLLMBackend):
    @property
    def name(self) -> str:
        return "real"

    def _resolve_api_key(self) -> str:
        api_key = str(self.cfg.llm.api_key or "").strip()
        if (not api_key) or ("请替换" in api_key) or api_key.endswith("你的ChatAnywhere密钥"):
            raise RuntimeError(
                "Please edit USER_EDITABLE_LLM['api_key'] in the code and replace the placeholder with your real ChatAnywhere key."
            )
        return api_key

    def _resolve_api_base(self) -> str:
        raw_base = str(self.cfg.llm.api_base or USER_EDITABLE_LLM["api_base"]).strip().rstrip("/")
        if not raw_base:
            raw_base = "https://api.chatanywhere.tech"
        if raw_base.endswith("/chat/completions"):
            return raw_base
        if raw_base.endswith("/v1"):
            return raw_base + "/chat/completions"
        if raw_base.endswith("/v1/chat"):
            return raw_base + "/completions"
        return raw_base + "/v1/chat/completions"

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
            raise RuntimeError("RealLLMBackend requires cfg.llm.api_key, CHATANYWHERE_API_KEY, or OPENAI_API_KEY.")
        if not api_base:
            raise RuntimeError("RealLLMBackend requires cfg.llm.api_base, CHATANYWHERE_API_BASE, or OPENAI_API_BASE.")
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

    def _call_real_api_with_retry(self, prompt: str) -> str:
        prompt_hash = _sha256_text(prompt)
        retry_times = max(1, int(self.cfg.llm.retry_times))
        backoffs = list(self.cfg.llm.retry_backoff_sec) or [1.0]
        last_error: Optional[Exception] = None
        for attempt in range(1, retry_times + 1):
            try:
                response = self._call_real_api(prompt)
                self._log_call(
                    status="success",
                    prompt_hash=prompt_hash,
                    response_hash=_sha256_text(response),
                    attempt=attempt,
                    fallback_used=False,
                )
                return response
            except Exception as e:
                last_error = e
                self._log_call(
                    status="error",
                    prompt_hash=prompt_hash,
                    response_hash="",
                    attempt=attempt,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    fallback_used=False,
                )
                if attempt < retry_times:
                    time.sleep(float(backoffs[min(attempt - 1, len(backoffs) - 1)]))
        if last_error is not None:
            raise last_error
        raise RuntimeError("RealLLMBackend failed without explicit exception")

    def generate(self, prompt: str, scene: dict) -> str:
        try:
            return self._call_real_api_with_retry(prompt)
        except Exception as e:
            if self.cfg.planner.print_real_llm_error:
                print(f"[REAL-LLM][WARN] {type(e).__name__}: {e}")
            allow_fallback = bool(self.cfg.llm.fallback_to_mock_on_error and str(self.cfg.workflow.stage).lower() != "formal")
            if allow_fallback:
                self._log_call(
                    status="fallback_to_mock",
                    prompt_hash=_sha256_text(prompt),
                    response_hash="",
                    attempt=max(1, int(self.cfg.llm.retry_times)),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    fallback_used=True,
                )
                return MockLLMBackend(self.cfg).generate(prompt, scene)
            raise


class LLMPlanner:

    def __init__(self, cfg, backend: BaseLLMBackend):
        self.cfg = cfg
        self.prompt_builder = PromptBuilder(cfg)
        self.backend = backend
        self.state = {"last_change_step": -10**9}

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

    def _normalize_decision(self, scene: dict, decision: dict) -> dict:
        lane_default = int(scene["lane_id"])
        speed_default = float(self.cfg.planner.target_speed_free)

        lane = int(decision.get("target_lane", lane_default))
        lane = int(np.clip(lane, 0, max(self.cfg.env.lanes_count - 1, 0)))

        speed = float(decision.get("target_speed", speed_default))
        speed = float(np.clip(speed, self.cfg.env.min_speed, self.cfg.env.max_speed))

        return {
            "target_lane": lane,
            "target_speed": speed,
            "reason": str(decision.get("reason", "llm decision")),
        }

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

        out["min_headway"] = float(np.clip(out["min_headway"], 5.0, 40.0))
        out["speed_max"] = float(np.clip(out["speed_max"], self.cfg.env.min_speed, self.cfg.env.max_speed))
        out["acc_max"] = float(np.clip(out["acc_max"], 0.2, 5.0))
        out["steer_max"] = float(np.clip(out["steer_max"], 0.05, 0.8))
        out["comfort_acc_delta"] = float(np.clip(out["comfort_acc_delta"], 0.05, 5.0))
        out["risk_mode"] = int(np.clip(int(out.get("risk_mode", 0)), 0, 2))
        return out

    def _normalize_waypoints(self, scene: dict, waypoints: list, decision: dict) -> np.ndarray:
        normalized = []
        target_lane = int(np.clip(decision.get("target_lane", scene["lane_id"]), 0, max(self.cfg.env.lanes_count - 1, 0)))
        target_y_default = target_lane * self.cfg.env.lane_width
        target_speed = float(np.clip(decision.get("target_speed", self.cfg.planner.target_speed_free), self.cfg.env.min_speed, self.cfg.env.max_speed))
        road_y_min = 0.0
        road_y_max = (self.cfg.env.lanes_count - 1) * self.cfg.env.lane_width

        prev_x = float(scene["ego_x"])
        for i in range(self.cfg.planner.waypoint_horizon):
            min_x = float(scene["ego_x"] + (i + 1) * self.cfg.planner.waypoint_gap * 0.5)
            default_x = float(scene["ego_x"] + (i + 1) * self.cfg.planner.waypoint_gap)
            default_alpha = (i + 1) / self.cfg.planner.waypoint_horizon
            default_y = (1.0 - default_alpha) * scene["ego_y"] + default_alpha * target_y_default
            if i < len(waypoints):
                wp = waypoints[i]
                x = float(wp[0])
                y = float(wp[1])
                v = float(wp[2])
            else:
                x = default_x
                y = default_y
                v = target_speed

            x = float(max(x, min_x, prev_x + 1e-3))
            y = float(np.clip(y, road_y_min, road_y_max))
            v = float(np.clip(v, self.cfg.env.min_speed, self.cfg.env.max_speed))
            normalized.append([x, y, v])
            prev_x = x

        return np.asarray(normalized, dtype=np.float32)

    def _parse_or_fallback(self, raw_response: str, scene: dict, prompt_hash: str) -> dict:
        try:
            json_str = self._extract_json_string(raw_response)
            obj = json.loads(json_str)
            decision = self._normalize_decision(scene, obj.get("decision", {}))
            constraints = self._normalize_constraints(obj.get("constraints", {}))
            decision["target_speed"] = min(decision["target_speed"], constraints["speed_max"])
            waypoints = self._normalize_waypoints(scene, obj.get("waypoints", []), decision)
            return {
                "decision": decision,
                "constraints": constraints,
                "waypoints": waypoints,
            }
        except Exception as e:
            if self.cfg.planner.print_real_llm_error:
                print(f"[LLM-PARSE][WARN] {type(e).__name__}: {e}")
            append_llm_call_log(self.cfg, {"timestamp_utc": _utc_now_iso(), "stage": str(self.cfg.workflow.stage), "mode": str(self.cfg.train.mode), "backend": f"{self.backend.name}_parse", "status": "parse_error", "prompt_hash": str(prompt_hash), "response_hash": _sha256_text(raw_response), "error_type": type(e).__name__, "error_message": str(e), "fallback_used": False})
            if self.cfg.llm.fallback_to_mock_on_parse_error and str(self.cfg.workflow.stage).lower() != "formal":
                fallback = heuristic_llm_decision(
                    self.cfg,
                    scene,
                    style="real" if self.backend.name == "real" else "mock",
                )
                decision = self._normalize_decision(scene, fallback["decision"])
                constraints = self._normalize_constraints(fallback["constraints"])
                decision["target_speed"] = min(decision["target_speed"], constraints["speed_max"])
                waypoints = self._normalize_waypoints(scene, fallback["waypoints"], decision)
                return {
                    "decision": decision,
                    "constraints": constraints,
                    "waypoints": waypoints,
                }
            raise

    def plan(self, scene: dict, step_idx: Optional[int] = None) -> dict:
        prompt = self.prompt_builder.build(scene)
        prompt_hash = _sha256_text(prompt)
        raw_response = self.backend.generate(prompt, scene)
        response_hash = _sha256_text(raw_response)
        parsed = self._parse_or_fallback(raw_response, scene, prompt_hash=prompt_hash)
        record_llm_interaction(self.cfg, scene, prompt, raw_response, self.cfg.train.mode, self.cfg.train.seed, step_idx)
        decision = dict(parsed["decision"])
        decision["target_lane"], stabilization_tag = stabilize_lane_decision(
            self.cfg, scene, int(decision["target_lane"]), self.state, step_idx
        )
        parsed["waypoints"] = build_waypoints_from_target(
            self.cfg, scene, int(decision["target_lane"]), float(decision["target_speed"])
        )

        return {
            "waypoints": parsed["waypoints"],
            "constraints": parsed["constraints"],
            "planner_info": {
                "planner_name": f"{self.backend.name}_llm",
                "target_lane": int(decision["target_lane"]),
                "target_speed": float(decision["target_speed"]),
                "reason": decision["reason"],
                "trigger": f"{self.backend.name}_llm_planning",
                "stabilization": stabilization_tag,
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
            },
            "prompt": prompt,
            "raw_response": raw_response,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
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
        acc_bound, steer_bound = physical_to_normalized_action_bounds(self.cfg, constraints)
        over_acc = abs(acc) > acc_bound
        over_steer = abs(steer) > steer_bound

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

        acc_bound, steer_bound = physical_to_normalized_action_bounds(self.cfg, constraints)
        acc = float(np.clip(acc, -acc_bound, acc_bound))
        steer = float(np.clip(steer, -steer_bound, steer_bound))

        safe_action = np.array([acc, steer], dtype=np.float32)
        return safe_action, {
            "shield_modified": bool(modified),
            "shield_reasons": reasons,
        }


def _squared_barrier_cost(value: float, threshold: float) -> float:
    threshold = max(float(threshold), 1e-6)
    gap = max(0.0, threshold - float(value))
    return float((gap / threshold) ** 2)


def compute_costs(cfg, scene: dict, next_scene: dict, action, constraints: dict) -> dict:
    action = np.asarray(action, dtype=np.float32)
    last_action = np.asarray(scene.get("last_action", np.zeros_like(action)), dtype=np.float32)

    if cfg.ablation.disable_constraint_costs:
        return {
            "collision_cost": 0.0,
            "headway_cost": 0.0,
            "overspeed_cost": 0.0,
            "comfort_cost": 0.0,
            "total_cost": 0.0,
        }

    # Second-priority fix: make collision cost dense enough to activate before an actual crash.
    crash_cost = 1.0 if next_scene["collision"] else 0.0
    ttc_risk_cost = _squared_barrier_cost(
        float(next_scene["ttc"]),
        float(cfg.safety.collision_risk_ttc_threshold),
    )
    front_risk_cost = _squared_barrier_cost(
        float(next_scene["front_distance"]),
        float(cfg.safety.collision_risk_front_distance),
    )
    collision_cost = max(crash_cost, ttc_risk_cost, front_risk_cost)

    headway_gap = max(0.0, float(constraints["min_headway"]) - float(next_scene["front_distance"]))
    headway_cost = headway_gap / max(float(constraints["min_headway"]), 1e-6)

    overspeed_gap = max(0.0, float(next_scene["ego_speed"]) - float(constraints["speed_max"]))
    overspeed_cost = overspeed_gap / max(float(constraints["speed_max"]), 1e-6)

    acc_delta = abs(float(action[0] - last_action[0]))
    steer_delta = abs(float(action[1] - last_action[1]))
    comfort_cost = 0.6 * acc_delta + 0.4 * steer_delta

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
    if (not cfg.ablation.disable_waypoint_reward) and planner_used and current_waypoints is not None and len(current_waypoints) > 0:
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
    """
    Second-priority stabilized constrained SAC + lightweight diagnostics:
    - actor penalty still uses only collision + headway
    - overspeed + comfort critics/lambdas are kept as diagnostic branches
    - lambda updates use EMA-smoothed immediate costs for collision/headway
    - collision/headway lambdas have non-zero lower bounds
    - collision cost is made denser via TTC / front-distance risk before a crash happens
    """

    def __init__(self, state_dim, action_dim, cfg):
        super().__init__(state_dim, action_dim, cfg)

        init_lambda = max(float(self.cfg.sac.initial_lambda), 1e-6)
        self.log_lambda_collision = torch.tensor(np.log(init_lambda), requires_grad=True, device=self.device)
        self.log_lambda_headway = torch.tensor(np.log(init_lambda), requires_grad=True, device=self.device)
        self.log_lambda_overspeed = torch.tensor(np.log(init_lambda), requires_grad=True, device=self.device)
        self.log_lambda_comfort = torch.tensor(np.log(init_lambda), requires_grad=True, device=self.device)

        self.qc_collision = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.qc_headway = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.qc_overspeed = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.qc_comfort = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)

        self.qc_collision_target = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.qc_headway_target = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.qc_overspeed_target = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.qc_comfort_target = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)

        self.qc_collision_target.load_state_dict(self.qc_collision.state_dict())
        self.qc_headway_target.load_state_dict(self.qc_headway.state_dict())
        self.qc_overspeed_target.load_state_dict(self.qc_overspeed.state_dict())
        self.qc_comfort_target.load_state_dict(self.qc_comfort.state_dict())

        self.qc_collision_opt = torch.optim.Adam(self.qc_collision.parameters(), lr=cfg.sac.critic_lr)
        self.qc_headway_opt = torch.optim.Adam(self.qc_headway.parameters(), lr=cfg.sac.critic_lr)
        self.qc_overspeed_opt = torch.optim.Adam(self.qc_overspeed.parameters(), lr=cfg.sac.critic_lr)
        self.qc_comfort_opt = torch.optim.Adam(self.qc_comfort.parameters(), lr=cfg.sac.critic_lr)

        self.lambda_opt = torch.optim.Adam(
            [
                self.log_lambda_collision,
                self.log_lambda_headway,
                self.log_lambda_overspeed,
                self.log_lambda_comfort,
            ],
            lr=self.cfg.sac.lambda_lr,
        )

        self.lambda_ema_beta = float(np.clip(self.cfg.sac.lambda_ema_beta, 0.0, 0.9999))
        self.ema_collision_cost = None
        self.ema_headway_cost = None
        self.ema_overspeed_cost = None
        self.ema_comfort_cost = None

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

    def _update_single_cost_critic(self, critic, critic_target, optimizer, state, action, done, next_state, next_action, immediate_cost):
        with torch.no_grad():
            next_cost = critic_target(next_state, next_action)
            target_cost = immediate_cost + (1.0 - done) * self.gamma * next_cost

        current_cost = critic(state, action)
        loss = F.mse_loss(current_cost, target_cost)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, current_cost.detach().mean(), target_cost.detach().mean()

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

        qc_collision_loss, _, qc_collision_target_mean = self._update_single_cost_critic(
            self.qc_collision, self.qc_collision_target, self.qc_collision_opt,
            state, action, done, next_state, next_action, cost_collision
        )
        qc_headway_loss, _, qc_headway_target_mean = self._update_single_cost_critic(
            self.qc_headway, self.qc_headway_target, self.qc_headway_opt,
            state, action, done, next_state, next_action, cost_headway
        )
        qc_overspeed_loss, _, qc_overspeed_target_mean = self._update_single_cost_critic(
            self.qc_overspeed, self.qc_overspeed_target, self.qc_overspeed_opt,
            state, action, done, next_state, next_action, cost_overspeed
        )
        qc_comfort_loss, _, qc_comfort_target_mean = self._update_single_cost_critic(
            self.qc_comfort, self.qc_comfort_target, self.qc_comfort_opt,
            state, action, done, next_state, next_action, cost_comfort
        )

        new_action, log_prob, _ = self.actor.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        qc_collision_new = self.qc_collision(state, new_action)
        qc_headway_new = self.qc_headway(state, new_action)
        qc_overspeed_new = self.qc_overspeed(state, new_action)
        qc_comfort_new = self.qc_comfort(state, new_action)

        # First-priority fix: actor is constrained only by collision + headway.
        actor_penalty = (
            self.lambda_collision.detach() * qc_collision_new
            + self.lambda_headway.detach() * qc_headway_new
        )

        actor_loss = (self.alpha.detach() * log_prob - q_new + actor_penalty).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        batch_collision_mean = cost_collision.mean().detach()
        batch_headway_mean = cost_headway.mean().detach()
        batch_overspeed_mean = cost_overspeed.mean().detach()
        batch_comfort_mean = cost_comfort.mean().detach()

        def _ema_update(prev, current_tensor):
            current_scalar = float(current_tensor.item())
            if prev is None:
                return current_scalar
            return float(self.lambda_ema_beta * prev + (1.0 - self.lambda_ema_beta) * current_scalar)

        self.ema_collision_cost = _ema_update(self.ema_collision_cost, batch_collision_mean)
        self.ema_headway_cost = _ema_update(self.ema_headway_cost, batch_headway_mean)
        self.ema_overspeed_cost = _ema_update(self.ema_overspeed_cost, batch_overspeed_mean)
        self.ema_comfort_cost = _ema_update(self.ema_comfort_cost, batch_comfort_mean)

        ema_collision_tensor = torch.tensor(self.ema_collision_cost, dtype=torch.float32, device=self.device)
        ema_headway_tensor = torch.tensor(self.ema_headway_cost, dtype=torch.float32, device=self.device)
        ema_overspeed_tensor = torch.tensor(self.ema_overspeed_cost, dtype=torch.float32, device=self.device)
        ema_comfort_tensor = torch.tensor(self.ema_comfort_cost, dtype=torch.float32, device=self.device)

        lambda_loss = -(
            self.log_lambda_collision * (ema_collision_tensor - self.cfg.sac.cost_limit_collision)
            + self.log_lambda_headway * (ema_headway_tensor - self.cfg.sac.cost_limit_headway)
            + self.log_lambda_overspeed * (ema_overspeed_tensor - self.cfg.sac.cost_limit_overspeed)
            + self.log_lambda_comfort * (ema_comfort_tensor - self.cfg.sac.cost_limit_comfort)
        )

        self.lambda_opt.zero_grad()
        lambda_loss.backward()
        self.lambda_opt.step()

        max_log_lambda = float(np.log(self.cfg.sac.lambda_max))
        min_log_lambda_collision = float(np.log(max(self.cfg.sac.min_lambda_collision, 1e-8)))
        min_log_lambda_headway = float(np.log(max(self.cfg.sac.min_lambda_headway, 1e-8)))
        self.log_lambda_collision.data.clamp_(min=min_log_lambda_collision, max=max_log_lambda)
        self.log_lambda_headway.data.clamp_(min=min_log_lambda_headway, max=max_log_lambda)
        self.log_lambda_overspeed.data.clamp_(min=-20.0, max=max_log_lambda)
        self.log_lambda_comfort.data.clamp_(min=-20.0, max=max_log_lambda)

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self._soft_update(self.qc_collision, self.qc_collision_target)
        self._soft_update(self.qc_headway, self.qc_headway_target)
        self._soft_update(self.qc_overspeed, self.qc_overspeed_target)
        self._soft_update(self.qc_comfort, self.qc_comfort_target)

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
            "qc_collision_loss": float(qc_collision_loss.item()),
            "qc_headway_loss": float(qc_headway_loss.item()),
            "qc_overspeed_loss": float(qc_overspeed_loss.item()),
            "qc_comfort_loss": float(qc_comfort_loss.item()),
            "qc_collision": float(qc_collision_new.mean().item()),
            "qc_headway": float(qc_headway_new.mean().item()),
            "qc_overspeed": float(qc_overspeed_new.mean().item()),
            "qc_comfort": float(qc_comfort_new.mean().item()),
            "qc_collision_target": float(qc_collision_target_mean.item()),
            "qc_headway_target": float(qc_headway_target_mean.item()),
            "qc_overspeed_target": float(qc_overspeed_target_mean.item()),
            "qc_comfort_target": float(qc_comfort_target_mean.item()),
            "batch_cost_collision": float(batch_collision_mean.item()),
            "batch_cost_headway": float(batch_headway_mean.item()),
            "batch_cost_overspeed": float(batch_overspeed_mean.item()),
            "batch_cost_comfort": float(batch_comfort_mean.item()),
            "ema_collision_cost": float(self.ema_collision_cost if self.ema_collision_cost is not None else float("nan")),
            "ema_headway_cost": float(self.ema_headway_cost if self.ema_headway_cost is not None else float("nan")),
            "ema_overspeed_cost": float(self.ema_overspeed_cost if self.ema_overspeed_cost is not None else float("nan")),
            "ema_comfort_cost": float(self.ema_comfort_cost if self.ema_comfort_cost is not None else float("nan")),
            "lambda_loss": float(lambda_loss.item()),
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
    if step_idx <= 0:
        return False
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


def _evaluate_policy_once(cfg, agent, mode: str, use_shield_eval: bool, prefix: str):
    print(f"\n================ {prefix} START ================\n")
    print(f"[{prefix}-CONFIG] mode={mode} shield_enabled_eval={int(use_shield_eval)}")

    eval_env = HighwayEnvWrapper(cfg, render=False)
    verifier = SafetyVerifier(cfg) if cfg.safety.verifier_enabled else None
    shield = SafetyShield(cfg) if use_shield_eval else None

    eval_results = []

    for ep in range(cfg.eval.episodes):
        state, _ = eval_env.reset(seed=cfg.train.seed + cfg.eval.seed_offset + ep)
        planner = build_planner(cfg)

        metrics = EpisodeMetrics()

        if planner is not None:
            init_scene = eval_env.get_scene_dict()
            plan = planner.plan(init_scene, step_idx=None)
            eval_env.apply_plan(plan["waypoints"], plan["constraints"])
            print_plan_debug(cfg, plan, ep=ep, t=None, prefix=prefix + "-PLAN")

        for t in range(cfg.train.max_steps_per_episode):
            scene = eval_env.get_scene_dict()
            replanned = False

            if planner is not None and need_replan(cfg, t, scene):
                plan = planner.plan(scene, step_idx=t)
                eval_env.apply_plan(plan["waypoints"], plan["constraints"])
                replanned = True
                print_plan_debug(cfg, plan, ep=ep, t=t, prefix=prefix + "-PLAN")

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
                    f"[{prefix}-STEP][ep={ep}][t={t}] mode={mode} "
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
            f"[{prefix}][ep={ep}] "
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
    print(f"\n================ {prefix} SUMMARY ================\n")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    eval_env.close()
    return {"episodes": eval_results, "summary": summary, "use_shield": bool(use_shield_eval)}


def _print_dual_eval_delta(raw_summary: dict, shielded_summary: dict) -> None:
    keys = [
        "success",
        "collision",
        "mean_speed",
        "unsafe_headway_rate",
        "mean_cost_total",
        "interventions",
    ]
    print("\n================ RAW vs SHIELDED DELTA ================\n")
    for key in keys:
        raw_v = float(raw_summary.get(key, 0.0))
        shield_v = float(shielded_summary.get(key, 0.0))
        delta = shield_v - raw_v
        print(f"{key}: raw={raw_v:.4f} shielded={shield_v:.4f} delta={delta:+.4f}")


def evaluate_policy(cfg, agent, mode: str):
    if cfg.eval.report_raw_and_shielded:
        raw = _evaluate_policy_once(cfg, agent, mode, use_shield_eval=False, prefix="EVAL-RAW")
        shielded = _evaluate_policy_once(cfg, agent, mode, use_shield_eval=True, prefix="EVAL-SHIELD")
        _print_dual_eval_delta(raw["summary"], shielded["summary"])
        primary_report = str(cfg.eval.primary_report).lower()
        primary = shielded if primary_report == "shielded" else raw
        return {
            "raw": raw,
            "shielded": shielded,
            "primary": primary["summary"],
            "primary_report": primary_report,
        }

    single = _evaluate_policy_once(
        cfg,
        agent,
        mode,
        use_shield_eval=bool(cfg.safety.shield_enabled_eval),
        prefix="EVAL",
    )
    primary_report = "shielded" if cfg.safety.shield_enabled_eval else "raw"
    return {
        "raw": single if primary_report == "raw" else None,
        "shielded": single if primary_report == "shielded" else None,
        "primary": single["summary"],
        "primary_report": primary_report,
    }


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
            f" lambda_col={update_info.get('lambda_collision', 0.0):.4f}"
            f" lambda_head={update_info.get('lambda_headway', 0.0):.4f}"
            f" lambda_over={update_info.get('lambda_overspeed', 0.0):.4f}"
            f" lambda_comf={update_info.get('lambda_comfort', 0.0):.4f}"
        )
        if 'qc_collision' in update_info:
            msg += (
                f" qc_col={update_info.get('qc_collision', 0.0):.4f}"
                f" qc_head={update_info.get('qc_headway', 0.0):.4f}"
                f" qc_over={update_info.get('qc_overspeed', 0.0):.4f}"
                f" qc_comf={update_info.get('qc_comfort', 0.0):.4f}"
            )
    print(msg)


TRAIN_DIAGNOSTIC_FIELDS = [
    "episode",
    "episode_return",
    "success",
    "collision",
    "unsafe_headway_rate",
    "mean_cost_total",
    "lambda_collision",
    "lambda_headway",
    "lambda_overspeed",
    "lambda_comfort",
    "qc_collision",
    "qc_headway",
    "qc_overspeed",
    "qc_comfort",
    "ema_collision_cost",
    "ema_headway_cost",
]


def _diag_nan() -> float:
    return float("nan")


def build_train_diagnostic_row(ep: int, mode: str, ep_summary: dict, agent, last_update_info: dict) -> dict:
    row = {
        "episode": int(ep),
        "episode_return": float(ep_summary.get("episode_return", 0.0)),
        "success": float(ep_summary.get("success", 0.0)),
        "collision": float(ep_summary.get("collision", 0.0)),
        "unsafe_headway_rate": float(ep_summary.get("unsafe_headway_rate", 0.0)),
        "mean_cost_total": float(ep_summary.get("mean_cost_total", 0.0)),
        "lambda_collision": _diag_nan(),
        "lambda_headway": _diag_nan(),
        "lambda_overspeed": _diag_nan(),
        "lambda_comfort": _diag_nan(),
        "qc_collision": _diag_nan(),
        "qc_headway": _diag_nan(),
        "qc_overspeed": _diag_nan(),
        "qc_comfort": _diag_nan(),
        "ema_collision_cost": _diag_nan(),
        "ema_headway_cost": _diag_nan(),
    }

    if mode_is_constrained(mode):
        row["lambda_collision"] = float(last_update_info.get("lambda_collision", float(getattr(agent, "lambda_collision", torch.tensor(float("nan"))).item())))
        row["lambda_headway"] = float(last_update_info.get("lambda_headway", float(getattr(agent, "lambda_headway", torch.tensor(float("nan"))).item())))
        row["lambda_overspeed"] = float(last_update_info.get("lambda_overspeed", float(getattr(agent, "lambda_overspeed", torch.tensor(float("nan"))).item())))
        row["lambda_comfort"] = float(last_update_info.get("lambda_comfort", float(getattr(agent, "lambda_comfort", torch.tensor(float("nan"))).item())))
        row["qc_collision"] = float(last_update_info.get("qc_collision", _diag_nan()))
        row["qc_headway"] = float(last_update_info.get("qc_headway", _diag_nan()))
        row["qc_overspeed"] = float(last_update_info.get("qc_overspeed", _diag_nan()))
        row["qc_comfort"] = float(last_update_info.get("qc_comfort", _diag_nan()))
        row["ema_collision_cost"] = float(last_update_info.get("ema_collision_cost", _diag_nan()))
        row["ema_headway_cost"] = float(last_update_info.get("ema_headway_cost", _diag_nan()))
    return row


def export_train_diagnostics_csv(rows: List[dict], save_path: str) -> str:
    if not save_path:
        return ""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_DIAGNOSTIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TRAIN_DIAGNOSTIC_FIELDS})
    print(f"[DIAG] saved train diagnostics csv to {save_path}")
    return os.path.abspath(save_path)


def export_train_diagnostics_json(rows: List[dict], save_path: str) -> str:
    if not save_path:
        return ""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[DIAG] saved train diagnostics json to {save_path}")
    return os.path.abspath(save_path)


def _safe_float_list(rows: List[dict], key: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        try:
            vals.append(float(row.get(key, float("nan"))))
        except Exception:
            vals.append(float("nan"))
    return vals


def _safe_path_component(text_value: str) -> str:
    out = []
    for ch in str(text_value):
        if ch.isalnum() or ch in {'-', '_', '.'}:
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('._') or 'run'


def resolve_tensorboard_run_dir(cfg) -> str:
    base_dir = str(cfg.diagnostics.tensorboard_log_dir or '').strip()
    if not base_dir:
        return ''
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    stage = _safe_path_component(str(cfg.workflow.stage))
    mode = _safe_path_component(str(cfg.train.mode))
    seed = f"seed_{int(cfg.train.seed)}"
    run_name = f"{mode}_{seed}_{stamp}"
    return os.path.join(base_dir, stage, mode, run_name)


def create_tensorboard_writer(cfg):
    run_dir = resolve_tensorboard_run_dir(cfg)
    if not run_dir:
        return None, ''
    if SummaryWriter is None:
        raise RuntimeError('TensorBoard requested but torch.utils.tensorboard.SummaryWriter is unavailable.')
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir, flush_secs=max(1, int(cfg.diagnostics.tensorboard_flush_secs)))
    writer.add_text('run/mode', str(cfg.train.mode), 0)
    writer.add_text('run/workflow_stage', str(cfg.workflow.stage), 0)
    writer.add_text('run/seed', str(cfg.train.seed), 0)
    return writer, os.path.abspath(run_dir)


def write_train_diagnostics_tensorboard(writer, row: dict, mode: str) -> None:
    if writer is None:
        return
    ep = int(row.get('episode', 0))
    writer.add_scalar('train_episode/episode_return', float(row.get('episode_return', 0.0)), ep)
    writer.add_scalar('train_episode/success', float(row.get('success', 0.0)), ep)
    writer.add_scalar('train_episode/collision', float(row.get('collision', 0.0)), ep)
    writer.add_scalar('train_episode/unsafe_headway_rate', float(row.get('unsafe_headway_rate', 0.0)), ep)
    writer.add_scalar('train_episode/mean_cost_total', float(row.get('mean_cost_total', 0.0)), ep)
    if mode_is_constrained(mode):
        for key in ['lambda_collision', 'lambda_headway', 'lambda_overspeed', 'lambda_comfort']:
            value = float(row.get(key, float('nan')))
            if np.isfinite(value):
                writer.add_scalar(f'constraint_lambda/{key}', value, ep)
        for key in ['qc_collision', 'qc_headway', 'qc_overspeed', 'qc_comfort']:
            value = float(row.get(key, float('nan')))
            if np.isfinite(value):
                writer.add_scalar(f'cost_critic/{key}', value, ep)
        for key in ['ema_collision_cost', 'ema_headway_cost']:
            value = float(row.get(key, float('nan')))
            if np.isfinite(value):
                writer.add_scalar(f'constraint_ema/{key}', value, ep)


def plot_train_diagnostics(rows: List[dict], save_path: str, mode: str = "", include_plot_grid: bool = True) -> str:
    if not save_path or len(rows) == 0:
        return ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = [int(row.get("episode", i)) for i, row in enumerate(rows)]

    if mode_is_constrained(mode):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = np.asarray(axes).reshape(3, 2)

        ax = axes[0, 0]
        ax.plot(episodes, _safe_float_list(rows, "episode_return"), marker="o")
        ax.set_title("Episode Return")
        ax.set_xlabel("Episode")
        ax.set_ylabel("episode_return")
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(episodes, _safe_float_list(rows, "success"), marker="o", label="success")
        ax.plot(episodes, _safe_float_list(rows, "collision"), marker="o", label="collision")
        ax.set_title("Success / Collision")
        ax.set_xlabel("Episode")
        ax.legend()
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(episodes, _safe_float_list(rows, "unsafe_headway_rate"), marker="o")
        ax.set_title("Unsafe Headway Rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("unsafe_headway_rate")
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(episodes, _safe_float_list(rows, "mean_cost_total"), marker="o")
        ax.set_title("Mean Cost Total")
        ax.set_xlabel("Episode")
        ax.set_ylabel("mean_cost_total")
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        ax.plot(episodes, _safe_float_list(rows, "lambda_collision"), marker="o", label="lambda_collision")
        ax.plot(episodes, _safe_float_list(rows, "lambda_headway"), marker="o", label="lambda_headway")
        ax.plot(episodes, _safe_float_list(rows, "lambda_overspeed"), marker="o", label="lambda_overspeed")
        ax.plot(episodes, _safe_float_list(rows, "lambda_comfort"), marker="o", label="lambda_comfort")
        ax.set_title("Constraint Lambdas")
        ax.set_xlabel("Episode")
        ax.legend()
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        ax.plot(episodes, _safe_float_list(rows, "qc_collision"), marker="o", label="qc_collision")
        ax.plot(episodes, _safe_float_list(rows, "qc_headway"), marker="o", label="qc_headway")
        ax.plot(episodes, _safe_float_list(rows, "qc_overspeed"), marker="o", label="qc_overspeed")
        ax.plot(episodes, _safe_float_list(rows, "qc_comfort"), marker="o", label="qc_comfort")
        ax.set_title("Cost Critic Means")
        ax.set_xlabel("Episode")
        ax.legend()
        if include_plot_grid:
            ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = np.asarray(axes).reshape(2, 2)

        ax = axes[0, 0]
        ax.plot(episodes, _safe_float_list(rows, "episode_return"), marker="o")
        ax.set_title("Episode Return")
        ax.set_xlabel("Episode")
        ax.set_ylabel("episode_return")
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(episodes, _safe_float_list(rows, "success"), marker="o", label="success")
        ax.plot(episodes, _safe_float_list(rows, "collision"), marker="o", label="collision")
        ax.set_title("Success / Collision")
        ax.set_xlabel("Episode")
        ax.legend()
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(episodes, _safe_float_list(rows, "unsafe_headway_rate"), marker="o")
        ax.set_title("Unsafe Headway Rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("unsafe_headway_rate")
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(episodes, _safe_float_list(rows, "mean_cost_total"), marker="o")
        ax.set_title("Mean Cost Total")
        ax.set_xlabel("Episode")
        ax.set_ylabel("mean_cost_total")
        if include_plot_grid:
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Train Diagnostics - {mode or 'run'}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"[DIAG] saved train diagnostics plot to {save_path}")
    return os.path.abspath(save_path)


def load_train_diagnostics_csv(csv_path: str) -> List[dict]:

    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k == "episode":
                    try:
                        parsed[k] = int(v)
                    except Exception:
                        parsed[k] = 0
                else:
                    try:
                        parsed[k] = float(v)
                    except Exception:
                        parsed[k] = float("nan")
            rows.append(parsed)
    return rows


def run(cfg):
    enforce_workflow_stage_policy(cfg, cfg.train.mode, is_compare=False)
    configure_llm_for_stage(cfg)
    reset_runtime_llm_log(cfg)
    set_seed(cfg.train.seed)
    print(
        f"[RUN-CONFIG] mode={cfg.train.mode} seed={cfg.train.seed} "
        f"device={describe_runtime_device(cfg.train.device)} "
        f"planner_interval={cfg.planner.planner_interval} "
        f"target_speed_free={cfg.planner.target_speed_free:.1f} "
        f"normal_headway={cfg.planner.normal_headway:.1f}"
    )
    print(f"[ABLATION] {cfg.ablation}")
    print(
        f"[WORKFLOW] stage={cfg.workflow.stage} "
        f"paper_eligible={int(cfg.workflow.paper_eligible)} "
        f"protocol_hash={cfg.workflow.protocol_hash or 'none'}"
    )

    env = HighwayEnvWrapper(cfg, render=cfg.train.render)
    verifier = SafetyVerifier(cfg) if cfg.safety.verifier_enabled else None
    shield = SafetyShield(cfg) if cfg.safety.shield_enabled_train else None

    state, _ = env.reset(seed=cfg.train.seed)
    state_dim = state.shape[0]
    action_dim = env.action_dim()
    agent = build_agent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)
    tb_writer, tb_run_dir = create_tensorboard_writer(cfg)
    if tb_run_dir:
        print(f"[DIAG] tensorboard run dir: {tb_run_dir}")

    total_steps = 0
    train_episode_results = []
    train_diagnostic_rows = []

    try:
        for ep in range(cfg.train.episodes):
            state, _ = env.reset(seed=cfg.train.seed + ep)
            planner = build_planner(cfg)
            metrics = EpisodeMetrics()
            last_update_info = {}

            if planner is not None:
                init_scene = env.get_scene_dict()
                plan = planner.plan(init_scene, step_idx=None)
                env.apply_plan(plan["waypoints"], plan["constraints"])
                print_plan_debug(cfg, plan, ep=ep, t=None, prefix="PLAN")

            for t in range(cfg.train.max_steps_per_episode):
                scene = env.get_scene_dict()
                replanned = False

                if planner is not None and need_replan(cfg, t, scene):
                    plan = planner.plan(scene, step_idx=t)
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
                    if update_info:
                        last_update_info = dict(update_info)

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
            train_diag_row = build_train_diagnostic_row(
                ep=ep,
                mode=cfg.train.mode,
                ep_summary=ep_summary,
                agent=agent,
                last_update_info=last_update_info,
            )
            train_diagnostic_rows.append(train_diag_row)
            write_train_diagnostics_tensorboard(tb_writer, train_diag_row, cfg.train.mode)
            print(
                f"[EPISODE END] ep={ep} return={ep_summary['episode_return']:.3f} "
                f"steps={ep_summary['steps']} collision={ep_summary['collision']} "
                f"success={ep_summary['success']} interventions={ep_summary['interventions']} "
                f"mean_cost_total={ep_summary['mean_cost_total']:.3f} "
                f"replay_size={agent.replay.size}"
            )
            print(
                f"[TRAIN-DIAG] ep={ep} "
                f"success={train_diag_row['success']:.0f} "
                f"collision={train_diag_row['collision']:.0f} "
                f"unsafe_headway_rate={train_diag_row['unsafe_headway_rate']:.3f} "
                f"mean_cost_total={train_diag_row['mean_cost_total']:.3f} "
                f"return={train_diag_row['episode_return']:.3f}"
                + (
                    f" lambda_col={train_diag_row['lambda_collision']:.4f}"
                    f" lambda_head={train_diag_row['lambda_headway']:.4f}"
                    f" lambda_over={train_diag_row['lambda_overspeed']:.4f}"
                    f" lambda_comf={train_diag_row['lambda_comfort']:.4f}"
                    f" qc_col={train_diag_row['qc_collision']:.4f}"
                    f" qc_head={train_diag_row['qc_headway']:.4f}"
                    f" qc_over={train_diag_row['qc_overspeed']:.4f}"
                    f" qc_comf={train_diag_row['qc_comfort']:.4f}"
                    if mode_is_constrained(cfg.train.mode) else ""
                )
            )
    except Exception as e:
        env.close()
        if tb_writer is not None:
            tb_writer.close()
        raise

    env.close()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    train_summary = summarize_eval_results(train_episode_results)
    train_log_path = ""
    train_json_path = ""
    train_plot_path = ""
    if cfg.diagnostics.export_train_log and cfg.diagnostics.train_log_path:
        train_log_path = export_train_diagnostics_csv(train_diagnostic_rows, cfg.diagnostics.train_log_path)
    if cfg.diagnostics.save_train_json and cfg.diagnostics.train_json_path:
        train_json_path = export_train_diagnostics_json(train_diagnostic_rows, cfg.diagnostics.train_json_path)
    if cfg.diagnostics.train_plot_path:
        train_plot_path = plot_train_diagnostics(
            train_diagnostic_rows,
            cfg.diagnostics.train_plot_path,
            mode=cfg.train.mode,
            include_plot_grid=bool(cfg.diagnostics.include_plot_grid),
        )

    result = {
        "mode": cfg.train.mode,
        "paper_eligible": bool(cfg.workflow.paper_eligible),
        "train_episodes": train_episode_results,
        "train_summary": train_summary,
        "eval": None,
        "config": asdict(cfg),
        "metadata": build_run_metadata(cfg, cfg.train.mode, cfg.train.seed, None, __file__, run_status="success"),
        "llm_call_log": get_llm_call_log(cfg),
        "llm_call_log_path": "",
        "train_diagnostics": train_diagnostic_rows,
        "train_diagnostics_csv_path": train_log_path,
        "train_diagnostics_json_path": train_json_path,
        "train_diagnostics_plot_path": train_plot_path,
        "train_tensorboard_dir": tb_run_dir,
    }
    if cfg.eval.enabled:
        result["eval"] = evaluate_policy(cfg, agent, mode=cfg.train.mode)
    llm_call_log_path = export_llm_call_log(cfg, cfg.llm.call_log_path, run_tag=f"{cfg.train.mode}_seed{cfg.train.seed}")
    result["llm_call_log_path"] = llm_call_log_path
    return result



# ============================================================
# compare module
# ============================================================

COMPARE_METRIC_SPECS = [
    ("episode_return", "return"),
    ("success", "success"),
    ("collision", "collision"),
    ("mean_speed", "mean_speed"),
    ("min_front_distance", "min_front_d"),
    ("interventions", "interventions"),
    ("unsafe_headway_rate", "unsafe_headway"),
    ("mean_cost_total", "cost_total"),
    ("replans", "replans"),
    ("elapsed_sec", "time_sec"),
]


def _extract_report_summary(exp_result: dict, report_variant: str) -> dict:
    report_variant = str(report_variant).lower()
    if exp_result.get("eval") is None:
        return exp_result.get("train_summary", {})
    eval_payload = exp_result["eval"]
    report_block = eval_payload.get(report_variant)
    if report_block is not None and report_block.get("summary"):
        return report_block["summary"]
    return eval_payload.get("primary", exp_result.get("train_summary", {}))


def _metric_from_result(exp_result: dict, key: str, report_variant: str) -> float:
    if key == "elapsed_sec":
        return float(exp_result.get("elapsed_sec", 0.0))
    summary = _extract_report_summary(exp_result, report_variant)
    return float(summary.get(key, 0.0))


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.3f}"


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{float(mean):.3f}±{float(std):.3f}"



def resolve_compare_seeds(cfg) -> List[int]:
    stage = str(cfg.workflow.stage).lower()
    if stage == "formal":
        return _normalize_seed_list(cfg.freeze_protocol.formal_seeds, [142, 242, 342])
    if len(cfg.compare.explicit_seeds) > 0:
        return _normalize_seed_list(cfg.compare.explicit_seeds, cfg.freeze_protocol.dev_seeds)
    if stage in {"dev", "freeze"}:
        return _normalize_seed_list(cfg.freeze_protocol.dev_seeds, [cfg.train.seed])
    return [int(cfg.train.seed + i * cfg.compare.seed_stride) for i in range(max(1, cfg.compare.num_seeds))]


def build_compare_run_row(exp_result: dict, report_variant: str) -> dict:
    row = {
        "mode": exp_result["mode"],
        "seed": int(exp_result.get("seed", -1)),
        "report": str(report_variant),
    }
    for metric_key, alias in COMPARE_METRIC_SPECS:
        row[alias] = float(_metric_from_result(exp_result, metric_key, report_variant))
    return row


def aggregate_compare_rows(mode: str, run_rows: List[dict], report_variant: str) -> dict:
    agg = {
        "mode": mode,
        "report": str(report_variant),
        "num_runs": len(run_rows),
        "seeds": [int(r["seed"]) for r in run_rows],
    }
    for _, alias in COMPARE_METRIC_SPECS:
        values = np.array([float(r[alias]) for r in run_rows], dtype=np.float32)
        mean = float(values.mean()) if len(values) > 0 else 0.0
        std = float(values.std()) if len(values) > 0 else 0.0
        agg[f"{alias}_mean"] = mean
        agg[f"{alias}_std"] = std
        agg[alias] = _fmt_mean_std(mean, std)
    return agg


def format_compare_table(aggregated_rows: List[dict], title: str) -> str:
    headers = [
        "mode",
        "runs",
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
    for r in aggregated_rows:
        rows.append([
            r["mode"],
            str(int(r.get("num_runs", 0))),
            r["return"],
            r["success"],
            r["collision"],
            r["mean_speed"],
            r["min_front_d"],
            r["interventions"],
            r["unsafe_headway"],
            r["cost_total"],
            r["replans"],
            r["time_sec"],
        ])

    widths = []
    for col_idx, header in enumerate(headers):
        max_len = len(header)
        for row in rows:
            max_len = max(max_len, len(str(row[col_idx])))
        widths.append(max_len)

    out = [title]
    line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    out.extend([line, sep])
    for row in rows:
        out.append(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def merge_compare_reports(raw_rows: List[dict], shielded_rows: List[dict]) -> List[dict]:
    shielded_map = {row["mode"]: row for row in shielded_rows}
    merged = []
    for raw in raw_rows:
        shield = shielded_map.get(raw["mode"], {})
        merged.append({
            "mode": raw["mode"],
            "num_runs": int(raw.get("num_runs", 0)),
            "seeds": raw.get("seeds", []),
            "raw_return": raw.get("return", "0.000±0.000"),
            "raw_success": raw.get("success", "0.000±0.000"),
            "raw_collision": raw.get("collision", "0.000±0.000"),
            "raw_mean_speed": raw.get("mean_speed", "0.000±0.000"),
            "raw_unsafe_headway": raw.get("unsafe_headway", "0.000±0.000"),
            "shield_success": shield.get("success", "0.000±0.000"),
            "shield_collision": shield.get("collision", "0.000±0.000"),
            "shield_interventions": shield.get("interventions", "0.000±0.000"),
            "shield_cost_total": shield.get("cost_total", "0.000±0.000"),
            "time_sec": raw.get("time_sec", shield.get("time_sec", "0.000±0.000")),
        })
    return merged


def format_combined_compare_table(merged_rows: List[dict]) -> str:
    headers = [
        "mode",
        "runs",
        "raw_return",
        "raw_success",
        "raw_collision",
        "raw_speed",
        "raw_unsafe",
        "shield_success",
        "shield_collision",
        "shield_interv",
        "shield_cost",
        "time_sec",
    ]
    body_rows = []
    for row in merged_rows:
        body_rows.append([
            row["mode"],
            str(int(row["num_runs"])),
            row["raw_return"],
            row["raw_success"],
            row["raw_collision"],
            row["raw_mean_speed"],
            row["raw_unsafe_headway"],
            row["shield_success"],
            row["shield_collision"],
            row["shield_interventions"],
            row["shield_cost_total"],
            row["time_sec"],
        ])
    widths = []
    for col_idx, header in enumerate(headers):
        max_len = len(header)
        for row in body_rows:
            max_len = max(max_len, len(str(row[col_idx])))
        widths.append(max_len)
    out = ["COMBINED RAW / SHIELDED TABLE"]
    line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    out.extend([line, sep])
    for row in body_rows:
        out.append(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def export_compare_csv(merged_rows: List[dict], save_path: str) -> None:
    if not save_path:
        return
    fieldnames = [
        "mode", "num_runs", "seeds",
        "raw_return", "raw_success", "raw_collision", "raw_mean_speed", "raw_unsafe_headway",
        "shield_success", "shield_collision", "shield_interventions", "shield_cost_total", "time_sec",
    ]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({
                "mode": row["mode"],
                "num_runs": int(row["num_runs"]),
                "seeds": ",".join(str(s) for s in row.get("seeds", [])),
                "raw_return": row["raw_return"],
                "raw_success": row["raw_success"],
                "raw_collision": row["raw_collision"],
                "raw_mean_speed": row["raw_mean_speed"],
                "raw_unsafe_headway": row["raw_unsafe_headway"],
                "shield_success": row["shield_success"],
                "shield_collision": row["shield_collision"],
                "shield_interventions": row["shield_interventions"],
                "shield_cost_total": row["shield_cost_total"],
                "time_sec": row["time_sec"],
            })
    print(f"\n[COMPARE] saved csv to {save_path}")


def _latex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def export_compare_latex(merged_rows: List[dict], save_path: str) -> str:
    if not save_path:
        return ""
    headers = [
        ("mode", "Mode"),
        ("num_runs", "Runs"),
        ("raw_return", "RawReturn"),
        ("raw_success", "RawSuccess"),
        ("raw_collision", "RawCollision"),
        ("raw_mean_speed", "RawSpeed"),
        ("shield_success", "ShieldSuccess"),
        ("shield_collision", "ShieldCollision"),
        ("shield_interventions", "ShieldInterv"),
        ("shield_cost_total", "ShieldCost"),
        ("time_sec", "TimeSec"),
    ]
    lines = []
    lines.append("\\begin{tabular}{lrrrrrrrrrr}")
    lines.append("\\hline")
    lines.append(" & ".join(display for _, display in headers) + " \\")
    lines.append("\\hline")
    for row in merged_rows:
        values = []
        for key, _ in headers:
            if key == "mode":
                values.append(_latex_escape(str(row[key])))
            elif key == "num_runs":
                values.append(str(int(row[key])))
            else:
                values.append(_latex_escape(str(row[key])))
        lines.append(" & ".join(values) + " \\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    latex_str = "\n".join(lines)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"\n[COMPARE] saved latex to {save_path}")
    return latex_str



def run_compare(cfg, modes: Optional[List[str]] = None) -> dict:
    enforce_workflow_stage_policy(cfg, "compare", is_compare=True)
    configure_llm_for_stage(cfg)
    stage = str(cfg.workflow.stage).lower()
    if stage == "formal":
        modes = list(cfg.freeze_protocol.main_table_modes)
        ablation_modes = list(cfg.freeze_protocol.ablation_modes)
        cfg.eval.report_raw_and_shielded = True
        cfg.eval.primary_report = "raw"
    else:
        modes = modes or cfg.compare.modes
        ablation_modes = list(cfg.compare.ablation_modes)

    def _sanitize_modes(mode_list: List[str]) -> List[str]:
        out = []
        for mode in mode_list:
            m = str(mode).strip()
            if m in ALL_MODES and m not in out:
                out.append(m)
        return out

    mode_groups = {
        "main": _sanitize_modes(list(modes)),
        "ablation": _sanitize_modes(list(ablation_modes)),
    }
    seeds = resolve_compare_seeds(cfg)
    raw_results = []
    grouped_payloads = {}
    merged_rows_all = []
    failed_runs = []

    print("\n================ COMPARE START ================\n")
    print(f"[COMPARE] stage={cfg.workflow.stage}")
    print(f"[COMPARE] device={describe_runtime_device(cfg.train.device)}")
    print(f"[COMPARE] mode_groups={mode_groups}")
    print(f"[COMPARE] seeds={seeds}")

    for group_name in ["main", "ablation"]:
        group_modes = mode_groups.get(group_name, [])
        if not group_modes:
            continue
        group_raw_rows = []
        group_shield_rows = []
        agg_raw_rows = []
        agg_shield_rows = []
        mode_success_counts = {}
        print(f"\n================ COMPARE GROUP: {group_name.upper()} START ================\n")
        for idx, mode in enumerate(group_modes):
            mode_raw_rows = []
            mode_shield_rows = []
            mode_success_counts[mode] = 0
            print(f"[COMPARE][GROUP={group_name}][{idx + 1}/{len(group_modes)}] run mode={mode}")
            for seed_idx, seed in enumerate(seeds):
                exp_cfg = copy.deepcopy(cfg)
                exp_cfg.train.mode = mode
                exp_cfg.train.seed = int(seed)
                exp_cfg.eval.report_raw_and_shielded = True
                exp_cfg.eval.primary_report = str(exp_cfg.compare.primary_report).lower()
                if exp_cfg.compare.disable_step_print_during_compare:
                    exp_cfg.train.print_every_step = False
                try:
                    print(f"[COMPARE][GROUP={group_name}][RUN] mode={mode} seed={seed} ({seed_idx + 1}/{len(seeds)})")
                    start = time.time()
                    result = run(exp_cfg)
                    elapsed = time.time() - start
                    result["elapsed_sec"] = float(elapsed)
                    result["seed"] = int(seed)
                    result["run_index"] = int(seed_idx)
                    result["group"] = str(group_name)
                    result.setdefault("metadata", {})["run_status"] = "success"
                    raw_results.append(result)
                    raw_row = build_compare_run_row(result, report_variant="raw")
                    shield_row = build_compare_run_row(result, report_variant="shielded")
                    group_raw_rows.append(raw_row)
                    group_shield_rows.append(shield_row)
                    mode_raw_rows.append(raw_row)
                    mode_shield_rows.append(shield_row)
                    mode_success_counts[mode] += 1
                    print(
                        f"[COMPARE][GROUP={group_name}][DONE-RUN] mode={mode} seed={seed} "
                        f"raw_success={raw_row['success']:.3f} shield_success={shield_row['success']:.3f} "
                        f"raw_collision={raw_row['collision']:.3f} shield_collision={shield_row['collision']:.3f} "
                        f"time_sec={raw_row['time_sec']:.2f}"
                    )
                except Exception as e:
                    failed = {
                        "group": str(group_name),
                        "mode": str(mode),
                        "seed": int(seed),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "metadata": build_run_metadata(exp_cfg, mode, int(seed), None, __file__, run_status="failed", failure_reason=str(e)),
                    }
                    failed_runs.append(failed)
                    print(f"[COMPARE][GROUP={group_name}][FAIL-RUN] mode={mode} seed={seed} {type(e).__name__}: {e}")
            if mode_raw_rows:
                agg_raw = aggregate_compare_rows(mode, mode_raw_rows, report_variant="raw")
                agg_shield = aggregate_compare_rows(mode, mode_shield_rows, report_variant="shielded")
                agg_raw_rows.append(agg_raw)
                agg_shield_rows.append(agg_shield)
        raw_table = format_compare_table(agg_raw_rows, title=f"{group_name.upper()} RAW POLICY TABLE")
        shielded_table = format_compare_table(agg_shield_rows, title=f"{group_name.upper()} SHIELDED POLICY TABLE")
        merged_rows = merge_compare_reports(agg_raw_rows, agg_shield_rows)
        for row in merged_rows:
            row["group"] = str(group_name)
        combined_table = format_combined_compare_table(merged_rows)
        grouped_payloads[group_name] = {
            "group": str(group_name),
            "modes": [str(m) for m in group_modes],
            "aggregated_raw_rows": agg_raw_rows,
            "aggregated_shielded_rows": agg_shield_rows,
            "merged_rows": merged_rows,
            "raw_table": raw_table,
            "shielded_table": shielded_table,
            "combined_table": combined_table,
            "successful_runs_per_mode": mode_success_counts,
        }
        merged_rows_all.extend(merged_rows)
        print(raw_table)
        print()
        print(shielded_table)
        print()
        print(combined_table)

    main_payload = grouped_payloads.get("main", {})
    compare_valid = True
    invalid_reasons = []
    min_success = int(cfg.compare.min_successful_runs_per_mode)
    if stage == "formal":
        for mode in main_payload.get("modes", []):
            success_count = int(main_payload.get("successful_runs_per_mode", {}).get(mode, 0))
            if success_count < min_success:
                compare_valid = False
                invalid_reasons.append(
                    f"mode {mode} successful_runs={success_count} < required {min_success}"
                )
        if cfg.compare.require_all_modes_success and failed_runs:
            compare_valid = False
            invalid_reasons.append("formal compare has failed runs")

    payload = {
        "seeds": seeds,
        "paper_eligible": bool(cfg.workflow.paper_eligible),
        "config": asdict(cfg),
        "metadata": build_result_metadata(cfg, run_type="compare", seeds=seeds),
        "groups": grouped_payloads,
        "raw_results": raw_results,
        "merged_rows": merged_rows_all,
        "failed_runs": failed_runs,
        "compare_valid": bool(compare_valid),
        "invalid_reasons": list(invalid_reasons),
        "raw_table": main_payload.get("raw_table", ""),
        "shielded_table": main_payload.get("shielded_table", ""),
        "combined_table": main_payload.get("combined_table", ""),
        "latex_table": "",
    }
    if cfg.compare.save_json_path:
        os.makedirs(os.path.dirname(cfg.compare.save_json_path) or ".", exist_ok=True)
        with open(cfg.compare.save_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[COMPARE] saved json to {cfg.compare.save_json_path}")
    export_compare_csv(merged_rows_all, cfg.compare.save_csv_path)
    payload["latex_table"] = export_compare_latex(merged_rows_all, cfg.compare.save_latex_path)
    if stage == "formal" and not compare_valid:
        raise RuntimeError("Formal compare invalid: " + "; ".join(invalid_reasons))
    return payload



# ============================================================
# CLI

# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-stage", type=str, default="dev", choices=["dev", "freeze", "formal"])
    parser.add_argument("--freeze-save", type=str, default="frozen_protocol.json")
    parser.add_argument("--freeze-load", type=str, default="")
    parser.add_argument("--formal-strict", action="store_true")
    parser.add_argument("--allow-real-llm-smoke", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--formal-modes", type=str, default="")
    parser.add_argument("--formal-ablation-modes", type=str, default="")
    parser.add_argument("--dev-seeds", type=str, default="")
    parser.add_argument("--formal-seeds", type=str, default="")
    parser.add_argument("--min-successful-runs-per-mode", type=int, default=1)

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
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--train-log-path", type=str, default="")
    parser.add_argument("--train-json-path", type=str, default="")
    parser.add_argument("--train-plot-path", type=str, default="")
    parser.add_argument("--tensorboard-dir", type=str, default="")
    parser.add_argument("--tensorboard-flush-secs", type=int, default=10)
    parser.add_argument("--plot-diagnostics-from-csv", type=str, default="")
    parser.add_argument("--plot-diagnostics-to", type=str, default="")
    parser.add_argument("--plot-diagnostics-mode", type=str, default="")
    parser.add_argument("--disable-diagnostic-plot-grid", action="store_true")

    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--eval-print-step", action="store_true")
    parser.add_argument("--disable-dual-eval-report", action="store_true")
    parser.add_argument("--eval-primary-report", type=str, default="raw", choices=["raw", "shielded"])
    parser.add_argument("--print-prompt-on-replan", action="store_true")

    parser.add_argument("--shield-train", action="store_true")
    parser.add_argument("--disable-shield-eval", action="store_true")

    parser.add_argument("--llm-backend", type=str, default=str(USER_EDITABLE_LLM["backend"]), choices=["mock", "real"])
    parser.add_argument("--llm-model", type=str, default=str(USER_EDITABLE_LLM["model_name"]))
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-timeout", type=int, default=20)
    parser.add_argument("--llm-retry-times", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=str, default="1,3,5")
    parser.add_argument("--llm-call-log-path", type=str, default="")
    parser.add_argument("--disable-llm-fallback", action="store_true")

    parser.add_argument("--compare-modes", type=str, default="")
    parser.add_argument("--compare-ablation-modes", type=str, default="")
    parser.add_argument("--compare-seeds", type=str, default="")
    parser.add_argument("--compare-num-seeds", type=int, default=1)
    parser.add_argument("--compare-seed-stride", type=int, default=100)
    parser.add_argument("--compare-use-shield-eval", action="store_true")
    parser.add_argument("--compare-primary-report", type=str, default="raw", choices=["raw", "shielded"])
    parser.add_argument("--compare-save-json", type=str, default="")
    parser.add_argument("--compare-save-csv", type=str, default="")
    parser.add_argument("--compare-save-latex", type=str, default="")

    parser.add_argument("--lambda-ema-beta", type=float, default=0.97)
    parser.add_argument("--min-lambda-collision", type=float, default=0.010)
    parser.add_argument("--min-lambda-headway", type=float, default=0.010)
    parser.add_argument("--collision-risk-ttc-threshold", type=float, default=4.0)
    parser.add_argument("--collision-risk-front-distance", type=float, default=10.0)

    parser.add_argument("--ablate-waypoint-features", action="store_true")
    parser.add_argument("--ablate-constraint-features", action="store_true")
    parser.add_argument("--ablate-waypoint-reward", action="store_true")
    parser.add_argument("--ablate-constraint-costs", action="store_true")
    parser.add_argument("--ablate-lane-stabilization", action="store_true")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    if args.plot_diagnostics_from_csv.strip():
        plot_rows = load_train_diagnostics_csv(args.plot_diagnostics_from_csv)
        plot_save_path = args.plot_diagnostics_to.strip() or "train_diagnostics.png"
        plot_train_diagnostics(
            plot_rows,
            plot_save_path,
            mode=str(args.plot_diagnostics_mode).strip(),
            include_plot_grid=not bool(args.disable_diagnostic_plot_grid),
        )
        sys.exit(0)

    cfg = get_config()
    # Centralized forwarding API settings: edit USER_EDITABLE_LLM near the top of this file.
    cfg.llm.backend = str(USER_EDITABLE_LLM["backend"]).lower()
    cfg.llm.model_name = str(USER_EDITABLE_LLM["model_name"])
    cfg.llm.api_base = str(USER_EDITABLE_LLM["api_base"])
    cfg.llm.api_key = str(USER_EDITABLE_LLM["api_key"])
    cfg.workflow.stage = str(args.workflow_stage).lower()
    cfg.workflow.freeze_save_path = args.freeze_save
    cfg.workflow.freeze_load_path = args.freeze_load
    cfg.workflow.formal_strict = bool(args.formal_strict)
    cfg.workflow.allow_real_llm_smoke = bool(args.allow_real_llm_smoke)
    cfg.workflow.smoke_test = bool(args.smoke_test)

    cfg.train.mode = args.mode if args.mode != "compare" else cfg.train.mode
    cfg.train.episodes = args.episodes
    cfg.train.max_steps_per_episode = args.max_steps
    cfg.train.render = args.render
    cfg.train.seed = args.seed
    cfg.train.device = resolve_runtime_device(args.device)
    configure_torch_runtime(cfg.train.device)
    cfg.train.use_shield_during_train = args.shield_train

    cfg.diagnostics.export_train_log = bool(args.train_log_path.strip())
    cfg.diagnostics.train_log_path = args.train_log_path
    cfg.diagnostics.save_train_json = bool(args.train_json_path.strip())
    cfg.diagnostics.train_json_path = args.train_json_path
    cfg.diagnostics.train_plot_path = args.train_plot_path
    cfg.diagnostics.include_plot_grid = not bool(args.disable_diagnostic_plot_grid)
    cfg.diagnostics.tensorboard_log_dir = args.tensorboard_dir
    cfg.diagnostics.tensorboard_flush_secs = max(1, int(args.tensorboard_flush_secs))

    cfg.eval.episodes = args.eval_episodes
    cfg.eval.enabled = not args.disable_eval
    cfg.eval.print_step = args.eval_print_step
    cfg.eval.report_raw_and_shielded = not args.disable_dual_eval_report
    cfg.eval.primary_report = args.eval_primary_report
    cfg.planner.print_prompt_on_replan = args.print_prompt_on_replan

    cfg.safety.shield_enabled_eval = not args.disable_shield_eval
    cfg.safety.shield_enabled_train = args.shield_train

    cfg.llm.backend = str(args.llm_backend).lower() if str(args.llm_backend).strip() else cfg.llm.backend
    cfg.llm.model_name = str(args.llm_model).strip() if str(args.llm_model).strip() else cfg.llm.model_name
    cfg.llm.temperature = float(args.llm_temperature)
    cfg.llm.timeout_sec = args.llm_timeout
    cfg.llm.retry_times = max(1, int(args.llm_retry_times))
    cfg.llm.retry_backoff_sec = [float(x) for x in args.llm_retry_backoff.split(",") if x.strip()]
    cfg.llm.call_log_path = args.llm_call_log_path
    cfg.llm.fallback_to_mock_on_error = not args.disable_llm_fallback
    cfg.llm.fallback_to_mock_on_parse_error = not args.disable_llm_fallback

    cfg.sac.lambda_ema_beta = float(args.lambda_ema_beta)
    cfg.sac.min_lambda_collision = float(args.min_lambda_collision)
    cfg.sac.min_lambda_headway = float(args.min_lambda_headway)
    cfg.safety.collision_risk_ttc_threshold = float(args.collision_risk_ttc_threshold)
    cfg.safety.collision_risk_front_distance = float(args.collision_risk_front_distance)

    if args.compare_modes.strip():
        cfg.compare.modes = [m.strip() for m in args.compare_modes.split(",") if m.strip()]
    if args.compare_ablation_modes.strip():
        cfg.compare.ablation_modes = [m.strip() for m in args.compare_ablation_modes.split(",") if m.strip()]
    if args.compare_seeds.strip():
        cfg.compare.explicit_seeds = [int(s.strip()) for s in args.compare_seeds.split(",") if s.strip()]
    cfg.compare.num_seeds = max(1, args.compare_num_seeds)
    cfg.compare.seed_stride = max(1, args.compare_seed_stride)
    cfg.compare.use_shield_in_eval = args.compare_use_shield_eval
    cfg.compare.primary_report = args.compare_primary_report
    cfg.compare.save_json_path = args.compare_save_json
    cfg.compare.save_csv_path = args.compare_save_csv
    cfg.compare.save_latex_path = args.compare_save_latex
    cfg.compare.min_successful_runs_per_mode = max(1, int(args.min_successful_runs_per_mode))

    if args.formal_modes.strip():
        cfg.freeze_protocol.main_table_modes = [m.strip() for m in args.formal_modes.split(",") if m.strip()]
        cfg.compare.formal_modes = list(cfg.freeze_protocol.main_table_modes)
    if args.formal_ablation_modes.strip():
        cfg.freeze_protocol.ablation_modes = [m.strip() for m in args.formal_ablation_modes.split(",") if m.strip()]
    if args.dev_seeds.strip():
        cfg.freeze_protocol.dev_seeds = [int(s.strip()) for s in args.dev_seeds.split(",") if s.strip()]
    if args.formal_seeds.strip():
        cfg.freeze_protocol.formal_seeds = [int(s.strip()) for s in args.formal_seeds.split(",") if s.strip()]
    cfg.freeze_protocol.min_successful_runs_per_mode = max(1, int(args.min_successful_runs_per_mode))
    cfg.freeze_protocol.primary_report = "raw"
    cfg.freeze_protocol.report_raw_and_shielded = True
    cfg.freeze_protocol.llm_retry_times = cfg.llm.retry_times
    cfg.freeze_protocol.llm_retry_backoff_sec = list(cfg.llm.retry_backoff_sec)

    cfg.ablation.disable_waypoint_features = args.ablate_waypoint_features
    cfg.ablation.disable_constraint_features = args.ablate_constraint_features
    cfg.ablation.disable_waypoint_reward = args.ablate_waypoint_reward
    cfg.ablation.disable_constraint_costs = args.ablate_constraint_costs
    cfg.ablation.disable_lane_stabilization = args.ablate_lane_stabilization

    initialize_workflow(cfg, run_mode=args.mode)
    if cfg.workflow.stage == "freeze":
        sys.exit(0)

    if args.mode == "compare":
        run_compare(cfg, modes=cfg.compare.modes)
    else:
        run(cfg)
