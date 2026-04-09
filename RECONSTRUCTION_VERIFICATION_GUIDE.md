# 🔧 Phase 3A 重构修复验证指南 (v1.1)

**修复时间**: 2026-03-18  
**原因**: 22轮训练后Reward恶化 (70→-50) 的根本原因修复

---

## 📌 修复内容速览

### 3个核心修复

```
修复1: 温度参数恢复 ⭐⭐⭐ (最关键)
  TRAJ_QUERY_TEMPERATURE: 0.25 → 0.8
  效果: 模式分布从[0.95,0.03,0.01,0.01] → [0.35,0.25,0.20,0.20]

修复2: 轨迹参数规范化 ⭐⭐⭐
  TRAJ_GUIDE_ACCEL_REF: 20.0 → 12.0  (参考点移到中点)
  TRAJ_GUIDE_ACCEL_NORM: 12.0 → 16.0 (范围扩大,避免饱和)
  效果: guide_acc从恒-0.83 → 均匀分布[-0.375, +0.125]

修复3: 权重回归基线 ⭐⭐
  TRAJ_QUERY_COUNT: 5→4 (去除质量差的模式)
  TRAJ_GUIDE_ONLINE_BLEND: 0.28→0.18 (权重回到0.18)
  TRAJ_GUIDE_LOSS_WEIGHT: 3.5→2.0 (约束回到2.0)
  效果: 策略恢复自由度,学习正确的加速/减速决策
```

---

## 🚀 立即启动

```bash
# 方式1: 从头开始验证修复 (推荐)
python main.py

# 方式2: 从最后一个checkpoint (如果有)
# python main.py  # 自动加载checkpoints/hybrid_sac_ep22.pth

# 在另一个终端监控
tensorboard --logdir=./runs
# 浏览器访问: http://localhost:6006
```

---

## ✅ 修复验证清单

### 第0步: 参数检查 (启动前)

```python
# 检查参数是否正确
import sys
sys.path.insert(0, 'd:/Program Files (x86)/paper/SafeDrive-main')
from main import SystemConfig

config = SystemConfig()
print(f"TRAJ_QUERY_TEMPERATURE: {config.TRAJ_QUERY_TEMPERATURE}")          # 应该是 0.8
print(f"TRAJ_QUERY_COUNT: {config.TRAJ_QUERY_COUNT}")                      # 应该是 4
print(f"TRAJ_GUIDE_ACCEL_REF: {config.TRAJ_GUIDE_ACCEL_REF}")              # 应该是 12.0
print(f"TRAJ_GUIDE_ACCEL_NORM: {config.TRAJ_GUIDE_ACCEL_NORM}")            # 应该是 16.0
```

### 第1步: 第1轮检查 (启动后监控5分钟)

| 检查项 | 期望 | 状态 |
|--------|------|------|
| 无Cuda错误 | ✓ | ⏳ |
| 无形状错误 | ✓ | ⏳ |
| Episode 1开始 | ✓ | ⏳ |
| 初始Reward | ~-10 | ⏳ |

**TensorBoard指标**:
```
Mean_Guide_Accel:
  预期: 急剧上升从-0.27 → ~0.0±0.1
  这表明轨迹参数化修复成功
  
Reward:
  预期: 从-50快速恢复到-10~-20
  如果继续下降,说明修复不足
```

### 第5轮检查 (关键评估点)

```
检查项                预期                状态
─────────────────────────────────────────────────
Mean_Guide_Accel    -0.05 ~ +0.05      [ ]
Reward              -15 ± 5            [ ]
Crash_Rate          0% (保持)          [ ]
Near_Miss_Rate      保持改善趋势        [ ]
Action_Smoothness   0.15-0.22          [ ]
Uncertainty_Mean    < 0.15             [ ]
```

**判断标准**:
- ✅ 通过: 所有3个✓ OR Reward>-20
- 🟡 部分: 2个✓ 但Reward在-20~-30
- ❌ 失败: Reward仍<-40

### 第15轮最终验收

```
指标              v1.0基线    v1.1恢复目标   判断
─────────────────────────────────────────────────
Reward           -15~-20      -15~-20      ✓ 匹配
Min_TTC          ≥2.0s        ≥2.0s        ✓ 匹配
Crash_Rate       0%           0%           ✓ 匹配
Action_Smooth    0.18-0.22    0.18-0.25    ✓ 匹配
Uncertainty      0.08-0.15    0.08-0.15    ✓ 匹配
```

---

## 🔍 关键指标详解

### 1. Mean_Guide_Accel (诊断轨迹参数化)

```
含义: 平均轨迹建议的加速度成分

修复前:
  ├─ 值: -0.27 (恒定负值)
  └─ 问题: 轨迹始终建议减速

修复后:
  ├─ 值: -0.05 ~ +0.05 (正常分布)
  └─ 说明: 轨迹均衡建议加速/减速
  
验证方法:
  □ 第1轮: 应从-0.27快速上升
  □ 应在-0.1到+0.1之间波动
  □ 均值应接近0 (无偏差)
```

### 2. Reward (诊断整体性能)

```
修复前: -50 (完全失败)
修复后目标: -10 ~ -20 (恢复正常)

恢复曲线:
  Epoch  1: -50 → -30 (大幅改善)
  Epoch  5: -30 → -15 (继续恢复)
  Epoch 15: -15 ± 3 (稳定)

如果卡在-35以下,说明:
  ① 温度修复不足
  ② 参数化仍有问题
  ③ 需要应用微调1
```

### 3. Crash_Rate (验证安全性)

```
期望: 始终为 0% (不能出现碰撞)
如果Crash_Rate > 0:
  说明参数可能过于激进
  应检查REVIEW_MIN_TTC是否被应用
```

---

## 🎯 微调路线 (如果修复不完全)

### 微调A: 若Reward仍在-35以下

```python
进一步提升温度:
  TRAJ_QUERY_TEMPERATURE: 0.8 → 0.9
  
或调整权重:
  TRAJ_GUIDE_ONLINE_BLEND: 0.18 → 0.15
  TRAJ_GUIDE_LOSS_WEIGHT: 2.0 → 1.5
```

### 微调B: 若Mean_Guide_Accel未改善

```python
检查参数是否应用:
  guide_acc = (first_wp[:, 0] - 12.0) / 16.0
  
测试guide_accel范围:
  dx_min=6:  (6-12)/16 = -0.375  ✓
  dx_mid=12: (12-12)/16 = 0.0   ✓
  dx_max=14: (14-12)/16 = +0.125 ✓
```

### 微调C: 若不稳定波动

```python
增加batch size:
  BATCH_SIZE: 128 → 256
  
降低学习率:
  ACTOR_LR: 3e-4 → 1e-4
```

---

## 📊 监控dashboard设置

### TensorBoard重点关注 (6个曲线)

```
第1行:
├─ Episode/Reward (红线,应从-50→-15)
├─ Mean_Guide_Accel (蓝线,应从-0.27→0)
└─ Crash_Rate (绿线,应保持0%)

第2行:
├─ Loss/TrajectoryConsistency (应稳定<0.01)
├─ Uncertainty_Mean (应<0.15)
└─ Action_Smoothness (应0.18-0.22)
```

### 快速诊断技巧

```
症状1: Reward横盘不动
  → 温度可能仍需调整
  → 尝试微调A

症状2: Mean_Guide_Accel仍为负
  → 参数化没有生效
  → 检查代码第1125行是否正确

症状3: Action_Smoothness恶化
  → TRAJ_GUIDE_LOSS_WEIGHT可能太高
  → 降低到1.5试试

症状4: Crash_Rate > 0
  → 安全参数太激进
  → 恢复REVIEW_MIN_TTC到2.2
```

---

## 💾 版本管理

### 保存快照

```bash
# 修复前备份 (已有ep22)
cp checkpoints/hybrid_sac_ep22.pth backups/v2.0_broken_ep22.pth

# 修复后从新baseline开始 (推荐)
rm -rf checkpoints/*.pth  # 清空旧checkpoint
python main.py  # 开始训练新序列

# 记录版本
echo "v1.1修复: 温度0.8,参数化12.0/16.0,权重0.18/2.0" > VERSION.txt
```

---

## 📋 最终检查表

```
启动前:
□ main.py 已保存
□ 后有25K行代码
□ 无Pylance错误 (已验证)
□ 参数值已确认正确

启动后5分钟:
□ 无Cuda crash
□ Episode 1正常运行
□ Mean_Guide_Accel开始上升

第5轮评估:
□ Reward > -20
□ Mean_Guide_Accel ∈ [-0.1, 0.1]
□ Crash_Rate = 0%

第15轮验收:
□ 所有指标恢复正常
□ 无新问题出现
□ 已准备Phase 3B集成
```

---

## 🎓 理论回顾

**核心修复原理**:

```
三个参数的作用:

1. 温度T (softmax多样性):
   低T(0.25) → 一个模式被完全选中 → 失败
   高T(0.8)  → 4个模式均衡选中 → 成功

2. 参考值R (轨迹0点):
   错误R(20) → (10-20)/12 = -0.83 (太负)
   正确R(12) → (12-12)/16 = 0.0  (中立)

3. 权重W (引导影响):
   高W(0.28) → 坏指导权力大 → 策略被带坏
   低W(0.18) → 轨迹是参考 → 策略自由学习
```

---

## 📞 快速问题排查

| Q | A |
|---|---|
| 启动后仍然报错? | 检查TRAJ_QUERY_COUNT是否为4,确保anchor_bank只有4个 |
| Mean_Guide_Accel没变化? | 确认L1125的公式中参考值是12.0,范数是16.0 |
| Reward继续下降? | 尝试微调A,或检查是否有其他参数被意外改过 |
| 总是Crash_Rate>0? | 检查REVIEW_MIN_TTC(应为2.0) |
| 资源不足? | 降低BATCH_SIZE: 128 → 64 |

---

**预期**: 启动后第1-5轮应看到明显改善,Mean_Guide_Accel从-0.27→0.0,Reward从-50→-15。

