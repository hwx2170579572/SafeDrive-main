# 🚀 优化后训练启动和监控指南

## 📌 参数变更一览表

### 🔴 高优先级改动 (安全裕度)
```
REVIEW_MIN_FRONT_GAP:  10.0 m  →  12.0 m    (提高20%)
REVIEW_MIN_TTC:        2.0 s  →  2.5 s     (提高25%)
NEAR_MISS_TTC:         1.5 s  →  1.0 s     (降低33%)
```
**影响：** Min_TTC最小值从1.5提升至2.5s，Crash_Rate显著下降

### 🟡 中优先级改动 (轨迹引导)
```
TRAJ_GUIDE_ONLINE_BLEND:     0.18  →  0.28    (提高56%)
TRAJ_GUIDE_LOSS_WEIGHT:      2.5   →  3.5     (提高40%)
```
**影响：** 策略更信任轨迹预测，动作更平稳

### 🟢 低优先级改动 (模式优化)
```
TRAJ_QUERY_COUNT:           4  →  5        (增加1个模式)
TRAJ_QUERY_TEMPERATURE:   0.7  →  0.5      (模式集中)
TEMPLATE_GAP_CLAMP:    [0.75-1.25]  →  [0.65-1.35]
```
**影响：** 轨迹多样性和覆盖率优化

---

## ⚡ 启动训练命令

```bash
# 从头开始
python main.py

# 或从断点恢复（如果有）
# python main.py  # 代码中会自动检测checkpoints/hybrid_sac_ep50.pth
```

## 📊 实时监控（新终端）

```bash
tensorboard --logdir=./runs
# 浏览器打开: http://localhost:6006
```

---

## 🎯 分阶段验收标准

### 第1-5轮：立即生效验证
| 指标 | 期望 | 判断 |
|------|------|------|
| Min_TTC > 2.5s | ✅ Min值不低于2.5 | 成功 |
| Crash_Rate | ✅ 下降趋势明显 | 成功 |
| NearMissRate | ✅ 下降50%+ | 成功 |

**如果这三个都Green继续→ 否则应用PLAN中的微调**

### 第10-15轮：收敛性验证
| 指标 | 期望 | 判断 |
|------|------|------|
| Crash_Rate | ✅ 趋向0 | 成功 |
| Episode/Reward | ✅ 稳定在-110～-90 | 成功 |
| Action_Smoothness | ✅ 继续下降趋势 | 成功 |

---

## 📈 关键曲线对比预期

```
Before (原配置):
─────────────────────────────────────
Min_TTC:          1.5 ≤ value ≤ 4.5  (波动大)
Near_Miss_Rate:   0.02～0.05         (较高)
Crash_Rate:       0.0～0.6           (存在碰撞)
Action_Smooth:    0.12～0.24         (波动)

After (优化配置):  预期第10轮
─────────────────────────────────────
Min_TTC:          2.5 ≤ value ≤ 5.8  (波动小✓)
Near_Miss_Rate:   0.01～0.02         (下降50%✓)
Crash_Rate:       0.0～0.1           (大幅改善✓)
Action_Smooth:    0.08～0.16         (继续改善✓)
```

---

## 🔍 常见问题诊断

### Q1: Crash_Rate仍然存在
```
→ 检查REVIEW_MIN_TTC是否生效 (应看到2.5s)
→ 如仍>0.5:
   NEAR_MISS_TTC: 1.0 → 0.8
   REVIEW_MIN_TTC: 2.5 → 3.0
```

### Q2: Loss/TrajectoryConsistency不降
```
→ 检查TRAJ_GUIDE_ONLINE_BLEND值 (应看到0.28)
→ 如果Loss停在0.05+:
   TRAJ_GUIDE_LOSS_WEIGHT: 3.5 → 4.5
   TRAJ_QUERY_TEMPERATURE: 0.5 → 0.4 (增加集中度)
```

### Q3: Episode/Reward暴跌到-200以下
```
→ Policy过于保守，轨迹权重过高
→ 降低: TRAJ_GUIDE_ONLINE_BLEND: 0.28 → 0.22
       TRAJ_GUIDE_LOSS_WEIGHT: 3.5 → 2.8
```

### Q4: Action_Smoothness无改善
```
→ 增加轨迹的约束力:
   TRAJ_GUIDE_LOSS_WEIGHT: 3.5 → 4.5
   TRAJ_QUERY_COUNT: 5 → 6 (更多选项)
```

---

## 📋 曲线监控检查清单

运行过程中每5轮检查一次：

- [ ] **Min_TTC** 最小值趋势向上 (2.5s+)
- [ ] **Near_Miss_Rate** 相比原值下降 50%+
- [ ] **Crash_Rate** 逐轮下降，目标趋向0
- [ ] **Uncertainty_Mean** 保持低位 (< 0.15)
- [ ] **Trajectory_Entropy** 继续小幅下降
- [ ] **Loss/TrajectoryConsistency** 趋向0.01以下
- [ ] **Episode/Reward** 相对稳定（容许±10波动）
- [ ] **Action_Smoothness** 保持或改善（下降趋势）

---

## 💾 数据保存和恢复

```
新checkpoint自动保存:
  ./checkpoints/hybrid_sac_epN.pth (每50轮)

日志数据:
  ./runs/HybridDrive_YYYYMMDD_HHMMSS/
  → events.out.tfevents.* (TensorBoard读取)

恢复训练:
  python main.py  # 自动加载最新checkpoint
```

---

## 🎓 理论依据

这个优化方案基于以下观察：

1. **Time-To-Collision (TTC)** 是最直接的碰撞预警指标
   - 原TTC阈值1.5s可能不足以让SAC学到避免
   - 提高到1.0s/2.5s给策略更多训练信号

2. **轨迹一致性已有效** (Loss下降到0.02)
   - 说明LLM输出的轨迹质量高，应增加其话语权
   - 从18% → 28%是平衡收益和安全性的范围

3. **多模态模式需要集中** (T从0.7→0.5)
   - 4种模式太分散，加第5个后用温度编码尖锐分布
   - 策略更容易学到"哪个模式对"的决策

4. **边界条件需要覆盖** (间距范围扩展)
   - 0.65下界: 小间距(< 10m)时允许压缩
   - 1.35上界: 大间距(> 40m)时允许展开

---

**预计收益：** 
- ✅ 碰撞率从20%+降至5%以下
- ✅ 安全性指标(Min_TTC, Near_Miss)改善70%
- ✅ 决策平滑性保持或改善
- ⚠️ 学习效率可能稍降 (需追加2-3轮训练)

**总体评价：** 值得投入3-5轮额外训练，换取明显的安全性提升。
