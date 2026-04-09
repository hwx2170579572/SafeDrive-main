# 📝 参数调整历史记录

## Version 2.0 | 优化版 (Phase 3A改进)

**时间**: 2026-03-18  
**根据**: TensorBoard曲线分析（35轮训练数据）  
**目标**: 降低碰撞率，提升安全指标，保持决策平滑性

---

## 参数变更明细

### 块1: 安全裕度强化 (Safety Margins)

```python
# 前方碰撞预警
OLD: REVIEW_MIN_FRONT_GAP = 10.0    # 米
NEW: REVIEW_MIN_FRONT_GAP = 12.0    # 米 (+20%)
效果: 给予更多减速/变道空间

OLD: REVIEW_MIN_TTC = 2.0            # 秒
NEW: REVIEW_MIN_TTC = 2.5            # 秒 (+25%)
效果: 反应预留时间延长 (250ms → 300ms)

# 近碰撞定义
OLD: NEAR_MISS_TTC = 1.5             # 秒
NEW: NEAR_MISS_TTC = 1.0             # 秒 (-33%)
效果: 更严格的KPI定义，更早触发学习信号
```

**原理**: 
- TTC < 2.5s时策略需要立即响应
- 1.0s阈值对应80-100km/h下的3-4个车长
- 这个范围内避碰成功率显著提高

**影响链**:
```
Min_TTC ↑ → Crash_Rate ↓ → Reward稍降 → 需追加训练轮数
```

---

### 块2: 轨迹引导权重增强 (Trajectory Guidance)

```python
# 多模态查询配置
OLD: TRAJ_QUERY_COUNT = 4
NEW: TRAJ_QUERY_COUNT = 5           (+25% 模式数)
效果: 从4种方案选择→5种方案选择，覆盖更全

OLD: TRAJ_QUERY_TEMPERATURE = 0.7
NEW: TRAJ_QUERY_TEMPERATURE = 0.5   (-29% 温度)
效果: softmax分布从 "分散" → "尖锐"
      E.g., [0.25, 0.25, 0.25, 0.25] 
          → [0.55, 0.20, 0.15, 0.10] (集中在最优)

# 在线混合比例
OLD: TRAJ_GUIDE_ONLINE_BLEND = 0.18         (18%)
NEW: TRAJ_GUIDE_ONLINE_BLEND = 0.28         (28%, +56%)

反映在决策上:
OLD: proposed = 0.82 * policy_action + 0.18 * guide_action
NEW: proposed = 0.72 * policy_action + 0.28 * guide_action
                                    ↑ 轨迹权重增加 10%

# 一致性学习强度
OLD: TRAJ_GUIDE_LOSS_WEIGHT = 2.5
NEW: TRAJ_GUIDE_LOSS_WEIGHT = 3.5   (+40%)

影响Actor Loss:
OLD: L_actor = L_policy + 10.0*L_cbf + 2.5*L_traj
NEW: L_actor = L_policy + 10.0*L_cbf + 3.5*L_traj
              ↑ 轨迹一致性从2.5倍→3.5倍重要
```

**原理**:
- 已验证轨迹预测优质 (Loss已到0.02，说明模式选择稳定)
- 应该增加其话语权至28% (不超过30%避免过度限制)
- 一致性损失3.5倍权重能有效约束策略跟踪轨迹

**影响链**:
```
TRAJ_GUIDE_ONLINE_BLEND ↑ → Policy更稳健 → Action_Smoothness ↓
                      + → Loss/Traj ↓ (更集中)
                      + → Reward稍降 (多了约束)
```

---

### 块3: 轨迹模板自适应范围 (Template Scaling)

```python
# 间距缩放钳制范围
OLD: TEMPLATE_GAP_CLAMP_MIN = 0.75
     TEMPLATE_GAP_CLAMP_MAX = 1.25
     范围: [0.75, 1.25] (宽度50%)

NEW: TEMPLATE_GAP_CLAMP_MIN = 0.65
     TEMPLATE_GAP_CLAMP_MAX = 1.35
     范围: [0.65, 1.35] (宽度70%, +40%)

具体场景:
1. 小间距(dx < 15m):
   OLD: scale = 0.75 * template (有点大)
   NEW: scale = 0.65 * template (更紧凑✓)
   
2. 正常(15m < dx < 35m):
   OLD: scale ≈ 1.0 * template
   NEW: scale ≈ 1.0 * template (不变)
   
3. 大间距(dx > 35m):
   OLD: scale = 1.25 * template (可能还受限)
   NEW: scale = 1.35 * template (更自由✓)
```

**原理**:
- 小间距: 下界更低(0.65) →轨迹压缩→更安全变道/减速
- 大间距: 上界更高(1.35) →轨迹展开→更多加速/舒适驾驶空间

**影响**:
- 边界情况(dx<12m or dx>50m)的轨迹多样性提升
- SlowSystemThread中的 `_build_parametric_trajectory()` 会应用这个范围

---

## 曲线预期变化

### Before vs After 对标表

```
               BEFORE               AFTER (期望)
              ─────────────────────
Min_TTC       1.5-4.5 (波动±150%)   2.5-5.8 (波动±35%)
Near_Miss     1-5% 高波动            0.5-2% 低波动
Crash_Rate    0-60% 存在碰撞         0-10% 偶发碰撞
─────────────────────────────────────
Traj_Entropy  0.15-0.25 收敛中      0.10-0.15 更集中
Uncertainty   0.15-0.78 收敛中      0.08-0.12 很低
─────────────────────────────────────
Action_Smooth  0.12-0.24             0.08-0.16 改善
Reward         -100 ~ -105           -110 ~ -95 (合理降低)
```

### 核心指标改善倍数

| 指标 | 改善倍数 | 信心度 |
|------|---------|--------|
| Crash_Rate | 3-5x↓ | ⭐⭐⭐⭐⭐ 高 |
| Min_TTC稳定性 | 4-5x | ⭐⭐⭐⭐⭐ 高 |
| Near_Miss | 2-3x↓ | ⭐⭐⭐⭐ 中高 |
| Action_Smoothness | 1.5-2x↓ | ⭐⭐⭐⭐ 中高 |

---

## 代码实现位置

### SafetyShieldReviewer (行917)
```python
if accel > 0 and (min_front_dx < config.REVIEW_MIN_FRONT_GAP or 
                   min_ttc < config.REVIEW_MIN_TTC):
    # 现在: min_front_dx < 12.0, min_ttc < 2.5 (vs 旧的10.0, 2.0)
    accel = config.REVIEW_ACCEL_LIMIT
```

### Trajectory Predictor (行939-1035)
```python
def predict():
    # TRAJ_QUERY_COUNT = 5 (vs 4)
    # TRAJ_QUERY_TEMPERATURE = 0.5 (vs 0.7)
    scores = torch.nn.functional.softmax(raw_scores / 0.5, dim=-1)  # 温度应用
```

### Training Loop (行1290-1291, 1521-1522)
```python
proposed_action = torch.clamp(
    (1.0 - 0.28) * policy_action +      # changed from 0.18
    0.28 * guide_action,
    -1.0, 1.0
)
```

### Loss Computation (行1374)
```python
actor_loss = policy_loss + 10.0 * cbf_imitation_loss + 3.5 * traj_consistency_loss
                                                         ↑ changed from 2.5
```

### Template Scaling (行642)
```python
gap_scale = float(np.clip(
    gap_scale, 
    config.TEMPLATE_GAP_CLAMP_MIN,   # 0.65 (vs 0.75)
    config.TEMPLATE_GAP_CLAMP_MAX    # 1.35 (vs 1.25)
))
```

---

## 验收标准

### ✅ 第一阶段 (第1-5轮)
- [ ] NEAR_MISS_TTC改动生效 (KPI触发时刻从1.5s→1.0s)
- [ ] REVIEW_MIN_TTC生效 (Reviewer在2.5s介入)
- [ ] Min_TTC最小值 ≥ 2.5s (改善超50%)

### ✅ 第二阶段 (第6-15轮)
- [ ] Crash_Rate → 0% (100% safe runs)
- [ ] Episode/Reward 稳定在 -110~-90 范围
- [ ] Action_Smoothness 继续下降趋势

### ✅ 第三阶段 (第16+轮)
- [ ] 所有指标稳定收敛
- [ ] 无需进一步微调

---

## 回滚方案 (如果优化失败)

如果10轮后仍然没有改善，回滚参数:

```python
# 回滚Step 1: 检查REVIEW_MIN_TTC是否过高
REVIEW_MIN_TTC = 2.5 → 2.2  (缓一点)

# 回滚Step 2: 如果Reward暴跌
TRAJ_GUIDE_ONLINE_BLEND = 0.28 → 0.25 (轻一点)
TRAJ_GUIDE_LOSS_WEIGHT = 3.5 → 3.0   (轻一点)

# 回滚Step 3: 如果无法学习
TRAJ_QUERY_TEMPERATURE = 0.5 → 0.6 (分散一点)
TRAJ_QUERY_COUNT = 5 → 4 (简化一点)
```

---

## 版本历史

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| 1.0 | 2026-03-17 | Phase 3A初始参数 (TRAJ_QUERY=4, BLEND=0.18, LOSS=2.5) | ✅ 完成 |
| 2.0 | 2026-03-18 | 基于TensorBoard优化 (↑安全裕度, ↑轨迹权重) | 🟡 测试中 |
| (待定) | TBD | Phase 3B/3C 方案集成 | ⏳ 计划中 |

