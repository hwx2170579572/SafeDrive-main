# LLM-Guided Hierarchical Constrained SAC for Highway Driving

## 实验流程说明（Markdown版）

本文档用于指导论文 **LLM-Guided Hierarchical Constrained SAC for Highway Driving** 的实验实施。

本文默认使用的代码文件为：

```bash
single_file_llm_guided_rl_v6.py
```

---

## 1. 论文中的方法对应关系

本论文主实验包含 4 组方法：

- **Baseline A：纯 SAC**  
  对应代码模式：`baseline_sac`

- **Baseline B：SAC + reward shaping**  
  对应代码模式：`shaping_sac`

- **Baseline C：规则高层 + SAC**  
  对应代码模式：`rule_hier`

- **Ours：LLM 高层 + 约束低层 SAC**  
  对应代码模式：`constrained_real_llm_hier`

---

## 2. 关于 Ours 中 LLM 的使用方式

这是本实验最重要的设定之一：

- Ours 中的 **LLM 不是训练参数的一部分**，而是通过 **API 调用** 的方式作为高层规划器使用。
- LLM 不参与反向传播，不更新自身权重。
- **LLM 在训练阶段就接入**，持续输出高层信息（如目标车道、目标速度、局部 waypoints、安全约束），从而影响低层 constrained SAC 的学习过程。
- **LLM 在测试阶段也继续接入**，保持与训练时一致的高层决策方式。

因此，本论文的 Ours 并不是“测试时临时加一个 LLM 帮忙”，而是：

> **LLM 作为高层规划器，在训练和测试中持续引导低层强化学习策略。**

这才符合题目中“LLM-Guided”的含义。

---

## 3. 正式主实验时，到底冻结什么，训练什么

正式论文实验中，需要区分 3 类东西：

### 3.1 冻结的内容

冻结的是 **实验协议**，包括：

- 学习率等超参数
- reward / cost 权重
- planner interval 与重规划阈值
- 正式比较的方法集合
- seeds 集合
- 训练 budget 和评估 budget
- prompt 模板
- formal 阶段的评估规则

这些在正式实验开始后不应再改。

### 3.2 仍然要训练的内容

仍然训练的是 **RL 模型参数**，也就是：

- actor
- critic
- constrained SAC 中的 cost critics
- 相关温度参数 / 乘子参数

也就是说，正式实验不是“拿已经训练好的模型直接测一次”，而是：

> **在冻结协议下，从头训练每个方法，再进行正式评估。**

### 3.3 不训练的内容

- LLM 本身的参数不训练
- 只是通过 API 调用使用其输出

---

## 4. 总实验顺序

整个实验建议按以下 5 个阶段执行：

1. **最小运行检查**：确认代码和模式都能正常运行
2. **开发调参**：先不用真实 LLM 做主调参
3. **冻结协议**：固定正式实验设置
4. **正式试跑 Ours**：先单独检查真实 LLM 主方法能否稳定运行
5. **正式 compare**：输出论文主表结果

---

## 5. 第一步：最小运行检查

这一步的目的只是确认：

- 代码没有明显错误
- 环境能正常初始化
- 各模式都能启动
- 基本评估流程正常

### 5.1 Baseline A

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode baseline_sac --episodes 2 --eval-episodes 1
```

### 5.2 Baseline B

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode shaping_sac --episodes 2 --eval-episodes 1
```

### 5.3 Baseline C

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode rule_hier --episodes 2 --eval-episodes 1
```

### 5.4 Ours 的开发版（先用 mock）

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode constrained_llm_hier --episodes 2 --eval-episodes 1 --llm-backend mock
```

### 5.5 这一步看什么

只看以下几件事：

- 程序能不能正常跑完
- 是否报维度错误、模式名错误、依赖错误
- 评估日志是否能正常打印

如果这里都过不了，不要往下做正式实验。

---

## 6. 第二步：开发阶段调参

### 6.1 为什么先不用真实 LLM 调参

因为真实 LLM 会带来额外不确定性，例如：

- API 波动
- 网络超时
- 响应差异
- 成本较高

所以更稳妥的做法是：

- 先用 `rule_hier` 调系统主结构
- 再用 `constrained_llm_hier`（mock）检查 LLM 风格高层接口
- 不用真实 LLM 反复调参

这样做的目的是让：

- reward/cost 稳定
- 高低层接口顺畅
- 策略先学会“跑起来、别乱换道、别太危险”

---

### 6.2 开发阶段先跑 Baseline C

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode rule_hier  --episodes 120 --max-steps 200 --eval-episodes 5 --eval-primary-report raw 
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode rule_hier  --episodes 120 --max-steps 200 --eval-episodes 5 --eval-primary-report raw --device auto
python single_file_llm_guided_rl_v6.py \
  --workflow-stage dev \
  --mode rule_hier \
  --episodes 120 \
  --max-steps 200 \
  --eval-episodes 5 \
  --eval-primary-report raw

跑多组随机种子
  python single_file_llm_guided_rl_v6_gpu.py --workflow-stage dev --mode compare --compare-modes rule_hier --compare-seeds 42,52,62,72,82 --episodes 120 --max-steps 200 --eval-episodes 5 --eval-primary-report raw --compare-primary-report raw --compare-save-json results/dev_rule_hier_5seeds.json --compare-save-csv results/dev_rule_hier_5seeds.csv   
```

---

### 6.3 再跑 Ours 的开发版（mock LLM）

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage dev \
  --mode constrained_llm_hier \
  --episodes 120 \
  --max-steps 200 \
  --eval-episodes 5 \
  --eval-primary-report raw \
  --llm-backend mock
```
python single_file_llm_guided_rl_v6_gpu_priority1_tensorboard.py --workflow-stage dev --mode compare --compare-modes constrained_llm_hier --compare-seeds 42,52,62,72,82 --episodes 120 --max-steps 200 --eval-episodes 5 --eval-primary-report raw --compare-primary-report raw --llm-backend mock --tensorboard-dir results/tb_dev_constrained_llm_hier_5seeds --compare-save-json results/dev_constrained_llm_hier_5seeds.json --compare-save-csv results/dev_constrained_llm_hier_5seeds.csv      

python single_file_llm_guided_rl_v6_gpu_priority2_tensorboard.py --workflow-stage dev --mode compare --compare-modes constrained_llm_hier --compare-seeds 42,52,62,72,82 --episodes 120 --max-steps 200 --eval-episodes 5 --eval-primary-report raw --compare-primary-report raw --llm-backend mock --lambda-ema-beta 0.97 --min-lambda-collision 0.01 --min-lambda-headway 0.01 --collision-risk-ttc-threshold 4.0 --collision-risk-front-distance 10.0 --tensorboard-dir results/tb_dev_constrained_llm_hier_p2_5seeds --compare-save-json results/dev_constrained_llm_hier_p2_5seeds.json --compare-save-csv results/dev_constrained_llm_hier_p2_5seeds.csv 

python single_file_llm_guided_rl_v6_gpu_priority2_tensorboard.py \
  --workflow-stage dev \
  --mode compare \
  --compare-modes llm_hier,constrained_llm_hier,constrained_rule_hier \
  --compare-seeds 42,52,62,72,82 \
  --episodes 120 \
  --max-steps 200 \
  --eval-episodes 5 \
  --eval-primary-report raw \
  --compare-primary-report raw \
  --llm-backend mock \
  --lambda-ema-beta 0.97 \
  --min-lambda-collision 0.01 \
  --min-lambda-headway 0.01 \
  --collision-risk-ttc-threshold 4.0 \
  --collision-risk-front-distance 10.0 \
  --tensorboard-dir results/tb_dev_p2_crosscheck_5seeds \
  --compare-save-json results/dev_p2_crosscheck_5seeds.json \
  --compare-save-csv results/dev_p2_crosscheck_5seeds.csv
---

### 6.4 开发阶段重点看哪些指标

不要只看 return，重点看：

- `success`
- `collision`
- `mean_speed`
- `unsafe_headway_rate`
- `lane_changes`
- `mean_smooth_steer`
- `mean_cost_total`

开发阶段优先看 **raw policy**，因为 shielded 容易把主体策略的问题遮住。

---

### 6.5 什么时候说明调参差不多了

当你发现：

- `rule_hier` 和 `constrained_llm_hier` 的结果已经比较稳定
- 改一点参数不会让结论大变
- raw policy 已经具备基本能力
- 不再需要频繁改 reward/cost/planner 设置

这时就可以进入冻结阶段。

---

## 7. 第三步：冻结正式实验协议

这一步的意思是：

> **从这里开始，不再随便改正式实验设置。**

你要把下面这些固定下来：

- 正式比较哪些方法
- 用哪些随机种子
- 每个方法训练多少 episodes
- 每个 episode 最多多少步
- 评估多少 episode

### 推荐冻结命令

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage freeze \
  --freeze-save frozen_protocol.json \
  --formal-modes baseline_sac,shaping_sac,rule_hier,constrained_real_llm_hier \
  --formal-seeds 142,242,342 \
  --episodes 150 \
  --max-steps 200 \
  --eval-episodes 8
```

### 为什么 formal 只保留这 4 个主方法

因为这 4 个方法正好对应论文主线：

- 纯 RL
- RL + shaping
- 非 LLM 的层次化引导
- LLM 引导的层次化 constrained SAC

这样论文主表最清晰。

---

## 8. 第四步：正式实验前，先单独试跑 Ours

在开始正式 compare 之前，建议先单独检查一次真实 LLM 主方法能否稳定运行。

因为 Ours 中的 LLM 是 API 调用，所以正式实验前要确认：

- API key 正确
- 网络连接正常
- formal 阶段不会误 fallback 到 mock
- LLM 返回内容能够通过解析和归一化

### 命令

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage formal \
  --freeze-load frozen_protocol.json \
  --formal-strict \
  --mode constrained_real_llm_hier \
  --llm-backend real \
  --llm-api-key YOUR_KEY
```

### 这一步看什么

主要看：

- 是否能顺利开始训练
- 是否频繁出现 API 错误
- 是否发生 LLM 输出解析失败
- 是否出现 formal 阶段不允许的 fallback

如果这一步不稳定，不要立刻做正式 compare，先把 real LLM 接口稳定下来。

---

## 9. 第五步：正式跑论文主实验

正式阶段需要比较 4 个主方法。

---

### 9.1 Baseline A：纯 SAC

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage formal \
  --freeze-load frozen_protocol.json \
  --formal-strict \
  --mode baseline_sac
```

---

### 9.2 Baseline B：SAC + reward shaping

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage formal \
  --freeze-load frozen_protocol.json \
  --formal-strict \
  --mode shaping_sac
```

---

### 9.3 Baseline C：规则高层 + SAC

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage formal \
  --freeze-load frozen_protocol.json \
  --formal-strict \
  --mode rule_hier
```

---

### 9.4 Ours：LLM 高层 + 约束低层 SAC

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage formal \
  --freeze-load frozen_protocol.json \
  --formal-strict \
  --mode constrained_real_llm_hier \
  --llm-backend real \
  --llm-api-key YOUR_KEY
```

---

## 10. 第六步：正式 compare（推荐）

如果不想一个一个单独跑，也可以直接使用 compare 一次性输出主结果。

### 推荐命令

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage formal \
  --freeze-load frozen_protocol.json \
  --formal-strict \
  --mode compare \
  --compare-save-json final_compare.json \
  --compare-save-csv final_compare.csv \
  --compare-save-latex final_compare.tex
```

### 这一步会得到什么

通常会得到：

- `final_compare.json`
- `final_compare.csv`
- `final_compare.tex`

这些文件可以用来：

- 整理论文表格
- 保存完整实验记录
- 生成 LaTeX 表格

---

## 11. 论文主表应该怎么整理

正式实验完成后，建议论文里至少放两张主表。

### 11.1 主表一：Raw Policy

行：

- Baseline A: SAC
- Baseline B: SAC + reward shaping
- Baseline C: Rule + SAC
- Ours: Real-LLM + constrained SAC

列建议：

- Return
- Success
- Collision
- Mean speed
- Unsafe headway rate
- Mean cost total

这张表最重要，因为它直接回答：

> **LLM 引导下的层次化 constrained SAC 是否优于纯 RL。**

---

### 11.2 主表二：Shielded Policy

还是同样四个方法，但汇报 shielded 结果。

这张表说明：

- 如果加上安全层，最终系统整体表现如何
- Ours 在部署层面是否同样占优

---

## 12. 建议补做的附加实验

虽然论文主表只有 4 个方法，但建议再补两类实验。

---

### 12.1 补充实验：mock LLM 版本

这个实验不是主表，但可以帮助说明：

- 你的方法不是只对真实 API 有效
- LLM 风格高层接口本身就有价值

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage dev \
  --mode constrained_llm_hier \
  --episodes 120 \
  --max-steps 200 \
  --eval-episodes 5 \
  --llm-backend mock
```

---

### 12.2 消融实验

建议至少做以下 3 个：

#### 去掉 waypoint reward

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage dev \
  --mode constrained_llm_hier \
  --episodes 120 \
  --eval-episodes 5 \
  --ablate-waypoint-reward
```

#### 去掉 constraint costs

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage dev \
  --mode constrained_llm_hier \
  --episodes 120 \
  --eval-episodes 5 \
  --ablate-constraint-costs
```

#### 去掉 lane stabilization

```bash
python single_file_llm_guided_rl_v6.py \
  --workflow-stage dev \
  --mode constrained_llm_hier \
  --episodes 120 \
  --eval-episodes 5 \
  --ablate-lane-stabilization
```

这些实验有助于解释：

- 哪个模块贡献最大
- 你的方法为什么有效

---

## 13. 最推荐的一套完整执行顺序

如果你现在就开始做实验，建议按下面顺序执行。

### 第 1 轮：最小检查

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode baseline_sac --episodes 2 --eval-episodes 1
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode shaping_sac --episodes 2 --eval-episodes 1
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode rule_hier --episodes 2 --eval-episodes 1
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode constrained_llm_hier --episodes 2 --eval-episodes 1 --llm-backend mock
```

### 第 2 轮：开发调参

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode rule_hier --episodes 120 --max-steps 200 --eval-episodes 5 --eval-primary-report raw
python single_file_llm_guided_rl_v6.py --workflow-stage dev --mode constrained_llm_hier --episodes 120 --max-steps 200 --eval-episodes 5 --eval-primary-report raw --llm-backend mock
```

### 第 3 轮：冻结协议

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage freeze --freeze-save frozen_protocol.json --formal-modes baseline_sac,shaping_sac,rule_hier,constrained_real_llm_hier --formal-seeds 142,242,342 --episodes 150 --max-steps 200 --eval-episodes 8
```

### 第 4 轮：先单试 Ours

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage formal --freeze-load frozen_protocol.json --formal-strict --mode constrained_real_llm_hier --llm-backend real --llm-api-key YOUR_KEY
```

### 第 5 轮：正式 compare

```bash
python single_file_llm_guided_rl_v6.py --workflow-stage formal --freeze-load frozen_protocol.json --formal-strict --mode compare --compare-save-json final_compare.json --compare-save-csv final_compare.csv --compare-save-latex final_compare.tex
```

---

## 14. 最后一句话总结

这篇论文的主实验逻辑应该始终保持一致：

- 开发阶段：先不用真实 LLM 反复调参
- 正式阶段：Ours 中的真实 LLM 在训练和测试都接入
- 正式阶段冻结的是实验协议，不是 RL 模型权重
- 最终用 raw 和 shielded 两套结果共同支撑论文结论

如果后续还需要，可以在此基础上继续整理成：

- 论文中的 **Experimental Protocol** 小节
- 论文中的 **Implementation Details** 小节
- 论文表格模板

启动 TensorBoard 看曲线：
tensorboard --logdir results/tb_dev_constrained_llm_hier_5seeds --port 6006
http://localhost:6006