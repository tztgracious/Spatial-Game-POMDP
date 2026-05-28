# Project Guide: Spatial-Game-POMDP (FPS Microstructure & MARL)

## 🎯 核心目标 (Core Objective)
构建一个非完全信息下的动态多阶段空间博弈马尔可夫决策过程（POMDP），通过引入视线遮挡（Line-of-Sight Discontinuity）模拟 FPS 游戏中的微观结构，并使用多智能体强化学习（自我博弈 PPO）求解该局部空间博弈下的近似纳什均衡。

---

## 🕒 Phase 1: 微观环境引擎与状态空间抽象 (Day 1)
**目标：** 摒弃所有图形渲染，用纯数学和矩阵运算构建最底层的物理与碰撞引擎。

### 📦 核心模块 (Modules)
1. **`core/geometry.py` - 空间几何计算模块**
   * 实现基于向量叉乘的线段相交算法，用于判定两点之间是否有视线（LoS）遮挡。
   * **输入:** $A(x_1, y_1), B(x_2, y_2)$, 墙体坐标。
   * **输出:** `Boolean` (True 即可见，False 即被遮挡)。
2. **`env/arena.py` - 环境状态机模块 (基于 Gymnasium / PettingZoo)**
   * 定义连续状态空间 $S \in \mathbb{R}^n$：包含双方的 $(x, y)$ 坐标、生命值、视野状态。
   * 定义离散动作空间 $A$：上下左右移动（步长 $\Delta d$）、开火、静止。
   * 构建 `step()` 和 `reset()` 函数，处理状态转移概率 $P(s'|s,a)$。

### 🏁 达到效果 (Expected Outcomes)
* 能够实例化环境，随机输入 action，环境能正确打印出当前的 State 矩阵，并正确计算 LoS 和 HP 扣减。
* 物理逻辑达到无 bug 闭环，耗时必须控制在极低水平（单步 step < 1ms），为后续高频蒙特卡洛采样做准备。

---

## 🧠 Phase 2: 算法基建与自我博弈训练环 (Day 2)
**目标：** 引入强化学习算法基线，建立“左手打右手”的自我进化机制。

### 📦 核心模块 (Modules)
1. **`agents/ppo_baseline.py` - 策略网络接入**
   * 避免手撕底层 PPO（时间受限），直接接入 `stable-baselines3` 或 `Ray RLlib` 的 PPO 算子。
   * 定义 Actor-Critic 网络的 MLP 架构（例如 `[128, 128]` 隐藏层）。
2. **`training/self_play_loop.py` - 自我博弈池引擎**
   * 建立模型历史池（Model Pool）。
   * 实现机制：当前的 Agent A 不仅与最新的 Agent B 训练，还要以一定概率（如 20%）与过去的 Agent B 历史快照（Checkpoints）对战，防止策略遗忘（Catastrophic Forgetting）和陷入局部最优。

### 🏁 达到效果 (Expected Outcomes)
* 训练循环能够顺利跑通，模型可以不断输出 Checkpoints 并更新 Tensorboard 曲线。
* 此时大概率会观察到**模型无法收敛**（由于奖励稀疏，双方都在原地发呆），这属于正常现象，进入 Phase 3。

---

## 🔧 Phase 3: 奖励塑形与博弈均衡突破 (Day 3 - Day 4)
**目标：** 解决多智能体强化学习最核心的稀疏奖励与非平稳性问题，逼迫 Agent 进化出高级战术。

### 📦 核心模块 (Modules)
1. **`env/reward_shaping.py` - 奖励函数精调模块**
   * **基础函数:** 击杀 $+10$，被击杀 $-10$，每步时间惩罚 $-0.01$。
   * **势能引导 (Potential-based Reward):** 在早期训练中，引入双方距离的势能差，引导进攻方靠近转角区。
   * **行为惩罚 (Action Penalty):** 当 $LoS = False$ 时持续开火，给予额外惩罚（模拟后坐力或暴露位置的代价）。
2. **`utils/callbacks.py` - 课程学习 (Curriculum Learning) 调度器**
   * 随着训练步数（Timesteps）的增加，逐渐衰减势能引导奖励，最终平滑过渡到纯粹的胜负零和博弈（Zero-Sum Game）奖励。

### 🏁 达到效果 (Expected Outcomes)
* 监控 Tensorboard，观察到 Episode Length 逐渐变短，Win Rate 开始在 50% 附近震荡（证明双方实力交替上升）。
* 抽样观察对局，Agent 展现出类似人类玩家的“预瞄（Pre-aim）”和“边缘试探（Jiggle Peeking）”行为。

---

## 📊 Phase 4: 极客可视化与量化特征提取 (Day 5)
**目标：** 停止训练，将神经网络里的“黑盒”黑话，翻译成量化研究员最爱的数学图表。

### 📦 核心模块 (Modules)
1. **`analysis/value_heatmap.py` - 状态价值函数空间映射**
   * 固定防守方 Agent B 于地图某一点 $s_b$。
   * 遍历地图上所有可能的 $x, y$ 坐标，将生成的观测向量输入 Critic 网络，提取期望回报值 $V(s)$。
   * 利用 `matplotlib` 绘制 2D 热力图，直观展示高危区域与绝对安全区。
2. **`analysis/trajectory_plot.py` - 策略轨迹演变分析**
   * 导出第 10k, 50k, 500k 步的对局坐标时序数据 $(x_t, y_t)$。
   * 绘制轨迹散点图，证明模型动作从“随机游走（Random Walk）”收敛至“有目的的局部博弈路径”。

### 🏁 达到效果 (Expected Outcomes)
* 产出两张极其硬核的高清图表（Heatmap & Trajectory Evolution）。
* 完善 `README.md`，用清晰的 LaTeX 公式列出 $S, A, R$ 的定义，将这几天踩过的坑提炼为项目的 Highlights。