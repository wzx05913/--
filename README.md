## XJTU-SY 滚动轴承端到端 PHM 在线诊断流水线

本仓库实现了基于 XJTU-SY 滚动轴承加速寿命试验数据集的**端到端在线 PHM（故障预测与健康管理）流水线**，包含：

> 数据加载 → 特征提取 → TabPFN 故障诊断 → 健康指标（HI）构造 → 剩余寿命（RUL）预测

所有核心代码位于 `src/` 目录下，可通过命令行一键运行，并将结果保存至 `output/` 目录。

---

### 1. 目录结构

```text
XJTU-SY-Engineering/
  XJTU-SY_Bearing_Datasets/      # 官方数据集（原始结构）
    Data/                        # 三个工况 15 个轴承的 CSV 数据
  model/                         # TabPFN 模型 checkpoint
  src/
    __init__.py
    config.py                    # 全局配置与故障特征频率计算
    data_loader.py               # CSV 读取 + 窗口生成器
    feature_engineering.py       # 时域/频域/小波/差分特征
    fault_diagnosis.py           # TabPFN 分类 + 弱标签 + SHAP
    health_index.py              # HI 构造与分级
    rul.py                       # RUL 预测与置信区间
    run.py                       # CLI 入口，串联全流程
  output/
    features/                    # 历史特征表 CSV
    diagnosis/                   # 故障诊断结果 CSV
    hi/                          # 健康指标与分级 CSV
    rul/                         # RUL 预测结果 CSV
    logs/                        # 运行日志
```

---

### 2. 功能概述

#### 2.1 数据与几何配置（`src/config.py`）

- **轴承几何参数**：
  - 滚动体数 `n_b = 8`
  - 滚动体直径 `d = 7.92 mm`
  - 节径 `D = 34.55 mm`
  - 接触角 `α = 0°`
- **数据参数**：
  - 采样率 `f_s = 25600 Hz`
  - 单 CSV 行数 `32768`（约 1 分钟）
  - 窗口比例 `p ∈ (0,1]`，默认 `p=1.0`
- **工况与轴承元数据**：
  - 内置 15 个轴承（Bearing1_1 ~ Bearing3_5）的工况编号、转速 `f_r`、载荷、故障类型（Inner/Outer/Cage/Mixed）
- **故障特征频率**：
  - 几何比 `λ = (d/D)·cosα`
  - `compute_fault_freqs(speed_hz)` 返回 BPFO/BPFI/BSF/FTF 四种特征频率
- **HI / RUL 参数**：
  - 健康 95% 分位、失效 5% 分位
  - HI 非线性系数 `λ_hi = 3.0`
  - 失效阈值 `τ_fail = 0.25`
  - 最大外推步数 `rul_max_steps = 200`
- **TabPFN 模型路径**：
  - 回归器：`model/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt`
  - 分类器：`model/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt`
- **设备选择**：
  - `resolve_device("auto")` 自动检测 CUDA，不可用时回退 CPU

#### 2.2 数据加载（`src/data_loader.py`）

- 以**轴承目录**为单位读取数据：
  - 通过 `BEARING_CONFIG` 中的 `condition` 字段，映射到 `35Hz12kN / 37.5Hz11kN / 40Hz10kN` 目录
  - 在该目录下按**数字文件名顺序**加载 `1.csv → 2.csv → ...`
- 每个 CSV 取**两通道（水平/竖直）**，自动跳过 NaN/非数值项。
- 窗口生成：
  - 窗口长度 `N = floor(32768 * p)`
  - 当 `p = 1.0`：每个窗口是一个完整 CSV
  - 当 `p < 1.0`：窗口在 CSV 之间顺序滑动，无重叠，跨文件续读
- 接口：
  - `iter_windows(bearing_id, cfg)` 是一个生成器，依次 `yield WindowRecord`：
    - `history: np.ndarray (N, 2)`
    - `bearing_id, window_id, speed_hz, load_kn, condition, fault`

#### 2.3 特征工程（`src/feature_engineering.py`）

对每个窗口的水平/竖直通道分别提取：

1. **时域特征**：RMS、Peak、MAV、Kurtosis、Crest、Impulse、Skew
2. **跨通道特征**：`rms_combined`、`kurt_max`、`ρ_hv`、`energyratio_hv`
3. **频域特征**：
   - Hann 窗 + RFFT，计算：
     - 故障特征频率（BPFO/BPFI/BSF/FTF）附近 ±5 Hz 的最大幅值
     - 频谱重心 `spectral_centroid`
     - 频谱熵 `spectral_entropy`
4. **小波特征**：
   - 默认小波基 `db4`，分解 4 层
   - 各层能量比 + 小波熵 `wavelet_entropy`
5. **差分特征**：
   - 所有特征 `x_t` 的一阶差分 `Δx_t = x_t - x_{t-1}`，首窗口置 0

#### 2.4 故障诊断（`src/fault_diagnosis.py`）

1. **弱标签（0 样本模式）**：
   - 前 `healthy_reference_windows` 个窗口视为健康参考（类别 0）
   - 健康阶段估计峰值参考 `A_h`，若 `Peak > 10·A_h` 则按照元数据故障类型标记失效类别
   - 损伤比超过 `weak_label_degrade_ratio` 阈值时，切换为相应故障类别
2. **两轮 TabPFN 推理**：
   - 使用 `TabPFNClassifier`，将历史概率作为滞后特征拼接到 X 中
   - 第一轮使用弱标签先验，得到 `p^(1)`；第二轮用 `p^(1)` 更新当前先验再推理得到最终 `p^(2)`
3. **SHAP 解释**：
   - 优先调用 `shap.KernelExplainer` 对当前类别概率做解释，输出 top-k 特征及其 SHAP 值
   - 若 SHAP 不可用，则退化为基于因果 z-score 的解释权重（仅用于可解释性与 HI 权重）
4. **日志与指标**：
   - 每个窗口输出：类别概率向量、Top-k SHAP、累积混淆矩阵、Accuracy、Macro-F1、Weighted-F1、Brier Score、校准曲线点。

#### 2.5 健康指标（`src/health_index.py`）

1. **单特征损伤度**：
   - 根据健康阶段 95% 分位 `q_m^{(H)}` 和失效阶段 5% 分位 `q_m^{(F)}`，对正向/反向特征分别使用给定公式计算 `d_{m,t}`。
2. **综合损伤指数**：
   - 默认使用 SHAP 绝对值归一化作为权重 `w_m`，否则退化为等权重；
   - `D_t = Σ w_m d_{m,t}`，可在映射前用 EMA 平滑。
3. **指数 HI 与分级**：
   - `HI_t = exp(-λ · D_t)`，保证 HI ∈ (0,1]；
   - 分级：HI ≥0.75 为健康，0.5–0.75 轻度，0.25–0.5 中度，<0.25 重度。

#### 2.6 RUL 预测（`src/rul.py`）

1. **HI 单调化**：
   - 对 HI 做 EMA + 运行最小值，得到单调非增序列。
2. **RUL 估计**：
   - 若已低于阈值 `τ_fail`，RUL=1；否则拟合最近若干点线性趋势，求到阈值的步数；
   - 若斜率近 0，改用指数衰减外推；结果截断到 `[1, max_steps]`。
3. **置信区间**：
   - 根据拟合残差 σ_r 和斜率 β：`CI = RUL ± 1.96·σ_r/|β|`。

---

### 3. 安装依赖

```bash
pip install numpy pywt shap tabpfn torch
```

> 如未安装 TabPFN/SHAP，流水线将自动降级为弱标签 soft one-hot 概率及 z-score 解释，仅用于调试。

---

### 4. 运行方法

在项目根目录下执行：

```bash
# 完整运行（示例）
python -m src.run --bearings Bearing3_2 --p 0.5 --wavelet db4 --lambda-hi 3.0

# 快速调试，仅前 5 个窗口
python -m src.run --bearings Bearing1_1 --p 0.1 --max-windows 5 --device cpu

# 同时处理多个轴承
python -m src.run --bearings Bearing1_1 Bearing3_2 --p 0.5
```

#### 4.1 主要 CLI 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--bearings` | 要处理的轴承 ID 列表（`Bearing1_1` 等） | `['Bearing1_1']` |
| `--p` | 窗口占单 CSV 的比例，`(0,1]` | `1.0` |
| `--wavelet` | 小波基（`db4`/`haar`/`sym4` 等） | `db4` |
| `--lambda-hi` | HI 指数映射系数 λ | `3.0` |
| `--device` | 计算设备：`auto/cuda/cpu` | `auto` |
| `--max-windows` | 每轴承最多处理窗口数，0 表示全量 | `0` |
| `--data-root` | XJTU-SY 数据根目录 | `XJTU-SY_Bearing_Datasets/Data` |
| `--output-dir` | 输出目录 | `output` |

运行结束后，将在 `output/` 下生成：

- `features/*_features.csv`：历史特征表
- `diagnosis/*_diagnosis.csv`：诊断与概率、SHAP、指标
- `hi/*_hi.csv`：HI、综合损伤度、等级
- `rul/*_rul.csv`：RUL 点估计与置信区间

---

### 5. 关键约束与建议

1. **严格因果性**：不使用任何未来信息；滞后特征仅使用历史概率；HI/RUL 仅基于过去和当前。
2. **不跨轴承混合**：每次运行仅在指定轴承内部累积特征与指标。
3. **TabPFN 使用规范**：概率全部来自 `predict_proba`；SHAP 仅用于解释，不参与预测或标签修正。
4. **长时间运行**：某些轴承（如 `Bearing3_2`）窗口较多，建议先使用小 `--max-windows` 试跑，再进行全量实验。
