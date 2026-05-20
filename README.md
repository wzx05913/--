## XJTU-SY 轴承 LOBO 离线 PHM 流水线（Diagnosis → HI → RUL）

本仓库已重构为**离线 Leave-One-Bearing-Out (LOBO)** 评估流程：

- 诊断任务 A2：预测 `future_health_stage`（默认 10 步前瞻）
- HI：固定等权重（H1）
- RUL：保持基于 HI 趋势外推（非直接监督回归）

并保留原有链式流程：**Diagnosis → HI → RUL**。

---

### 1. 关键变更

1. **完全移除单轴承在线学习与 weak label 注入路径**。  
2. **按轴承 LOBO 切分**：每个目标轴承测试，训练集使用“其余全部轴承”。
3. **规则标签预处理**（无全寿命泄漏）：
   - 基线 `baseline_rms`：前 `baseline_windows`（默认 50）窗口均值
   - `rms_over_baseline = rms_combined / baseline_rms`
   - `health_stage`：按 `stage_thresholds`（默认 `(1.6, 3.0)`）划分 0/1/2
   - `future_health_stage = health_stage.shift(-forecast_horizon)`（默认 horizon=10，尾部丢弃）
   - `is_fault = (health_stage > 0)`
4. **时间序列增强特征**：
   - 时间位置：`t_index`, `t_sqrt`, `t_log1p`
   - 固定周期编码：`sin/cos` for periods `[5,10,20,60,120]`
   - 选定信号的 lag/diff/rolling（仅过去信息，`fillna(0.0)`，无 `bfill`）
5. **输出升级**：
   - `output_v2/diagnosis/{bearing}_diagnosis.csv`
   - `output_v2/hi/{bearing}_hi.csv`
   - `output_v2/rul/{bearing}_rul.csv`
   - `output_v2/figures/{bearing}_dashboard.png`
   - `output_v2/GLOBAL_SUMMARY.csv`

---

### 2. 模块说明

- `preprocess.py`：构建每轴承窗口级 DataFrame（特征 + 规则标签）
- `fault_diagnosis.py`：离线分类器封装（优先 TabPFN；不可用时回退 LogisticRegression）
- `health_index.py`：等权重 HI 计算
- `rul.py`：基于 HI 轨迹预测 RUL
- `run.py`：LOBO 主流程，写 CSV + 全局汇总 + dashboard

---

### 3. 运行

```bash
python -m run --bearings Bearing1_1 Bearing1_2 --max-windows 300 --device cpu
```

常用参数：

- `--bearings`: 目标轴承列表（默认全部）
- `--max-windows`: 每轴承最多窗口，0 为全量
- `--output-dir`: 默认 `output_v2`
- `--data-root`: 数据根目录

---

### 4. Dashboard 内容

每个轴承会输出一张仪表图：

1. `future_health_stage` 真值 vs 预测曲线（x 轴 `file_index`）
2. HI 曲线
3. RUL 曲线
4. 第一处预测退化点（`pred_future_health_stage > 0`）的垂线标记（`fpt_index`）

---

### 5. 评估指标

诊断指标改为对 `future_health_stage` 真值进行评估：

- Accuracy
- Macro-F1
- Weighted-F1
- Confusion Matrix

并在 `GLOBAL_SUMMARY.csv` 汇总所有轴承结果。
