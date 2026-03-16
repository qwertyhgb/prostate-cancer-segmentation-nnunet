# BPH 失败病例可视化复核完整指南

## 文档概述

本文档详细说明如何使用可视化工具对 BPH (良性前列腺增生) 失败病例进行系统性复核，以找出模型在 BPH 病灶上失效的根本原因。

**目标读者**: 研究人员、临床医生、算法工程师  
**前置条件**: 已完成第一折 (fold_0) 训练并生成验证结果  
**预计时间**: 完整复核流程约 2-4 小时

---

## 1. 背景与问题定义

### 1.1 核心发现

第一折验证结果显示严重的 **BPH/PCA 性能撕裂**：

| 指标 | BPH (38例) | PCA (34例) | 差异倍数 |
|------|------------|------------|----------|
| Mean Dice | 0.0819 | 0.5230 | **6.4x** |
| Recall | 0.0388 | 0.5055 | **13x** |
| Dice=0 比例 | 78.9% | 14.7% | - |

### 1.2 待解答问题

1. **标签一致性**: BPH 和 PCA 的病灶标注风格是否一致？
2. **影像表型**: BPH 病灶在 T2/ADC/DWI 上是否与 PCA 有显著差异？
3. **失败模式**: BPH 是完全漏检、误检、还是边界偏差？
4. **病灶特征**: BPH 失败是否与病灶大小、位置、对比度相关？

---

## 2. 工具脚本说明

### 2.1 脚本清单

| 脚本 | 路径 | 功能 |
|------|------|------|
| `visualize_bph_failures.py` | `script/v1/visualize_bph_failures.py` | 失败病例影像可视化 |
| `compare_bph_pca_features.py` | `script/v1/compare_bph_pca_features.py` | BPH vs PCA 特征对比分析 |

### 2.2 数据路径配置

脚本自动识别的路径：

```
PROJECT_ROOT = /opt/data/private/lm/projects/nnunet

# 输入数据
SUMMARY_JSON = nnUNet_results/Dataset101_Prostate/.../fold_0/validation/summary.json
IMAGES_DIR   = nnUNet_raw/Dataset101_Prostate/imagesTr/
LABELS_DIR   = nnUNet_raw/Dataset101_Prostate/labelsTr/
PREDICTIONS  = nnUNet_results/.../fold_0/validation/

# 默认输出
viz/bph_failures/          # 可视化图像
analysis/bph_pca_comparison/  # 特征分析结果
```

---

## 3. 可视化复核流程

### 步骤 1: 生成失败病例清单（5分钟）

**目的**: 快速了解失败病例分布

```bash
python script/v1/visualize_bph_failures.py \
    --all-failures \
    --report-only \
    --output viz/report
```

**输出**: `viz/report/bph_failures_report.txt`

**报告内容示例**：
```
Case Name            Dice     n_ref     n_pred       TP       FP       FN   Recall  Precision
----------------------------------------------------------------------------------------------------
BPH_0000661886       0.0000       4752          0        0        0     4752    0.000      0.000
BPH_0000044758       0.0000       3705        496        0      496     3705    0.000      0.000
BPH_0000909756       0.0000       3554          0        0        0     3554    0.000      0.000
...

失败模式分类:
----------------------------------------------------------------------------------------------------
完全漏检 (Dice=0, n_pred=0): 11 例
  - BPH_0000661886: n_ref=4752
  - BPH_0000044758: n_ref=3705
  ...

零交并+误检 (Dice=0, n_pred>0): 19 例
  - BPH_0000027417: n_ref=2372, n_pred=2651
  ...
```

---

### 步骤 2: BPH vs PCA 特征对比分析（15分钟）

**目的**: 量化 BPH 和 PCA 的影像特征差异

```bash
python script/v1/compare_bph_pca_features.py \
    --output analysis/bph_pca_comparison
```

**输出文件**：
- `comparison_report.txt` - 统计报告
- `feature_comparison.png` - 箱线图对比
- `scatter_analysis.png` - 散点图分析
- `features_data.json` - 原始数据

**关键指标解读**：

| 指标 | 意义 | 预期差异 |
|------|------|----------|
| `T2_contrast` | 病灶与背景的 T2 信号对比度 | BPH 可能更低 |
| `ADC_mean` | 病灶 ADC 均值 | BPH 可能更高 (弥散受限轻) |
| `DWI_mean` | 病灶 DWI 均值 | BPH 可能更低 |
| `n_voxels` | 病灶体积 | BPH 可能更大 |
| `centroid_z` | 病灶轴向位置 | 分布可能不同 |

**示例报告**：
```
Metric               BPH (mean±std)            PCA (mean±std)            Ratio (PCA/BPH)
--------------------------------------------------------------------------------
n_voxels             1197.60±1523.45           2536.20±4123.67           2.12x
T2_contrast          0.15±0.12                 0.32±0.18                 2.13x
ADC_mean             1456.30±523.40            892.50±312.60             0.61x
DWI_mean             423.80±156.30             687.40±234.50             1.62x
dice                 0.08±0.15                 0.52±0.35                 6.50x
recall               0.04±0.11                 0.51±0.38                 12.75x
```

**关键发现示例**：
1. **Dice 差异**: PCA 是 BPH 的 6.5 倍
2. **T2 对比度**: BPH 病灶对比度显著低于 PCA (0.15 vs 0.32)
3. **ADC 差异**: BPH 的 ADC 值更高，说明弥散受限程度轻

---

### 步骤 3: P0 优先级病例可视化（30分钟）

**目的**: 复核大病灶完全漏检病例

**筛选标准**: Dice=0, n_pred=0, n_ref > 1000

```bash
python script/v1/visualize_bph_failures.py \
    --all-failures \
    --mode complete_miss \
    --top-n 10 \
    --output viz/p0_complete_miss
```

**输出图像命名**：
```
BPH_0000661886_axial_slice045.png
BPH_0000661886_axial_slice046.png
BPH_0000661886_axial_slice047.png
...
```

**图像解读指南**：

```
┌─────────────────────────────────────────────────────────────┐
│  布局（2行 x 3列）                                            │
├─────────────┬─────────────┬─────────────────────────────────┤
│   T2原图     │   ADC原图    │   DWI原图                        │
│   (灰度)     │   (灰度)     │   (灰度)                         │
├─────────────┼─────────────┼─────────────────────────────────┤
│ T2 + GT     │ T2 + Pred   │ T2 + GT + Pred (叠加)             │
│ (红色=金标)  │ (绿色=预测)  │  红色 = FN (漏检)                 │
│             │             │  绿色 = FP (误检)                 │
│             │             │  黄色 = TP (正确)                 │
└─────────────┴─────────────┴─────────────────────────────────┘
```

**复核检查清单**：

- [ ] 金标准标签 (红色) 是否在 T2 上可见？
- [ ] 病灶边界是否清晰？
- [ ] 预测图 (绿色) 是否完全空白？
- [ ] ADC/DWI 上病灶信号是否异常？
- [ ] 病灶位置是否在前列腺典型区域？

---

### 步骤 4: P1 优先级病例可视化（30分钟）

**目的**: 复核零交并+误检病例

**筛选标准**: Dice=0, n_pred>0

```bash
python script/v1/visualize_bph_failures.py \
    --all-failures \
    --mode false_positive \
    --top-n 10 \
    --output viz/p1_false_positive
```

**重点关注**：
- 误检区域 (绿色) 是否集中在特定解剖结构？
- 是否存在系统性空间偏移？
- 假阳性是否与真病灶有相似强度特征？

---

### 步骤 5: 生成多视图对比（可选，20分钟）

**目的**: 获取矢状位和冠状位视图，确认空间关系

```bash
python script/v1/visualize_bph_failures.py \
    --all-failures \
    --mode zero_dice \
    --top-n 5 \
    --views axial sagittal coronal \
    --output viz/multi_view
```

---

### 步骤 6: 单个病例深度分析（按需）

**目的**: 对特定病例进行详细检查

```bash
# 指定病例分析
python script/v1/visualize_bph_failures.py \
    --case BPH_0000044758 \
    --views axial sagittal coronal \
    --output viz/single_case
```

---

## 4. 完整复核命令速查

### 快速版（1小时）

```bash
# 1. 生成报告
python script/v1/visualize_bph_failures.py --all-failures --report-only --output viz/report

# 2. 特征对比
python script/v1/compare_bph_pca_features.py --output analysis/bph_pca_comparison

# 3. 可视化前5个完全漏检病例
python script/v1/visualize_bph_failures.py --all-failures --mode complete_miss --top-n 5 --output viz/p0_top5
```

### 标准版（2-3小时）

```bash
# 1. 报告 + 特征分析
python script/v1/visualize_bph_failures.py --all-failures --report-only --output viz/report
python script/v1/compare_bph_pca_features.py --output analysis/bph_pca_comparison

# 2. P0 优先级（完全漏检）
python script/v1/visualize_bph_failures.py --all-failures --mode complete_miss --top-n 10 --output viz/p0

# 3. P1 优先级（零交并+误检）
python script/v1/visualize_bph_failures.py --all-failures --mode false_positive --top-n 10 --output viz/p1

# 4. 低Dice病例（0 < Dice < 0.3）
python script/v1/visualize_bph_failures.py --all-failures --mode all --top-n 10 --output viz/low_dice
```

### 完整版（4-5小时）

```bash
# 包含所有失败病例的多视图可视化
python script/v1/visualize_bph_failures.py --all-failures --mode zero_dice --views axial sagittal coronal --output viz/complete
```

---

## 5. 复核结果分析框架

### 5.1 标签质量检查

| 检查项 | 方法 | 预期结果 |
|--------|------|----------|
| 标注一致性 | 对比 BPH_0000044758 和 PCA_0000166190 | 边界风格应一致 |
| 病灶定义 | 检查 BPH 是否多为弥漫性 | 若与 PCA 局灶性不同，需记录 |
| 标注完整性 | 检查 T2 上可见但无标签区域 | 发现漏标需反馈 |

### 5.2 影像表型分析

| 特征 | BPH 典型表现 | PCA 典型表现 | 模型影响 |
|------|-------------|-------------|----------|
| T2 信号 | 等/稍高信号 | 低信号 | BPH 对比度低，难检测 |
| ADC 值 | 较高 (限制轻) | 较低 (限制重) | BPH 与正常组织重叠 |
| DWI 信号 | 不高 | 高信号 | BPH 缺乏明显高信号 |
| 边界 | 模糊/弥漫 | 清晰/局灶 | BPH 边界难界定 |

### 5.3 失败模式分类

根据可视化结果，将失败病例归类：

```
失败模式分类：
├─ A类: 完全漏检 (n_pred=0)
│   ├─ A1: 病灶在影像上不可见 → 可能是标签错误
│   ├─ A2: 病灶可见但信号弱 → 对比度不足
│   └─ A3: 病灶明显但被忽略 → 模型偏见
│
├─ B类: 零交并+误检 (Dice=0, n_pred>0)
│   ├─ B1: FP在固定区域 → 解剖结构误识别
│   ├─ B2: FP分散随机 → 噪声敏感
│   └─ B3: FP与病灶相似 → 特征混淆
│
└─ C类: 低Dice (0 < Dice < 0.3)
    ├─ C1: 欠分割为主 → 边界保守
    ├─ C2: 过分割为主 → 边界激进
    └─ C3: 位置偏移 → 配准或定位问题
```

---

## 6. 常见问题排查

### 6.1 脚本运行错误

**错误**: `FileNotFoundError: Image not found`

**解决**: 检查数据路径是否正确
```bash
# 验证路径
ls nnUNet_raw/Dataset101_Prostate/imagesTr/BPH_0000044758_0000.nii.gz
ls nnUNet_raw/Dataset101_Prostate/labelsTr/BPH_0000044758.nii.gz
```

**错误**: `ModuleNotFoundError: No module named 'nibabel'`

**解决**: 安装依赖
```bash
pip install nibabel numpy matplotlib
```

### 6.2 图像质量问题

**问题**: 图像太暗/太亮

**解决**: 脚本已使用百分位数归一化 (1%-99%)，如仍有问题可手动调整：
```python
# 在 visualize_bph_failures.py 中修改 normalize_image 函数
p1, p99 = np.percentile(image[image > 0], [1, 99])  # 可改为 [5, 95]
```

**问题**: 切片位置不对

**解决**: 脚本自动定位病灶区域，如需要特定切片，可手动指定：
```python
# 修改 find_roi_slices 函数中的切片选择逻辑
```

---

## 7. 输出结果归档

建议的目录结构：

```
nnunet/
├── docs/
│   └── v1/
│       ├── fold0_training_summary.md
│       ├── fold_0_failure_analysis.md
│       └── bph_visualization_guide.md  # 本文档
│
├── viz/                          # 可视化结果
│   ├── report/
│   │   └── bph_failures_report.txt
│   ├── p0_complete_miss/         # P0 优先级
│   │   ├── BPH_0000661886_axial_slice045.png
│   │   └── ...
│   ├── p1_false_positive/        # P1 优先级
│   │   └── ...
│   └── multi_view/               # 多视图
│       └── ...
│
├── analysis/                     # 特征分析
│   └── bph_pca_comparison/
│       ├── comparison_report.txt
│       ├── feature_comparison.png
│       ├── scatter_analysis.png
│       └── features_data.json
│
└── script/
    └── v1/
        ├── visualize_bph_failures.py
        └── compare_bph_pca_features.py
```

---

## 8. 后续行动建议

完成可视化复核后，根据发现采取相应行动：

### 如果发现标签问题
- [ ] 整理问题病例清单
- [ ] 与标注团队/临床医生复核
- [ ] 考虑重新标注或排除问题数据

### 如果发现影像表型差异
- [ ] 在文档中记录 BPH 和 PCA 的典型特征差异
- [ ] 考虑修改任务定义（多类标签或分开建模）
- [ ] 设计针对性的数据增强策略

### 如果发现模型偏见
- [ ] 尝试 lesion-aware sampling
- [ ] 调整损失函数（Tversky/Focal Loss）
- [ ] 考虑级联或多尺度策略

---

## 附录 A: 命令行参数完整参考

### visualize_bph_failures.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--case` | str | None | 指定单个病例 |
| `--all-failures` | flag | False | 批量处理所有失败病例 |
| `--mode` | choice | zero_dice | 筛选模式: all/zero_dice/false_positive/complete_miss |
| `--top-n` | int | None | 只处理前N个（按病灶大小排序） |
| `--output` | Path | viz/bph_failures | 输出目录 |
| `--views` | list | [axial] | 视图: axial/sagittal/coronal |
| `--report-only` | flag | False | 只生成报告 |

### compare_bph_pca_features.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | Path | analysis/bph_pca_comparison | 输出目录 |
| `--max-cases` | int | None | 最大处理病例数（测试用） |

---

## 附录 B: 关键病例速查

### P0 优先级（完全漏检 + 大病灶）

| 病例 | n_ref | 备注 |
|------|-------|------|
| BPH_0000661886 | 4752 | 最大失败病灶 |
| BPH_0000044758 | 3705 | 同时有 FP=496 |
| BPH_0000909756 | 3554 | 完全无预测 |
| BPH_0001010008 | 3328 | 完全无预测 |
| BPH_0000542541 | 2102 | 完全无预测 |
| BPH_0000748787 | 1840 | 完全无预测 |

### P1 优先级（零交并 + 误检）

| 病例 | n_ref | n_pred | 备注 |
|------|-------|--------|------|
| BPH_0000027417 | 2372 | 2651 | FP > FN |
| BPH_0000117901 | 1268 | 1228 | FP ≈ FN |
| BPH_0000090107 | 1037 | 545 | 部分误检 |

### 对照组（成功案例）

| 病例 | Dice | 类型 | 备注 |
|------|------|------|------|
| PCA_0000166190 | 0.9323 | PCA | 最佳案例 |
| PCA_0000887724 | 0.8796 | PCA | 小病灶成功 |
| PCA_0000647594 | 0.8556 | PCA | 大病灶成功 |

---

**文档版本**: v1.0  
**最后更新**: 2026-03-13  
**作者**: AI Assistant
