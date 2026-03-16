#!/usr/bin/env python3
"""
BPH vs PCA 影像特征快速对比分析

功能：
1. 对比 BPH 和 PCA 病灶的影像强度特征
2. 分析病灶大小、位置分布差异
3. 生成统计报告和对比图

用法：
    python script/compare_bph_pca_features.py --output analysis/bph_pca_comparison
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # 从 v1 目录向上到项目根目录
SUMMARY_JSON = PROJECT_ROOT / "nnUNet_results/Dataset101_Prostate/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/summary.json"
IMAGES_DIR = PROJECT_ROOT / "nnUNet_raw/Dataset101_Prostate/imagesTr"
LABELS_DIR = PROJECT_ROOT / "nnUNet_raw/Dataset101_Prostate/labelsTr"


def load_summary() -> Dict:
    """加载验证结果"""
    with open(SUMMARY_JSON, 'r') as f:
        return json.load(f)


def extract_case_features(case_name: str, case_metrics: Dict) -> Dict:
    """提取单个病例的影像特征"""
    try:
        # 加载标签
        label_path = LABELS_DIR / f"{case_name}.nii.gz"
        if not label_path.exists():
            return None
        
        label = np.asarray(nib.load(str(label_path)).dataobj)
        label_mask = label > 0
        
        if not label_mask.any():
            return None
        
        # 加载各模态影像
        features = {
            "case_name": case_name,
            "category": "BPH" if case_name.startswith("BPH_") else "PCA",
            "n_voxels": int(label_mask.sum()),
            "volume_mm3": int(label_mask.sum()),  # 假设体素大小为 1mm^3
        }
        
        # 病灶位置（质心）
        centroid = np.mean(np.argwhere(label_mask), axis=0)
        features["centroid_z"] = centroid[2]  # 轴向位置
        
        # 各模态强度统计
        for mod_idx, mod_name in [(0, "T2"), (1, "ADC"), (2, "DWI")]:
            img_path = IMAGES_DIR / f"{case_name}_{mod_idx:04d}.nii.gz"
            if not img_path.exists():
                continue
            
            image = np.asarray(nib.load(str(img_path)).dataobj)
            lesion_voxels = image[label_mask]
            
            if len(lesion_voxels) > 0:
                features[f"{mod_name}_mean"] = float(np.mean(lesion_voxels))
                features[f"{mod_name}_std"] = float(np.std(lesion_voxels))
                features[f"{mod_name}_median"] = float(np.median(lesion_voxels))
                
                # 计算与背景的对比度
                background_voxels = image[~label_mask & (image > 0)]
                if len(background_voxels) > 0:
                    bg_mean = np.mean(background_voxels)
                    contrast = (features[f"{mod_name}_mean"] - bg_mean) / bg_mean if bg_mean != 0 else 0
                    features[f"{mod_name}_contrast"] = float(contrast)
        
        # 添加验证指标
        metrics = case_metrics["metrics"]["1"]
        features["dice"] = metrics["Dice"]
        features["recall"] = metrics["TP"] / (metrics["TP"] + metrics["FN"]) if (metrics["TP"] + metrics["FN"]) > 0 else 0
        features["precision"] = metrics["TP"] / (metrics["TP"] + metrics["FP"]) if (metrics["TP"] + metrics["FP"]) > 0 else 0
        
        return features
        
    except Exception as e:
        print(f"Error processing {case_name}: {e}")
        return None


def analyze_features(features_list: List[Dict]) -> Tuple[Dict, Dict]:
    """分析 BPH 和 PCA 的特征差异"""
    bph_features = [f for f in features_list if f["category"] == "BPH"]
    pca_features = [f for f in features_list if f["category"] == "PCA"]
    
    def compute_stats(features, key):
        """计算统计量"""
        values = [f[key] for f in features if key in f]
        if not values:
            return None
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "min": np.min(values),
            "max": np.max(values),
            "n": len(values)
        }
    
    # 要分析的指标
    metrics = ["n_voxels", "T2_mean", "T2_contrast", "ADC_mean", "ADC_contrast", 
               "DWI_mean", "DWI_contrast", "centroid_z", "dice", "recall", "precision"]
    
    bph_stats = {m: compute_stats(bph_features, m) for m in metrics}
    pca_stats = {m: compute_stats(pca_features, m) for m in metrics}
    
    return bph_stats, pca_stats


def print_comparison_report(bph_stats: Dict, pca_stats: Dict, output_path: Path):
    """打印对比报告"""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BPH vs PCA 影像特征对比分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Metric':<20} {'BPH (mean±std)':<25} {'PCA (mean±std)':<25} {'Ratio (PCA/BPH)'}\n")
        f.write("-" * 80 + "\n")
        
        for metric in ["n_voxels", "T2_mean", "T2_contrast", "ADC_mean", "ADC_contrast",
                       "DWI_mean", "DWI_contrast", "centroid_z", "dice", "recall", "precision"]:
            bph = bph_stats.get(metric)
            pca = pca_stats.get(metric)
            
            if bph and pca:
                bph_str = f"{bph['mean']:.2f}±{bph['std']:.2f}"
                pca_str = f"{pca['mean']:.2f}±{pca['std']:.2f}"
                ratio = pca['mean'] / bph['mean'] if bph['mean'] != 0 else float('inf')
                ratio_str = f"{ratio:.2f}x"
                
                f.write(f"{metric:<20} {bph_str:<25} {pca_str:<25} {ratio_str}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("关键发现:\n")
        f.write("-" * 80 + "\n")
        
        # 自动发现关键差异
        if bph_stats.get("dice") and pca_stats.get("dice"):
            dice_ratio = pca_stats["dice"]["mean"] / bph_stats["dice"]["mean"] if bph_stats["dice"]["mean"] > 0 else float('inf')
            f.write(f"1. Dice 差异: PCA 是 BPH 的 {dice_ratio:.1f} 倍\n")
        
        if bph_stats.get("recall") and pca_stats.get("recall"):
            recall_ratio = pca_stats["recall"]["mean"] / bph_stats["recall"]["mean"] if bph_stats["recall"]["mean"] > 0 else float('inf')
            f.write(f"2. Recall 差异: PCA 是 BPH 的 {recall_ratio:.1f} 倍\n")
        
        if bph_stats.get("T2_contrast") and pca_stats.get("T2_contrast"):
            bph_contrast = bph_stats["T2_contrast"]["mean"]
            pca_contrast = pca_stats["T2_contrast"]["mean"]
            f.write(f"3. T2 对比度: BPH={bph_contrast:.3f}, PCA={pca_contrast:.3f}\n")
        
        if bph_stats.get("ADC_mean") and pca_stats.get("ADC_mean"):
            bph_adc = bph_stats["ADC_mean"]["mean"]
            pca_adc = pca_stats["ADC_mean"]["mean"]
            f.write(f"4. ADC 均值: BPH={bph_adc:.2f}, PCA={pca_adc:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    # 同时打印到控制台
    with open(output_path, 'r') as f:
        print(f.read())


def plot_feature_distributions(features_list: List[Dict], output_dir: Path):
    """绘制特征分布对比图"""
    bph_features = [f for f in features_list if f["category"] == "BPH"]
    pca_features = [f for f in features_list if f["category"] == "PCA"]
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("BPH vs PCA Feature Comparison", fontsize=16, fontweight='bold')
    
    def plot_comparison(ax, metric, title, xlabel):
        bph_values = [f[metric] for f in bph_features if metric in f]
        pca_values = [f[metric] for f in pca_features if metric in f]
        
        if not bph_values or not pca_values:
            ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            return
        
        # 箱线图
        bp = ax.boxplot([bph_values, pca_values], labels=['BPH', 'PCA'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        
        ax.set_title(title)
        ax.set_ylabel(xlabel)
        ax.grid(True, alpha=0.3)
        
        # 添加均值标记
        ax.scatter([1], [np.mean(bph_values)], color='red', s=100, marker='D', zorder=5, label='Mean')
        ax.scatter([2], [np.mean(pca_values)], color='blue', s=100, marker='D', zorder=5)
    
    # 绘制各特征对比
    plot_comparison(axes[0, 0], "n_voxels", "Lesion Volume (voxels)", "Voxels")
    plot_comparison(axes[0, 1], "T2_contrast", "T2 Contrast (vs Background)", "Contrast Ratio")
    plot_comparison(axes[0, 2], "ADC_mean", "ADC Mean Intensity", "Intensity")
    plot_comparison(axes[1, 0], "DWI_mean", "DWI Mean Intensity", "Intensity")
    plot_comparison(axes[1, 1], "dice", "Validation Dice Score", "Dice")
    plot_comparison(axes[1, 2], "recall", "Validation Recall", "Recall")
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature comparison plot saved to: {output_dir / 'feature_comparison.png'}")


def plot_scatter_analysis(features_list: List[Dict], output_dir: Path):
    """绘制散点图分析关系"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("BPH vs PCA Scatter Analysis", fontsize=16, fontweight='bold')
    
    bph_features = [f for f in features_list if f["category"] == "BPH"]
    pca_features = [f for f in features_list if f["category"] == "PCA"]
    
    # 1. Volume vs Dice
    ax = axes[0]
    bph_vol = [f["n_voxels"] for f in bph_features if "n_voxels" in f and "dice" in f]
    bph_dice = [f["dice"] for f in bph_features if "n_voxels" in f and "dice" in f]
    pca_vol = [f["n_voxels"] for f in pca_features if "n_voxels" in f and "dice" in f]
    pca_dice = [f["dice"] for f in pca_features if "n_voxels" in f and "dice" in f]
    
    ax.scatter(bph_vol, bph_dice, c='red', alpha=0.6, label='BPH', s=50)
    ax.scatter(pca_vol, pca_dice, c='blue', alpha=0.6, label='PCA', s=50)
    ax.set_xlabel("Lesion Volume (voxels)")
    ax.set_ylabel("Dice Score")
    ax.set_title("Volume vs Segmentation Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. T2 Contrast vs Recall
    ax = axes[1]
    bph_contrast = [f["T2_contrast"] for f in bph_features if "T2_contrast" in f and "recall" in f]
    bph_recall = [f["recall"] for f in bph_features if "T2_contrast" in f and "recall" in f]
    pca_contrast = [f["T2_contrast"] for f in pca_features if "T2_contrast" in f and "recall" in f]
    pca_recall = [f["recall"] for f in pca_features if "T2_contrast" in f and "recall" in f]
    
    ax.scatter(bph_contrast, bph_recall, c='red', alpha=0.6, label='BPH', s=50)
    ax.scatter(pca_contrast, pca_recall, c='blue', alpha=0.6, label='PCA', s=50)
    ax.set_xlabel("T2 Contrast (vs Background)")
    ax.set_ylabel("Recall")
    ax.set_title("Contrast vs Detection Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter analysis plot saved to: {output_dir / 'scatter_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="BPH vs PCA 影像特征对比分析")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "analysis" / "bph_pca_comparison",
                       help="输出目录")
    parser.add_argument("--max-cases", type=int, default=None,
                       help="最多处理多少病例（用于测试）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading summary...")
    summary = load_summary()
    
    print("Extracting features from validation cases...")
    features_list = []
    
    cases = summary.get("metric_per_case", [])
    if args.max_cases:
        cases = cases[:args.max_cases]
    
    for i, case_data in enumerate(cases):
        pred_file = case_data["prediction_file"]
        case_name = Path(pred_file).stem
        
        # 只处理 BPH 和 PCA
        if not (case_name.startswith("BPH_") or case_name.startswith("PCA_")):
            continue
        
        print(f"  [{i+1}/{len(cases)}] Processing {case_name}...")
        features = extract_case_features(case_name, case_data)
        if features:
            features_list.append(features)
    
    print(f"\nSuccessfully extracted features from {len(features_list)} cases")
    
    # 统计分析
    print("\nAnalyzing features...")
    bph_stats, pca_stats = analyze_features(features_list)
    
    # 生成报告
    report_path = output_dir / "comparison_report.txt"
    print_comparison_report(bph_stats, pca_stats, report_path)
    
    # 生成可视化
    print("\nGenerating visualizations...")
    plot_feature_distributions(features_list, output_dir)
    plot_scatter_analysis(features_list, output_dir)
    
    # 保存原始数据
    import json
    with open(output_dir / "features_data.json", 'w') as f:
        json.dump(features_list, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
