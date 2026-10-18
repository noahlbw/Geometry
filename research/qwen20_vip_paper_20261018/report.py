"""Build the auditable report for the Qwen20 -> VIP paper-rule run."""
from __future__ import annotations

import json
from pathlib import Path


DATASETS = ("vdd", "potsdam", "udd5", "oem", "loveda", "vaihingen", "landcoverai", "flair1")
ROOT = Path(__file__).resolve().parent
VOCAB = ROOT.parents[1] / "vocabularies" / "qwen20_vip_20261018"
OLD = ROOT.parents[0] / "vip_paper_distillation_full_20261003"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def protocols(dataset: str, row: dict):
    return ("P", "D") if dataset == "loveda" else (dataset,)


def value(row: dict, protocol: str, arm: str) -> float:
    return float(row["metrics"][protocol][arm]["mean_iou_percent"])


def old_value(dataset: str, protocol: str):
    path = OLD / dataset / "merged.json"
    if not path.is_file():
        return None
    row = load(path)
    metric = row.get("metrics", {}).get(protocol)
    if metric is None:
        return None
    if "VIP_All20" in metric:
        return float(metric["VIP_All20"]["mean_iou_percent"])
    if "mean_iou_percent" in metric:
        return float(metric["mean_iou_percent"])
    return None


def main() -> None:
    rows = []
    all_class_rows = []
    summary = {"implementation": "qwen20-vip-paper-distillation-v1-20261018",
               "datasets": {}, "all_protocols_loveda_d_once": {}}
    for dataset in DATASETS:
        current = load(ROOT / dataset / "merged.json")
        selected = load(VOCAB / "selected" / (dataset + ".json"))
        counts = {p: list(bank["selected_counts"]) for p, bank in selected["banks"].items()}
        retention = {p: sum(c) / (20 * len(c)) for p, c in counts.items()}
        for protocol in protocols(dataset, current):
            all20 = value(current, protocol, "VIP_All20")
            distilled = value(current, protocol, "VIP_Distilled")
            rows.append({"dataset": dataset, "protocol": protocol, "images": current["processed_images"],
                         "all20": all20, "distilled": distilled, "screening_gain_pp": distilled - all20,
                         "historical_vip_all20": old_value(dataset, protocol),
                         "selected_counts": counts[protocol], "retention_fraction": retention[protocol],
                         "selection_seconds": selected["selection_parallel_wall_seconds"],
                         "selection_peak_mib": selected["selection_peak_cuda_memory_mb"],
                         "evaluation_seconds": current["evaluation_parallel_wall_seconds"],
                         "evaluation_peak_mib": current["evaluation_peak_cuda_memory_mb"]})
            for all20_class, distilled_class in zip(current["metrics"][protocol]["VIP_All20"]["per_class"],
                                                    current["metrics"][protocol]["VIP_Distilled"]["per_class"]):
                if all20_class["iou_percent"] is not None and distilled_class["iou_percent"] is not None:
                    all_class_rows.append({"dataset": dataset, "protocol": protocol,
                        "class": distilled_class["name"], "all20": all20_class["iou_percent"],
                        "distilled": distilled_class["iou_percent"],
                        "delta_pp": distilled_class["iou_percent"] - all20_class["iou_percent"]})
        summary["datasets"][dataset] = {"selected_counts": counts, "retention_fraction": retention,
            "selection_signature": selected["source_signature"],
            "selection_seconds": selected["selection_parallel_wall_seconds"],
            "selection_peak_mib": selected["selection_peak_cuda_memory_mb"],
            "evaluation_seconds": current["evaluation_parallel_wall_seconds"],
            "evaluation_peak_mib": current["evaluation_peak_cuda_memory_mb"]}
    scored = [r for r in rows if not (r["dataset"] == "loveda" and r["protocol"] == "P")]
    summary["all_protocols_loveda_d_once"] = {
        "count": len(scored), "all20_mean": sum(r["all20"] for r in scored) / len(scored),
        "distilled_mean": sum(r["distilled"] for r in scored) / len(scored),
        "screening_gain_mean_pp": sum(r["screening_gain_pp"] for r in scored) / len(scored),
        "distilled_wins": sum(r["distilled"] > r["all20"] for r in scored),
    }
    (ROOT / "comparison_summary.json").write_text(json.dumps({"rows": rows, "classes": all_class_rows,
        "summary": summary}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = [
        "# Qwen20 candidates → VIP paper-rule screening: 8-domain full evaluation", "",
        "本实验不读取 VIP 官方词表。每个类别由 Qwen2.5-VL-7B 生成一个固定 20-query bank：canonical 类别名 + 19 个文本候选。生成阶段不读取图像、mask、mIoU 或 VIP 分数；只做 JSON 格式、重复和 exact-rival 的机械检查。原始 prompt/response 保存在 `vocabularies/qwen20_vip_20261018/generation.json`。", "",
        "随后对每个数据集的全部无标签评测图像计算 VIP paper-rule 的 VG/SC：保留 canonical；候选还必须满足 cosine≥0.7、VG 高于 canonical、SC 低于 canonical。没有数量 quota。冻结后的 bank 再对同一完整图像序列评测，因此这是 transductive 的 VIP 筛词复现，不应当写成独立测试集结果。mask 只在验证阶段加载。", "",
        "## 结果", "",
        "`VIP_All20` 是 Qwen20 全候选；`VIP_Distilled` 是完全相同观测经过 VG/SC 筛选后的 bank。", "",
        "| 数据集/协议 | 图像数 | Qwen All20 | VIP Distilled | 筛选增益 pp | 保留数/类 | 保留率 | 历史 VIP All20* |", "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"
    ]
    for row in rows:
        old = "--" if row["historical_vip_all20"] is None else f"{row['historical_vip_all20']:.4f}"
        counts = ",".join(map(str, row["selected_counts"]))
        lines.append(f"| {row['dataset']}/{row['protocol']} | {row['images']} | {row['all20']:.4f} | {row['distilled']:.4f} | {row['screening_gain_pp']:+.4f} | [{counts}] | {100*row['retention_fraction']:.1f}% | {old} |")
    mean = summary["all_protocols_loveda_d_once"]
    lines += ["", f"8 协议均值（LoveDA 只计 D）：All20 **{mean['all20_mean']:.4f}**，Distilled **{mean['distilled_mean']:.4f}**，平均筛选增益 **{mean['screening_gain_mean_pp']:+.4f} pp**；Distilled 胜出 **{mean['distilled_wins']}/{mean['count']}** 个协议。", "", "*历史 VIP All20 仅在仓库存在对应冻结记录时列出；不同记录的输入/数值修复口径不完全相同，不能把它当作本次 Qwen bank 的同词表对照。", "", "## 各类筛选变化", "", "| 数据集/协议 | 类别 | All20 IoU | Distilled IoU | Δ pp |", "| --- | --- | ---: | ---: | ---: |"]
    for row in all_class_rows:
        lines.append(f"| {row['dataset']}/{row['protocol']} | {row['class']} | {row['all20']:.4f} | {row['distilled']:.4f} | {row['delta_pp']:+.4f} |")
    lines += ["", "## 筛选与推理成本", "", "selection 是 VG/SC 全图统计；evaluation 使用缓存 logits，同时重建 All20 和 Distilled。峰值显存为单个 worker 的 allocated MiB，初始化和 Qwen 离线生成不计入。", "", "| 数据集 | 筛选秒 | 筛选峰值 MiB | 验证秒 | 验证峰值 MiB |", "| --- | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        row = next(x for x in rows if x["dataset"] == dataset)
        lines.append(f"| {dataset} | {row['selection_seconds']:.2f} | {row['selection_peak_mib']:.1f} | {row['evaluation_seconds']:.2f} | {row['evaluation_peak_mib']:.1f} |")
    lines += ["", "## 复现边界", "", "- VIP 的公开仓库没有公开完整 distillation implementation；这里实现并固定了论文文字规则，代码和 config 已保存。", "- VDD、Potsdam、Vaihingen 也显式走本次 Qwen JSON，不会回退到 VIP 官方 short-query 文件。", "- Qwen generation、VG/SC selection、冻结 bank、full evaluation 的 JSON 和 SHA256 都保留；selection 阶段没有加载 mask。", "- 结果包含负增益（例如 Potsdam、Vaihingen），因此不能把 VG/SC 筛选描述成对所有域必然有效。", "", "完整机器可读汇总：`comparison_summary.json`；候选词表：`vocabularies/qwen20_vip_20261018/`。"]
    (ROOT / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["all_protocols_loveda_d_once"], ensure_ascii=False))


if __name__ == "__main__":
    main()
