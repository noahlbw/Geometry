# Qwen20 → VIP comparison vocabulary (2026-10-18)

Each dataset JSON contains exactly 20 queries per class: the canonical taxonomy name followed by 19 text-only Qwen2.5-VL-7B proposals. `selected/` contains the frozen VIP paper-rule banks after full-image VG/SC screening. `comparison_vocabulary.json` records candidate/selected SHA256 identities and selected counts.

Generation did not read images, masks, labels, mIoU, or VIP scores. Screening uses all unlabeled evaluation images, so the distilled bank is transductive; see `research/qwen20_vip_paper_20261018/REPORT_ZH.md`.
