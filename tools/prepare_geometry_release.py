"""Stage the Geometry source and measured evidence without weights or credentials."""
import argparse
import json
from pathlib import Path
import re
import shutil


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT = WORKSPACE / 'artifacts/geometry_release_20261009'
SOURCE_SUFFIXES = {'.py', '.sh', '.json', '.toml', '.yaml', '.yml'}
REPORT_TOPICS = ('ALIAS', 'GEAR', 'GEOMETRY', 'RIVAL', 'SHARED', 'SEMANTIC', 'LOCAL',
    'VARIABLE', 'CROSS', 'TAXONOMY', 'DEVELOPMENT', 'NATURAL', 'BOUNDED', 'PATCH',
    'READOUT', 'RESIDUAL', 'FAMILY', 'FINAL', 'CVPR', 'VIP', 'COUNT', 'ONE_SIDED',
    'SUPPORTED', 'SURVIVAL', 'TARGET_CONTEXT', 'EXCESS', 'STRATIFIED', 'GENERATION',
    'MATCHED_CONTRIBUTION', 'WITNESS', 'CANONICAL', 'TEMPLATE', 'REGION_RESIDUAL',
    'FOOTPRINT', 'RECIPROCAL', 'POOL_ISOLATED', 'RESPONSE_COLLISION', 'OWN_PEER')
SECRET = re.compile(r'(?<![A-Za-z0-9_])(?:github_pat_[A-Za-z0-9_]{20,}|'
    r'gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{24,})|'
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def write_index(destination):
    lines = ['# 实验、证据与源码索引', '',
        '本页索引保留探索过程，不把所有历史模块加入终版。当前精度/速度比较以 [RESULTS.md](RESULTS.md) 为准，设计以 [ARCHITECTURE.md](ARCHITECTURE.md) 为准。', '',
        '历史文档可能含当时的pilot、NaN污染VIP、旧输入/词库、被否定的候选和本机绝对路径；不能将它们拼接为最新主表。仅有protocol/plan不表示该候选完成全量实验。逐图缓存、旧原始日志与所有历史输出未全部打包。', '',
        '## 核心方法到源码', '',
        '| 方法/职责 | 实现 |', '| --- | --- |']
    methods = [
        ('冻结mask-free入口', 'taxonomy_inference'),
        ('任务配置与聚合', 'taxonomy_readout'),
        ('Geometry执行/关系', 'geometry_execution'),
        ('Geometry语义head干预', 'geometry_readout_trace'),
        ('耦合写回与既定profile', 'development_readout'),
        ('局部LME与stream归约', 'packed_alias_readout'),
        ('VIP有限值观测', 'finite_vip_observer'),
        ('VIP论文规则蒸馏', 'vip_alias_distillation'),
        ('细视野竞争硬准入', 'fine_alias_view'),
        ('RivalFine全量路线', 'rival_fine_coupling'),
        ('等价缓存加速', 'rival_alias_fast'),
        ('细视野软风险/正增量', 'rival_alias_influence_soft'),
        ('无新增RGB的特征复用', 'sparse_alias_reuse'),
        ('SharedLocal语义头', 'shared_local_alias'),
        ('流式alias处理', 'one_sided_alias_stream'),
        ('词族质量与类偏移', 'family_mass_alias'),
        ('局部词源分配', 'local_role_alias'),
        ('跨支持增量诊断', 'cross_support_alias_diagnostic'),
        ('稀疏竞争语义约束', 'semantic_contract_alias'),
        ('coverage-soft最终开关', 'coverage_soft_alias'),
        ('变长词表输入', 'variable_alias_vocabulary'),
        ('独立残余本体', 'residual_ontology'),
    ]
    for label, name in methods:
        relative = Path('DINOtool/dinotool') / (name + '.py')
        if not (destination / relative).is_file():
            raise FileNotFoundError(relative)
        lines.append(f'| {label} | [{name}.py](../{relative.as_posix()}) |')
    lines += ['', '## 全部历史报告（按日期）', '']
    reports = sorted((destination / 'research').glob('*.md'),
                     key=lambda p: (re.search(r'2026\d{4}', p.name)[0], p.name))
    previous = None
    for report in reports:
        date = re.search(r'2026\d{4}', report.name)[0]
        if date != previous:
            lines += [f'### {date}', '']
            previous = date
        lines.append(f'- [{report.stem}](../research/{report.name})')
    lines += ['', '## 全部词处理与Geometry实现', '',
        '下面按文件名列出实现；方法是否有效及是否保留，看综述和对应配对报告。', '']
    for source in sorted((destination / 'DINOtool/dinotool').glob('*.py')):
        if any(word in source.stem for word in ('alias', 'rival', 'geometry', 'family', 'taxonomy', 'semantic')):
            lines.append(f'- [{source.name}](../DINOtool/dinotool/{source.name})')
    lines += ['', '## 冻结证据', '',
        '- [完整协议、词库与模板](../research/alias_finalization_20261009/protocol.json)',
        '- [最终策略、各VIP来源与汇总](../research/alias_finalization_20261009/final_model.json)',
        '- [覆盖及计时汇总](../research/alias_finalization_20261009/comparison_summary.json)',
        '- [逐类别竞争与误差](../research/alias_finalization_20261009/class_competition.json)',
        '- [配对区间](../research/alias_finalization_20261009/statistics_full/PAIRED_UNCERTAINTY.md)',
        '- [LoveDA参考恢复](../research/alias_finalization_20261009/reference_replay/loveda/verification.json)',
        '- [Geometry近邻创新边界](../research/geometry_publication_20261001/NOVELTY_AND_MECHANISM_AUDIT.md)', '']
    (destination / 'docs/EXPERIMENT_INDEX.md').write_text('\n'.join(lines), encoding='utf-8')


def stage(destination):
    destination = destination.resolve()
    destination.relative_to((WORKSPACE / 'artifacts').resolve())
    destination.mkdir(parents=True, exist_ok=True)
    copied = []

    def copy(relative):
        source = WORKSPACE / relative
        target = destination / relative
        text = source.read_text(encoding='utf-8', errors='replace')
        if SECRET.search(text):
            raise RuntimeError('Credential-like material; inspect privately: ' + str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative.as_posix())

    for folder in ('dinotool', 'scripts', 'tests', 'configs', 'dinov3_hub'):
        for source in sorted((WORKSPACE / 'DINOtool' / folder).rglob('*')):
            if (source.is_file() and source.suffix in SOURCE_SUFFIXES
                    and not any(part.startswith('.') or part=='__pycache__' for part in source.relative_to(WORKSPACE).parts)):
                copy(source.relative_to(WORKSPACE))
    for name in ('README.md', 'RESEARCH.md', 'pyproject.toml'):
        copy(Path('DINOtool') / name)
    for source in sorted((WORKSPACE / 'research').glob('*.md')):
        # Pre-Geometry projects in this shared workspace are not this release.
        match = re.search(r'(2026\d{4})', source.name)
        if source.name.startswith(REPORT_TOPICS) and match and match[1] >= '20260929':
            copy(source.relative_to(WORKSPACE))
    base = Path('research/alias_finalization_20261009')
    for name in ('protocol.json', 'comparison_summary.json', 'final_model.json', 'class_competition.json',
                 'COMPLETION_AUDIT.md', 'TRANSFER_COMPETITION_DIAGNOSIS.md',
                 'reference_recovery_results.json', 'reference_recovery_status.json',
                 'natural_vip_cost_reference_recovery_results.json',
                 'reference_replay/loveda/verification.json',
                 'statistics_full/paired_statistics.json', 'statistics_full/PAIRED_UNCERTAINTY.md'):
        copy(base / name)
    manifest = json.loads((WORKSPACE / base / 'final_model.json').read_text(encoding='utf-8'))
    comparators = set()
    for dataset in manifest['datasets']:
        copy(base / dataset / 'merged.json')
        for source in sorted((WORKSPACE / base / dataset).glob('cost_*.json')):
            copy(source.relative_to(WORKSPACE))
        for values in manifest['vip_comparators'][dataset].values():
            for comparator in values.values():
                if comparator:
                    comparators.add(Path(comparator['source'].replace('\\', '/')))
    for relative in sorted(comparators):
        copy(relative)
    for relative in ('research/geometry_publication_20261001/NOVELTY_AND_MECHANISM_AUDIT.md',
                     'tools/finalize_geometry_model.py', 'tools/prepare_geometry_release.py',
                     'tools/export_geometry_tables.py'):
        copy(Path(relative))
    metadata = dict(date='2026-10-09', implementation=manifest['implementation'],
        ordinary_alias_policy=manifest['ordinary_alias_policy'], copied_files=copied,
        scope='Model/source/test archive, frozen language/protocol/config, aggregate confusion matrices, standalone timings, paired statistics and research reports.',
        excluded='Credentials/askpass/SSH configuration, weights, image/mask datasets, binary text caches, raw GPU logs and per-image NPZ shards.',
        exact_replay='Aggregate evidence and vocabulary are included. Per-image NPZ arrays and binary text caches remain outside Git; bitwise historical replay requires those original caches.',
        status='Measured exploratory frozen release; no universal SOTA or semantic-reliability claim.')
    (destination / 'RELEASE_MANIFEST.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    write_index(destination)
    print(json.dumps(dict(destination=str(destination), source_evidence_files=len(copied))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DEFAULT)
    stage(parser.parse_args().destination)
