"""Publish audited component-wise gain search and its frozen source closure."""
import json
from pathlib import Path
import shutil
from publish_query_slot_best_alias import source_dependencies,WORKSPACE

NAME='geov2_coco81_bestg_ablation_20261014'


def main():
    root=WORKSPACE/'research'/NAME
    repo=WORKSPACE/'artifacts/geometry_github'
    s=json.loads((root/'SUMMARY.json').read_text(encoding='utf-8'))
    if s['status']!='complete' or s['images']!=5000 or not s['all_shards_audited'] or not s['all_per_image_sums_verified']:
        raise ValueError('Only publish audited complete precision and cost.')
    for subdir,suffix in (('configs','.yaml'),('vocabularies','.json')):
        source=WORKSPACE/'research/query_slot_best_alias_20261013'/subdir/('coco_object81'+suffix)
        dest=root/subdir/source.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
    dest=root/'reference/previous_g0_sweep.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(WORKSPACE/'research/geov2_g0_sweep_20261014/coco_object81_audited.json',dest)
    files=source_dependencies([WORKSPACE/'DINOtool/scripts/eval_geov2_component_gain_sweep.py'])
    files.update([WORKSPACE/'tools/report_coco81_bestg_ablation.py',Path(__file__).resolve()])
    for path in root.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.json','.npz','.csv','.md','.png','.pdf','.yaml','.py','.sh'):
            files.add(path)
    changed=[]
    for source in sorted(files):
        dest=repo/source.relative_to(WORKSPACE)
        if dest.exists() and dest.read_bytes()==source.read_bytes():
            continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        changed.append(source.relative_to(WORKSPACE).as_posix())
    marker='2026-10-14更新：完成 **COCO-Object81'
    readme=repo/'README.md'
    body=readme.read_text(encoding='utf-8')
    if marker not in body:
        rows={r['method']:r for r in s['matched_rows']}
        verdict='四个主部件在允许对照各自重选g0后仍有正向配对区间' if s['all_four_supported_after_gain_retuning'] else '允许各对照重选g0后，不能声称所有主部件均优于替代方案'
        cost=s['costs']['Full']
        paragraph=(marker+f"全量5,000张、99配置的最佳g0与部件消融**，见[完整报告、全步长表及逐图证据](research/{NAME}/REPORT_ZH.md)。"
            f"扫描8档后Full最佳g0={s['full_best_g0']:g}，mIoU={s['full_miou_percent']:.4f}%；同g0下QuerySlot净差{rows['NoQueryRouting']['full_minus_arm_pp']:+.4f}pp、DonorBefore {rows['NoDonorBefore']['full_minus_arm_pp']:+.4f}pp、前景扩词{rows['CanonicalForeground']['full_minus_arm_pp']:+.4f}pp。"
            f"{verdict}。全部三份已完成0.5消融的事后等域汇总中，四项平均差的区间均高于0，单域不确定性仍保留。Full为{cost['median_ms']:.2f}ms／{cost['peak_allocated_mib']:.2f}MiB allocated。g0使用已查看验证标签选择，属于开发分析。")
        title,rest=body.split('\n\n',1)
        readme.write_text(title+'\n\n'+paragraph+'\n\n'+rest,encoding='utf-8')
        changed.append('README.md')
    print(json.dumps(dict(copied=len(changed),files=changed),indent=2))


if __name__=='__main__':
    main()
