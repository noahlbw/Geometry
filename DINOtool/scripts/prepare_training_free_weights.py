"""Fetch original pretrained resources once, recording hashes and pinned sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

RESOURCES={
    'ViT-B-16.pt':('https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
        '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'),
    'dino_vitbase8_pretrain.pth':('https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',None),
    'b16_fullcc2.5b.pt':('https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt',None),
    'sam2_hiera_large.pt':('https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',None),
    'xclip_jbu_one_million_aid.ckpt':('https://raw.githubusercontent.com/likyoo/SegEarth-OV/3e22a969b32c6d751bdbba64a88a0b670e630f55/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt',None),
}
DEPENDENCIES={
    'dino':('https://github.com/facebookresearch/dino.git','7c446df5b9f45747937fb0d72314eb9f7b66930a'),
    'sam2':('https://github.com/facebookresearch/sam2.git','2b90b9f5ceec907a1c18123530e92e794ad901a4'),
    # Native JBU CUDA operator is required; no slow Python surrogate.
    'SimFeatUp':('https://github.com/likyoo/SimFeatUp.git','78a0ba70b1d6ea7283684a88c98ce338af4593ca'),
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root).resolve();checkpoint=root/'checkpoints';checkpoint.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        name,(url,expected)=item;target=checkpoint/name
        if not target.exists():
            temporary=target.with_suffix(target.suffix+'.part')
            subprocess.run(['curl','--fail','-L','--retry','3','--retry-delay','5','--connect-timeout','30',
                '--max-time','1800','--speed-time','180','--speed-limit','8192','-C','-',url,'-o',str(temporary)],check=True)
            temporary.rename(target)
        actual=sha(target)
        if expected and expected!=actual:raise RuntimeError('Checkpoint SHA mismatch: '+name)
        print('CHECKPOINT_READY '+name,flush=True)
        return name,dict(url=url,sha256=actual,bytes=target.stat().st_size)
    with ThreadPoolExecutor(max_workers=3) as pool:weights=dict(pool.map(fetch,RESOURCES.items()))
    deps={};(root/'dependencies').mkdir(exist_ok=True)
    for name,(url,commit) in DEPENDENCIES.items():
        path=root/'dependencies'/name
        # Commit archives avoid long Git smart-HTTP transfers on this host.
        # A partial .git clone, if present, is left intact as diagnostic evidence.
        identity=path/'BASELINE_SOURCE.json'
        if not identity.exists():
            archive=root/'dependencies'/(name+'-'+commit+'.zip')
            archive_url='https://codeload.github.com/'+url.split('github.com/')[1].removesuffix('.git')+'/zip/'+commit
            if not archive.exists():subprocess.run(['curl','--fail','-sS','-L','--max-time','600','--retry','2',archive_url,'-o',str(archive)],check=True)
            path.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(archive) as data:
                for member in data.infolist():
                    relative=Path(*Path(member.filename).parts[1:])
                    target=(path/relative).resolve();target.relative_to(path.resolve())
                    if member.is_dir():target.mkdir(parents=True,exist_ok=True)
                    elif relative.parts:
                        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data.read(member))
            identity.write_text(json.dumps(dict(url=url,commit=commit,archive_sha256=sha(archive))))
        recorded=json.loads(identity.read_text())
        if recorded['commit']!=commit:raise RuntimeError('Existing dependency source differs')
        deps[name]=recorded
    # All native downloads are prepopulated before parallel workers are allowed.
    for folder,names in [(Path('/root/.cache/clip'),('ViT-B-16.pt','b16_fullcc2.5b.pt')),
        (Path('/root/.cache/torch/hub/checkpoints'),('dino_vitbase8_pretrain.pth',)),
        (root/'sources/CorrCLIP/CorrCLIPv1',('sam2_hiera_large.pt',))]:
        folder.mkdir(parents=True,exist_ok=True)
        for name in names:
            target=folder/name
            if target.is_symlink() and not target.exists():
                target.unlink()
            if target.exists():
                if sha(target)!=weights[name]['sha256']:raise RuntimeError('Existing checkpoint cache differs: '+str(target))
            else:target.symlink_to(checkpoint/name)
    (root/'checkpoint_manifest.json').write_text(json.dumps(dict(weights=weights,dependencies=deps),indent=2))
    (root/'weights_ready').touch()


if __name__=='__main__':main()
