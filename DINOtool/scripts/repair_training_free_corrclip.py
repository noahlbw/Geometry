"""Isolate CorrCLIP's TIFF RGB decoder fix; preserve model and SAM2 operators."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from eval_training_free_baseline import save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--diagnosis',required=True)
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    root=Path(args.root)
    diagnosis=json.loads(Path(args.diagnosis).read_text())
    path=root/'source_repairs.json'
    repairs=json.loads(path.read_text()) if path.exists() else {}
    if args.validate:
        if repairs['CorrCLIP']['status']!='prepared_pending_validation':
            raise RuntimeError('Pending repair required')
        result_path=root/'corrclip_repaired_validation.json'
        subprocess.run([sys.executable,str(Path(__file__).with_name('diagnose_training_free_corrclip.py')),
            '--root',str(root),'--output',str(result_path),'--prepared-repair'],check=True)
        validation=json.loads(result_path.read_text())
        if validation['target_masks_loaded'] or any(r['status']!='finite' for r in validation['outcomes']):
            raise RuntimeError('Repaired image-only checks failed')
        native=next(r for r in diagnosis['outcomes'] if r['dataset']=='vdd')
        fixed=next(r for r in validation['outcomes'] if r['dataset']=='vdd')
        if native['prediction_sha256']!=fixed['prediction_sha256']:
            raise RuntimeError('PIL-readable native-image prediction changed')
        repairs['CorrCLIP'].update(status='validated',validation=validation,native_image_replay_exact=True)
        save(path,repairs)
        print('TIFF RGB decoder validated; PIL-readable native image replays exactly.')
        return
    failed=next(r for r in diagnosis['outcomes'] if r['dataset']=='flair1')
    if diagnosis['target_masks_loaded'] or 'UnidentifiedImageError' not in failed.get('error',''):
        raise RuntimeError('Expected TIFF decoder failure not reproduced')
    if 'CorrCLIP' in repairs:raise RuntimeError('Existing repair preserved')
    original=root/'sources/CorrCLIP/CorrCLIPv1'
    destination=root/'sources_repaired/CorrCLIPv1_tiff_rgb_v1_crlf'
    if destination.exists():raise RuntimeError('Existing repaired source preserved')
    shutil.copytree(original,destination,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source=destination/'corrclip_segmentor.py'
    raw=source.read_bytes();text=raw.decode()
    before='''        elif self.mask_generator_type == 'sam2':
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
'''
    after='''        elif self.mask_generator_type == 'sam2':
            from PIL import UnidentifiedImageError
            try:
                image = Image.open(img_path).convert("RGB")
                image = np.array(image)
            except UnidentifiedImageError:
                # FLAIR uint8 RGB bands, identical to the frozen image loader.
                import rasterio
                with rasterio.open(img_path) as source:
                    image = np.moveaxis(source.read((1, 2, 3)), 0, -1)
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError("TIFF fallback requires uint8 RGB bands 1,2,3")
'''
    newline='\r\n' if b'\r\n' in raw else '\n'
    before=before.replace('\n',newline);after=after.replace('\n',newline)
    if text.count(before)!=1:raise RuntimeError('Unexpected original SAM2 decoder source')
    changed=text.replace(before,after);source.write_bytes(changed.encode())
    repairs['CorrCLIP']=dict(id='corrclip-native-sam2-tiff-rgb-reader-v1',
        source_relative=destination.relative_to(root).as_posix(),status='prepared_pending_validation',
        diagnosis=diagnosis,files={'corrclip_segmentor.py':dict(
            original_sha256=hashlib.sha256(raw).hexdigest(),repaired_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            diff=''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),
                fromfile='upstream/corrclip_segmentor.py',tofile='repaired/corrclip_segmentor.py')))},
        note='Only PIL-unreadable uint8 TIFF uses RGB bands 1,2,3; SAM2, views and parameters unchanged.')
    save(path,repairs)
    print('Prepared isolated TIFF decoder compatibility fix.')


if __name__=='__main__':main()
