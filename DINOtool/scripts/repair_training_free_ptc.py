"""Prepare an isolated PTC copy with two documented execution-only fixes."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from eval_training_free_baseline import save


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--diagnosis', required=True)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    diagnosis = json.loads(Path(args.diagnosis).read_text())
    manifest_path = root/'source_repairs.json'
    if args.validate:
        repairs = json.loads(manifest_path.read_text())
        if repairs['PTC']['status'] != 'prepared_pending_validation':
            raise RuntimeError('Only a pending repair can be validated')
        result_path = root/'ptc_repaired_validation.json'
        subprocess.run([sys.executable, str(Path(__file__).with_name('diagnose_training_free_ptc.py')),
            '--root', str(root), '--output', str(result_path), '--prepared-repair'], check=True)
        validation = json.loads(result_path.read_text())
        if validation['target_masks_loaded'] or any(row['status'] != 'finite' for row in validation['outcomes']):
            raise RuntimeError('Repaired image-only outputs must all be finite')
        repaired = {row['key']:row for row in validation['outcomes']}
        finite = [row for row in diagnosis['outcomes'] if row['status']=='finite']
        if not finite or any(row['prediction_sha256'] != repaired[row['key']]['prediction_sha256'] for row in finite):
            raise RuntimeError('Originally finite image must replay exactly after execution repair')
        repairs['PTC'].update(status='validated', validation=validation,
            finite_prediction_replay_exact=True, finite_replay_images=len(finite))
        save(manifest_path, repairs)
        print('Image-only repair validated; originally finite predictions replay exactly.')
        return
    events = [event for row in diagnosis['outcomes'] for event in row['events']]
    if diagnosis['target_masks_loaded'] or not events or any(
            event['input_nan'] or event['input_positive_infinity'] or
            not event['all_negative_infinity_rows'] for event in events):
        raise RuntimeError('First nonfinite operator is not verified as empty-row softmax')
    source = root/'sources/PTC'
    destination = root/'sources_repaired/PTC_empty_rows_rectangular_v1'
    if manifest_path.exists():
        raise RuntimeError('Existing repair manifest preserved')
    if destination.exists():
        raise RuntimeError('Existing repaired source preserved')
    shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    replacements = {
        'ptc_segmentor.py': (
            'I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size',
            'I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size'),
        'open_clip/transformer.py': (
            '        attn_weights = F.softmax(mask, dim=-1)',
            '        # Numerical repair: self-Value only for rows with no finite source.\n'
            '        empty_rows = torch.isneginf(mask).all(dim=-1)\n'
            '        if empty_rows.any():\n'
            '            batch, row = empty_rows.nonzero(as_tuple=True)\n'
            '            mask[batch, row, row] = 0.0\n'
            '        attn_weights = F.softmax(mask, dim=-1)')}
    files = {}
    for relative, (before, after) in replacements.items():
        path = destination/relative
        raw = path.read_bytes()
        text = raw.decode()
        if text.count(before) != 1:
            raise RuntimeError('Unexpected upstream source: '+relative)
        changed = text.replace(before, after)
        path.write_bytes(changed.encode())
        files[relative] = dict(original_sha256=sha(raw), repaired_sha256=sha(path.read_bytes()),
            diff=''.join(difflib.unified_diff(text.splitlines(True), changed.splitlines(True),
                fromfile='upstream/'+relative, tofile='repaired/'+relative)))
    save(manifest_path, {'PTC': dict(id='ptc-empty-row-self-value-and-rectangular-grid-v1',
        source_relative=destination.relative_to(root).as_posix(), files=files,
        diagnosis=diagnosis, status='prepared_pending_validation',
        note='Original finite attention rows unchanged; no parameter, vocabulary or threshold change.')})
    print('Prepared isolated PTC execution repair; validation required before evaluation.')


if __name__ == '__main__':
    main()
