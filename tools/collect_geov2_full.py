"""Retrieve the explicitly enumerated valid Geov2 shards from A800."""
import argparse
from pathlib import Path

from training_free_baseline_experiment import checked_connect as connect, REMOTE, WORKSPACE

SHARDS = {'vdd': 1, 'potsdam': 1, 'ade150': 4, 'voc20': 2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('primary', 'supplement'), default='primary')
    args = p.parse_args()
    relative = 'results/geov2_full_ablation_20261014'
    local = WORKSPACE / 'research/geov2_full_ablation_20261014'
    if args.kind == 'supplement':
        relative += '/supplement'
        local /= 'supplement'
    client = connect()
    try:
        with client.open_sftp() as sftp:
            for ds, count in SHARDS.items():
                for shard in range(count):
                    folder = f'{ds}_shard{shard}'
                    for name in ('results.json', 'per_image_confusions.npz'):
                        remote = REMOTE + '/' + relative + '/' + folder + '/' + name
                        path = local / folder / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(path.suffix + '.part')
                        # Bound SFTP read requests: unlimited prefetch can stall
                        # through this host's gateway, even for small archives.
                        sftp.get(remote, str(temporary), prefetch=False)
                        if temporary.stat().st_size != sftp.stat(remote).st_size:
                            raise RuntimeError('Incomplete transfer: ' + remote)
                        temporary.replace(path)
                    print(args.kind, folder, 'downloaded', flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
