"""Untar WSI sample tiles to node's local storage temporarily for training"""

import glob
import tarfile
from os import makedirs
from os.path import basename, dirname, join
from time import perf_counter
import random
from argparse import ArgumentParser

from joblib import Parallel, delayed


def untar_tiles(tarname: str, dst_path: str):
    """Untar tiles in a specific folder structure"""
    with tarfile.open(tarname, mode='r') as tar:
        for member in tar.getmembers():
            member.name = basename(member.name)
        sample_dst_path = join(dst_path, basename(tarname).split('.')[0])
        makedirs(sample_dst_path, exist_ok=True)
        tar.extractall(path=sample_dst_path)

def untar_samples(src_path: str, dst_path: str, skip_stained: bool, num_cores: int):
    """Untar all the tar (wsi samples) packages to the destination path"""
    start_time = perf_counter()
    filenames = glob.glob(join(src_path,'**/*.tar'), recursive=True)
    if skip_stained:
        filenames = [filename for filename in filenames if '/stained/' not in filename]
    Parallel(n_jobs=num_cores)(
            delayed(untar_tiles)(
                tarname,
                dirname(tarname.replace(src_path, dst_path))
            )
        for tarname in filenames)
    elapsed = (perf_counter() - start_time) // 60
    print(f'Untarring took {elapsed} minutes')

def untar_limited_tiles_randomly(tarname: str, dst_path: str, count: int):
    """Untar tiles in a specific folder structure"""
    makedirs(dst_path, exist_ok=True)
    with tarfile.open(tarname, mode='r') as tar:
        members = tar.getmembers()
        random.seed(1984)
        random.shuffle(members)
        member_subset = members if count == -1 else members[:count]
        for member in member_subset:
            member.name = basename(member.name)
            tar.extract(member=member, path=dst_path)

def main():
    """Main function for temporary random sample tile untarring"""

    parser = ArgumentParser()
    parser.add_argument(
        '--src_path', help='Source path to tar files', type=str, required=True)
    parser.add_argument(
        '--dst_path', help='Destination path for the tiles', type=str, required=True)
    parser.add_argument(
        '--tile_count', help='Tiles extracted per sample, -1 for all', type=int, required=True)
    parser.add_argument(
        '--samples', nargs='+', help='List of sample names', required=True
    )
    param = parser.parse_args()
    start_time = perf_counter()
    Parallel(n_jobs=10)(
            delayed(untar_limited_tiles_randomly)(
                join(param.src_path, sample),
                param.dst_path,
                param.tile_count
            )
        for sample in param.samples)
    elapsed = (perf_counter() - start_time) // 60
    print(f'Untarring took {elapsed} minutes')

if __name__ == '__main__':
    main()
