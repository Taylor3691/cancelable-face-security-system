"""Download, extract, and reorganize CelebA-HQ or VGGFace2 for FaceAnonyMixer.

Produces the per-identity layout expected by ``anonymize.py``, ``CelebAHQ``,
and ``VGGFace2``::

    <output-dir>/
    └── train/
        └── <identity_id>/
            ├── 0.jpg
            └── ...

Usage
-----
    # CelebA-HQ (from CelebAMask-HQ.zip)
    python scripts/prepare_dataset.py --dataset celebahq \
        --archive-path datasets/CelebAMask-HQ.zip \
        --output-dir datasets/celebahq

    # VGGFace2 (from a local zip/tar archive, or HuggingFace / Kaggle)
    python scripts/prepare_dataset.py --dataset vggface2 \
        --vggface2-source kaggle \
        --output-dir datasets/vggface2
"""

import argparse
import os
import os.path as osp
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from tqdm import tqdm


CELEBAMASK_HQ_URL = (
    'https://drive.usercontent.google.com/download'
    '?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv&export=download&confirm=t'
)
IDENTITY_CELEBA_URL = (
    'https://drive.usercontent.google.com/download'
    '?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS&export=download&confirm=t'
)

VGGFACE2_HF_REPO = 'ProgramComputer/VGGFace2'
VGGFACE2_KAGGLE_SLUG = 'greatgamedota/vggface2-test'


def download_file(url: str, dest: str):
    """Download *url* to *dest* using curl/wget or urllib fallback."""
    os.makedirs(osp.dirname(osp.abspath(dest)), exist_ok=True)
    if shutil.which('curl'):
        subprocess.run(['curl', '-L', '-C', '-', '-o', dest, url], check=True)
    elif shutil.which('wget'):
        subprocess.run(['wget', '-c', '-O', dest, url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)


# ── CelebA-HQ helpers ─────────────────────────────────────────────────────────

def extract_celebahq_files(zip_path: str, extract_dir: str):
    """Extract only images and annotation txt files (skipping mask PNGs)."""
    wanted_prefixes = (
        'CelebAMask-HQ/CelebA-HQ-img/',
        'CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt',
        'CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt',
    )
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = [
            m for m in zf.namelist()
            if m.startswith(wanted_prefixes) and not m.endswith('/')
        ]
        for member in tqdm(members, desc='Extracting CelebAMask-HQ'):
            target = osp.join(extract_dir, member)
            if not osp.exists(target):
                zf.extract(member, extract_dir)


def load_hq_to_orig_mapping(mapping_path: str) -> dict:
    """Return {celebahq_filename (str): orig_celeba_filename (str)}."""
    hq_to_orig = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3 or parts[0] == 'idx':
                continue
            hq_idx, _, orig_file = parts[0], parts[1], parts[2]
            hq_to_orig[f'{hq_idx}.jpg'] = orig_file
    return hq_to_orig


def load_orig_to_identity(identity_path: str) -> dict:
    """Return {orig_celeba_filename (str): identity_id (str)}."""
    orig_to_id = {}
    with open(identity_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                orig_file, identity_id = parts
                orig_to_id[orig_file] = identity_id
    return orig_to_id


def prepare_celebahq(args):
    archive_path = args.archive_path or 'datasets/CelebAMask-HQ.zip'
    output_dir = args.output_dir or 'datasets/celebahq'

    if not osp.isfile(archive_path):
        print(f'Downloading CelebAMask-HQ.zip to {archive_path} ...')
        download_file(CELEBAMASK_HQ_URL, archive_path)
    else:
        print(f'Found existing archive: {archive_path}')

    tmp_extract_dir = osp.join(output_dir, '_raw_extract')
    os.makedirs(tmp_extract_dir, exist_ok=True)
    extract_celebahq_files(archive_path, tmp_extract_dir)

    raw_root = osp.join(tmp_extract_dir, 'CelebAMask-HQ')
    img_dir = osp.join(raw_root, 'CelebA-HQ-img')
    mapping_src = osp.join(raw_root, 'CelebA-HQ-to-CelebA-mapping.txt')
    attr_src = osp.join(raw_root, 'CelebAMask-HQ-attribute-anno.txt')

    anno_dir = osp.join(output_dir, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    mapping_dst = osp.join(anno_dir, 'CelebA-HQ-to-CelebA-mapping.txt')
    attr_dst = osp.join(anno_dir, 'CelebAMask-HQ-attribute-anno.txt')
    identity_dst = osp.join(anno_dir, 'identity_CelebA.txt')

    if osp.isfile(mapping_src) and not osp.isfile(mapping_dst):
        shutil.copy2(mapping_src, mapping_dst)
    if osp.isfile(attr_src) and not osp.isfile(attr_dst):
        shutil.copy2(attr_src, attr_dst)
    if not osp.isfile(identity_dst):
        print(f'Downloading identity_CelebA.txt to {identity_dst} ...')
        download_file(IDENTITY_CELEBA_URL, identity_dst)

    hq_to_orig = load_hq_to_orig_mapping(mapping_dst)
    orig_to_id = load_orig_to_identity(identity_dst)

    train_dir = osp.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    img_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )

    moved, skipped = 0, 0
    op = shutil.copy2 if args.copy else shutil.move
    for fname in tqdm(img_files, desc='Reorganizing CelebA-HQ'):
        orig_file = hq_to_orig.get(fname)
        identity_id = orig_to_id.get(orig_file) if orig_file else None
        if not identity_id:
            skipped += 1
            continue
        id_dir = osp.join(train_dir, identity_id)
        os.makedirs(id_dir, exist_ok=True)
        dst = osp.join(id_dir, fname)
        if not osp.exists(dst):
            op(osp.join(img_dir, fname), dst)
        moved += 1

    if not args.keep_extracted:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)

    print(f'Done: {moved} images organized into {train_dir} (skipped {skipped}).')
    return output_dir


# ── VGGFace2 helpers ──────────────────────────────────────────────────────────

def _download_vggface2_kaggle(dest_dir: str):
    if not shutil.which('kaggle'):
        raise RuntimeError(
            'Kaggle CLI not found. Install with `pip install kaggle` and '
            'place your API token at ~/.kaggle/kaggle.json.'
        )
    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', VGGFACE2_KAGGLE_SLUG,
         '-p', dest_dir, '--unzip'],
        check=True,
    )


def _download_vggface2_hf(dest_dir: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            'huggingface_hub not installed. Run `pip install huggingface_hub` '
            'and login via `huggingface-cli login` (repo is gated).'
        ) from exc
    os.makedirs(dest_dir, exist_ok=True)
    snapshot_download(
        repo_id=VGGFACE2_HF_REPO,
        repo_type='dataset',
        local_dir=dest_dir,
    )


def _extract_archive(archive_path: str, extract_dir: str):
    os.makedirs(extract_dir, exist_ok=True)
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            for m in tqdm(zf.namelist(), desc='Extracting zip'):
                zf.extract(m, extract_dir)
    elif archive_path.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f'Unsupported archive format: {archive_path}')


def _find_identity_parent(root: str) -> str:
    """Locate the directory containing identity folders (e.g. n000001/)."""
    for candidate in ('train', 'test', ''):
        p = osp.join(root, candidate) if candidate else root
        if osp.isdir(p):
            subdirs = [d for d in os.listdir(p) if osp.isdir(osp.join(p, d))]
            if subdirs and any(d.startswith('n') and d[1:].isdigit() for d in subdirs):
                return p
    for dirpath, dirnames, _ in os.walk(root):
        if any(d.startswith('n') and d[1:].isdigit() for d in dirnames):
            return dirpath
    raise FileNotFoundError(f'Could not locate VGGFace2 identity folders under {root}')


def prepare_vggface2(args):
    output_dir = args.output_dir or 'datasets/vggface2'
    tmp_extract_dir = osp.join(output_dir, '_raw_extract')

    if args.archive_path:
        if not osp.isfile(args.archive_path):
            raise FileNotFoundError(f'Archive not found: {args.archive_path}')
        _extract_archive(args.archive_path, tmp_extract_dir)
        search_root = tmp_extract_dir
    elif args.vggface2_source == 'kaggle':
        _download_vggface2_kaggle(tmp_extract_dir)
        search_root = tmp_extract_dir
    elif args.vggface2_source == 'hf':
        _download_vggface2_hf(tmp_extract_dir)
        for root, _, files in os.walk(tmp_extract_dir):
            for f in files:
                if f.endswith(('.zip', '.tar.gz', '.tgz', '.tar')):
                    _extract_archive(osp.join(root, f), tmp_extract_dir)
        search_root = tmp_extract_dir
    else:
        raise ValueError('Provide --archive-path or --vggface2-source {kaggle,hf}.')

    src_id_parent = _find_identity_parent(search_root)
    train_dir = osp.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    op = shutil.copy2 if args.copy else shutil.move
    id_folders = sorted(
        d for d in os.listdir(src_id_parent)
        if osp.isdir(osp.join(src_id_parent, d))
    )

    total = 0
    for id_name in tqdm(id_folders, desc='Reorganizing VGGFace2'):
        src_id_dir = osp.join(src_id_parent, id_name)
        dst_id_dir = osp.join(train_dir, id_name)
        os.makedirs(dst_id_dir, exist_ok=True)

        imgs = sorted(
            f for f in os.listdir(src_id_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        # anonymize.py requires integer filenames (e.g. 0001.jpg)
        for idx, fname in enumerate(imgs, start=1):
            ext = osp.splitext(fname)[1].lower()
            stem = osp.splitext(fname)[0]
            out_name = fname if stem.isdigit() else f'{idx:04d}{ext}'
            dst = osp.join(dst_id_dir, out_name)
            if not osp.exists(dst):
                op(osp.join(src_id_dir, fname), dst)
            total += 1

    if not args.keep_extracted:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)

    print(f'Done: {total} images across {len(id_folders)} identities in {train_dir}.')
    return output_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Download, extract, and reorganize CelebA-HQ or VGGFace2.'
    )
    p.add_argument('--dataset', type=str, default='celebahq',
                   choices=['celebahq', 'vggface2'],
                   help='Dataset to prepare')
    p.add_argument('--archive-path', '--zip-path', dest='archive_path',
                   type=str, default=None,
                   help='Path to local archive (.zip / .tar.gz)')
    p.add_argument('--vggface2-source', type=str, default='kaggle',
                   choices=['kaggle', 'hf'],
                   help='Download source for VGGFace2 when --archive-path is omitted')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Target dataset root directory')
    p.add_argument('--copy', action='store_true',
                   help='Copy images instead of moving them from extracted folder')
    p.add_argument('--keep-extracted', action='store_true',
                   help='Keep temporary extracted folder')
    return p.parse_args()


def main():
    args = parse_args()
    if args.dataset == 'celebahq':
        out_dir = prepare_celebahq(args)
    else:
        out_dir = prepare_vggface2(args)

    print(
        f'\nNext: update DATASETS["{args.dataset}"] in lib/config.py to:\n'
        f'    {osp.abspath(out_dir)}'
    )


if __name__ == '__main__':
    main()
