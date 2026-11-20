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
        --output-dir datasets/celebahq

    # VGGFace2 (from existing dir, local archive, or Kaggle test split)
    python scripts/prepare_dataset.py --dataset vggface2 \
        --output-dir datasets/vggface2
"""

import argparse
import os
import os.path as osp
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

from tqdm import tqdm


CELEBAMASK_HQ_URL = (
    'https://drive.usercontent.google.com/download'
    '?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv&export=download&confirm=t'
)
IDENTITY_CELEBA_URLS = [
    'https://drive.usercontent.google.com/download'
    '?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS&export=download&confirm=t',
    'https://raw.githubusercontent.com/KaiserW/bald-recognition/'
    'master/dataset/celeba/identity_CelebA.txt',
]
EVAL_PARTITION_URLS = [
    'https://drive.usercontent.google.com/download'
    '?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk&export=download&confirm=t',
    'https://raw.githubusercontent.com/KaiserW/bald-recognition/'
    'master/dataset/celeba/list_eval_partition.txt',
]

VGGFACE2_KAGGLE_SLUG = 'greatgamedota/vggface2-test'


def download_file(url: str, dest: str, allow_resume: bool = True):
    """Download *url* to *dest* using curl/wget or urllib fallback."""
    os.makedirs(osp.dirname(osp.abspath(dest)), exist_ok=True)
    if shutil.which('curl'):
        cmd = ['curl', '-fsSL']
        if allow_resume:
            cmd += ['-C', '-']
        cmd += ['-o', dest, url]
        subprocess.run(cmd, check=True)
    elif shutil.which('wget'):
        cmd = ['wget']
        if allow_resume:
            cmd.append('-c')
        cmd += ['-O', dest, url]
        subprocess.run(cmd, check=True)
    else:
        urllib.request.urlretrieve(url, dest)


def _is_valid_txt(path: str) -> bool:
    """Return True if *path* exists, is non-empty, and is not an HTML error page."""
    if not osp.isfile(path) or osp.getsize(path) == 0:
        return False
    with open(path, 'r', errors='ignore') as f:
        head = f.read(256).lstrip().lower()
    if head.startswith(('<!doctype', '<html', '<head', '<script')):
        return False
    return True


def download_txt_with_fallback(urls: list, dest: str):
    """Download a text annotation file, falling back if a mirror returns HTML/quota error."""
    if _is_valid_txt(dest):
        return
    if osp.exists(dest):
        os.remove(dest)
    for url in urls:
        try:
            print(f'Downloading {osp.basename(dest)} from {url} ...')
            download_file(url, dest, allow_resume=False)
            if _is_valid_txt(dest):
                return
            print(f'Warning: {url} returned invalid/HTML content (e.g. Drive quota), trying fallback...')
            if osp.exists(dest):
                os.remove(dest)
        except Exception as exc:
            print(f'Warning: download failed from {url} ({exc}), trying fallback...')
            if osp.exists(dest):
                os.remove(dest)
    raise RuntimeError(f'Failed to download valid {osp.basename(dest)} from all mirrors.')


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


def load_eval_partition(partition_path: str) -> dict:
    """Return {orig_celeba_filename (str): split_int (0=train, 1=val, 2=test)}."""
    splits = {}
    with open(partition_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                splits[parts[0]] = int(parts[1])
    return splits


def _manual_celebahq_msg(archive_path: str, output_dir: str) -> str:
    return (
        f'\nCould not automatically download a valid CelebAMask-HQ.zip '
        f'(Google Drive may have hit a quota limit).\n'
        f'Please download CelebAMask-HQ.zip manually from the official repo: '
        f'https://github.com/switchablenorms/CelebAMask-HQ\n'
        f'Save it to "{archive_path}" (or pass --archive-path /path/to/CelebAMask-HQ.zip) '
        f'and re-run:\n'
        f'  python scripts/prepare_dataset.py --dataset celebahq '
        f'--archive-path {archive_path} --output-dir {output_dir}'
    )


def prepare_celebahq(args):
    archive_path = args.archive_path or 'datasets/CelebAMask-HQ.zip'
    output_dir = args.output_dir or 'datasets/celebahq'

    if osp.isfile(archive_path):
        if not zipfile.is_zipfile(archive_path):
            os.remove(archive_path)
            print(
                f'Error: Removed invalid/HTML file at {archive_path} (not a valid zip archive).'
                + _manual_celebahq_msg(archive_path, output_dir),
                file=sys.stderr,
            )
            sys.exit(1)
        print(f'Found existing archive: {archive_path}')
    else:
        print(f'Downloading CelebAMask-HQ.zip to {archive_path} ...')
        try:
            download_file(CELEBAMASK_HQ_URL, archive_path, allow_resume=False)
        except Exception as exc:
            if osp.exists(archive_path):
                os.remove(archive_path)
            print(
                f'Error: Download failed ({exc}).'
                + _manual_celebahq_msg(archive_path, output_dir),
                file=sys.stderr,
            )
            sys.exit(1)

        if not zipfile.is_zipfile(archive_path):
            os.remove(archive_path)
            print(
                'Error: Downloaded file is not a valid zip archive.'
                + _manual_celebahq_msg(archive_path, output_dir),
                file=sys.stderr,
            )
            sys.exit(1)

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
    partition_dst = osp.join(anno_dir, 'list_eval_partition.txt')

    if osp.isfile(mapping_src) and not osp.isfile(mapping_dst):
        shutil.copy2(mapping_src, mapping_dst)
    if osp.isfile(attr_src) and not osp.isfile(attr_dst):
        shutil.copy2(attr_src, attr_dst)
    download_txt_with_fallback(IDENTITY_CELEBA_URLS, identity_dst)
    need_partition = args.split in ('train', 'val', 'test')
    if need_partition:
        download_txt_with_fallback(EVAL_PARTITION_URLS, partition_dst)

    hq_to_orig = load_hq_to_orig_mapping(mapping_dst)
    orig_to_id = load_orig_to_identity(identity_dst)
    orig_to_split = load_eval_partition(partition_dst) if need_partition else {}
    if not orig_to_id:
        raise RuntimeError(f'Parsed 0 entries from {identity_dst}.')
    if need_partition and not orig_to_split:
        raise RuntimeError(f'Parsed 0 entries from {partition_dst}.')
    split_code = {'train': 0, 'val': 1, 'test': 2}.get(args.split)

    img_files = sorted(
        (f for f in os.listdir(img_dir)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))),
        key=lambda x: int(osp.splitext(x)[0]) if osp.splitext(x)[0].isdigit() else x,
    )

    # Group candidate images by identity first so splits & filters apply cleanly
    id_to_files = {}
    skipped = 0
    for fname in img_files:
        orig_file = hq_to_orig.get(fname)
        identity_id = orig_to_id.get(orig_file) if orig_file else None
        if not identity_id:
            skipped += 1
            continue
        if split_code is not None and orig_to_split.get(orig_file) != split_code:
            continue
        id_to_files.setdefault(identity_id, []).append(fname)

    if args.split == 'paper':
        # Na et al. (ECCV 2022) / FaceAnonyMixer Sec. 4.1:
        # 307 identities with >=15 images, keeping the last 20% per identity (1,215 test images)
        min_imgs = max(args.min_images, 15)
        selected_ids = sorted(k for k, v in id_to_files.items() if len(v) >= min_imgs)
        for k in selected_ids:
            files = id_to_files[k]
            n_train = int(len(files) * 0.8)
            id_to_files[k] = files[n_train:]
    else:
        selected_ids = sorted(
            k for k, v in id_to_files.items() if len(v) >= args.min_images
        )

    if args.max_identities is not None:
        selected_ids = selected_ids[:args.max_identities]

    train_dir = osp.join(output_dir, 'train')
    if args.clean and osp.isdir(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)
    moved = 0
    op = shutil.copy2 if args.copy else shutil.move
    for identity_id in tqdm(selected_ids, desc='Reorganizing CelebA-HQ'):
        id_dir = osp.join(train_dir, identity_id)
        os.makedirs(id_dir, exist_ok=True)
        for fname in id_to_files[identity_id]:
            dst = osp.join(id_dir, fname)
            if not osp.exists(dst):
                op(osp.join(img_dir, fname), dst)
            moved += 1

    if not args.keep_extracted:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)

    print(
        f'Done: {moved} images across {len(selected_ids)} identities '
        f'organized into {train_dir} (skipped {skipped} unmapped).'
    )
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
    train_dir = osp.join(output_dir, 'train')
    tmp_extract_dir = osp.join(output_dir, '_raw_extract')
    using_existing = False

    if args.source_dir:
        src_id_parent = _find_identity_parent(args.source_dir)
        using_existing = True
    elif args.archive_path:
        if not osp.isfile(args.archive_path):
            raise FileNotFoundError(f'Archive not found: {args.archive_path}')
        _extract_archive(args.archive_path, tmp_extract_dir)
        src_id_parent = _find_identity_parent(tmp_extract_dir)
    elif osp.isdir(train_dir) and os.listdir(train_dir):
        print(f'Found existing VGGFace2 train directory: {train_dir}')
        src_id_parent = train_dir
        using_existing = True
    else:
        _download_vggface2_kaggle(tmp_extract_dir)
        src_id_parent = _find_identity_parent(tmp_extract_dir)

    id_folders = sorted(
        d for d in os.listdir(src_id_parent)
        if osp.isdir(osp.join(src_id_parent, d))
    )

    rng = random.Random(args.seed)
    if args.split == 'paper':
        # Sample ~1,500 images across test identities (default: 300 IDs x 5 imgs = 1,500)
        target_total = args.total_images if args.total_images is not None else 1500
        num_ids = args.max_identities if args.max_identities is not None else 300
        eligible = []
        for id_name in id_folders:
            imgs = sorted(
                f for f in os.listdir(osp.join(src_id_parent, id_name))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            if len(imgs) >= args.min_images:
                eligible.append((id_name, imgs))
        if len(eligible) < num_ids:
            num_ids = len(eligible)
        rng.shuffle(eligible)
        chosen = sorted(eligible[:num_ids], key=lambda x: x[0])

        base_k, rem = divmod(target_total, len(chosen))
        selected_map = {}
        for idx, (id_name, imgs) in enumerate(chosen):
            k = min(len(imgs), base_k + (1 if idx < rem else 0))
            imgs_copy = list(imgs)
            rng.shuffle(imgs_copy)
            selected_map[id_name] = sorted(imgs_copy[:k])
    else:
        if args.max_identities is not None:
            id_folders = id_folders[:args.max_identities]
        selected_map = {}
        for id_name in id_folders:
            imgs = sorted(
                f for f in os.listdir(osp.join(src_id_parent, id_name))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            if len(imgs) >= args.min_images:
                selected_map[id_name] = imgs

    # Stage into a temp dir if modifying existing train_dir in-place
    in_place = osp.abspath(src_id_parent) == osp.abspath(train_dir)
    stage_dir = osp.join(output_dir, '_staged_train') if in_place else train_dir
    if in_place and osp.exists(stage_dir):
        shutil.rmtree(stage_dir)
    elif not in_place and args.clean and osp.isdir(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(stage_dir, exist_ok=True)

    op = shutil.copy2 if (args.copy or (using_existing and not in_place)) else shutil.move
    total = 0
    for id_name, imgs in tqdm(selected_map.items(), desc='Reorganizing VGGFace2'):
        src_id_dir = osp.join(src_id_parent, id_name)
        dst_id_dir = osp.join(stage_dir, id_name)
        os.makedirs(dst_id_dir, exist_ok=True)

        for fname in imgs:
            total += 1
            ext = osp.splitext(fname)[1].lower()
            # Globally unique integer filename prevents invert.py latents/<stem>.pt collisions
            out_name = f'{total:06d}{ext}'
            dst = osp.join(dst_id_dir, out_name)
            if not osp.exists(dst):
                op(osp.join(src_id_dir, fname), dst)

    if in_place:
        shutil.rmtree(train_dir)
        os.rename(stage_dir, train_dir)

    if not args.keep_extracted and osp.isdir(tmp_extract_dir):
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)

    print(f'Done: {total} images across {len(selected_map)} identities in {train_dir}.')
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
    p.add_argument('--source-dir', type=str, default=None,
                   help='Existing extracted dataset directory (e.g. datasets/vggface2) to subsample from')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Target dataset root directory')
    p.add_argument('--split', type=str, default='paper',
                   choices=['paper', 'all', 'train', 'val', 'test'],
                   help='Split to prepare: "paper" = 307 IDs / 1,215 imgs (CelebA-HQ) '
                        'or 1,500 imgs (VGGFace2); or CelebA eval partition')
    p.add_argument('--min-images', type=int, default=1,
                   help='Minimum images per identity to keep')
    p.add_argument('--max-identities', type=int, default=None,
                   help='Cap total number of identities (default: 300 for VGGFace2 paper split)')
    p.add_argument('--total-images', type=int, default=None,
                   help='Target total images for VGGFace2 paper split (default: 1500)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducible VGGFace2 sampling')
    p.add_argument('--clean', action='store_true',
                   help='Remove existing train/ directory before reorganizing')
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
