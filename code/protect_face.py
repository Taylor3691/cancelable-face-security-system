"""Standalone FaceAnonyMixer pipeline for practical face protection.

Runs the full 5-step pipeline end-to-end on one or more input face images of
a subject and one or more revocable keys, without requiring dataset configs or
pre-generated fake pools:

  1. Align and invert input face(s) into StyleGAN2's W+ space (e4e + PTI).
  2. Deterministically map each key to a synthetic W+ code z_f = S(k).
  3. Generate mild augmentations for single-image inputs (or use real views).
  4. Mix layers (0:3 real, 3:8 fake, 8:18 real; Eq. 1) and optimize layers 3:8
     under L_anon + L_attr + L_idp.
  5. Save native 1024x1024 protected face(s) and optional comparison grid.

Usage
-----
    python protect_face.py \
        --images path/to/face.jpg \
        --keys 42 my_secret_key \
        --output-dir outputs/demo \
        --visualize --verbose
"""

import argparse
import hashlib
import os
import os.path as osp
import re

import lpips
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from anonymize import _build_optimizer
from invert import _build_e4e, _get_latents, _pivot_tune
from lib import (
    AttrLoss,
    ConsistencyLoss,
    FaceAligner,
    IDLoss,
    ImageAugmenter,
    LatentCode,
    tensor2image,
)
from models.load_generator import load_generator


# ── helpers ───────────────────────────────────────────────────────────────────

def _key_to_seed(key: str) -> int:
    """Convert an integer or arbitrary string key to a deterministic 63-bit seed."""
    try:
        return int(key) & 0x7FFFFFFFFFFFFFFF
    except ValueError:
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], 'big') & 0x7FFFFFFFFFFFFFFF


def _safe_key_tag(key: str) -> str:
    """Sanitize key string for filenames."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', key).strip('_') or 'key'


def _load_and_align(image_paths, aligner, skip_align, transform, device):
    """Return list of (1, 3, 256, 256) tensors in [-1, 1] and PIL aligned images."""
    tensors, aligned_pils = [], []
    for path in image_paths:
        if not osp.isfile(path):
            raise FileNotFoundError(f'Input image not found: {path}')
        if skip_align or aligner is None:
            pil_img = Image.open(path).convert('RGB')
        else:
            aligned_np = aligner.align_face(image_file=path)
            pil_img = Image.fromarray(aligned_np)
        aligned_pils.append(pil_img.resize((256, 256), Image.BICUBIC))
        tensors.append(transform(pil_img).unsqueeze(0).to(device))
    return tensors, aligned_pils


def _build_comparison_grid(
    aligned_pils,
    recon_pils,
    key_labels,
    key_face_pils,
    protected_pils_by_key,
    img_size: int = 256,
    header_h: int = 30,
    font_path: str = 'lib/fonts/RobotoCondensed-Bold.ttf',
) -> Image.Image:
    """Render Original | Inverted | [Key k | Protected (k)]... grid."""
    columns = ['Original', 'Inverted']
    for k in key_labels:
        columns.extend([f'Key: {k}', f'Protected ({k})'])

    n_rows = len(aligned_pils)
    grid = Image.new('RGB', (img_size * len(columns), header_h + n_rows * img_size), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype(font_path, 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for col_idx, title in enumerate(columns):
        draw.text((col_idx * img_size + 4, 4), title, font=font, fill=(0, 0, 0))

    for row in range(n_rows):
        y = header_h + row * img_size
        row_imgs = [
            aligned_pils[row].resize((img_size, img_size), Image.BICUBIC),
            recon_pils[row].resize((img_size, img_size), Image.BICUBIC),
        ]
        for k_idx in range(len(key_labels)):
            row_imgs.append(key_face_pils[k_idx].resize((img_size, img_size), Image.BICUBIC))
            row_imgs.append(protected_pils_by_key[k_idx][row].resize((img_size, img_size), Image.BICUBIC))

        for col_idx, pil_img in enumerate(row_imgs):
            grid.paste(pil_img, (col_idx * img_size, y))

    return grid


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Standalone FaceAnonyMixer pipeline for image + key(s).'
    )
    p.add_argument('--images', nargs='+', required=True,
                   help='One or more face image paths of the same subject')
    p.add_argument('--keys', nargs='+', default=['42'],
                   help='One or more revocable keys (integers or strings)')
    p.add_argument('--output-dir', type=str, default='outputs/protected',
                   help='Directory to save protected images (and optional grid/latents)')
    p.add_argument('--inv-steps', type=int, default=150,
                   help='PTI refinement steps per input image (0 = e4e only)')
    p.add_argument('--inv-lr', type=float, default=5e-4,
                   help='Learning rate for PTI refinement')
    p.add_argument('--epochs', type=int, default=50,
                   help='Latent mixing optimization steps per key')
    p.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lr-milestones', nargs='+', type=float, default=[0.75, 0.9])
    p.add_argument('--lr-gamma', type=float, default=0.8)
    p.add_argument('-m', '--id-margin', type=float, default=0.0)
    p.add_argument('--lambda-id', type=float, default=10.0)
    p.add_argument('--lambda-attr', type=float, default=0.15)
    p.add_argument('--lambda-consistency', type=float, default=10.0)
    p.add_argument('--n-augmentations', type=int, default=5,
                   help='Augmented views generated when a single input image is provided')
    p.add_argument('--truncation', type=float, default=0.7,
                   help='Truncation psi for sampling key-derived synthetic W+ code')
    p.add_argument('--attr-backbone', type=str, default='farl',
                   choices=['farl', 'clip', 'dino'])
    p.add_argument('--skip-align', action='store_true',
                   help='Skip face alignment if input images are already aligned 256x256 crops')
    p.add_argument('--save-latents', action='store_true',
                   help='Save inverted and protected W+ latent codes (.pt)')
    p.add_argument('--visualize', action='store_true',
                   help='Save side-by-side comparison grid PNG')
    p.add_argument('--cuda', dest='cuda', action='store_true')
    p.add_argument('--no-cuda', dest='cuda', action='store_false')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(cuda=True)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
    device = f'cuda:{args.gpu_id}' if use_cuda else 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)
    latents_dir = osp.join(args.output_dir, 'latents')
    if args.save_latents:
        os.makedirs(latents_dir, exist_ok=True)

    img_transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ── Stage 1: Generator + Alignment + Inversion ────────────────────────────
    if args.verbose:
        print('#. Stage 1: Loading generator, aligner, and e4e ...')

    G = load_generator('stylegan2_ffhq1024', latent_is_w=True, verbose=args.verbose).eval().to(device)
    aligner = None if args.skip_align else FaceAligner(device=device)
    e4e = _build_e4e(device)
    percept = lpips.LPIPS(net='vgg').to(device)

    real_tensors, aligned_pils = _load_and_align(
        args.images, aligner, args.skip_align, img_transform, device
    )
    aligned_batch = torch.cat(real_tensors, dim=0)

    with torch.no_grad():
        init_latents = _get_latents(e4e, aligned_batch)

    if args.inv_steps > 0:
        if args.verbose:
            print(f'#. Running PTI refinement ({args.inv_steps} steps) ...')
        real_latents = _pivot_tune(
            G, e4e, init_latents, aligned_batch,
            args.inv_steps, args.inv_lr, percept, device,
        )
    else:
        real_latents = init_latents

    recon_pils = []
    if args.visualize or args.save_latents:
        with torch.no_grad():
            recons = G(real_latents)
        for i, path in enumerate(args.images):
            stem = osp.splitext(osp.basename(path))[0]
            if args.visualize:
                recon_pils.append(tensor2image(recons[i].cpu(), adaptive=True))
            if args.save_latents:
                torch.save(
                    real_latents[i].detach().cpu().clone(),
                    osp.join(latents_dir, f'{stem}_inv.pt'),
                )

    # Build augmented views for single-image identity consistency (Algorithm 1, Step 4)
    all_imgs = list(real_tensors)
    all_latents = [real_latents[i:i + 1] for i in range(len(real_tensors))]
    n_real = len(real_tensors)

    if n_real == 1 and args.n_augmentations > 0:
        augmenter = ImageAugmenter(n_augmentations=args.n_augmentations)
        aug_views = augmenter(real_tensors[0])
        aug_batch = torch.cat(aug_views, dim=0).to(device)
        with torch.no_grad():
            aug_latents = _get_latents(e4e, aug_batch)
        all_imgs.extend(aug_views)
        all_latents.extend(aug_latents[i:i + 1] for i in range(len(aug_views)))

    # Offload inversion-only models before loading loss networks
    del aligner, e4e, percept
    if use_cuda:
        torch.cuda.empty_cache()

    # ── Stage 2: Loss Networks + Per-Key Mixing & Optimization ────────────────
    if args.verbose:
        print('#. Stage 2: Loading loss networks (ArcFace, attribute backbone) ...')

    id_criterion = IDLoss(id_margin=args.id_margin).eval().to(device)
    attr_loss_fn = AttrLoss(feat_ext=args.attr_backbone, use_cuda=use_cuda).eval().to(device)
    cons_loss_fn = ConsistencyLoss().eval().to(device)

    G.zero_grad()
    key_face_pils = []
    protected_pils_by_key = []

    for key in args.keys:
        seed = _key_to_seed(key)
        safe_tag = _safe_key_tag(key)
        if args.verbose:
            print(f'#. Protecting with key="{key}" (seed={seed}) ...')

        gen = torch.Generator().manual_seed(seed)
        z_fake = torch.randn(1, G.dim_z, generator=gen).to(device)
        with torch.no_grad():
            w_fake = G.get_w(z_fake, truncation=args.truncation)
            if args.visualize:
                key_face_pils.append(tensor2image(G(w_fake).cpu(), adaptive=True))

        latent_codes, optimizers, schedulers = [], [], []
        code_dir = latents_dir if args.save_latents else args.output_dir
        for i, w_real in enumerate(all_latents):
            lc = LatentCode(
                latent_code_real=w_real,
                latent_code_fake_nn=w_fake,
                img_id=i,
                out_code_dir=code_dir,
                latent_space='W+',
            ).to(device)
            opt = _build_optimizer(lc.parameters(), args.optim, args.lr)
            sched = MultiStepLR(
                optimizer=opt,
                milestones=[int(m * args.epochs) for m in args.lr_milestones],
                gamma=args.lr_gamma,
            )
            latent_codes.append(lc)
            optimizers.append(opt)
            schedulers.append(sched)

        for _ in range(args.epochs):
            anon_images = []
            for lc, opt in zip(latent_codes, optimizers):
                opt.zero_grad()
                anon_images.append(G(lc()))

            anon_batch = torch.cat(anon_images, dim=0)
            total_id = sum(id_criterion(ai, oi) for ai, oi in zip(anon_images, all_imgs))
            total_attr = sum(attr_loss_fn(ai, oi) for ai, oi in zip(anon_images, all_imgs))
            cons = cons_loss_fn(anon_batch)

            loss = (
                args.lambda_id * total_id
                + args.lambda_attr * total_attr
                + args.lambda_consistency * cons
            )
            loss.backward()

            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()

        key_protected_pils = []
        for i in range(n_real):
            stem = osp.splitext(osp.basename(args.images[i]))[0]
            with torch.no_grad():
                anon_img = G(latent_codes[i]())
            pil_out = tensor2image(anon_img.cpu(), adaptive=True)
            out_img_path = osp.join(args.output_dir, f'{stem}_key_{safe_tag}.jpg')
            pil_out.save(out_img_path, 'JPEG', quality=95, subsampling=0, progressive=True)
            if args.verbose:
                print(f'   Saved protected image: {out_img_path}')
            if args.save_latents:
                torch.save(
                    latent_codes[i]().squeeze(0).detach().cpu().clone(),
                    osp.join(latents_dir, f'{stem}_key_{safe_tag}.pt'),
                )
            if args.visualize:
                key_protected_pils.append(pil_out)

        if args.visualize:
            protected_pils_by_key.append(key_protected_pils)

    if args.visualize:
        first_stem = osp.splitext(osp.basename(args.images[0]))[0]
        grid_path = osp.join(args.output_dir, f'{first_stem}_comparison.png')
        grid = _build_comparison_grid(
            aligned_pils, recon_pils, args.keys, key_face_pils, protected_pils_by_key
        )
        grid.save(grid_path)
        if args.verbose:
            print(f'#. Saved comparison grid: {grid_path}')


if __name__ == '__main__':
    main()
