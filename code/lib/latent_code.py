import os.path as osp
import torch
import torch.nn as nn


class LatentCode(nn.Module):
    """Learnable anonymization latent code in StyleGAN2's W+ space.

    Follows FaceAnonyMixer Eq. (1) and Algorithm 1:

        layers  0 – 2  : coarse global structure   → initialised from real code,       frozen
        layers  3 – 7  : identity-related          → initialised from fake (key) code, trainable
        layers  8 – 17 : fine details / background → initialised from real code,       frozen

    Args:
        latent_code_real    (torch.Tensor): (1, 18, 512) W+ code of the real image
                                            (from GAN inversion).
        latent_code_fake_nn (torch.Tensor): (1, 18, 512) W+ code of the
                                            fake (key) image.
        img_id       (str | int): image identifier used for the checkpoint filename.
        out_code_dir (str): directory where the optimised code is saved.
        gan          (str): generator model name (currently unused; for future use
                            with different generator configs).
        latent_space (str): must be ``'W+'`` (S-space not yet implemented).
        freeze_non_identity (bool): if True (default), freeze layers 0–2 and 8–17
                            and only optimise layers 3–7.
    """

    LAYER_START = 3   # first identity-related layer
    LAYER_END   = 8   # first fine-detail layer

    def __init__(self,
                 latent_code_real: torch.Tensor,
                 latent_code_fake_nn: torch.Tensor,
                 img_id,
                 out_code_dir: str,
                 gan: str = 'stylegan2_ffhq1024',
                 latent_space: str = 'W+',
                 freeze_non_identity: bool = True):
        super().__init__()

        if not osp.isdir(out_code_dir):
            raise NotADirectoryError(
                f'Invalid output latent code directory: {out_code_dir}'
            )
        if latent_space != 'W+':
            raise NotImplementedError('Only W+ latent space is currently supported.')

        self.img_id       = str(img_id)
        self.gan          = gan
        self.latent_space = latent_space
        self.out_code_dir = out_code_dir

        s, e = self.LAYER_START, self.LAYER_END
        grad_non_id = not freeze_non_identity

        # Naïve mixing (Eq. 1): [z_r^(0:2), z_f^(3:7), z_r^(8:17)]
        self.nontrainable_layers_start = nn.Parameter(
            latent_code_real[:, :s, :].clone(), requires_grad=grad_non_id
        )
        self.trainable_layers = nn.Parameter(
            latent_code_fake_nn[:, s:e, :].clone(), requires_grad=True
        )
        self.nontrainable_layers_end = nn.Parameter(
            latent_code_real[:, e:, :].clone(), requires_grad=grad_non_id
        )

    # ── public helpers ────────────────────────────────────────────────────────

    def do_optim(self) -> bool:
        """Return ``True`` if no saved checkpoint exists for this image yet."""
        return not osp.exists(osp.join(self.out_code_dir, f'{self.img_id}.pt'))

    def save(self):
        """Persist the optimised latent code to disk."""
        torch.save(
            self.forward().detach(),
            osp.join(self.out_code_dir, f'{self.img_id}.pt'),
        )

    # ── nn.Module forward ────────────────────────────────────────────────────

    def forward(self) -> torch.Tensor:
        """Concatenate the three segments and return the full (1, 18, 512) code."""
        return torch.cat(
            [self.nontrainable_layers_start,
             self.trainable_layers,
             self.nontrainable_layers_end],
            dim=1,
        )
