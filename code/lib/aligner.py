import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image

try:
    import face_alignment
    _FA_AVAILABLE = True
except ImportError:
    _FA_AVAILABLE = False


# FFHQ reference alignment helper (matches StyleGAN2 / e4e canonical framing).
def _ffhq_quad_from_68(lm68: np.ndarray) -> np.ndarray:
    """Compute the 4 FFHQ crop corners (TL, BL, BR, TR) from 68 landmarks."""
    eye_left = lm68[36:42].mean(axis=0)
    eye_right = lm68[42:48].mean(axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm68[48]
    mouth_right = lm68[54]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x_norm = np.hypot(*x)
    if x_norm < 1e-6:
        raise ValueError('Degenerate landmarks')
    x /= x_norm
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)


class FaceAligner:
    """Detect and align a face image to a canonical 256×256 crop.

    Uses ``face_alignment`` (pip install face-alignment) for landmark
    detection.  Falls back to a centre-crop if the library is unavailable
    or no face is detected.

    Args:
        device (str): ``'cuda'`` or ``'cpu'``.
        output_size (int): size of the square output crop (default 256).
    """

    def __init__(self, device='cuda', output_size=256):
        self.output_size = output_size
        self._fa = None

        if _FA_AVAILABLE:
            try:
                self._fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    device=device,
                    face_detector='sfd',
                )
            except Exception:
                self._fa = None

        if self._fa is None:
            print(
                '[FaceAligner] face_alignment not available or failed to initialise. '
                'Falling back to centre-crop alignment.'
            )

    # ── public API ────────────────────────────────────────────────────────────

    def align_face(self, image_file: str,
                   alignment_errors_file: str = None,
                   face_detection_errors_file: str = None) -> np.ndarray:
        """Detect, align, and crop the face in *image_file*.

        Args:
            image_file (str): path to an image file.
            alignment_errors_file (str): optional path to a text file where
                alignment failures are logged.
            face_detection_errors_file (str): optional path to a text file
                where detection failures are logged.

        Returns:
            np.ndarray: RGB image of shape (output_size, output_size, 3),
                        dtype uint8.
        """
        img_rgb = np.array(Image.open(image_file).convert('RGB'))

        if self._fa is not None:
            try:
                preds = self._fa.get_landmarks(img_rgb)
                if preds is not None and len(preds) > 0:
                    lm68 = preds[0]  # (68, 2)
                    quad = _ffhq_quad_from_68(lm68)
                    return self._warp(img_rgb, quad)
            except Exception as exc:
                if alignment_errors_file:
                    self._log(alignment_errors_file,
                              f'{image_file}: alignment error — {exc}')

        # No landmarks found
        if face_detection_errors_file:
            self._log(face_detection_errors_file,
                      f'{image_file}: no face detected')

        return self._centre_crop(img_rgb)

    # ── private helpers ───────────────────────────────────────────────────────

    def _warp(self, img_rgb: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Warp the FFHQ crop *quad* (TL, BL, BR, TR) to (output_size, output_size)."""
        src_pts = quad[[0, 1, 3]]  # TL, BL, TR
        dst_pts = np.array([
            [0.0, 0.0],
            [0.0, float(self.output_size)],
            [float(self.output_size), 0.0],
        ], dtype=np.float32)
        M = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(
            img_rgb, M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _centre_crop(self, img_rgb: np.ndarray) -> np.ndarray:
        """Fallback: resize and centre-crop to output_size."""
        h, w = img_rgb.shape[:2]
        s = min(h, w)
        top  = (h - s) // 2
        left = (w - s) // 2
        cropped = img_rgb[top:top + s, left:left + s]
        return cv2.resize(cropped, (self.output_size, self.output_size),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _log(filepath: str, message: str):
        os.makedirs(osp.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'a') as f:
            f.write(message + '\n')
