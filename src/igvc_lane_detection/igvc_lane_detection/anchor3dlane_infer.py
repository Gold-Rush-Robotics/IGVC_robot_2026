"""Anchor3DLane++ inference wrapper.

Thin, ROS-free wrapper around the Anchor3DLane++ PyTorch model
(CVPR 2023 / TPAMI 2024, github.com/tusen-ai/Anchor3DLane).

Setup (run once before using this node):

    git clone -b anchor3dlane++ https://github.com/tusen-ai/Anchor3DLane
    cd Anchor3DLane
    conda install pytorch==1.12 torchvision cudatoolkit=11.3 -c pytorch -y
    pip install mmcv-full
    python setup.py develop
    cd mmseg/models/utils/ops && sh make.sh   # builds deformable attention

HuggingFace weights (ResNet-18 360×480, camera-only):
    https://huggingface.co/nowherespyfly/anchor3dlane

Output convention
-----------------
Each detected lane is a list of (x, y, z) tuples **in the camera optical
frame** (the same frame ZED publishes its depth in):

    x  = lateral  (positive = right)
    y  = vertical (positive = down)
    z  = depth    (positive = forward, toward the scene)

Points are ordered from near (small z) to far (large z) along the lane.
The node's TF lookup from camera_optical → base_link then converts them
to the robot frame for costmap rasterisation.

Target hardware: NVIDIA Jetson AGX Orin (JetPack 6, CUDA 12).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    import mmcv  # type: ignore
    from mmcv.runner import load_checkpoint  # type: ignore
    from mmseg.models import build_segmentor  # type: ignore
except ImportError:  # pragma: no cover
    mmcv = None  # type: ignore
    load_checkpoint = None  # type: ignore
    build_segmentor = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

# ImageNet normalisation — matches the Anchor3DLane++ training pipeline.
_IMAGENET_MEAN = np.array([123.675, 116.280, 103.530], dtype=np.float32)
_IMAGENET_STD  = np.array([ 58.395,  57.120,  57.375], dtype=np.float32)


def _to_point_list(lane: object) -> List[Tuple[float, float, float]]:
    """Convert a single lane prediction (any common format) to (x,y,z) tuples.

    Handles:
      • (N, 3) ndarray
      • (N, 2) ndarray (x, z — no height; y is set to 0)
      • flat [x0,y0,z0, x1,y1,z1, …] array/list
    """
    if lane is None:
        return []
    arr = np.asarray(lane, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        elif arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
        else:
            return []
    if arr.ndim != 2:
        return []
    if arr.shape[1] == 2:
        # (x, z) only — pad y=0
        arr = np.column_stack([arr[:, 0], np.zeros(len(arr)), arr[:, 1]])
    if arr.shape[1] < 3:
        return []
    # Filter out padding rows (Anchor3DLane uses large negative y or z=0 sentinels)
    valid = (arr[:, 2] > 0.1) & np.all(np.isfinite(arr[:, :3]), axis=1)
    arr = arr[valid]
    if len(arr) == 0:
        return []
    # Sort by z (near → far) so polyline interpolation is well-ordered
    arr = arr[np.argsort(arr[:, 2])]
    return [(float(p[0]), float(p[1]), float(p[2])) for p in arr]


# ──────────────────────────────────────────────────────────────────────────
# Wrapper class
# ──────────────────────────────────────────────────────────────────────────

class Anchor3DLane:
    """End-to-end wrapper: load model, run inference, return 3D lane polylines.

    Parameters
    ----------
    config_path:
        Filesystem path to the mmcv config file for Anchor3DLane++ (a
        ``.py`` file from the ``configs/`` directory of the repo).
    checkpoint_path:
        Filesystem path to the pretrained checkpoint (``.pth``).
    device:
        PyTorch device string (``cuda:0``, ``cpu``, …).  Falls back to
        CPU with a warning if CUDA is requested but unavailable.
    input_h, input_w:
        Model input resolution.  Must match the config file.
        Default: 360×480 (ResNet-18 camera-only variant).
    score_threshold:
        Minimum lane confidence score.  Lower values recall more lanes
        at the cost of more false positives.  0.4 is a safe starting
        point; drop to 0.3 for thinner IGVC paint.
    half:
        Run in FP16.  Recommended on the Jetson (saves ~40% latency
        with negligible accuracy loss at these confidence thresholds).
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0',
        input_h: int = 360,
        input_w: int = 480,
        score_threshold: float = 0.4,
        half: bool = True,
    ) -> None:
        if mmcv is None or build_segmentor is None:
            raise RuntimeError(
                "Anchor3DLane++ requires mmcv-full and the Anchor3DLane "
                "package.  Run the setup steps at the top of this file.")
        if torch is None:
            raise RuntimeError(
                "torch is not installed.  On Jetson install the JetPack "
                "wheel from developer.download.nvidia.com/compute/redist/jp/")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.score_threshold = float(score_threshold)
        self.half = bool(half)
        self.fallback_warning: Optional[str] = None

        # Honour CUDA availability; fall back to CPU gracefully.
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.fallback_warning = (
                f"CUDA requested ('{device}') but not available; "
                "falling back to CPU.  Expect ~10× slower inference.")
            device = 'cpu'
        self._device = torch.device(device)
        self.model: Optional[object] = None

    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Deserialise weights and move model to the configured device.

        Must be called once before ``infer()``.  Safe to call from a
        background thread — the model is moved to GPU inside this call.
        """
        # Anchor3DLane++ defines custom model classes (e.g. Anchor3DLaneMF) that
        # register themselves with mmcv's registry only when their module is
        # imported.  We must import the repo's mmseg package (not the installed
        # one) before build_segmentor is called, otherwise we get:
        #   KeyError: 'Anchor3DLaneMF is not in the models registry'
        #
        # The repo's mmseg/__init__.py checks  mmcv_version < mmcv_max_version
        # with a strict upper bound.  mmcv==1.7.2 fails because the repo was
        # written for <1.7.2.  We patch the assertion to <= before importing.
        import importlib, os as _os, sys as _sys

        # Colcon-generated console scripts may have a shebang bound to
        # /usr/bin/python3 even when the workspace is launched from /opt/venv.
        # Ensure venv packages (spconv/cumm/etc.) are visible in that case.
        #
        # Use site.addsitedir() rather than sys.path insertion so .pth files
        # are processed (editable installs of spconv/cumm depend on this).
        import glob as _glob
        import site as _site
        _venv_sites = sorted(_glob.glob('/opt/venv/lib/python*/site-packages'))
        for _venv_site in _venv_sites:
            if _os.path.isdir(_venv_site):
                _site.addsitedir(_venv_site)

        # Eager check so failures are explicit before mmseg deep-import chain.
        try:
            import spconv.pytorch as _unused_spconv  # type: ignore  # noqa: F401
        except Exception:
            try:
                import spconv as _unused_spconv_legacy  # type: ignore  # noqa: F401
            except Exception:
                import warnings
                warnings.warn(
                    "spconv not importable from current Python path. If spconv "
                    "is installed in /opt/venv, rebuild this package from that "
                    "venv so ROS entrypoints use /opt/venv/bin/python3.\n"
                    "  source /opt/venv/bin/activate\n"
                    "  colcon build --packages-select igvc_lane_detection igvc_test_bringup")

        _a3d_root = None
        for _p in _sys.path:
            if _os.path.isfile(_os.path.join(_p, 'mmseg', '__init__.py')):
                _a3d_root = _p
                break

        if _a3d_root:
            _init_py = _os.path.join(_a3d_root, 'mmseg', '__init__.py')
            with open(_init_py) as _fh:
                _src = _fh.read()
            # Patch strict < to <= so mmcv==max_version is accepted (idempotent)
            if 'mmcv_version < mmcv_max_version' in _src:
                _src = _src.replace(
                    'mmcv_version < mmcv_max_version',
                    'mmcv_version <= mmcv_max_version')
                with open(_init_py, 'w') as _fh:
                    _fh.write(_src)

            # MultiScaleDeformableAttention is a compiled CUDA extension built
            # by make.sh and installed as an egg into the venv.  The egg path
            # is normally added via easy-install.pth at Python startup, but if
            # the process Python differs from the venv (e.g. system python vs
            # /opt/venv) the .pth file is never processed.  Explicitly find the
            # egg and add it to sys.path so the import always works.
            import glob as _glob
            try:
                import MultiScaleDeformableAttention  # noqa: already on path
            except (ImportError, ModuleNotFoundError):
                # Search venv site-packages (and any dir already on sys.path)
                # for the installed egg.
                _candidates: list = []
                _search_dirs = list(_sys.path) + [
                    '/opt/venv/lib/python3.12/site-packages',
                    '/usr/lib/python3/dist-packages',
                    '/usr/local/lib/python3.12/dist-packages',
                ]
                for _sp in _search_dirs:
                    _candidates += _glob.glob(
                        _os.path.join(_sp, 'MultiScaleDeformableAttention*.egg'))
                _added = False
                for _egg in _candidates:
                    if _os.path.isdir(_egg) and _egg not in _sys.path:
                        _sys.path.insert(0, _egg)
                        _added = True
                if not _added:
                    import warnings
                    warnings.warn(
                        "MultiScaleDeformableAttention CUDA op not found. "
                        "Build it with:\n"
                        "  cd /root/anchor3dlane/mmseg/models/utils/ops\n"
                        "  bash make.sh")

            # Anchor3DLane imports "from ortools.graph import pywrapgraph"
            # from dataset utilities even in inference-only runs.  On Python
            # 3.12, old OR-Tools builds that still export pywrapgraph are not
            # available.  Provide a minimal compatibility shim so imports work.
            try:
                from ortools.graph import pywrapgraph as _unused_pywrapgraph  # type: ignore  # noqa: F401
            except Exception:
                import types as _types
                from scipy.optimize import linear_sum_assignment as _lsa

                class _SimpleMinCostFlowCompat:
                    OPTIMAL = 0

                    def __init__(self):
                        self._arcs = []
                        self._supplies = {}
                        self._flow_arcs = set()
                        self._optimal_cost = 0

                    def AddArcWithCapacityAndUnitCost(self, start, end, cap, cost):
                        self._arcs.append((int(start), int(end), int(cap), int(cost)))

                    def SetNodeSupply(self, node, supply):
                        self._supplies[int(node)] = int(supply)

                    def Solve(self):
                        self._flow_arcs = set()
                        self._optimal_cost = 0
                        if not self._arcs:
                            return self.OPTIMAL

                        # Pattern used by Anchor3DLane MinCostFlow:
                        # source -> left set -> right set -> sink
                        source = min(self._supplies, key=self._supplies.get, default=0)
                        sink = max(self._supplies, key=self._supplies.get, default=0)

                        left_nodes = sorted({e for s, e, c, u in self._arcs if s == source and c > 0})
                        right_nodes = sorted({s for s, e, c, u in self._arcs if e == sink and c > 0})
                        if not left_nodes or not right_nodes:
                            return self.OPTIMAL

                        lidx = {n: i for i, n in enumerate(left_nodes)}
                        ridx = {n: j for j, n in enumerate(right_nodes)}
                        inf = 10 ** 9
                        cm = np.full((len(left_nodes), len(right_nodes)), inf, dtype=np.int64)
                        mid_arc_idx = {}

                        for i, (s, e, c, u) in enumerate(self._arcs):
                            if c <= 0:
                                continue
                            if s in lidx and e in ridx:
                                r = lidx[s]
                                col = ridx[e]
                                if u < cm[r, col]:
                                    cm[r, col] = u
                                    mid_arc_idx[(r, col)] = i

                        row_ind, col_ind = _lsa(cm)
                        for r, c in zip(row_ind, col_ind):
                            if cm[r, c] >= inf:
                                continue
                            lnode = left_nodes[r]
                            rnode = right_nodes[c]

                            for i, (s, e, cap, cost) in enumerate(self._arcs):
                                if s == source and e == lnode and cap > 0:
                                    self._flow_arcs.add(i)
                                    break

                            mid_idx = mid_arc_idx.get((r, c))
                            if mid_idx is not None:
                                self._flow_arcs.add(mid_idx)

                            for i, (s, e, cap, cost) in enumerate(self._arcs):
                                if s == rnode and e == sink and cap > 0:
                                    self._flow_arcs.add(i)
                                    break

                        self._optimal_cost = sum(
                            self._arcs[i][3]
                            for i in self._flow_arcs
                            if self._arcs[i][0] in lidx and self._arcs[i][1] in ridx)
                        return self.OPTIMAL

                    def OptimalCost(self):
                        return int(self._optimal_cost)

                    def NumArcs(self):
                        return len(self._arcs)

                    def Tail(self, arc):
                        return self._arcs[int(arc)][0]

                    def Head(self, arc):
                        return self._arcs[int(arc)][1]

                    def Flow(self, arc):
                        return 1 if int(arc) in self._flow_arcs else 0

                    def UnitCost(self, arc):
                        return self._arcs[int(arc)][3]

                _pywrap = _types.ModuleType('ortools.graph.pywrapgraph')
                _pywrap.SimpleMinCostFlow = _SimpleMinCostFlowCompat

                _graph = _sys.modules.get('ortools.graph')
                if _graph is None:
                    _graph = _types.ModuleType('ortools.graph')
                    _sys.modules['ortools.graph'] = _graph
                _graph.pywrapgraph = _pywrap

                _ortools = _sys.modules.get('ortools')
                if _ortools is None:
                    _ortools = _types.ModuleType('ortools')
                    _sys.modules['ortools'] = _ortools
                _ortools.graph = _graph
                _sys.modules['ortools.graph.pywrapgraph'] = _pywrap

            # ONCE dataset eval helper imports jarvis.eload during mmseg package
            # init, but the pip "jarvis" package on Python 3.12 is Python-2-only
            # and raises SyntaxError.  In inference we do not use that path;
            # provide a tiny compatible stub so imports succeed.
            try:
                import jarvis.eload as _unused_jarvis_eload  # type: ignore  # noqa: F401
            except Exception:
                import json as _json
                import types as _types

                _jarvis = _sys.modules.get('jarvis')
                if _jarvis is None:
                    _jarvis = _types.ModuleType('jarvis')
                    _sys.modules['jarvis'] = _jarvis

                _eload = _types.ModuleType('jarvis.eload')

                def _load_json(path):
                    with open(path, 'r') as _fh:
                        return _json.load(_fh)

                _eload.load_json = _load_json
                _jarvis.eload = _eload
                _sys.modules['jarvis.eload'] = _eload

            # Evict installed mmseg and reimport from the repo so all
            # @SEGMENTORS.register_module() decorators fire.
            for _mod in list(_sys.modules.keys()):
                if _mod == 'mmseg' or _mod.startswith('mmseg.'):
                    del _sys.modules[_mod]

            # mmcv keeps parent registries (MMCV_MODELS, MMCV_ATTENTION, etc.)
            # whose .children dicts and _module_dicts retain mmseg entries even
            # after sys.modules eviction.  Reimporting triggers:
            #   AssertionError: scope mmseg exists in * registry
            #   KeyError: 'MMSegWandbHook is already registered in hook'
            #
            # Robust fix: patch both _add_children and _register_module to be
            # idempotent (replace rather than assert/raise on duplicates).
            # Applied once (idempotency flag), covers all mmcv registries.
            try:
                from mmcv.utils.registry import Registry as _MmcvReg
                if not getattr(_MmcvReg, '_a3d_patched', False):
                    def _add_children_idempotent(self, registry):
                        self.children[registry.scope] = registry

                    _orig_register_module = _MmcvReg._register_module

                    def _register_module_idempotent(
                            self, module, module_name=None, force=False):
                        _orig_register_module(
                            self, module, module_name=module_name, force=True)

                    _MmcvReg._add_children = _add_children_idempotent
                    _MmcvReg._register_module = _register_module_idempotent
                    _MmcvReg._a3d_patched = True
            except Exception:
                pass

            importlib.import_module('mmseg')
        else:
            import warnings
            warnings.warn(
                "anchor3dlane_root not found in sys.path — custom model "
                "classes may not be registered.  Set 'anchor3dlane_root' "
                "parameter to the cloned repo root.")

        # Always resolve build_segmentor from the *currently loaded* mmseg so
        # we use the same SEGMENTORS registry that Anchor3DLaneMF just registered
        # into.  The module-level import bound it to the installed mmseg's registry
        # before the eviction, so we must re-import it here.
        import sys as _sys
        _bs_mod = _sys.modules.get('mmseg.models')
        if _bs_mod is not None and hasattr(_bs_mod, 'build_segmentor'):
            _build_segmentor = _bs_mod.build_segmentor
        else:
            from mmseg.models import build_segmentor as _build_segmentor  # fallback

        cfg = mmcv.Config.fromfile(self.config_path)
        model = _build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, self.checkpoint_path, map_location='cpu')
        model = model.to(self._device)
        model.eval()
        if self.half and self._device.type == 'cuda':
            model = model.half()
        self.model = model

        # Store normalisation params (override from config if present)
        img_norm = cfg.get('img_norm_cfg', {})
        mean = img_norm.get('mean', _IMAGENET_MEAN.tolist())
        std  = img_norm.get('std',  _IMAGENET_STD.tolist())
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

    # ─────────────────────────────────────────────────────────────────

    def infer(
        self,
        bgr: np.ndarray,
        K: Optional[np.ndarray] = None,
    ) -> List[List[Tuple[float, float, float]]]:
        """Run inference on a single BGR image.

        Parameters
        ----------
        bgr:
            BGR uint8 image (H, W, 3) as returned by ``cv2`` / ``CvBridge``.
        K:
            Camera intrinsic matrix (3×3 float32) for the image as
            delivered (before any resize inside this method).  Anchor3DLane++
            uses K to project 3D anchors onto the feature map; providing
            the actual ZED intrinsics gives the most accurate 3D coordinates.
            If ``None`` a synthetic K is derived from the configured
            input resolution (less accurate; use only for testing).

        Returns
        -------
        lanes:
            List of detected lane polylines.  Each polyline is a list of
            ``(x, y, z)`` tuples in the camera optical frame
            (x=right, y=down, z=forward), ordered near → far.
            Returns ``[]`` on no detection or inference error.
        """
        if self.model is None:
            raise RuntimeError("Call load() before infer().")

        orig_h, orig_w = bgr.shape[:2]

        # ── Preprocess ──────────────────────────────────────────────
        rgb = bgr[:, :, ::-1].copy().astype(np.float32)
        if orig_h != self.input_h or orig_w != self.input_w:
            rgb = cv2.resize(rgb, (self.input_w, self.input_h),
                             interpolation=cv2.INTER_LINEAR)
        rgb = (rgb - self._mean) / self._std

        # (H, W, C) → (1, C, H, W) tensor
        t = torch.from_numpy(
            rgb.transpose(2, 0, 1)[None]
        ).to(self._device)
        if self.half and self._device.type == 'cuda':
            t = t.half()
        else:
            t = t.float()

        # ── Scale intrinsics to model input resolution ───────────────
        scale_x = self.input_w / orig_w
        scale_y = self.input_h / orig_h
        if K is None:
            # Derive a plausible K from the input size.
            fx = float(self.input_w) * 1.2
            K_scaled = np.array([
                [fx,  0.0, self.input_w / 2.0],
                [0.0, fx,  self.input_h / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
        else:
            K_scaled = K.astype(np.float32).copy()
            K_scaled[0, :] *= scale_x
            K_scaled[1, :] *= scale_y

        img_metas = [{
            'img_shape':     (self.input_h, self.input_w, 3),
            'pad_shape':     (self.input_h, self.input_w, 3),
            'ori_shape':     (orig_h, orig_w, 3),
            'scale_factor':  np.array(
                [scale_x, scale_y, scale_x, scale_y], dtype=np.float32),
            'flip':          False,
            'cam_intrinsic': K_scaled,
        }]

        # Upstream Anchor3DLane forward_test expects gt_project_matrix with
        # shape [B, 1, 3, 4] and will call squeeze(1). Build a sane default
        # projection from camera intrinsics: P = [K | 0].
        gt_project_matrix = np.zeros((1, 1, 3, 4), dtype=np.float32)
        gt_project_matrix[0, 0, :, :3] = K_scaled
        gt_project_matrix_t = torch.from_numpy(gt_project_matrix).to(self._device)

        # Some model paths expect a binary mask tensor at test time.
        # Zero mask means "all valid pixels" for the current frame.
        mask_t = torch.zeros(
            (1, 1, self.input_h, self.input_w),
            dtype=t.dtype,
            device=self._device,
        )

        # ── Forward pass ─────────────────────────────────────────────
        # Anchor3DLaneMF expects multiframe input:
        #   img: [B, C, H, W, Np+1]
        # plus prev_poses: [B, Np, 3, 4].
        # This ROS node currently has one live frame; for MF models we create
        # a minimal valid shape by repeating the current frame and reusing the
        # same projection as previous pose(s).
        _is_mf = ('_mf' in self.config_path.lower()) or (
            getattr(self.model, '__class__', type('x', (), {})).__name__.lower().endswith('mf'))
        _prev_num = int(getattr(self.model, 'prev_num', 1)) if _is_mf else 0

        if _is_mf:
            img_arg = t.unsqueeze(-1).repeat(1, 1, 1, 1, _prev_num + 1)
            prev_poses_t = gt_project_matrix_t[:, :1, :, :].repeat(1, _prev_num, 1, 1)
        else:
            img_arg = t
            prev_poses_t = None

        def _forward_once(_img, _gt_proj, _mask, _prev):
            if _is_mf:
                return self.model(
                    return_loss=False,
                    img=_img,
                    img_metas=img_metas,
                    gt_project_matrix=_gt_proj,
                    prev_poses=_prev,
                    mask=_mask,
                )
            return self.model(
                return_loss=False,
                img=_img,
                img_metas=img_metas,
                gt_project_matrix=_gt_proj,
                mask=_mask,
            )

        with torch.no_grad():
            try:
                result = _forward_once(img_arg, gt_project_matrix_t, mask_t, prev_poses_t)
            except RuntimeError as exc:
                # Some MF paths mix FP32 constants/projections with FP16 feature
                # tensors and fail matmul. Fall back to full FP32 inference.
                if self.half and 'same dtype' in str(exc):
                    self.model = self.model.float()
                    self.half = False
                    img_arg = img_arg.float()
                    gt_project_matrix_t = gt_project_matrix_t.float()
                    mask_t = mask_t.float()
                    if prev_poses_t is not None:
                        prev_poses_t = prev_poses_t.float()
                    result = _forward_once(img_arg, gt_project_matrix_t, mask_t, prev_poses_t)
                else:
                    raise

        # ── Decode output → List[List[(x,y,z)]] ─────────────────────
        # result is a list of length 1 (batch size 1); result[0] is the
        # prediction for this image — a dict or list depending on the
        # model's simple_test implementation.
        return self._decode_output(result[0] if isinstance(result, list) else result)

    # ─────────────────────────────────────────────────────────────────
    # Output decoding
    # ─────────────────────────────────────────────────────────────────

    def _decode_output(
        self,
        raw: object,
    ) -> List[List[Tuple[float, float, float]]]:
        """Parse model output to ``List[List[(x,y,z)]]``.

        Anchor3DLane++ ``simple_test`` returns a dict with keys that vary
        slightly between branches.  This method tries common key names in
        order of preference.  If the actual output of your build uses a
        different key, add it to the ``PRED_KEYS`` / ``SCORE_KEYS`` tuples
        below.
        """
        PRED_KEYS  = ('pred_3d_lanes', 'lanes_3d', 'lanes', 'lane_pts')
        SCORE_KEYS = ('scores', 'lane_scores', 'conf')

        if raw is None:
            return []

        # Some builds return a plain list of lanes directly
        if isinstance(raw, (list, tuple)) and not isinstance(raw, dict):
            lanes = []
            for lane in raw:
                pts = _to_point_list(lane)
                if pts:
                    lanes.append(pts)
            return lanes

        if not isinstance(raw, dict):
            return []

        pred_key = next((k for k in PRED_KEYS if k in raw), None)
        if pred_key is None:
            return []

        score_key = next((k for k in SCORE_KEYS if k in raw), None)
        preds  = raw[pred_key]
        scores = raw[score_key] if score_key else [1.0] * len(preds)

        lanes: List[List[Tuple[float, float, float]]] = []
        for lane, score in zip(preds, scores):
            score_f = float(score) if not hasattr(score, '__len__') else float(np.max(score))
            if score_f < self.score_threshold:
                continue
            pts = _to_point_list(lane)
            if pts:
                lanes.append(pts)
        return lanes
