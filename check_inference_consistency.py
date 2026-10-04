"""Read-only single-record diagnosis. Run from the ECG project root.
No gateway, no history changes, no threshold changes. Report goes to a new JSON.
Uses the uploaded TestSet contract: crop [100:4900], unnormalized STFT.
"""
import argparse
import ast
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_evaluator(path, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == 'evaluate_testmy_style']
    if len(nodes) != 1:
        raise ValueError('Expected evaluate_testmy_style in test2.py')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['evaluate_testmy_style']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/Processed_PTBXL/test.npy')
    p.add_argument('--labels', default='data/Processed_PTBXL/label.npy')
    p.add_argument('--index', type=int, default=10)
    p.add_argument('--checkpoint', default='ckpt_shape_guided_shapex_gate/best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt')
    p.add_argument('--test-script', default='test2.py')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, default=-0.9182019432385763)
    args = p.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    import numpy as np
    import torch
    from scipy.signal import stft
    from src.inference.sgrf_adapter import SGRFDetector
    from lib.SGRFNet import TSRNet_ShapeGuided_ResidualFusion_SHAPEX

    data = np.load(args.data, mmap_mode='r', allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    if data.ndim != 3 or data.shape[2] != 12 or data.shape[1] < 4900:
        raise ValueError('Expected N x at least 4900 x 12 data')
    if not 0 <= args.index < len(data) or len(labels) != len(data):
        raise ValueError('Index or labels/data length mismatch')
    raw = np.array(data[args.index, 100:4900, :], copy=True)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite input')
    detector = SGRFDetector(checkpoint=args.checkpoint, device=args.device)
    norm = detector.normalize_instance(raw)
    evaluate = load_evaluator(args.test_script, {
        'np': np, 'torch': torch, 'copy': copy,
        'tqdm': lambda it, **kw: it,
    })
    cfg = argparse.Namespace(mask_loss=False, dims=12, mask_ratio_time=30, mask_ratio_spec=20)

    def score(model, time, spec):
        batch = [(torch.as_tensor(np.ascontiguousarray(time)).unsqueeze(0),
                  torch.as_tensor(np.ascontiguousarray(spec)).unsqueeze(0),
                  torch.empty((1, 0), dtype=torch.int64))]
        return float(evaluate(cfg, model, batch, detector.device, 'sgrf')[0])

    # Exact uploaded test2 model-construction arguments. No baseline model needed.
    legacy = TSRNet_ShapeGuided_ResidualFusion_SHAPEX(
        enc_in=12, channel=12, d_model=96, shapex_num_shapelets=16,
        shapex_shapelet_len=96, shapex_use_encoder=True).to(detector.device)
    legacy.load_state_dict(detector.model.state_dict(), strict=True)
    legacy.eval()
    raw_spec = np.abs(stft(raw.T, fs=500, window='hann', nperseg=125)[2]).transpose(1, 2, 0)
    norm_spec = detector.compute_spectrogram(norm)
    scores = {
        'test2_constructor_without_normalization': score(legacy, raw, raw_spec),
        'adapter_constructor_without_normalization': score(detector.model, raw, raw_spec),
        'adapter_constructor_with_normalization': score(detector.model, norm, norm_spec),
        'website_adapter_predict': float(detector.predict(raw).anomaly_score),
    }
    report = {
        'scope': 'single_record_default_config_no_mask_loss',
        'index': args.index, 'label': np.asarray(labels[args.index]).tolist(),
        'data_shape': list(data.shape), 'crop': [100, 4900],
        'data_file_sha256': sha(args.data), 'labels_sha256': sha(args.labels),
        'checkpoint_sha256': sha(args.checkpoint),
        'input_sha256': hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float32).tobytes()).hexdigest(),
        'source_hashes': {f: sha(f) for f in [args.test_script, 'dataloader.py', 'src/inference/sgrf_adapter.py']},
        'device': str(detector.device), 'numpy_version': np.__version__, 'torch_version': torch.__version__,
        'threshold_for_comparison_only': args.threshold,
        'scores': scores,
        'predictions_at_same_threshold': {k: int(v >= args.threshold) for k, v in scores.items()},
        'max_input_change_from_normalization': float(np.max(np.abs(raw.astype(np.float32) - norm))),
        'normalized_evaluator_matches_adapter': bool(np.isclose(
            scores['adapter_constructor_with_normalization'], scores['website_adapter_predict'], atol=1e-6, rtol=0)),
        'notes': ['Does not read or overwrite website history.',
                  'Compare input/checkpoint hashes against saved website provenance.',
                  'Does not establish which preprocessing matches training.',
                  'Raw STFT follows uploaded TestSet code, without executing HeartPy peak detection; mask_loss=False.',
                  'Non-default test2 model/mask settings require matching changes to this diagnostic.'],
    }
    out = Path('evaluation/inference_consistency')
    out.mkdir(parents=True, exist_ok=True)
    path = out / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with path.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Report:', path)


if __name__ == '__main__':
    main()
