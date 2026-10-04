# test1.py
import matplotlib
matplotlib.use('Agg')

import os
import argparse
import torch
import numpy as np
import copy
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

from dataloader import TestSet
from lib.TSRNet import TSRNet
from lib.SGRFNet import TSRNet_ShapeGuided_ResidualFusion_SHAPEX

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ================= Utility =================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/Processed_PTBXL/')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ckpt_baseline', type=str, default='best-checkpointsTSRNet-0860.pt')
    parser.add_argument('--ckpt_sgrf', type=str,
                        default='ckpt_shape_guided_shapex_gate/best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt')

    parser.add_argument('--dims', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=96)
    parser.add_argument('--shapex_num_shapelets', type=int, default=16)
    parser.add_argument('--shapex_shapelet_len', type=int, default=96)
    parser.add_argument('--shapex_no_encoder', action='store_true')
    parser.add_argument('--attention_heads', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--num_transformer_blocks', type=int, default=2)

    parser.add_argument('--mask_ratio_time', type=int, default=30)
    parser.add_argument('--mask_ratio_spec', type=int, default=20)
    parser.add_argument('--mask_loss', action='store_true')

    # 新增参数
    parser.add_argument('--out_dir', type=str, default='.',
                        help='错分清单、可视化图片输出目录')
    parser.add_argument('--max_show', type=int, default=50,
                        help='控制台最多打印多少条错分样本')
    parser.add_argument('--save_misclassified_figs', action='store_true',
                        help='是否把错分样本逐条画图')
    parser.add_argument('--num_misclassified_figs', type=int, default=10,
                        help='每个模型最多画多少条错分样本图')

    return parser.parse_args()


def normalize_instance(data):
    data_norm = copy.deepcopy(data)
    for ch in range(data.shape[1]):
        seq = data[:, ch]
        d_min = seq.min()
        d_max = seq.max()
        if d_max - d_min > 1e-6:
            data_norm[:, ch] = 2 * (seq - d_min) / (d_max - d_min) - 1
        else:
            data_norm[:, ch] = 0
    return data_norm


def compute_spectrogram(time_data):
    f, t, Zxx = stft(time_data.transpose(1, 0), fs=500, window='hann', nperseg=125)
    return np.abs(Zxx).transpose(1, 2, 0)


# ================= Phase 1: Metrics =================

def evaluate_testmy_style(args, model, test_loader, device, model_type='baseline'):
    model.eval()
    results = []
    with torch.no_grad():
        for i, (time_ecg, spectrogram_ecg, r_index) in tqdm(
                enumerate(test_loader), total=len(test_loader),
                desc=f"Eval {model_type}"):
            time_ecg = time_ecg.float().to(device)
            spectrogram_ecg = spectrogram_ecg.float().to(device)
            bs, time_length, dim = time_ecg.shape

            mask_loss_tensor = None
            if args.mask_loss:
                idx_length = r_index.shape[1]
                mask_loss_tensor = torch.zeros((time_length, args.dims),
                                               dtype=torch.bool).to(device)
                for r_idx in range(idx_length):
                    r_val = r_index[0][r_idx]
                    if 200 < r_val < 4800 - 400:
                        left = max(0, r_val - 240)
                        mask_loss_tensor[left:r_val + 240, :] = 1

            num_strategies = max(1, 100 // args.mask_ratio_time)
            instance_result = []

            for j in range(num_strategies):
                mask_time = copy.deepcopy(time_ecg)
                mask_t = torch.zeros((1, time_length, 1), dtype=torch.bool).to(device)
                patch_interval_time = 4800 // args.mask_ratio_time
                for k in range(args.mask_ratio_time):
                    cut_idx = 48 * j + patch_interval_time * k
                    if cut_idx + 48 <= time_length:
                        mask_t[:, cut_idx:cut_idx + 48] = 1
                mask_time = torch.mul(mask_time, ~mask_t)

                mask_spec = copy.deepcopy(spectrogram_ecg)
                bs_s, freq, t_dim, _ = spectrogram_ecg.shape
                mask_s = torch.zeros((bs_s, freq, t_dim, 1), dtype=torch.bool).to(device)
                patch_interval_spec = 66 // args.mask_ratio_spec
                for k in range(args.mask_ratio_spec):
                    cut_idx = 1 * j + patch_interval_spec * k
                    if cut_idx < t_dim:
                        mask_s[:, :, cut_idx:cut_idx + 1] = 1
                mask_spec = torch.mul(mask_spec, ~mask_s)

                if model_type == 'sgrf':
                    recon, sigma = model(mask_time, mask_spec, return_aux=False)
                    sigma = torch.clamp(sigma, min=-5.0, max=5.0)
                    target = time_ecg[:, :, 0:model.channel]
                    recon_err_sq = torch.square(recon - target)
                    sigma_bd = sigma.expand_as(recon_err_sq)
                    mse_score = torch.exp(-sigma_bd) * recon_err_sq + sigma_bd
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        recon, target, dim=1).unsqueeze(1)
                    shape_error = (1 - cosine_sim).expand(1, time_length, model.channel)
                    combined_map = mse_score + 0.15 * shape_error

                    if args.mask_loss and mask_loss_tensor is not None:
                        masked_score = torch.mul(combined_map, mask_loss_tensor)
                        loss_val = torch.sum(masked_score) / (
                            torch.sum(mask_loss_tensor) + 1e-6)
                    else:
                        loss_val = torch.mean(combined_map)
                else:
                    (gen_time, time_var) = model(mask_time, mask_spec)
                    time_err = (gen_time - time_ecg) ** 2
                    if args.mask_loss and mask_loss_tensor is not None:
                        l_time = torch.exp(-time_var) * time_err
                        l_time = torch.mul(l_time, mask_loss_tensor)
                        l_time = torch.sum(l_time) / (
                            torch.sum(mask_loss_tensor) + 1e-6)
                    else:
                        l_time = torch.mean(torch.exp(-time_var) * time_err)
                    loss_val = l_time

                instance_result.append(loss_val.item())

            results.append(np.mean(instance_result))
    return np.array(results)


def calculate_full_metrics(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)

    if labels.ndim == 2:
        labels = labels.argmax(axis=1)

    s_min, s_max = scores.min(), scores.max()
    scores_norm = (scores - s_min) / (s_max - s_min) if s_max - s_min > 1e-8 else scores

    try:
        auc_val = roc_auc_score(labels, scores_norm)
        fpr, tpr, thresholds = roc_curve(labels, scores_norm)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        preds = (scores_norm >= best_thresh).astype(int)
        f1 = f1_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
    except Exception as e:
        print(f"[calculate_full_metrics] error: {e}")
        auc_val, f1, prec, rec = 0.5, 0, 0, 0
        fpr, tpr = [0, 1], [0, 1]
        best_thresh = 0.5
        preds = np.zeros_like(labels)

    return {
        "AUC": auc_val, "F1": f1, "Precision": prec, "Recall": rec,
        "FPR": fpr, "TPR": tpr,
        "best_thresh": best_thresh,
        "scores_norm": scores_norm,
        "scores_raw": scores,
        "preds": preds,
    }


# ================= 错分样本报告 =================

def report_misclassified(metrics, labels, model_name='model',
                         save_dir='.', max_show=50):
    """
    打印并保存模型判断错误的样本。
      - FN (False Negative)：真异常判成正常 —— 漏报
      - FP (False Positive)：真正常判成异常 —— 误报
    """
    labels = np.array(labels)
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)

    preds       = metrics['preds']
    scores_norm = metrics['scores_norm']
    scores_raw  = metrics.get('scores_raw', scores_norm)
    best_thresh = metrics['best_thresh']

    idx_fn  = np.where((labels == 1) & (preds == 0))[0]
    idx_fp  = np.where((labels == 0) & (preds == 1))[0]
    idx_err = np.concatenate([idx_fn, idx_fp])

    print(f"\n{'=' * 70}")
    print(f"[{model_name}] best_thresh = {best_thresh:.6f}  "
          f"misclassified = {len(idx_err)} / {len(labels)}")
    print(f"  FN (anomaly -> normal) : {len(idx_fn)}")
    print(f"  FP (normal  -> anomaly): {len(idx_fp)}")
    print(f"{'=' * 70}")

    def _print_block(idxs, tag):
        print(f"\n[{model_name}] {tag}，共 {len(idxs)} 条:")
        print(f"{'idx':>6} | {'label':>5} | {'pred':>4} | "
              f"{'score_norm':>10} | {'score_raw':>10}")
        print('-' * 52)
        for i in idxs[:max_show]:
            print(f"{i:>6d} | {labels[i]:>5d} | {preds[i]:>4d} | "
                  f"{scores_norm[i]:>10.4f} | {scores_raw[i]:>10.4f}")
        if len(idxs) > max_show:
            print(f"  ... 还有 {len(idxs) - max_show} 条未显示")

    _print_block(idx_fn, 'FN (label=1, pred=0) 漏报')
    _print_block(idx_fp, 'FP (label=0, pred=1) 误报')

    # 落盘
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, f'misclassified_{model_name}.txt')
    npy_path = os.path.join(save_dir, f'misclassified_{model_name}_idx.npy')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'model: {model_name}\n')
        f.write(f'best_thresh: {best_thresh:.6f}\n')
        f.write(f'total misclassified: {len(idx_err)} / {len(labels)}\n')
        f.write(f'  FN (missed anomaly): {len(idx_fn)}\n')
        f.write(f'  FP (false alarm)   : {len(idx_fp)}\n\n')

        f.write('--- FN (label=1, pred=0) ---\n')
        f.write('idx,label,pred,score_norm,score_raw\n')
        for i in idx_fn:
            f.write(f'{i},{labels[i]},{preds[i]},'
                    f'{scores_norm[i]:.6f},{scores_raw[i]:.6f}\n')

        f.write('\n--- FP (label=0, pred=1) ---\n')
        f.write('idx,label,pred,score_norm,score_raw\n')
        for i in idx_fp:
            f.write(f'{i},{labels[i]},{preds[i]},'
                    f'{scores_norm[i]:.6f},{scores_raw[i]:.6f}\n')

    np.save(npy_path, idx_err)

    print(f"\n✅ Saved:")
    print(f"   {os.path.abspath(txt_path)}")
    print(f"   {os.path.abspath(npy_path)}")

    return idx_fn, idx_fp, idx_err


def plot_misclassified_samples(dataset, labels, metrics, model_name,
                               idxs, save_dir='.', max_plots=10):
    """把错分样本逐条画 12 导联（只画前 max_plots 条）"""
    os.makedirs(save_dir, exist_ok=True)
    labels = np.array(labels)
    if labels.ndim == 2:
        labels = labels.argmax(axis=1)

    preds       = metrics['preds']
    scores_norm = metrics['scores_norm']

    for k, i in enumerate(idxs[:max_plots]):
        sig = dataset.test_data[i]           # (5000, 12)
        fs  = 500
        t   = np.arange(sig.shape[0]) / fs

        fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)
        for ch in range(12):
            axes[ch].plot(t, sig[:, ch], lw=0.8, color='black')
            axes[ch].set_ylabel(LEAD_NAMES[ch], rotation=0,
                                labelpad=20, fontsize=11)
            axes[ch].grid(alpha=0.3)
        axes[-1].set_xlabel('Time (s)')

        tag = 'FN' if labels[i] == 1 else 'FP'
        fig.suptitle(
            f'[{model_name}/{tag}] idx={i}  label={labels[i]}  '
            f'pred={preds[i]}  score={scores_norm[i]:.4f}',
            fontsize=14, y=0.995)

        plt.tight_layout()
        out = os.path.join(save_dir,
                           f'misclassified_{model_name}_{tag}_idx{i}.png')
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  [{k+1}/{min(len(idxs), max_plots)}] saved: {out}')


# ================= Phase 2: Visualization =================

def pick_valid_random_samples(dataset, labels, seed=42):
    print("Picking RANDOM VALID samples...")
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset))
    rng.shuffle(all_indices)

    normal_idx, anom_idx = None, None
    for i in all_indices:
        sig = dataset.test_data[i][:1000, :]
        if np.std(sig) > 0.1:
            if labels[i] == 0 and normal_idx is None:
                normal_idx = i
            elif labels[i] == 1 and anom_idx is None:
                anom_idx = i
        if normal_idx is not None and anom_idx is not None:
            break

    print(f"Selected -> Normal: {normal_idx}, Anomaly: {anom_idx}")
    return normal_idx, anom_idx


def run_vis_inference(model, dataset, idx, device, model_type='baseline'):
    raw_sig = dataset.test_data[idx][100:4900, :]
    norm_sig = normalize_instance(raw_sig)
    norm_spec = compute_spectrogram(norm_sig)

    time_tensor = torch.tensor(norm_sig).unsqueeze(0).float().to(device)
    spec_tensor = torch.tensor(norm_spec).unsqueeze(0).float().to(device)

    _, time_len, _ = time_tensor.shape

    mask_t = torch.zeros((1, time_len, 1), dtype=torch.bool).to(device)
    mask_start, mask_end = 2000, 2400
    mask_t[:, mask_start:mask_end] = 1
    masked_time = torch.mul(time_tensor, ~mask_t)

    bs_s, f_d, t_d, _ = spec_tensor.shape
    mask_s = torch.zeros((bs_s, f_d, t_d, 1), dtype=torch.bool).to(device)
    for k in range(0, t_d, 10):
        mask_s[:, :, k:k + 1] = 1
    masked_spec = torch.mul(spec_tensor, ~mask_s)

    with torch.no_grad():
        if model_type == 'sgrf':
            recon, _ = model(masked_time, masked_spec, return_aux=False)
        else:
            recon, _ = model(masked_time, masked_spec)

    std_leads = np.std(norm_sig, axis=0)
    best_lead_idx = int(np.argmax(std_leads))
    lead_name = LEAD_NAMES[best_lead_idx]

    gt_np = time_tensor[0, :, best_lead_idx].detach().cpu().numpy()
    rec_np = recon[0, :, best_lead_idx].detach().cpu().numpy()
    mse = float(np.mean((gt_np - rec_np) ** 2))

    return gt_np, rec_np, mse, lead_name, (mask_start, mask_end)


# ================= Main =================

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Data...")
    dataset = TestSet(folder=args.data_path)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=True)
    labels = np.load(os.path.join(args.data_path, 'label.npy'))
    print(f"labels: {labels.shape}, dtype={labels.dtype}")

    print("Loading Models...")
    model_base = TSRNet(enc_in=args.dims).to(device)
    ckpt_b = torch.load(args.ckpt_baseline, map_location=device)
    model_base.load_state_dict(
        ckpt_b['model_state_dict'] if 'model_state_dict' in ckpt_b else ckpt_b)
    model_base.eval()

    model_sgrf = TSRNet_ShapeGuided_ResidualFusion_SHAPEX(
        enc_in=args.dims, channel=args.dims,
        d_model=args.d_model,
        shapex_num_shapelets=args.shapex_num_shapelets,
        shapex_shapelet_len=args.shapex_shapelet_len,
        shapex_use_encoder=not args.shapex_no_encoder
    ).to(device)
    ckpt_s = torch.load(args.ckpt_sgrf, map_location=device, weights_only=False)
    model_sgrf.load_state_dict(
        ckpt_s['model_state_dict'] if 'model_state_dict' in ckpt_s else ckpt_s)
    model_sgrf.eval()

    # ---------- Phase 1 ----------
    print("\n[Phase 1] Calculating Metrics...")
    scores_base = evaluate_testmy_style(args, model_base, test_loader, device, 'baseline')
    scores_sgrf = evaluate_testmy_style(args, model_sgrf, test_loader, device, 'sgrf')

    metrics_b = calculate_full_metrics(scores_base, labels)
    metrics_s = calculate_full_metrics(scores_sgrf, labels)

    idx = 10
    website_threshold = -0.9182019432385763
    raw_scores = np.asarray(scores_sgrf)
    lo, hi = raw_scores.min(), raw_scores.max()
    threshold = metrics_s["best_thresh"]
    raw_threshold = (
        lo + threshold * (hi - lo)
        if hi - lo > 1e-8 else threshold
    )

    print("\n样本10阈值对照")
    print("label:", labels[idx])
    print("test2原始分数:", raw_scores[idx])
    print("test2原始空间阈值:", raw_threshold)
    print("test2预测:", metrics_s["preds"][idx])
    print("网站阈值:", website_threshold)
    print("同一分数按网站阈值预测:",
        int(raw_scores[idx] >= website_threshold))
    print(f"\nBaseline AUC: {metrics_b['AUC']:.4f}  |  SGRF-Net AUC: {metrics_s['AUC']:.4f}")
    print(f"Baseline F1 : {metrics_b['F1']:.4f}  |  SGRF-Net F1 : {metrics_s['F1']:.4f}")

    # ---------- 错分样本报告 ----------
    print("\n" + "#" * 70)
    print("# 错分样本分析")
    print("#" * 70)

    idx_fn_b, idx_fp_b, idx_err_b = report_misclassified(
        metrics_b, labels, model_name='baseline',
        save_dir=args.out_dir, max_show=args.max_show)

    idx_fn_s, idx_fp_s, idx_err_s = report_misclassified(
        metrics_s, labels, model_name='sgrf',
        save_dir=args.out_dir, max_show=args.max_show)

    # 两个模型都判错的
    both_wrong = np.intersect1d(idx_err_b, idx_err_s)
    print(f"\n两个模型都判错的样本数: {len(both_wrong)}")
    if len(both_wrong) > 0:
        preview = both_wrong[:30].tolist()
        print(f"索引 (前30): {preview}{' ...' if len(both_wrong) > 30 else ''}")
        np.save(os.path.join(args.out_dir, 'both_wrong_idx.npy'), both_wrong)

    # 可选：画错分样本
    if args.save_misclassified_figs:
        print(f"\n画错分样本图 (每模型最多 {args.num_misclassified_figs} 条)...")
        plot_misclassified_samples(dataset, labels, metrics_b, 'baseline',
                                   idx_err_b, save_dir=args.out_dir,
                                   max_plots=args.num_misclassified_figs)
        plot_misclassified_samples(dataset, labels, metrics_s, 'sgrf',
                                   idx_err_s, save_dir=args.out_dir,
                                   max_plots=args.num_misclassified_figs)

    # ---------- Phase 2 ----------
    print("\n[Phase 2] Generating Visualization...")
    idx_norm, idx_anom = pick_valid_random_samples(dataset, labels)
    if idx_norm is None or idx_anom is None:
        print("未找到有效样本，跳过可视化")
        return

    gt_n, rec_b_n, mse_b_n, lead_n, mask_range = run_vis_inference(
        model_base, dataset, idx_norm, device, 'baseline')
    _, rec_s_n, mse_s_n, _, _ = run_vis_inference(
        model_sgrf, dataset, idx_norm, device, 'sgrf')

    gt_a, rec_b_a, mse_b_a, lead_a, _ = run_vis_inference(
        model_base, dataset, idx_anom, device, 'baseline')
    _, rec_s_a, mse_s_a, _, _ = run_vis_inference(
        model_sgrf, dataset, idx_anom, device, 'sgrf')

    # ---------- Phase 3 ----------
    print("Plotting...")
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1],
                           hspace=0.4, wspace=0.2)

    ax_metrics = fig.add_subplot(gs[0, 0])
    bar_width = 0.35
    metrics_names = ['AUC', 'F1', 'Precision', 'Recall']
    vals_b = [metrics_b[k] for k in metrics_names]
    vals_s = [metrics_s[k] for k in metrics_names]
    x = np.arange(len(metrics_names))
    rects1 = ax_metrics.bar(x - bar_width / 2, vals_b, bar_width,
                            label='Baseline', color='#87CEEB')
    rects2 = ax_metrics.bar(x + bar_width / 2, vals_s, bar_width,
                            label='Ours (SGRF-Net)', color='#FFA500')
    ax_metrics.set_ylabel('Score')
    ax_metrics.set_title('Performance Metrics', fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metrics_names)
    ax_metrics.set_ylim(0, 1.15)
    ax_metrics.legend(loc='lower right')

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax_metrics.annotate(f'{h:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    ax_roc = fig.add_subplot(gs[0, 1])
    ax_roc.plot(metrics_b['FPR'], metrics_b['TPR'], '--', color='navy',
                label=f'Baseline (AUC={metrics_b["AUC"]:.3f})')
    ax_roc.plot(metrics_s['FPR'], metrics_s['TPR'], color='darkorange',
                linewidth=2, label=f'Ours (AUC={metrics_s["AUC"]:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_title('ROC Curve', fontweight='bold')
    ax_roc.legend(loc="lower right")

    view_s, view_e = 1500, 2900
    x_axis = np.arange(view_s, view_e)

    ax_orig_n = fig.add_subplot(gs[1, 0])
    ax_orig_n.plot(x_axis, gt_n[view_s:view_e], 'k', linewidth=1.5)
    ax_orig_n.set_title(f'Normal Sample (Idx: {idx_norm}, Lead: {lead_n})',
                        fontweight='bold')

    ax_orig_a = fig.add_subplot(gs[1, 1])
    ax_orig_a.plot(x_axis, gt_a[view_s:view_e], 'k', linewidth=1.5)
    ax_orig_a.set_title(f'Anomaly Sample (Idx: {idx_anom}, Lead: {lead_a})',
                        fontweight='bold')

    ax_base_n = fig.add_subplot(gs[2, 0], sharex=ax_orig_n)
    ax_base_n.plot(x_axis, gt_n[view_s:view_e], 'k', alpha=0.3)
    ax_base_n.plot(x_axis, rec_b_n[view_s:view_e], 'b')
    ax_base_n.axvspan(mask_range[0], mask_range[1], color='yellow', alpha=0.15)
    ax_base_n.set_title(f'Baseline (MSE: {mse_b_n:.4f})', color='navy')

    ax_base_a = fig.add_subplot(gs[2, 1], sharex=ax_orig_a)
    ax_base_a.plot(x_axis, gt_a[view_s:view_e], 'k', alpha=0.3)
    ax_base_a.plot(x_axis, rec_b_a[view_s:view_e], 'b')
    ax_base_a.axvspan(mask_range[0], mask_range[1], color='yellow', alpha=0.15)
    ax_base_a.set_title(f'Baseline (MSE: {mse_b_a:.4f})', color='navy')

    ax_sgrf_n = fig.add_subplot(gs[3, 0], sharex=ax_orig_n)
    ax_sgrf_n.plot(x_axis, gt_n[view_s:view_e], 'k', alpha=0.3)
    ax_sgrf_n.plot(x_axis, rec_s_n[view_s:view_e], 'r')
    ax_sgrf_n.axvspan(mask_range[0], mask_range[1], color='yellow', alpha=0.15)
    ax_sgrf_n.set_title(f'SGRF-Net (MSE: {mse_s_n:.4f})', color='darkred')

    ax_sgrf_a = fig.add_subplot(gs[3, 1], sharex=ax_orig_a)
    ax_sgrf_a.plot(x_axis, gt_a[view_s:view_e], 'k', alpha=0.3)
    ax_sgrf_a.plot(x_axis, rec_s_a[view_s:view_e], 'r')
    ax_sgrf_a.axvspan(mask_range[0], mask_range[1], color='yellow', alpha=0.15)
    ax_sgrf_a.set_title(f'SGRF-Net (MSE: {mse_s_a:.4f})', color='darkred')

    out_path = os.path.join(args.out_dir, 'paper_final_figure_random_leads.jpg')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Success! Saved to {out_path}")


if __name__ == '__main__':
    main()