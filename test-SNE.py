import matplotlib

matplotlib.use('Agg')

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# --- [修改] 强力去除警告 ---
warnings.filterwarnings("ignore")
# 专门过滤 PyTorch 的 weights_only FutureWarning
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# 引入项目模块
from dataloader import TestSet
from lib.TSRNet import TSRNet
from lib.SGRFNet import TSRNet_ShapeGuided_ResidualFusion_SHAPEX

# 全局容器
features_storage = []


def get_features_hook(module, input, output):
    """
    Hook 函数：截获特征并进行全局平均池化
    """
    feat = output
    # 处理可能的 Tuple 输出
    if isinstance(feat, tuple):
        feat = feat[0]

    # 维度处理: 我们需要 (Batch, Dim)
    # 这里的 feat 可能是 (Batch, Time, Dim)

    if feat.dim() == 3:
        # Global Average Pooling: 对时间维度取平均
        # 假设形状是 [Batch, Time, Dim] -> 在 dim=1 平均 -> [Batch, Dim]
        # 或者是 [Batch, Channels, Time] -> 在 dim=2 平均
        # TSRNet 的 latent 通常是 (B, T, D)
        avg_feat = feat.mean(dim=1)

        # 二次检查，如果平均后还有多余维度（比如 (B, D, 1)），再压扁
        if avg_feat.dim() > 2:
            avg_feat = avg_feat.mean(dim=-1)
    else:
        avg_feat = feat

    features_storage.append(avg_feat.detach().cpu())


def register_smart_hook(model):
    """
    [智能适配] 自动寻找这一层：'在进入解码器之前，包含了样本核心语义的那一层'
    """
    handle = None
    model_name = model.__class__.__name__

    # 1. 针对 SHAPEX 模型
    # SHAPEX 用了 Transformer Encoder，最后一层的输出是最纯粹的语义特征
    if hasattr(model, 'transformer_blocks') and len(model.transformer_blocks) > 0:
        # print(f"[{model_name}] Hooking 'transformer_blocks[-1]'")
        handle = model.transformer_blocks[-1].register_forward_hook(get_features_hook)

    # 2. 针对 Baseline TSRNet
    # TSRNet.py 在解码前通过了一个 self.mlp
    elif hasattr(model, 'mlp'):
        # print(f"[{model_name}] Hooking 'mlp'")
        handle = model.mlp.register_forward_hook(get_features_hook)

    # 3. 兜底 (比如早期的 TSRNet 版本用的是 attn1)
    elif hasattr(model, 'attn1'):
        handle = model.attn1.register_forward_hook(get_features_hook)

    else:
        print(f"Error: Could not find suitable layer in {model_name}")
        raise AttributeError("Failed to register hook.")

    return handle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/Processed_PTBXL/')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ckpt_baseline', type=str, default='best-checkpointsTSRNet-0860.pt')
    parser.add_argument('--ckpt_shapex', type=str,
                        default='ckpt_shape_guided_shapex_gate/best_TSRNet_SHAPEXGate-epoch23-auc0.8871.pt')

    # 模型参数 (保持你的设置)
    parser.add_argument('--dims', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=96)
    parser.add_argument('--shapex_num_shapelets', type=int, default=16)
    parser.add_argument('--shapex_shapelet_len', type=int, default=96)
    parser.add_argument('--shapex_no_encoder', action='store_true')
    parser.add_argument('--attention_heads', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--num_transformer_blocks', type=int, default=2)

    return parser.parse_args()


def extract_features(model, loader, device, limit=1000):
    global features_storage
    features_storage = []

    # 1. 挂载窃听器 (Hook)
    handle = register_smart_hook(model)

    model.eval()

    count = 0
    # 使用 torch.serialization.safe_globals 或者简单地忽略 weights_only 警告
    # 在最外层我们已经 filter 了警告

    with torch.no_grad():
        for i, (time_ecg, spectrogram_ecg, _) in tqdm(enumerate(loader), total=min(len(loader), limit),
                                                      desc=f"Extracting {model.__class__.__name__}"):
            if count >= limit:
                break

            time_ecg = time_ecg.float().to(device)
            spectrogram_ecg = spectrogram_ecg.float().to(device)

            # 2. 前向传播 (Hook 会自动把特征存进 features_storage)
            if hasattr(model, 'fusion_gate_net'):
                _ = model(time_ecg, spectrogram_ecg, return_aux=False)
            else:
                _ = model(time_ecg, spectrogram_ecg)

            count += 1

    # 3. 拆除窃听器
    handle.remove()

    if len(features_storage) == 0:
        raise ValueError("No features captured!")

    # 4. 整理数据
    all_feats = torch.cat(features_storage, dim=0).numpy()
    if all_feats.ndim > 2:
        all_feats = all_feats.reshape(all_feats.shape[0], -1)

    return all_feats


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading Data...")
    dataset = TestSet(folder=args.data_path)
    # batch_size=1 保证最稳妥的提取
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    labels = np.load(os.path.join(args.data_path, 'label.npy'))

    SAMPLE_LIMIT = 1000
    subset_labels = labels[:SAMPLE_LIMIT]

    # --- Baseline ---
    print("\nProcessing Baseline...")
    model_base = TSRNet(enc_in=args.dims).to(device)
    # [修改] 显式设置 weights_only=False (虽然警告已屏蔽，但这是为了代码合规性，如果 torch版本支持)
    # 但为了兼容旧版本，我们直接 load，依靠开头的 warnings.filter 屏蔽提示
    ckpt_b = torch.load(args.ckpt_baseline, map_location=device)
    state_dict_b = ckpt_b['model_state_dict'] if isinstance(ckpt_b, dict) and 'model_state_dict' in ckpt_b else ckpt_b
    model_base.load_state_dict(state_dict_b)

    feats_base = extract_features(model_base, loader, device, limit=SAMPLE_LIMIT)

    # --- SHAPEX ---
    print("\nProcessing SHAPEX...")
    model_shapex = TSRNet_ShapeGuided_ResidualFusion_SHAPEX(
        enc_in=args.dims, channel=args.dims,
        d_model=args.d_model, shapex_num_shapelets=args.shapex_num_shapelets,
        shapex_shapelet_len=args.shapex_shapelet_len, shapex_use_encoder=not args.shapex_no_encoder
    ).to(device)
    ckpt_s = torch.load(args.ckpt_shapex, map_location=device, weights_only=False)
    state_dict_s = ckpt_s['model_state_dict'] if isinstance(ckpt_s, dict) and 'model_state_dict' in ckpt_s else ckpt_s
    model_shapex.load_state_dict(state_dict_s)

    feats_shapex = extract_features(model_shapex, loader, device, limit=SAMPLE_LIMIT)

    # --- t-SNE ---
    print(f"\nComputing t-SNE (Samples: {SAMPLE_LIMIT})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')

    tsne_base = tsne.fit_transform(feats_base)
    tsne_shapex = tsne.fit_transform(feats_shapex)

    # --- Plotting ---
    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = ['#2ecc71', '#e74c3c']  # Green (Normal), Red (Anomaly)
    label_map = {0: 'Normal', 1: 'Anomaly'}

    # Plot Baseline
    for lbl in [0, 1]:
        idx = (subset_labels == lbl)
        axes[0].scatter(tsne_base[idx, 0], tsne_base[idx, 1], c=colors[lbl], label=label_map[lbl],
                        alpha=0.6, s=20, edgecolors='none')
    axes[0].set_title('Baseline (TSRNet)\nLatent Space', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot SHAPEX
    for lbl in [0, 1]:
        idx = (subset_labels == lbl)
        axes[1].scatter(tsne_shapex[idx, 0], tsne_shapex[idx, 1], c=colors[lbl], label=label_map[lbl],
                        alpha=0.6, s=20, edgecolors='none')
    axes[1].set_title('Ours (SGRF-Net)\nLatent Space', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.suptitle("t-SNE Visualization: SGRF-Net vs Baseline\n(Feature Separation Capability)", fontsize=16)
    plt.tight_layout()

    save_path = 'tsne_comparison.jpg'
    plt.savefig(save_path, dpi=300)
    print(f"Done! Saved t-SNE plot to {save_path}")


if __name__ == '__main__':
    main()