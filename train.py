
import os
import random
import argparse
import time
import math
import copy
import warnings

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import time_string, convert_secs2time, AverageMeter
from dataloader import TrainSet, TestSet

# NEW model
from lib.SGRFNet import TSRNet_ShapeGuided_ResidualFusion_SHAPEX

gate_values_storage = []
model = None


def get_gate_activation_hook(module, input, output):
    # output is after sigmoid if you hook last layer; prefer hooking the gate tensor directly in training if needed.
    if model is not None and model.training:
        # output shape: (B, 2D)
        gate_values_storage.append(output.detach().cpu().mean(dim=0))


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def total_variation_loss(x):
    diff = torch.abs(x[:, 1:, :] - x[:, :-1, :])
    return torch.mean(diff)


def main(args):
    global model

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print(f"Using random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = torch.cuda.is_available()
    device = "cuda:" + args.gpu if use_cuda and args.gpu.isdigit() else 'cpu'
    print(f"Using device: {device}")

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}

    # Data
    dset = TrainSet(folder=args.data_path)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True, **kwargs)

    dtset = TestSet(folder=args.data_path)
    test_loader = torch.utils.data.DataLoader(dtset, batch_size=1, shuffle=False, **kwargs)
    labels = np.load(os.path.join(args.data_path, 'label.npy'))

    # Fixed dims for PTB-XL (keep same as your current script)
    time_seq_len_data = 4800
    spec_h_dim_data = 63
    spec_w_dim_data = 78

    print("Building Shapelet Energy Guided Fusion...")
    model = TSRNet_ShapeGuided_ResidualFusion_SHAPEX(
        enc_in=args.dims, channel=args.dims,
        h=args.attention_heads, d_model=args.d_model, dropout=args.dropout,
        time_seq_len=time_seq_len_data, spec_h_dim=spec_h_dim_data, spec_w_dim=spec_w_dim_data,
        d_ff=args.d_ff, num_transformer_blocks=args.num_transformer_blocks,
        shapex_num_shapelets=args.shapex_num_shapelets,
        shapex_shapelet_len=args.shapex_shapelet_len,
        shapex_use_encoder=not args.shapex_no_encoder,
        shapex_delta=args.shapex_delta,
        gate_mlp_dropout=args.gate_mlp_dropout
    ).to(device)

    # Gate diagnostic hook (hook on sigmoid output)
    # In our gating, self.mlp ends with Sigmoid -> output is gates
    model.fusion_gate_net.mlp[-1].register_forward_hook(get_gate_activation_hook)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.pth_path is not None:
        print(f"Loading checkpoint from {args.pth_path}")
        checkpoint = torch.load(args.pth_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded.")

    os.makedirs(args.save_path, exist_ok=True)

    start_time = time.time()
    epoch_time = AverageMeter()
    best_auc_result = 0.0
    patience_counter = 0

    for epoch in range(0, args.epochs + 1):
        adjust_learning_rate(optimizer, args.lr, epoch, args)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = f'[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]'
        print(f' {epoch:3d}/{args.epochs:3d} ----- [{time_string()}] {need_time}')

        train(args, model, epoch, train_loader, optimizer, device)
        auc_result = test(args, model, test_loader, labels, device)

        if auc_result > best_auc_result:
            best_auc_result = auc_result
            patience_counter = 0
            if args.save_model == 1:
                model_save_name = os.path.join(args.save_path, f'SHAPEXGate-epoch{epoch}-auc{best_auc_result:.4f}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'auc': best_auc_result
                }, model_save_name)
                print(f"Saved: {model_save_name} (New Best)")
        else:
            patience_counter += 1
            print(f"No improve. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping.")
            break

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    print(f"Final Best AUC: {best_auc_result:.4f}")


def train(args, model, epoch, train_loader, optimizer, device):
    global gate_values_storage
    gate_values_storage = []

    model.train()
    total_losses = AverageMeter()

    fixed_cos_weight = 0.15  # keep your current best setting

    for _, (time_ecg, spectrogram_ecg) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        time_ecg = time_ecg.float().to(device)
        clean_target = time_ecg[:, :, 0:model.channel].clone()

        # optional noise
        if args.noise_level > 0:
            noise = torch.randn_like(time_ecg) * args.noise_level
            noisy_time_ecg = time_ecg + noise
        else:
            noisy_time_ecg = time_ecg

        bs, time_length, _ = time_ecg.shape

        # --- Masking Time ---
        mask_time = copy.deepcopy(noisy_time_ecg)
        mask = torch.zeros((bs, time_length, 1), dtype=torch.bool).to(device)
        patch_length = time_length // 100
        for j in random.sample(range(0, 100), args.mask_ratio_time):
            mask[:, j * patch_length:(j + 1) * patch_length] = 1
        mask_time = torch.mul(mask_time, ~mask)

        # --- Masking Spec ---
        spec_ecg = spectrogram_ecg.float().to(device)
        bs2, freq_dim, time_dim_spec, _ = spec_ecg.shape
        assert bs2 == bs
        mask_spec = copy.deepcopy(spec_ecg)
        mask_s = torch.zeros((bs, freq_dim, time_dim_spec, 1), dtype=torch.bool).to(device)
        for j in random.sample(range(0, time_dim_spec), args.mask_ratio_spec):
            mask_s[:, :, j:(j + 1), :] = 1
        mask_spec = torch.mul(mask_spec, ~mask_s)

        # --- Forward (NEW: returns aux losses) ---
        reconstructed_signal, sigma, aux = model(mask_time, mask_spec, return_aux=True)
        sigma = torch.clamp(sigma, min=-5.0, max=5.0)

        # 1) MSE + uncertainty
        recon_err_squared = torch.square(reconstructed_signal - clean_target)
        sigma_broadcast = sigma.expand_as(recon_err_squared)
        mse_uncertainty_loss = torch.mean(torch.exp(-sigma_broadcast) * recon_err_squared + sigma_broadcast)

        # 2) Shape cosine loss (your current)
        flat_recon = reconstructed_signal.reshape(bs, -1)
        flat_clean = clean_target.reshape(bs, -1)
        cosine_loss = 1 - torch.nn.functional.cosine_similarity(flat_recon, flat_clean, dim=1).mean()

        # 3) TV
        tv_reg = total_variation_loss(reconstructed_signal)

        # 4) SHAPEX-inspired shapelet losses (match + div)
        l_match = aux["l_match_time"] + aux["l_match_spec"]
        l_div = aux["l_div_time"] + aux["l_div_spec"]

        loss = (
            mse_uncertainty_loss
            + fixed_cos_weight * cosine_loss
            + args.lambda_tv * tv_reg
            + args.lambda_match * l_match
            + args.lambda_div * l_div
        )

        loss.backward()
        optimizer.step()
        total_losses.update(loss.item(), bs)

    # Diagnostics every 5 epochs (optional)
    if len(gate_values_storage) > 0 and epoch % 5 == 0:
        all_gates = torch.stack(gate_values_storage)  # (steps, 2D)
        d_model = all_gates.shape[1] // 2
        avg_time_gate = all_gates[:, :d_model].mean().item()
        avg_spec_gate = all_gates[:, d_model:].mean().item()
        print(f"[Diagnostic] T_Gate: {avg_time_gate:.3f} | S_Gate: {avg_spec_gate:.3f} (Shapelet+Energy)")

    print(f"Train Epoch: {epoch} Total_Loss: {total_losses.avg:.6f}")


def test(args, model, test_loader, labels, device):
    model.eval()
    result = []

    # keep reproducibility as your current code does
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    np_rng_state = np.random.get_state()
    py_rng_state = random.getstate()

    eval_seed = 12345
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)

    with torch.no_grad():
        for _, (time_ecg, spectrogram_ecg, r_index) in enumerate(test_loader):
            instance_result = []
            time_ecg_on_device = time_ecg.float().to(device)
            spectrogram_ecg_on_device = spectrogram_ecg.float().to(device)
            time_length = time_ecg_on_device.shape[1]

            # optional masked scoring region
            batch_mask_loss = None
            if args.mask_loss:
                idx_length = r_index.shape[1]
                dims = time_ecg_on_device.shape[2]
                batch_mask_loss = torch.zeros((time_length, dims), dtype=torch.bool).to(device)
                for r_idx in range(idx_length):
                    r_index_value = r_index[0][r_idx]
                    if 200 < r_index_value < 4800 - 400:
                        left = max(0, r_index_value - 240)
                        batch_mask_loss[left:r_index_value + 240, :] = 1

            num_strategies = max(1, 100 // args.mask_ratio_time)

            for j_strategy in range(num_strategies):
                patch_interval_time = 4800 // args.mask_ratio_time
                time_ecg_batch = copy.deepcopy(time_ecg_on_device)
                current_time_mask = torch.zeros((1, time_length, 1), dtype=torch.bool).to(device)
                for k_patch_idx in range(args.mask_ratio_time):
                    cut_idx = 48 * j_strategy + patch_interval_time * k_patch_idx
                    cut_idx = min(cut_idx, time_length - 48)
                    current_time_mask[:, cut_idx:cut_idx + 48] = 1
                mask_time = torch.mul(time_ecg_batch, ~current_time_mask)

                bs_test, freq_dim_test, time_dim_spec_test, _ = spectrogram_ecg_on_device.shape
                spec_ecg_batch = copy.deepcopy(spectrogram_ecg_on_device)
                current_spec_mask = torch.zeros((bs_test, freq_dim_test, time_dim_spec_test, 1), dtype=torch.bool).to(device)
                for k_patch_idx_spec in range(args.mask_ratio_spec):
                    cut_idx_spec = 1 * j_strategy + (time_dim_spec_test // args.mask_ratio_spec) * k_patch_idx_spec
                    cut_idx_spec = min(cut_idx_spec, time_dim_spec_test - 1) if time_dim_spec_test > 0 else 0
                    current_spec_mask[:, :, cut_idx_spec:cut_idx_spec + 1] = 1
                mask_spec = torch.mul(spec_ecg_batch, ~current_spec_mask)

                # Forward (ignore aux during test scoring)
                reconstructed_signal, sigma = model(mask_time, mask_spec, return_aux=False)
                sigma = torch.clamp(sigma, min=-5.0, max=5.0)

                recon_err_squared = torch.square(reconstructed_signal - time_ecg_on_device[:, :, 0:model.channel])
                sigma_broadcast = sigma.expand_as(recon_err_squared)

                mse_score = torch.exp(-sigma_broadcast) * recon_err_squared + sigma_broadcast

                cosine_sim = torch.nn.functional.cosine_similarity(
                    reconstructed_signal,
                    time_ecg_on_device[:, :, 0:model.channel],
                    dim=1
                ).unsqueeze(1)

                shape_error = 1 - cosine_sim
                shape_error = shape_error.expand(1, time_length, model.channel)

                combined_score_map = mse_score + (0.15 * shape_error)

                if args.mask_loss and batch_mask_loss is not None:
                    masked_anomaly_score = torch.mul(combined_score_map, batch_mask_loss)
                    loss_val = torch.sum(masked_anomaly_score) / (torch.sum(batch_mask_loss) + 1e-6)
                else:
                    loss_val = torch.mean(combined_score_map)

                instance_result.append(loss_val.detach().cpu().numpy())

            result.append(np.mean(instance_result))

    # restore RNG
    torch.set_rng_state(rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(py_rng_state)

    scores = np.asarray(result)
    test_labels = np.array(labels).astype(int)

    if len(scores) == 0:
        return 0.5

    n_mean = scores[test_labels == 0].mean() if len(scores[test_labels == 0]) > 0 else 0
    a_mean = scores[test_labels == 1].mean() if len(scores[test_labels == 1]) > 0 else 0
    gap = a_mean - n_mean
    print(f"Stats: Gap={gap:.5f} (N:{n_mean:.3f} A:{a_mean:.3f})")

    if (scores.max() - scores.min()) > 1e-6:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        return 0.5

    auc_result = roc_auc_score(test_labels, scores)
    print(f"Test AUC: {auc_result:.4f}")
    return auc_result


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='ECG Anomaly Detection (SHAPEX-guided Fusion)')

    parser.add_argument('--data_path', type=str, default='data/Processed_PTBXL/')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--dims', type=int, default=12)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='ckpt_shape_guided_shapex_gate/')

    # masking (keep your SOTA defaults)
    parser.add_argument('--mask_ratio_time', type=int, default=30)
    parser.add_argument('--mask_ratio_spec', type=int, default=20)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=668)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--pth_path", type=str, default=None)
    parser.add_argument("--mask_loss", action='store_true')
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--d_model', type=int, default=96)
    parser.add_argument('--attention_heads', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--num_transformer_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--noise_level', type=float, default=0.2)
    # parser.add_argument('--noise_level', type=float, default=0.0)# AUC: 0.8771 0.8820
    parser.add_argument('--lambda_tv', type=float, default=0)

    # ---- SHAPEX-inspired loss weights ----
    parser.add_argument('--lambda_match', type=float, default=0.05)
    parser.add_argument('--lambda_div', type=float, default=0.01)
    parser.add_argument('--shapex_delta', type=float, default=0.5)

    # ---- Shapelet settings (important!) ----
    parser.add_argument('--shapex_num_shapelets', type=int, default=16)
    parser.add_argument('--shapex_shapelet_len', type=int, default=96)
    parser.add_argument('--shapex_no_encoder', action='store_true')
    parser.add_argument('--gate_mlp_dropout', type=float, default=0.0)

    args = parser.parse_args()
    main(args)
