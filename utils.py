import torch
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

def get_projected_features(proj_model, features, batch_size=256, device="cuda"):
    proj_model.eval()
    projected_list = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.tensor(features[i:i+batch_size]).float().to(device)
            projected = proj_model(batch)
            projected_list.append(projected.cpu())
    return torch.cat(projected_list, dim=0).numpy()

def create_index_splits(N, nc, n1u, n2u):
    """
    按照固定顺序划分索引：
    - idx_complete: [0, nc)
    - idx_imgonly: [nc, nc+n1u)
    - idx_textonly: [nc+n1u, nc+n1u+n2u)
    其余样本 (如果有) 视为未使用/保留。
    """
    assert nc + n1u + n2u <= N, "nc + n1u + n2u must be <= total samples"

    idx_complete = np.arange(0, nc)
    idx_imgonly = np.arange(nc, nc + n1u)
    idx_textonly = np.arange(nc + n1u, nc + n1u + n2u)

    return idx_complete, idx_imgonly, idx_textonly


def make_projection_dataloaders(I_tr, T_tr, L_tr, idx_complete, idx_imgonly, idx_textonly, batch_size=256):
    """
    返回三个 dataloader：
    - complete_loader: only complete pairs (for cross InfoNCE)
    - img_avail_loader: samples where image exists (complete + image-only) -> for intra-image loss
    - txt_avail_loader: samples where text exists (complete + text-only) -> for intra-text loss
    注意：数据返回 (raw_image_feat, raw_text_feat, label, mask_image, mask_text)
    """
    # Complete pairs
    comp_images = I_tr[idx_complete]
    comp_texts = T_tr[idx_complete]
    comp_labels = L_tr[idx_complete]
    comp_ds = TensorDataset(torch.from_numpy(comp_images).float(), torch.from_numpy(comp_texts).float(), torch.from_numpy(comp_labels).float())
    complete_loader = DataLoader(comp_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Image-available: complete + image-only
    img_idxs = np.concatenate([idx_complete, idx_imgonly])
    img_images = I_tr[img_idxs]
    img_texts = T_tr[img_idxs]
    img_labels = L_tr[img_idxs]
    img_ds = TensorDataset(torch.from_numpy(img_images).float(), torch.from_numpy(img_texts).float(), torch.from_numpy(img_labels).float())
    img_avail_loader = DataLoader(img_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Text-available: complete + text-only
    txt_idxs = np.concatenate([idx_complete, idx_textonly])
    txt_images = I_tr[txt_idxs]
    txt_texts = T_tr[txt_idxs]
    txt_labels = L_tr[txt_idxs]
    txt_ds = TensorDataset(torch.from_numpy(txt_images).float(), torch.from_numpy(txt_texts).float(), torch.from_numpy(txt_labels).float())
    txt_avail_loader = DataLoader(txt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return complete_loader, img_avail_loader, txt_avail_loader, img_idxs, txt_idxs

# ======= 训练投影网络（统一优化 cross + intra） ==========
def train_projection_networks(image_proj, text_proj,
                              complete_loader, img_avail_loader, txt_avail_loader,
                              contrastive_loss_fn, structure_reg_fn,
                              optimizer, device,
                              epochs=50, lambda_intra=1.0, tau=0.07):
    """
    每一步从三个 loader 中采样一个 batch（通过 iterators 循环），计算 cross-modal InfoNCE（只用 complete batch）
    以及两个模态的 intra-structure loss（用 img_avail 和 txt_avail），然后总损失一起优化。
    """
    image_proj.train(); text_proj.train()
    for epoch in range(epochs):
        total_loss = 0.0
        # iterators
        it_comp = iter(complete_loader)
        it_img = iter(img_avail_loader)
        it_txt = iter(txt_avail_loader)

        steps = min(len(complete_loader), len(img_avail_loader), len(txt_avail_loader))
        for _ in range(steps):
            # sample batches
            img_comp, txt_comp, lbl_comp = next(it_comp)
            img_img, txt_img, lbl_img = next(it_img)
            img_txt, txt_txt, lbl_txt = next(it_txt)

            img_comp = img_comp.to(device); txt_comp = txt_comp.to(device)
            img_img = img_img.to(device); txt_img = txt_img.to(device)
            img_txt = img_txt.to(device); txt_txt = txt_txt.to(device)

            # forward projections
            z_img_comp = image_proj(img_comp)      # (B, d)
            z_txt_comp = text_proj(txt_comp)
            z_img_img = image_proj(img_img)
            z_txt_txt = text_proj(txt_txt)

            # losses
            loss_cross = contrastive_loss_fn(z_img_comp, z_txt_comp)          # InfoNCE on complete pairs
            loss_intra_img = structure_reg_fn(img_img, z_img_img, tau=tau, L=3)
            loss_intra_txt = structure_reg_fn(txt_txt, z_txt_txt, tau=tau, L=3)

            loss = loss_cross + lambda_intra * (loss_intra_img + loss_intra_txt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / steps
        print(f"[Proj] Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f}")
    # 返回训练后的投影器
    return image_proj, text_proj

# ======= 在投影空间训练 predictor (img->txt, txt->img) ==========
def train_predictors_on_projections(image_proj, text_proj, I_tr, T_tr, idx_complete, device,
                                    hidden_dim=512, epochs=100, batch_size=256, lr=1e-3):
    """
    使用 complete 对在 projection 空间训练两个 MLP predictor（输入和输出维均为 proj dim）。
    返回训练好的 img2txt_mlp, txt2img_mlp。
    """
    image_proj.eval(); text_proj.eval()
    # 计算完整对的投影表示（一次性）
    with torch.no_grad():
        imgs = torch.from_numpy(I_tr[idx_complete]).float().to(device)
        txts = torch.from_numpy(T_tr[idx_complete]).float().to(device)
        z_imgs = image_proj(imgs).detach().cpu()
        z_txts = text_proj(txts).detach().cpu()

    ds = TensorDataset(z_imgs, z_txts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    proj_dim = z_imgs.shape[1]
    img2txt = nn.Sequential(
        nn.Linear(proj_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim)
    ).to(device)
    txt2img = nn.Sequential(
        nn.Linear(proj_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim)
    ).to(device)

    opt_i2t = optim.Adam(img2txt.parameters(), lr=lr, weight_decay=1e-5)
    opt_t2i = optim.Adam(txt2img.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for z_i, z_t in loader:
            z_i = z_i.to(device); z_t = z_t.to(device)

            # img -> txt
            hat_t = img2txt(z_i)
            loss_i2t = criterion(hat_t, z_t)
            # txt -> img
            hat_i = txt2img(z_t)
            loss_t2i = criterion(hat_i, z_i)

            loss = loss_i2t + loss_t2i
            opt_i2t.zero_grad(); opt_t2i.zero_grad()
            loss.backward()
            opt_i2t.step(); opt_t2i.step()

            total_loss += loss.item()
        print(f"[Pred] Epoch {epoch+1}/{epochs} avg_loss={total_loss/len(loader):.6f}")
    return img2txt, txt2img

# ======= 用训练好的 predictor 在 projection 空间重建缺失模态 ==========
def build_reconstructed_projection_dataset(image_proj, text_proj, img2txt, txt2img,
                                           I_tr, T_tr, idx_complete, idx_imgonly, idx_textonly, device):
    """
    将投影过的特征（完整的 projection）与 predictor 生成的伪投影拼接成最终训练集（投影空间）。
    返回 numpy arrays: images_proj_all, texts_proj_all, labels_all, M1, M2 (masks)
    """
    # 1) 全体样本先投影（一次性，避免重复）
    image_proj.eval(); text_proj.eval(); img2txt.eval(); txt2img.eval()
    with torch.no_grad():
        images_tensor = torch.from_numpy(I_tr).float().to(device)
        texts_tensor = torch.from_numpy(T_tr).float().to(device)
        all_z_img = image_proj(images_tensor).cpu().numpy()   # shape (N, d)
        all_z_txt = text_proj(texts_tensor).cpu().numpy()

    N = I_tr.shape[0]
    # create arrays copies to be possibly overwritten
    images_proj = all_z_img.copy()
    texts_proj = all_z_txt.copy()

    # For image-only samples (idx_imgonly): their text feature may be invalid/mean-filled in raw T_tr.
    # We replace text projection with predictor(img_proj)
    if len(idx_imgonly) > 0:
        with torch.no_grad():
            z_img_missing = torch.from_numpy(all_z_img[idx_imgonly]).float().to(device)
            pred_txt = img2txt(z_img_missing).cpu().numpy()
            texts_proj[idx_imgonly] = pred_txt

    # For text-only samples (idx_textonly): replace image projection using txt2img predictor
    if len(idx_textonly) > 0:
        with torch.no_grad():
            z_txt_missing = torch.from_numpy(all_z_txt[idx_textonly]).float().to(device)
            pred_img = txt2img(z_txt_missing).cpu().numpy()
            images_proj[idx_textonly] = pred_img

    # Masks: M1 (image present), M2 (text present)
    M1 = np.zeros((N,1), dtype=np.float32); M2 = np.zeros((N,1), dtype=np.float32)
    M1[idx_complete] = 1.0; M1[idx_imgonly] = 1.0
    M2[idx_complete] = 1.0; M2[idx_textonly] = 1.0

    return images_proj, texts_proj, M1, M2


# 设置随机种子，确保每次训练结果可复现
def set_random_seed(seed):
    # 设置 Python 的随机种子
    random.seed(seed)
    
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    
    # 设置 CUDA 的随机种子（如果使用 GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    
    # 保证每次初始化时生成的随机数是相同的（确保可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁止启用 cudnn 的自动优化

def calculate_hamming_distance(a, b):
    q = a.shape[-1]
    return 0.5 * (q - torch.mm(a, b.T))


def calculate_s(labels1, labels2):
    s = torch.mm(labels1, labels2.T)
    return s


def normalize(x):
    l2_norm = np.linalg.norm(x, axis=1)[:, None]
    l2_norm[np.where(l2_norm == 0)] = 1e-6
    x = x/l2_norm
    return x


def zero_mean(x, mean_val=None):
    if mean_val is None:
        mean_val = np.mean(x, axis=0)
    x -= mean_val
    return x, mean_val

