"""
Mã tham khảo: kiến trúc ResNet kiểu CIFAR theo kuangliu/pytorch-cifar (BasicBlock, stem conv3x3,
không maxpool); dropout trong block theo Wide-ResNet (Zagoruyko & Komodakis, 2016).
"""
import os

# Phải đặt TRƯỚC khi CUDA khởi tạo (cần cho use_deterministic_algorithms với cuBLAS).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import math
import random
import shutil
import subprocess
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


# 1. Hằng số và config
SPLIT_SEED = 2025            # chỉ dùng để chia train/val dataset
NUM_WORKERS = 2              
EVAL_BATCH = 1000
ES_PATIENCE = 5
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
OUT_DIR, OWNER, COMMIT, DATA_DIR = "outputs", "unknown", "unknown", "data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"       

# Config mặc định
DEFAULTS = dict(
    width=32, norm="bn", dropout=0.0,
    optimizer="sgdm", lr=0.1, weight_decay=5e-4,
    batch_size=128, epochs=20,
    schedule="cosine", warmup_epochs=0,
    aug_crop=True, aug_flip=True, aug_jitter=False,
    eval_train=False,
)

# Các hằng số khác (cố định cho mọi thí nghiệm):
#   momentum 0.9; GN 8 nhóm; warm-up bắt đầu từ 0.01 x lr; step decay x0.1 tại 50% và 75% tổng bước;
#   color jitter 0.4/0.4/0.4 (không hue); crop padding 4; flip p = 0.5.

RUNS_COLUMNS = ["config", "seed", "status", "start_time", "duration_min", "owner", "gpu_name", "commit",
                "config_json", "final_test_acc", "final_val_acc", "final_train_loss", "gen_gap",
                "es_stop_epoch", "test_acc_at_es", "notes"]
EPOCH_COLUMNS = ["config", "seed", "epoch", "lr", "train_loss", "train_acc", "train_eval_loss",
                 "train_eval_acc", "val_loss", "val_acc", "test_loss", "test_acc", "epoch_time_s"]
RESULT_COLUMNS = ["final_test_acc", "final_val_acc", "final_train_loss", "gen_gap",
                  "es_stop_epoch", "test_acc_at_es"]


def make_cfg(overrides: dict) -> dict:
    """Overidde config for training"""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Khóa cấu hình không hợp lệ: {sorted(unknown)}")
    return {**DEFAULTS, **overrides}


# 2. Các hàm khởi tạo
def init(owner: str, out_dir: str = "outputs", data_dir: str = "data") -> None:
    """Hàm khởi tạo để lưu kết quả chạy thí nghiệm, thu thập tên người chạy thí nghiệm 
    (owner, folder kết quả, folder data và commit hash của code sử dụng)"""
    global OWNER, OUT_DIR, DATA_DIR, COMMIT
    OWNER, OUT_DIR, DATA_DIR = owner, out_dir, data_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        COMMIT = "unknown"
    print(f"device={DEVICE} gpu={gpu_name()} torch={torch.__version__} commit={COMMIT} owner={OWNER}")


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # seed cả GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)



# 3. Các hàm data loader và chia dữ liệu
def split_indices(labels):
    """Chia phân tầng, tất định: trả (train_idx, val_idx, train_eval_idx) = 45k / 5k / 5k chỉ số."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(SPLIT_SEED)
    train, val, train_eval = [], [], []
    for c in range(10):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        val += list(idx_c[:500])
        train_eval += list(idx_c[500:1000])      # 5k ảnh train cố định để đo train-eval
        train += list(idx_c[500:])               # (gồm cả train_eval)
    return np.sort(train), np.sort(val), np.sort(train_eval)


def make_loaders(cfg: dict, seed: int, quick: bool = False) -> dict:
    """Tạo DataLoader train / val / test (và train_eval nếu cfg['eval_train'])."""
    eval_tf = [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    train_tf = []
    if cfg["aug_crop"]:
        train_tf.append(T.RandomCrop(32, padding=4))
    if cfg["aug_flip"]:
        train_tf.append(T.RandomHorizontalFlip())
    if cfg["aug_jitter"]:
        train_tf.append(T.ColorJitter(0.4, 0.4, 0.4))      # áp trước normalize
    train_tf += eval_tf

    kw = dict(root=DATA_DIR, download=True)
    train_aug = torchvision.datasets.CIFAR10(train=True, transform=T.Compose(train_tf), **kw)
    train_plain = torchvision.datasets.CIFAR10(train=True, transform=T.Compose(eval_tf), **kw)
    test = torchvision.datasets.CIFAR10(train=False, transform=T.Compose(eval_tf), **kw)

    train_idx, val_idx, te_idx = split_indices(train_aug.targets)
    sets = {"train": Subset(train_aug, train_idx), "val": Subset(train_plain, val_idx), "test": test}
    if cfg["eval_train"]:
        sets["train_eval"] = Subset(train_plain, te_idx)
    if quick:
        n = {"train": 2000, "val": 500, "test": 500, "train_eval": 500}
        sets = {k: Subset(v, list(range(n[k]))) for k, v in sets.items()}
        # train_aug/train_plain đã được shuffle theo lớp nên 2000 chỉ số đầu của Subset vẫn đủ đa dạng
    workers = 0 if quick else NUM_WORKERS
    loaders = {}
    for k, ds in sets.items():
        if k == "train":
            loaders[k] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                                    num_workers=workers, generator=torch.Generator().manual_seed(seed),
                                    persistent_workers=workers > 0)
        else:
            loaders[k] = DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=workers)
    return loaders



# 4. Các hàm implement mô hình Resnet18
def make_norm(kind: str, channels: int) -> nn.Module:
    """bn: BatchNorm2d. 
       ln: GroupNorm(1, C) = LayerNorm cho CNN (thống kê trên (C,H,W) từng ảnh), giữ
       gamma/beta theo kênh như BN/GN (nn.LayerNorm([C,H,W]) sẽ có gamma/beta theo từng vị trí không gian).
       gn: GroupNorm(8, C) (8 nhóm vì stage đầu chỉ có 32 kênh)."""
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "ln":
        return nn.GroupNorm(1, channels)
    if kind == "gn":
        return nn.GroupNorm(8, channels)
    raise ValueError(f"norm không hợp lệ: {kind}")


class BasicBlock(nn.Module):
    """conv1 -> norm1 -> ReLU -> Dropout(p) -> conv2 -> norm2, cộng shortcut rồi ReLU."""

    def __init__(self, in_ch, out_ch, stride, norm, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm1 = make_norm(norm, out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = make_norm(norm, out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                          make_norm(norm, out_ch))

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(self.drop(out)))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 kiểu CIFAR: stem conv3x3 (không maxpool), 4 stage x 2 BasicBlock, kênh w,2w,4w,8w."""

    def __init__(self, width=32, norm="bn", dropout=0.0, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.norm1 = make_norm(norm, width)
        chans, strides, in_ch = [width, 2 * width, 4 * width, 8 * width], [1, 2, 2, 2], width
        for i, (ch, st) in enumerate(zip(chans, strides), start=1):
            blocks = [BasicBlock(in_ch, ch, st, norm, dropout), BasicBlock(ch, ch, 1, norm, dropout)]
            setattr(self, f"layer{i}", nn.Sequential(*blocks))
            in_ch = ch
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


# 5. Optimizer và learning rate schedules.
def make_optimizer(model, cfg):
    """Set up optimizer và learning rate từ config."""
    kw = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    name, p = cfg["optimizer"], model.parameters()
    if name == "sgd":
        return torch.optim.SGD(p, momentum=0.0, **kw)
    if name == "sgdm":
        return torch.optim.SGD(p, momentum=0.9, **kw)
    if name == "nesterov":
        return torch.optim.SGD(p, momentum=0.9, nesterov=True, **kw)
    if name == "rmsprop":
        return torch.optim.RMSprop(p, alpha=0.99, eps=1e-8, **kw)
    if name == "adam":
        return torch.optim.Adam(p, betas=(0.9, 0.999), eps=1e-8, **kw)
    if name == "adamw":
        return torch.optim.AdamW(p, betas=(0.9, 0.999), eps=1e-8, **kw)
    raise ValueError(f"optimizer không hợp lệ: {name}")


def lr_at(step: int, cfg: dict, steps_per_epoch: int) -> float:
    """LR tại bước `step` ∈ [0, total)"""
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg["warmup_epochs"] * steps_per_epoch
    if step < warm:
        f = 0.01 + 0.99 * step / warm
    elif cfg["schedule"] == "const":
        f = 1.0
    elif cfg["schedule"] == "step":
        f = 0.1 ** sum(step >= m * total for m in (0.5, 0.75))    # mốc tính trên TỔNG ngân sách
    elif cfg["schedule"] == "cosine":
        t = (step - warm) / (total - warm)
        f = 0.5 * (1 + math.cos(math.pi * t))
    else:
        raise ValueError(f"schedule không hợp lệ: {cfg['schedule']}")
    return cfg["lr"] * f


# 6. Training model
def train_epoch(model, loader, optimizer, scaler, cfg, step0, spe):
    """Train model trong một epoch. Trả (step_losses [spe], step_lrs [spe], train_acc %)."""
    model.train()
    loss_buf, lrs, correct, seen = [], [], torch.zeros((), device=DEVICE), 0
    for k, (x, y) in enumerate(loader):
        lr = lr_at(step0 + k, cfg, spe)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_buf.append(loss.detach())
        correct += (logits.argmax(1) == y).sum()          # tensor GPU, không .item()
        lrs.append(lr)
        seen += y.numel()
    losses = torch.stack(loss_buf).cpu().numpy()          # một lần đồng bộ duy nhất
    return losses, np.array(lrs), 100.0 * correct.item() / seen


@torch.no_grad()
def evaluate(model, loader):
    """Trả (loss trung bình mỗi ảnh, acc %)."""
    model.eval()
    total_loss, correct, n = torch.zeros((), device=DEVICE), torch.zeros((), device=DEVICE), 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            logits = model(x)
        total_loss += F.cross_entropy(logits.float(), y, reduction="sum")
        correct += (logits.argmax(1) == y).sum()
        n += y.numel()
    return total_loss.item() / n, 100.0 * correct.item() / n



# 7. Chỉ số
def early_stopping(val_losses, patience: int = ES_PATIENCE):
    """Early stopping hậu kiểm theo val loss. Trả (stop_epoch, best_epoch), đánh số từ 1."""
    best, best_epoch, wait = float("inf"), 1, 0
    for e, v in enumerate(val_losses, start=1):
        if v < best:
            best, best_epoch, wait = v, e, 0
        else:
            wait += 1
        if wait == patience:
            return e, best_epoch
    return len(val_losses), best_epoch


def summarize(history: list) -> dict:
    """6 cột kết quả của một lượt completed từ các dòng epoch."""
    last = history[-1]
    stop, best = early_stopping([h["val_loss"] for h in history])
    return dict(final_test_acc=last["test_acc"], final_val_acc=last["val_acc"],
                final_train_loss=last["train_loss"],
                gen_gap=last["train_eval_acc"] - last["test_acc"],     # NaN khi không đo train-eval
                es_stop_epoch=stop, test_acc_at_es=history[best - 1]["test_acc"])



# 8. Hàm run() để chạy toàn bộ pipeline training

def _append_csv(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_checkpoint(model, name: str, seed: int, tag: str, cfg: dict, epoch: int, h: dict) -> str:
    """Lưu state_dict + metadata vào OUT_DIR/checkpoints/<name>_s<seed>_<tag>.pt (tag: 'last' | 'best')."""
    d = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}_s{seed}_{tag}.pt")
    torch.save(dict(model_state=model.state_dict(), config=name, seed=seed, cfg=cfg, epoch=epoch,
                    val_loss=h["val_loss"], val_acc=h["val_acc"], test_acc=h["test_acc"]), path)
    return path


def run(name: str, seed: int, quick: bool = False, log_steps: bool = False, **overrides):
    """Chạy một lượt (config, seed). Trả (row, model); model là None nếu lượt không completed.
    Khi không quick, lưu checkpoint: bản val_loss thấp nhất (_best.pt) và bản epoch cuối (_last.pt)."""
    cfg = make_cfg(overrides)            # ngoài try: lỗi cấu hình phải báo ngay, không thành 'crashed'
    if quick:
        cfg["epochs"] = 2
    start_time = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    status, notes, history, step_rows, model = "completed", "", [], [], None
    best_val = float("inf")
    results = {c: float("nan") for c in RESULT_COLUMNS}
    try:
        set_seed(seed)
        model = ResNet18(cfg["width"], cfg["norm"], cfg["dropout"]).to(DEVICE)   # ngay sau set_seed
        loaders = make_loaders(cfg, seed, quick)
        optimizer = make_optimizer(model, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP)
        spe = len(loaders["train"])
        for epoch in range(1, cfg["epochs"] + 1):
            te = time.time()
            step0 = (epoch - 1) * spe
            losses, lrs, train_acc = train_epoch(model, loaders["train"], optimizer, scaler, cfg, step0, spe)
            nan = float("nan")
            h = dict(config=name, seed=seed, epoch=epoch, lr=lrs[0], train_loss=float(losses.mean()),
                     train_acc=train_acc, train_eval_loss=nan, train_eval_acc=nan,
                     val_loss=nan, val_acc=nan, test_loss=nan, test_acc=nan, epoch_time_s=nan)
            if not np.isfinite(losses).all():                    # kiểm ở cuối epoch, trước khi eval
                h["epoch_time_s"] = time.time() - te
                history.append(h)
                status, notes = "diverged", f"nan loss at epoch {epoch}"
                print(f"[{name} s{seed}] DIVERGED at epoch {epoch}")
                break
            h["val_loss"], h["val_acc"] = evaluate(model, loaders["val"])
            h["test_loss"], h["test_acc"] = evaluate(model, loaders["test"])
            if cfg["eval_train"]:
                h["train_eval_loss"], h["train_eval_acc"] = evaluate(model, loaders["train_eval"])
            h["epoch_time_s"] = time.time() - te
            history.append(h)
            if not quick and h["val_loss"] < best_val:
                best_val = h["val_loss"]
                save_checkpoint(model, name, seed, "best", cfg, epoch, h)
            if log_steps:
                step_rows += [(name, seed, step0 + i, lrs[i], float(losses[i])) for i in range(spe)]
            print(f"[{name} s{seed}] ep {epoch:2d}/{cfg['epochs']} lr {h['lr']:.4g} "
                  f"train {h['train_loss']:.3f}/{train_acc:.1f}% val {h['val_loss']:.3f}/{h['val_acc']:.1f}% "
                  f"test {h['test_acc']:.2f}% ({h['epoch_time_s']:.0f}s)")
        if status == "completed":
            results = summarize(history)
            if not quick:
                save_checkpoint(model, name, seed, "last", cfg, cfg["epochs"], history[-1])
    except Exception:
        tb = traceback.format_exc()
        status, notes, model = "crashed", tb[-300:], None        # KHÔNG raise: chạy tiếp lượt sau
        print(tb)
    if status != "completed":
        model = None

    row = dict(config=name, seed=seed, status=status, start_time=start_time,
               duration_min=(time.time() - t0) / 60, owner=OWNER, gpu_name=gpu_name(), commit=COMMIT,
               config_json=json.dumps(cfg, sort_keys=True), **results, notes=notes)
    if not quick:
        _append_csv(pd.DataFrame([row], columns=RUNS_COLUMNS), "runs.csv")
        if history:
            _append_csv(pd.DataFrame(history, columns=EPOCH_COLUMNS), "epochs.csv")
        if step_rows:
            _append_csv(pd.DataFrame(step_rows, columns=["config", "seed", "step", "lr", "loss"]), "steps.csv")
    return row, model



# 9. Chẩn đoán D1 (chỉ notebook 04): nhiễu thống kê batch của BN theo kích thước batch
@torch.no_grad()
def bn_noise(model, batch_sizes=(2, 4, 8, 16, 32, 64, 128), n_batches=200, seed=0) -> pd.DataFrame:
    """Với mỗi B, đo sai lệch của mean/var theo kênh tính từ B ảnh so với thống kê trên cả 5k ảnh.
    Cột: layer, batch_size, mean_noise, var_noise. Chạy fp32, eval mode, trên đầu vào của lớp BN."""
    model = model.float().eval()
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=tf)
    _, _, te_idx = split_indices(ds.targets)
    images = torch.stack([ds[i][0] for i in te_idx]).to(DEVICE)        # [5000, 3, 32, 32]

    layers = {"stage1": model.layer1[0].norm1, "stage4": model.layer4[1].norm2}
    captured = {}
    hooks = [m.register_forward_pre_hook(lambda mod, inp, key=k: captured.__setitem__(key, inp[0]))
             for k, m in layers.items()]

    # thống kê quần thể: cộng dồn sum và sum bình phương theo kênh, theo khối
    s1 = {k: 0.0 for k in layers}
    s2 = {k: 0.0 for k in layers}
    cnt = {k: 0 for k in layers}
    for chunk in images.split(500):
        model(chunk)
        for k in layers:
            a = captured[k].double()
            s1[k] = s1[k] + a.sum(dim=(0, 2, 3))
            s2[k] = s2[k] + (a ** 2).sum(dim=(0, 2, 3))
            cnt[k] += a.shape[0] * a.shape[2] * a.shape[3]
    mean_pop = {k: s1[k] / cnt[k] for k in layers}
    var_pop = {k: s2[k] / cnt[k] - mean_pop[k] ** 2 for k in layers}

    g = torch.Generator().manual_seed(seed)
    acc = {(k, b): ([], []) for k in layers for b in batch_sizes}
    for b in batch_sizes:
        for _ in range(n_batches):
            idx = torch.randperm(len(images), generator=g)[:b].to(DEVICE)
            model(images[idx])
            for k in layers:
                a = captured[k].double()
                mean_b = a.mean(dim=(0, 2, 3))
                var_b = a.var(dim=(0, 2, 3), unbiased=False)
                acc[(k, b)][0].append(((mean_b - mean_pop[k]) / var_pop[k].sqrt()) ** 2)
                acc[(k, b)][1].append(((var_b - var_pop[k]) / var_pop[k]) ** 2)
    for h in hooks:
        h.remove()
    rows = []
    for (k, b), (m_err, v_err) in acc.items():
        rows.append(dict(layer=k, batch_size=b,
                         mean_noise=float(torch.stack(m_err).mean().sqrt()),
                         var_noise=float(torch.stack(v_err).mean().sqrt())))
    return pd.DataFrame(rows)



# 10. Hàm tiện ích
def zip_outputs(notebook: str) -> str:
    """Nén OUT_DIR thành '<notebook>_<OWNER>.zip' và in đường dẫn từ notebook kaggle"""
    base = f"{notebook}_{OWNER}".replace(" ", "_")
    path = shutil.make_archive(base, "zip", OUT_DIR)
    print("Đã nén kết quả:", os.path.abspath(path))
    return path
