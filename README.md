# BTL Học sâu — Đề tài 3: khảo sát kỹ thuật tối ưu hóa khi huấn luyện

ResNet-18 nhỏ (width 32, ~2,8M tham số) huấn luyện **từ đầu** trên CIFAR-10 (45k train / 5k val / 10k test).
Bốn thí nghiệm, mỗi cấu hình chạy 20 epoch với seed 0 và 1:

| Notebook | Nội dung | Số lượt |
|---|---|---:|
| `notebooks/01_optimizers.ipynb` | E1: SGD, SGD+momentum, Nesterov, RMSProp, Adam, AdamW (+1 lượt lặp kiểm tra tái lập) | 13 |
| `notebooks/02_lr_schedules.ipynb` | E2: const / step / cosine × có/không warm-up; ghi loss theo bước | 12 |
| `notebooks/03_regularization.ipynb` | E3: weight decay, dropout (0.1/0.3/0.5), flip/crop/jitter, early stopping; riêng và kết hợp | 20 |
| `notebooks/04_normalization.ipynb` | E4: BN/LN/GN × batch 8/32/128 + chẩn đoán D1 (nhiễu thống kê batch của BN) | 18 |

Tổng 63 lượt. Kế hoạch và quyết định thiết kế: [plans/](plans/); đề bài: [docs/](docs/).

## Cấu trúc

```text
btl.py            toàn bộ code huấn luyện (dữ liệu, mô hình, optimizer/LR, run(), chỉ số, D1)
aggregate.py      gộp log -> bảng, hình, checks.md
notebooks/        4 notebook mỏng, chỉ cấu hình + vòng chạy (xóa output trước khi commit)
results/<tên>/    kết quả nhận từ thành viên: runs.csv, epochs.csv, (steps.csv, d1.csv)   [commit]
analysis/         do aggregate.py sinh: experiment_log.csv, tables/, figures/
```

## Hướng dẫn cách chạy

1. Kaggle → import notebook từ repo → bật GPU **T4** (không chọn P100) và Internet.
2. Sửa `OWNER` (optional để lưu lại tên người chạy, `SEED` để nguyên là 0 và 1)
3. *Save & Run All*.
4. Lưu lại file `.zip` kết quả.


## Tổng hợp kết quả 

```bash
pip install -r requirements.txt
# giải nén mỗi zip vào results/<tên>/ (mỗi thư mục chứa runs.csv, epochs.csv, ...)
python aggregate.py
```

`aggregate.py` đọc mọi `results/**/runs.csv` và ghi vào `analysis/`:
`experiment_log.csv` (nhật ký thí nghiệm nộp theo đề), `tables/` (csv + md), `figures/` (PNG), `checks.md`

## Quy tắc nhiễu

Mỗi cấu hình có 2 seed; báo cáo mean ± std (`ddof=1`). Hai cấu hình khác nhau **rõ** khi `|Δ mean| ≥ 2σ_Δ` và Δ cùng dấu ở cả hai seed
(σ_Δ lấy từ σ_pooled của nhóm); ngược lại **trong nhiễu**.

## Nguồn tham khảo

- Kiến trúc ResNet kiểu CIFAR (BasicBlock, stem conv3×3, không maxpool): theo [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), viết lại gọn.
- Dropout trong block: theo Wide-ResNet (Zagoruyko & Komodakis, 2016).