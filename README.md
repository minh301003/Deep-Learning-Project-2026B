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
analysis/         do aggregate.py sinh: experiment_log.csv, tables/, figures/, checks.md  [commit]
```

## Chạy một notebook (thành viên chạy máy)

1. Kaggle → import notebook từ repo → bật GPU **T4** (không chọn P100) và Internet.
2. Sửa `OWNER` (và `SEEDS` nếu được dặn chia theo seed). Không sửa gì khác.
3. *Save & Run All*. Cuối notebook in bảng tóm tắt (đọc kỹ nếu có lượt `crashed`/`diverged`) và tạo `<notebook>_<OWNER>.zip`.
4. Gửi file `.zip` cho điều phối.

Notebook `git clone` repo (phải public) ở tag `v1` (điều phối tạo tag bằng `git tag v1 && git push origin v1` trước khi mọi người chạy); hash commit được `btl.init()` đọc tự động và ghi vào `runs.csv`.

## Tổng hợp kết quả (điều phối)

```bash
pip install -r requirements.txt
# giải nén mỗi zip vào results/<tên>/ (mỗi thư mục chứa runs.csv, epochs.csv, ...)
python aggregate.py
```

`aggregate.py` đọc mọi `results/**/runs.csv` và ghi vào `analysis/`:
`experiment_log.csv` (nhật ký thí nghiệm nộp theo đề), `tables/` (csv + md), `figures/` (PNG), `checks.md`
(đủ lượt, tái lập, môi trường, cảnh báo "không học"). Mọi số trong báo cáo lấy từ các file này.

## Tái lập

Cùng commit, cùng seed, cùng loại GPU (T4) và cùng `NUM_WORKERS` (hằng số trong `btl.py`) thì cho cùng kết quả:
`E1-sgdm-rerun` phải trùng `E1-sgdm` seed 0, và cấu hình "SGDm, lr 0.1, cosine, BN, batch 128" xuất hiện ở
E1-sgdm, E2-cos, E4-bn-b128. `checks.md` so các nhóm này. AMP fp16 và cuDNN có thể làm số lệch giữa các loại GPU.

## Quy tắc nhiễu

Mỗi cấu hình có 2 seed; báo cáo mean ± std (`ddof=1`). Hai cấu hình khác nhau **rõ** khi `|Δ mean| ≥ 2σ_Δ` và Δ cùng dấu ở cả hai seed
(σ_Δ lấy từ σ_pooled của nhóm); ngược lại **trong nhiễu**. Đây là quy tắc sàng lọc, không phải kiểm định thống kê chính thức.

## Nguồn tham khảo và phần tự viết

- Kiến trúc ResNet kiểu CIFAR (BasicBlock, stem conv3×3, không maxpool): theo [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), viết lại gọn.
- Dropout trong block: theo Wide-ResNet (Zagoruyko & Komodakis, 2016).
- Tự viết: toàn bộ `btl.py` (chia dữ liệu phân tầng, lịch LR theo bước, vòng huấn luyện, early stopping hậu kiểm, ghi log, chẩn đoán D1),
  `aggregate.py`, và các notebook.
- Công cụ AI: xem mục "Khai báo sử dụng công cụ AI" cuối báo cáo.
