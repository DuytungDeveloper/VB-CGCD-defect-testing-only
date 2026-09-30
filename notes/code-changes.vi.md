# Thay đổi code so với upstream — VB-CGCD fork

Mốc so sánh **duy nhất**: tag `upstream-baseline` (daihao42/VB-CGCD). Cập nhật 2026-09-30.

Nguyên tắc: **mặc định trong code Python = hành vi upstream**. Wrapper phase 1 đặt
`FT_BATCH_SIZE=32 FT_ACCUM=4` vì giới hạn 8 GB; cấu hình này đổi số ở mức nhỏ (PATCH 2).

Cách chia nhóm dựa trên `git diff --name-status upstream-baseline` (xem mục 5), không viết tay.

---

## 1a. Sửa file có sẵn của upstream (7 file)

### `feature_extractor/sl_finetuned_model.py`

| Vị trí | Thay đổi | Lý do | Đổi số? |
|---|---|---|---|
| dòng 32, mức module | `FT_SEED = int(os.environ.get("FT_SEED", 42))` rồi `torch.manual_seed(FT_SEED)` | Seed 42 cố định khiến hai lần chạy cùng cấu hình ra kết quả giống hệt → **không đo được nhiễu phase 1**. Đây là thay đổi quan trọng nhất về phương pháp | chỉ khi đổi seed |
| hàm mới `nt_xent(z, tau)` | NT-Xent (SimCLR): 2N neo, đối xứng, mẫu số gồm 2N−1 view | Cài Eq. 2. Paper viết thiếu 4 chỗ; hiểu theo chữ nghĩa cho loss âm | chỉ khi λ > 0 |
| `finetune_dino`, dòng 76 | `batch_size` đọc từ `FT_BATCH_SIZE` (mặc định 128 = upstream) — PATCH 2 | Batch 128 không vừa 8 GB | qua wrapper |
| `finetune_dino`, `epochs` | đọc từ `FT_EPOCHS` (mặc định 10) | Run validation dùng 3 epoch | chỉ khi đổi |
| vòng lặp epoch | gradient accumulation: `accum`, `loss/accum`, `step` mỗi `accum` batch, flush cuối epoch, in `mean CE` / `mean CON` mỗi epoch — PATCH 2 | Giữ batch hiệu dụng 128 và số bước như upstream | nhỏ, xem PATCH 2 |
| thân vòng lặp, nhánh `lam > 0` | nối 2 view thành một forward, CE trên view 1, `con_loss = nt_xent(...)`, `loss = ((1-lam)*cls + lam*con)/accum` | Cài Eq. 3 | có |
| cùng chỗ, nhánh `else` | nếu batch có `images2` thì vẫn nối 2 view, CE trên view 1 | Cho phép `FT_AUG=1` với λ = 0: có augmentation, không có L_con, giữ nguyên luồng RNG như nhánh λ > 0 | có |
| cuối mỗi epoch | lưu `ft_ckpt.pt` theo `FT_CKPT_PATH`, giữ 2 key pooler — PATCH 5 | Hai run từng chết giữa chừng không để lại gì | không |

### `feature_extractor/dino-cifar100.py`

| Vị trí | Thay đổi | Lý do | Đổi số? |
|---|---|---|---|
| dòng 17 | `@torch.no_grad()` trên `infer_features_labels` — PATCH 1 | `eval()` không tắt autograd; batch 128 cần ~15 GB | không |
| trước `train_dataset.set_format` | thêm `aug_transforms` và `two_view_func` | Augmentation hai view. Ảnh đã transform sẵn ở `trans_func`, nên phải cài ở đây | chỉ khi bật |
| nhánh `else` của `if (not args.finetuned)` | `select(FT_LABELED_IDX)` thay `filter(labels < 50)`; lọc theo `FT_FOLD`; bật augmentation theo `FT_AUG` / `FT_LAMBDA` | Sửa D9 (500 → 400 ảnh/lớp), tạo fold validation, tách augmentation khỏi λ | chỉ khi bật |

### `main.py`

| Vị trí | Thay đổi | Lý do |
|---|---|---|
| dòng 24 | `from clustering.kmeans import KMeansCluster` | — |
| `Clustering_alg`, dòng 33–36 | thêm nhánh `elif alg == 'kmeans'` | Cho phép `--clustering_alg kmeans` (D5, D18) |
| argparse | `--kmeans_n_init` (10), `--gmm_n_init` (1 = upstream) | Tách "số lần khởi tạo" khỏi "thuật toán" trong thiết kế 2×2 (D18) |
| sau `Clustering = Clustering_alg(...)` | `functools.partial(Clustering, n_init=...)` | `Clustering_alg` trả về lớp; dòng gọi ở vòng lặp stage không đổi |
| import, argparse | `hungarian_map`, `SimpleNamespace`; cờ `--hungarian_on` (`train`/`eval`, mặc định `train` = upstream) | Sửa D13 (`98d2c55`) |
| vòng lặp stage, `i > 0` | `predict_raw` + `pi_test` (Hungarian trên tập test lớp mới). `train`: đường cũ + log `pi_train`/`pi_test`/`same=`. `eval`: nhãn = thứ tự cụm; ánh xạ đóng băng; bản sao nhãn test đổi sang không gian cụm qua LUT; dòng S0 giữ nguyên | D13, Q10b |
| import, `Dataloader`, `load_mode` | `NEUDETLoader`; nhánh `--dataset neudet`; chế độ `--load_mode t1` | NEU-DET, GĐ3 (`ea9e24a`) |
| argparse + vòng lặp stage | `--print_confusion` (mặc định tắt): in ma trận nhầm lẫn test_all theo id thật sau mỗi phiên online, **sau** bảng kết quả | Chẩn đoán (`5692e3f`); không đổi số (đã kiểm) |

### `clustering/gmm.py`

`GMMCluster.__init__` thêm tham số `n_init=1` và truyền vào `GaussianMixture`. Mặc định 1 giữ nguyên
hành vi upstream (đã kiểm tra hồi quy). Cần cho ô còn thiếu của bảng 2×2 (D18).

### `clustering/clustering_base.py`

Thêm `predict_raw(features)` (id cụm, không dùng nhãn) và hàm `hungarian_map(cluster_ids, true_ids, k)`
(Hungarian k×k, chỉ gọi lúc đánh giá). Không đổi hàm cũ. 29/9, `4aa8ff3`.

### `README.md`, `.gitignore`

- README: tag dạng `c100_*`, quy ước `<dataset>_<purpose>[_<variant>][_s<N>]`; 29/9: lệnh phase 1 thêm `--inputs -` (`ad7314f`); 30/9: lệnh khuyến nghị thêm `--hungarian_on eval` và biến thể `FT_LABELED_IDX`.
- `.gitignore`: thêm `!splits/*.npy` để file chỉ số không bị quy tắc `*.npy` chặn.

---

## 1b. File mới, ảnh hưởng kết quả (6 file)

| File | Hàm chính | Lý do |
|---|---|---|
| `clustering/kmeans.py` | `KMeansCluster` (`fit`, `_pre_predict`), tham số `n_init` | Paper C.3 nói k-means, code chỉ có GMM (D5). Cùng interface với `GMMCluster` |
| `feature_extractor/ft_data.py` | `labeled_subset(idx_path, fold)`, `hf_to_npy_row(n, extract_bs)`, hằng `FOLDS` | Tách logic chọn dữ liệu ra khỏi `__main__` để kiểm tra trên CPU. Định nghĩa fold F1–F5 và ánh xạ hàng `.npy` ↔ chỉ số HF (D17) |
| `splits/c100_labeled_hf_idx.npy` | dữ liệu, int64 (20000,) | 20000 chỉ số HF mà phase 2 coi là có nhãn; đầu vào cho việc sửa D9 |
| `feature_extractor/dino-neudet.py` | `build`, `to_dataset`, `extract`, `main` | Phase 1 NEU: fine-tune chỉ trên 576 ảnh có nhãn (không rò rỉ D9 theo thiết kế); trích 1440/360 đặc trưng theo thứ tự file split; ghi thẳng 4 `.npy`, không shard (tránh D17) |
| `dataloaders/neudet.py` | `NEUDETLoader.makeT1Loader` | b3t1: base 3, increment 3, 576 có nhãn, 192 mới/lớp, 12 cũ/lớp/phiên; `assert` khớp `--base/--increment/--num_classes` và `num_dim < 192` (D21) |
| `splits/neu_split.json` | dữ liệu, sha256 `49c23576…` | Giao thức NEU cố định: lớp theo chữ cái, xáo seed 0, 80/20 mỗi lớp; `validation` làm test |

---

## 1c. Hạ tầng

Không sửa thuật toán, nhưng **hai file ảnh hưởng số liệu hoặc cách tính số**: `run_phase1_extract.sh`
(32×4 → PATCH 2; `--batch-size 128` quyết định tập có nhãn, D17) và `extract_vbcgcd.py` (M_f proxy, D15).
`run_phase2_cl.sh` viết tường minh mọi tham số, tất cả trùng mặc định `main.py` (kiểm 29/9), nên không đổi số.

**Tạo 14–16/09:**

| File | Vai trò | Sửa từ 17/9 |
|---|---|---|
| `.gitattributes` | quy tắc xuống dòng LF | — |
| `requirements-lock.txt` | môi trường pip. Commit đầu `fd8316b` là bản hỏng; bản dùng được tạo lại 16/9 (Part D) | — |
| `scripts/exp.py` | chạy mọi run, ghi `config.json`, `INDEX.txt`, entry LOG; `agg` tổng hợp seed | `cmd_agg`: cảnh báo khi `commit`/`env`/`inputs` khác nhau, có run dirty, trùng seed, hoặc có run bị bỏ qua. 29/9: `--inputs PATH…` hoặc `--inputs -`, trường `inputs_source` (`c4088a1`); `append_log` ghi `(no metrics.json)` và dùng mọi khoá khi không trùng `columns.txt` (`e66e0bd`). Lưu ý: `hash_path` băm cả tên file + nội dung, nên khác sha256 của file (vd. `splits/…npy`: `0a50cc0b…` so với `a11f243b…`) |
| `scripts/extract_vbcgcd.py` | trích chỉ số từ log phase 2; M_f là proxy trên tập test S0 (D15) | — |
| `scripts/probe.py` | xem đặc trưng, loader, checkpoint | đọc đường dẫn từ `runs/inputs.txt`; `ckpt` đọc `runs/<run>/artifacts/saved_models`; thêm repo root vào `sys.path` |
| `scripts/vram_probe.py` | đo VRAM theo batch size | chỉ đề xuất mốc `OK`, và chọn ước của 128 |
| `scripts/check_phase1.py` | 16 kiểm tra sau phase 1 | — |
| `scripts/run_phase1_extract.sh` | wrapper phase 1. **Ảnh hưởng số:** đặt 32×4 (PATCH 2), cố định `--batch-size 128` (D17) | thêm `FT_LAMBDA`, `FT_TAU`, `FT_EPOCHS`, `FT_SEED`, `FT_AUG`, `FT_LABELED_IDX`, `FT_FOLD`; in tất cả ra log; 2 lớp bảo vệ đặt trước `mkdir`; `--seed` lấy từ `FT_SEED`. 29/9: comment ví dụ dùng `--inputs -` (`ad7314f`) |
| `scripts/run_phase2_cl.sh` | wrapper phase 2; mọi tham số trùng mặc định `main.py` (kiểm 29/9) | — (tham số mới truyền qua `"$@"`) |

**Tạo 17–22/09:**

| File | Hàm chính | Lý do |
|---|---|---|
| `scripts/make_labeled_idx.py` (17/9) | `main()` | Sinh file chỉ số, tự kiểm chứng P là hoán vị và `hf[P]` khớp nhãn trước khi ghi |
| `scripts/eval_fold.py` (18/9) | `cluster_acc()`, `main()` | Tiêu chí validation trên fold: PCA 384 + k-means/GMM + Hungarian trên lớp mới giả, cộng acc head CE trên tập val |
| `scripts/pick_lambda.py` (19/9) | `acc()`, `main()` | Bản rút gọn, chỉ k-means `n_init=10`, xếp hạng nhanh lưới (λ, τ) |
| `scripts/queue_eq3.sh` (21/9) | `finished()` + hàng đợi | Thiết kế 8 so với 8 (đã bỏ, thay bằng lưới 3 seed). **Ngừng dùng**: dòng gọi phase 1 chưa có `--inputs -`, đừng chạy lại |
| `scripts/queue_grid.sh` (22/9) | `finished()` + vòng lặp `CELLS` × `SEEDS` | Hàng đợi lưới (λ, τ); tự bỏ qua bước đã xong; `trap` khôi phục `runs/inputs.txt`. 29/9: gọi phase 1 với `--inputs -` (`ad7314f`) |

**Tạo 30/09 (NEU-DET):**

| File | Hàm chính | Lý do |
|---|---|---|
| `scripts/make_neu_split.py` | `num()`, `main()` | Sinh `splits/neu_split.json`, tất định (chạy lại in `unchanged`, cùng sha256) |
| `scripts/run_phase1_neu.sh` | wrapper | Phase 1 NEU qua `exp.py`: kiểm GPU, đặt `FT_*` (32×4, seed 42), gọi `dino-neudet.py` |

Ghi chú tiếng Việt: `notes/code-changes.vi.md` (file này), `notes/plan.vi.md`.
Không có file `q6-q8-giai-thich.md` (được nhắc nhưng chưa từng tạo); lý do chọn Q6–Q8 nằm ở D19.

---

## 2. Khác biệt do môi trường, không do sửa code

**BOFT butterfly factor 2 → 1.** `load_model` khai báo `boft_n_butterfly_factor=2`, nhưng peft không build
được extension CUDA `fbd_cuda` nên tự hạ xuống 1 (chỉ báo `UserWarning`). Xác nhận:
`trainable params: 221,184` (factor 2 sẽ là 405,504). **Có ảnh hưởng kết quả.** Chi tiết: A5 trong
`reproducibility-issues.md`.

---

## 3. Bảng tra nhanh các cờ

| Tên | Mặc định trong Python | Wrapper đặt | Tác dụng |
|---|---|---|---|
| `FT_LAMBDA` | 0 | 0 | λ trong Eq. 3; > 0 bật NT-Xent và augmentation 2 view |
| `FT_TAU` | 0.2 | 0.2 | τ của NT-Xent, chỉ dùng khi λ > 0 |
| `FT_AUG` | rỗng | rỗng | rỗng = theo `FT_LAMBDA`; 1 = bật; 0 = tắt |
| `FT_EPOCHS` | 10 | 10 | số epoch fine-tune |
| `FT_SEED` | 42 | 42 | seed phase 1 |
| `FT_BATCH_SIZE` | **128** | **32** | batch fine-tune |
| `FT_ACCUM` | **1** | **4** | gradient accumulation |
| `FT_LABELED_IDX` | rỗng | rỗng | rỗng = `filter(labels < 50)` (upstream); có = `select(idx)`, sửa D9 |
| `FT_FOLD` | rỗng | rỗng | rỗng = đủ 50 lớp cơ sở; F1–F5 = giữ 10 lớp làm lớp mới giả |
| `--clustering_alg` | gmm | — | `gmm` hoặc `kmeans` |
| `--kmeans_n_init` | 10 | — | số lần khởi tạo k-means |
| `--gmm_n_init` | 1 | — | số lần khởi tạo GMM (1 = upstream) |
| `--hungarian_on` | train | — | D13: `train` = upstream (nhãn thật đặt tên cụm); `eval` = thứ tự cụm khi huấn luyện, Hungarian trên tập test khi đánh giá |
| `--print_confusion` | tắt | — | chẩn đoán: in ma trận nhầm lẫn test_all (id thật) sau mỗi phiên online |
| `--dataset neudet --load_mode t1` | — | — | NEU-DET b3t1; phải kèm `--base 3 --increment 3 --num_classes 6 --num_dim 128` |

Các run Eq. 3 dùng `FT_BATCH_SIZE=16 FT_ACCUM=8`, vì 32 × 2 view = 7.68 GB > 6.93 GB khả dụng.

**Hai điểm cần biết khi dùng cờ:**
- `FT_SEED=0` **không** truyền được xuống `dino-cifar100.py`: code bỏ qua `--seed` khi giá trị là 0 (D4).
  Vì vậy tránh dùng seed 0 cho phase 1.
- Với `FT_AUG=1` và λ = 0, dòng log in `views=1` dù thực tế forward 2 view, vì trường đó đi theo
  `FT_LAMBDA`. Lỗi hiển thị đã biết, chưa sửa. Bằng chứng thật là dòng `augmentation ON` và thời gian chạy.

---

## 4. Các phép kiểm tra hồi quy đã chạy

| Thay đổi | Cách kiểm | Kết quả |
|---|---|---|
| `--gmm_n_init` | `c100_regress_s0` so với `c100_base_s0` | trùng khít 43 chỉ số |
| `FT_SEED` | `c100_seedcheck` so với `c100_con0_probe` | CE 1.1756 trùng, đặc trưng lệch 1.15e-5 (nhiễu GPU) |
| `FT_AUG` | `c100_augcheck` so với `c100_con0_probe` | `augmentation OFF`, CE trùng, lệch 1.15e-5 |
| `KMeansCluster` *(sanity check, không phải hồi quy: upstream không có k-means)* | `c100_km1` so với `c100_base` (5 seed mỗi bên) | trùng 43 chỉ số. *Việc S0_All trùng nhau ở smoke test không phải bằng chứng: S0 vốn không dùng clustering* |
| `exp.py --inputs` (29/9) | `t_regress_old/new/none/cli1/cli2/bad` | không cờ = hành vi cũ (`ac9fb477…`); `-` = `{}`; hash lặp lại được; sai cú pháp in usage, không tạo run |
| `exp.py append_log` (29/9) | 3 run thử: không metrics / metrics trùng cột / metrics lạ | `(no metrics.json)`, `S0_All 91.50`, `acc 0.5000` |
| `predict_raw` + `hungarian_map` (29/9) | 3 cụm tổng hợp, offset 50 | acc 1.0, trùng đường cũ `predict(..., with_known=True)` |
| `--hungarian_on train` (29/9) | `c100_km_hungtrain_s0` so với `c100_km_s0` | trùng khít 46 chỉ số |
| `--print_confusion` (30/9) | `neu_km_evhung_d128_cm_s0` so với `neu_km_evhung_d128_s0` | trùng khít mọi chỉ số |
| `make_neu_split.py` (30/9) | chạy 2 lần | `unchanged`, cùng sha256 |

---

## 5. Danh sách file thay đổi

Không chép danh sách vào đây (chép tay sẽ lệch, xem lỗi "12 file" cũ). Lệnh lấy danh sách:

    git diff --name-status upstream-baseline -- . ':(exclude)runs' ':(exclude)notes'

Kiểm 2026-09-30, sau `5692e3f`: **29 dòng = 7 `M` + 22 `A`** (`scripts/` có 14 file; NEU thêm 5 file `A`).
Nếu lệnh cho số khác, tài liệu này đã lệch.

---

## 6. Lỗi đã biết

| Lỗi | Vị trí | Trạng thái |
|---|---|---|
| Trường `views=` in sai khi dùng `FT_AUG` | `sl_finetuned_model.py`, dòng in cuối epoch | **chưa sửa** (quyết định 29/9: chỉ đổi log, không đáng đụng phase 1), xem mục 3 |
| `exp.py` hash `runs/inputs.txt` cho cả phase 1 | `exp.py`, `inputs_info` | **đã sửa** 29/9 (`c4088a1`, `ad7314f`). Trường `inputs` của run phase 1 **trước `ad7314f`** là vô nghĩa; không sửa `config.json` cũ vì đó là bản ghi |
| LOG ghi `(no metrics.json yet; copy from runs/INDEX.txt later)` | `exp.py`, `append_log` | **đã sửa** 29/9 (`e66e0bd`); 38 entry cũ đổi ở `98bb890` |
| LOG ghi `result:` trống khi metrics không trùng `columns.txt` | `exp.py`, `append_log` | **đã sửa** 29/9 (`e66e0bd`) |
