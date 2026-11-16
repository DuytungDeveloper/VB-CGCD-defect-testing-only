# Thay đổi code so với upstream — VB-CGCD fork

Mốc so sánh **duy nhất**: tag `upstream-baseline` (daihao42/VB-CGCD). Cập nhật 2026-10-19: thêm các thay đổi từ 1/10 tới 18/10; mã theo QD22 (mã cũ: `notes/old-codes.vi.md`); bảng đổi thành danh sách. Cập nhật 2026-10-20: `_check_09` của `lab/replay.py` và kết quả QD21/C1–C4 (§1c, §4); sửa ghi chú về các phép kiểm hồi quy của CT1 (§4). Cập nhật 2026-10-21: `lab/variants.py`, `lab/score.py`; xoá `lab/pl_cluster.py` và bản đầu của bàn mổ; phép kiểm GPU trong `replay.py run` (§1c, §4, §5). Cập nhật 2026-10-22: `lab/data_io.py` (§1c, §4, §5). Cập nhật 2026-10-24: `lab/00-data-gc10.ipynb` (§1c, §5). Cập nhật 2026-11-01: xoá `scripts/probe.py`, `scripts/vram_probe.py`, `scripts/ckpt_vs_data.py`; `lab/data_io.py` đọc NEU-DET; `lab/00-data-neu.ipynb`; sửa ô Kết quả của cell 0 trong `lab/00-data-gc10.ipynb` (§1c, §4, §5); `lab/01-preprocess.ipynb`, `lab/02-finetune.ipynb` (§1c, §5); `lab/score.py` thêm `tier1_03` cho E03.02 (§1c); `lab/03-features.ipynb` (§1c, §5). Cập nhật 2026-11-03: `rerun` trong `lab/replay.py` (§1c, §4). Cập nhật 2026-11-04: `lab/04-loader.ipynb`, `lab/05-pca.ipynb`, `lab/06-gauss-svi.ipynb`, `lab/07-early-stop.ipynb`, `lab/05a-camera.ipynb`, `lab/08-classifier.ipynb` (§1c, §5); `lab/landscape.py` (§1c, §4, §5); `lab/09-cluster.ipynb`, `lab/09d-ownership.ipynb` (§1c, §5). Cập nhật 2026-11-05: `lab/score.py` thêm `ledger`, `ledger_summary`; `lab/10-naming.ipynb` (§1c, §5); `lab/score.py` thêm cột 11 của `ledger`; `lab/11-gauss-f.ipynb` (§1c, §5); `lab/score.py` thêm cột 12 của `ledger`; `lab/12-filter.ipynb` (§1c, §5); `lab/13-gauss-s.ipynb` (§1c, §5); `lab/score.py` thêm cột 14, tập `test_old`, `ledger_state`; `lab/14-eval.ipynb` (§1c, §5); `feature_extractor/dino-cub200.py` thêm `@torch.no_grad()` theo QD27 (§1a, §5). Cập nhật 2026-11-13: `lab/variants.py` thêm `multi_init`, `cluster_of`, `v09c_multi_init` (E09.07), `v04a_cub_t5` (QD27), `v09b_merge` và các hàm gộp (E09.06); `lab/score.py` thêm runner `tier2_e0907`, `tier2_e0604`, `tier2_e0906` và bảng quyết định `e0907_summary`, `e0604_summary`, `e0906_summary`; file mới `lab/niw.py` (E06.04) (§1c, §4, §5).

Nguyên tắc: **mặc định trong code Python = hành vi upstream**. Wrapper phase 1 đặt
`FT_BATCH_SIZE=32 FT_ACCUM=4` vì giới hạn 8 GB; cấu hình này đổi số ở mức nhỏ (PATCH 2).

Cách chia nhóm dựa trên `git diff --name-status upstream-baseline` (xem mục 5), không viết tay.

---

## 1a. Sửa file có sẵn của upstream (9 file)

### `feature_extractor/sl_finetuned_model.py`

- dòng 32, mức module: `FT_SEED = int(os.environ.get("FT_SEED", 42))` rồi `torch.manual_seed(FT_SEED)`
  Lý do: Seed 42 cố định khiến hai lần chạy cùng cấu hình ra kết quả giống hệt → **không đo được nhiễu phase 1**. Đây là thay đổi quan trọng nhất về phương pháp. Đổi số: chỉ khi đổi seed.
- hàm mới `nt_xent(z, tau)`: NT-Xent (SimCLR): 2N neo, đối xứng, mẫu số gồm 2N−1 view
  Lý do: Cài Eq. 2. Paper viết thiếu 4 chỗ; hiểu theo chữ nghĩa cho loss âm. Đổi số: chỉ khi λ > 0.
- `finetune_dino`, dòng 76: `batch_size` đọc từ `FT_BATCH_SIZE` (mặc định 128 = upstream) — PATCH 2
  Lý do: Batch 128 không vừa 8 GB. Đổi số: qua wrapper.
- `finetune_dino`, `epochs`: đọc từ `FT_EPOCHS` (mặc định 10)
  Lý do: Run validation dùng 3 epoch. Đổi số: chỉ khi đổi.
- vòng lặp epoch: gradient accumulation: `accum`, `loss/accum`, `step` mỗi `accum` batch, flush cuối epoch, in `mean CE` / `mean CON` mỗi epoch — PATCH 2
  Lý do: Giữ batch hiệu dụng 128 và số bước như upstream. Đổi số: nhỏ, xem PATCH 2.
- thân vòng lặp, nhánh `lam > 0`: nối 2 view thành một forward, CE trên view 1, `con_loss = nt_xent(...)`, `loss = ((1-lam)*cls + lam*con)/accum`
  Lý do: Cài Eq. 3. Đổi số: có.
- cùng chỗ, nhánh `else`: nếu batch có `images2` thì vẫn nối 2 view, CE trên view 1
  Lý do: Cho phép `FT_AUG=1` với λ = 0: có augmentation, không có L_con, giữ nguyên luồng RNG như nhánh λ > 0. Đổi số: có.
- cuối mỗi epoch: lưu `ft_ckpt.pt` theo `FT_CKPT_PATH`, giữ 2 key pooler — PATCH 5
  Lý do: Hai run từng chết giữa chừng không để lại gì. Đổi số: không.

### `feature_extractor/dino-cifar100.py`

- dòng 17: `@torch.no_grad()` trên `infer_features_labels` — PATCH 1
  Lý do: `eval()` không tắt autograd; batch 128 cần ~15 GB. Đổi số: không.
- trước `train_dataset.set_format`: thêm `aug_transforms` và `two_view_func`
  Lý do: Augmentation hai view. Ảnh đã transform sẵn ở `trans_func`, nên phải cài ở đây. Đổi số: chỉ khi bật.
- nhánh `else` của `if (not args.finetuned)`: `select(FT_LABELED_IDX)` thay `filter(labels < 50)`; lọc theo `FT_FOLD`; bật augmentation theo `FT_AUG` / `FT_LAMBDA`
  Lý do: Sửa D9 (500 → 400 ảnh/lớp), tạo fold validation, tách augmentation khỏi λ. Đổi số: chỉ khi bật.

### `feature_extractor/dino-cub200.py`

- trên `infer_features_labels`: `@torch.no_grad()`, như PATCH 1 của `dino-cifar100.py` (QD27)
  Lý do: `eval()` không tắt autograd; ở batch 128 bản upstream không vừa 8 GB. Đổi số: không, vì giá trị forward không phụ thuộc autograd. Không có phép hồi quy: bản upstream không chạy được trên máy 8 GB.
- trong `__main__`, chỗ chọn dữ liệu fine-tune (16/11, `4b36e47`): `FT_LABELED_IDX` có giá trị thì `select(idx)` trên 2500 ảnh có nhãn, kèm assert 2500 chỉ số khác nhau và đúng 25 ảnh mỗi lớp 0–99; rỗng thì `filter(labels < 100)` như upstream (2996 ảnh). `FT_FOLD` bị chặn trên CUB.
  Lý do: sửa D9 trên CUB (QD28). Đổi số: có khi đặt biến; không đặt thì như upstream (phép hồi quy ở §4).

### `main.py`

- dòng 24: `from clustering.kmeans import KMeansCluster`
- `Clustering_alg`, dòng 33–36: thêm nhánh `elif alg == 'kmeans'`
  Lý do: Cho phép `--clustering_alg kmeans` (D5, D18).
- argparse: `--kmeans_n_init` (10), `--gmm_n_init` (1 = upstream)
  Lý do: Tách "số lần khởi tạo" khỏi "thuật toán" trong thiết kế 2×2 (D18).
- sau `Clustering = Clustering_alg(...)`: `functools.partial(Clustering, n_init=...)`
  Lý do: `Clustering_alg` trả về lớp; dòng gọi ở vòng lặp stage không đổi.
- import, argparse: `hungarian_map`, `SimpleNamespace`; cờ `--hungarian_on` (`train`/`eval`, mặc định `train` = upstream)
  Lý do: Sửa D13 (`98d2c55`).
- vòng lặp stage, `i > 0`: `predict_raw` + `pi_test` (Hungarian trên tập test lớp mới). `train`: đường cũ + log `pi_train`/`pi_test`/`same=`. `eval`: nhãn = thứ tự cụm; ánh xạ đóng băng; bản sao nhãn test đổi sang không gian cụm qua LUT; dòng S0 giữ nguyên
  Lý do: D13, QD15.
- import, `Dataloader`, `load_mode`: `NEUDETLoader`; nhánh `--dataset neudet`; chế độ `--load_mode t1`
  Lý do: NEU-DET, GĐ3 (`ea9e24a`).
- argparse + vòng lặp stage: `--print_confusion` (mặc định tắt): in ma trận nhầm lẫn test_all theo id thật sau mỗi phiên online, **sau** bảng kết quả
  Lý do: Chẩn đoán (`5692e3f`); không đổi số (đã kiểm).
- import, `Dataloader`: `GC10Loader`; nhánh `--dataset gc10`
  Lý do: GC10-DET (`a542c78`).
- argparse + stage 0: `--s0_labels true|none` (mặc định `true` = upstream). `none`: phân cụm dữ liệu cơ sở (k = `--base`), Gaussian học id cụm; Hungarian trên tập test chỉ để đánh giá; bắt buộc kèm `--hungarian_on eval`
  Lý do: S0 không nhãn, E04.01 (`ed7a6d6`).
- argparse: `--cov_prior_kappa`, `--cov_prior_target`, `--estimator`, `--mask_unseen`, `--oracle_filter`; chỉ chuyển giá trị sang thuộc tính của `s_classifier` (xem `classifier/mngmm.py`)
  Lý do: CT1, E06.01–E06.03 (`f6d7841`, `bd70b17`, `3e45d55`, `0ea29f0`).
- argparse + vòng lặp stage: `--print_cluster_xtab` (mặc định tắt; CHỈ CHẨN ĐOÁN, dùng nhãn thật): sau bảng kết quả, in bảng chéo cụm × lớp thật trên pool train và tập test lớp mới, cùng hàm mục tiêu của phân cụm (inertia)
  Lý do: E09.03 (`c7bc7e7`); không đổi số (đã kiểm).

### `classifier/mngmm.py`

Mọi nhánh mới chỉ chạy khi bật cờ; mặc định giữ đúng đường upstream (kiểm hồi quy ở mục 4).

- `model()`: khi `--cov_prior_kappa` > 0, cộng `numpyro.factor("cov_prior", ...)`: tiên nghiệm co Inverse-Wishart (RDA) lên hiệp phương sai của các lớp thuộc lượt hiện tại
  Lý do: E06.01 (`f6d7841`); lát cắt lớp sửa ở `fe95908` (trước đó, từ S1 tiên nghiệm áp lên lớp cũ). Đổi số: chỉ khi κ > 0.
- `run_inference()`: đích co theo `--cov_prior_target`: `pooled` (hiệp phương sai trong lớp gộp của lượt) hoặc `sphere` = (tr/d) I; bỏ tiên nghiệm khi lượt không có mẫu
  Lý do: E06.01 (`bd70b17`). Đổi số: chỉ khi κ > 0.
- `train()`, `_fit_closed()`: `--estimator closed` thay SVI bằng ước lượng dạng đóng (`classifier/closed_form.py`)
  Lý do: E06.03 (`3e45d55`). Đổi số: chỉ khi bật.
- `train()`: `--oracle_filter` thay bộ lọc cũ/mới bằng phép tách theo nhãn thật trước lượt S
  Lý do: CHỈ CHẨN ĐOÁN, không thuộc phương pháp (`0ea29f0`). Đổi số: chỉ khi bật.
- `_predict()`: `--mask_unseen` lấy argmax chỉ trên các lớp đã gặp
  Lý do: chẩn đoán D24 (`0ea29f0`). Đổi số: chỉ khi bật.

### `clustering/gmm.py`

`GMMCluster.__init__` thêm tham số `n_init=1` và truyền vào `GaussianMixture`. Mặc định 1 giữ nguyên
hành vi upstream (đã kiểm tra hồi quy). Cần cho ô còn thiếu của bảng 2×2 (D18).

### `clustering/clustering_base.py`

Thêm `predict_raw(features)` (id cụm, không dùng nhãn) và hàm `hungarian_map(cluster_ids, true_ids, k)`
(Hungarian k×k, chỉ gọi lúc đánh giá). Không đổi hàm cũ. 29/9, `4aa8ff3`.

### `README.md`, `.gitignore`

- README: tag dạng `c100_*`, quy ước `<dataset>_<purpose>[_<variant>][_s<N>]`; 29/9: lệnh phase 1 thêm `--inputs -` (`ad7314f`); 30/9: lệnh khuyến nghị thêm `--hungarian_on eval` và biến thể `FT_LABELED_IDX`.
- `.gitignore`: thêm `!splits/*.npy` để file chỉ số không bị quy tắc `*.npy` chặn.
- README từ 1/10: lệnh GC10-DET (`fa5cafb`); dòng mục lục `lab/README.vi.md` và bản lưu `archive/pre-rerun-20260915` nay là file tar trên E: (`ebcdd49`); câu hỏi mở trỏ tới `notes/pl/questions.vi.md`, liệt kê các file `notes/pl/` (`e75889f`); cài `ipykernel` cho notebook trong `lab/` (`d18fef2`).
- `.gitignore` (15/10): `data/` thành `/data/`, chỉ bỏ qua thư mục `data/` ở gốc repo (`dd7a954`).

---

## 1b. File mới, ảnh hưởng kết quả (11 file, cộng 26 file split ngân sách nhãn)

- `clustering/kmeans.py`: Paper C.3 nói k-means, code chỉ có GMM (D5). Cùng interface với `GMMCluster`.
  Hàm chính: `KMeansCluster` (`fit`, `_pre_predict`), tham số `n_init`.
- `feature_extractor/ft_data.py`: Tách logic chọn dữ liệu ra khỏi `__main__` để kiểm tra trên CPU. Định nghĩa fold F1–F5 và ánh xạ hàng `.npy` ↔ chỉ số HF (D17).
  Hàm chính: `labeled_subset(idx_path, fold)`, `hf_to_npy_row(n, extract_bs)`, hằng `FOLDS`.
- `splits/c100_labeled_hf_idx.npy`: 20000 chỉ số HF mà phase 2 coi là có nhãn; đầu vào cho việc sửa D9.
  Hàm chính: dữ liệu, int64 (20000,).
- `feature_extractor/dino-neudet.py`: Phase 1 cho mọi file split (NEU: tên ảnh; GC10: `thư_mục/tên`, `n_base`/`n_use` từ file split). Fine-tune chỉ trên ảnh có nhãn (không rò rỉ D9 theo thiết kế); `--no_finetune` = DINO đóng băng, token CLS; `--resize crop|full`; `--img_size H W` (interpolate_pos_encoding); ghi thẳng 4 `.npy`, không shard (tránh D17).
  Hàm chính: `build`, `to_dataset`, `extract`, `main`.
- `dataloaders/neudet.py`: b3t1: base 3, increment 3, 576 có nhãn, 192 mới/lớp, 12 cũ/lớp/phiên; `assert` khớp `--base/--increment/--num_classes` và `num_dim < 192` (D21). Từ 1/10, bỏ qua `assert num_dim` khi `--cov_prior_kappa` > 0 hoặc `--estimator closed` (`f6d7841`, `3e45d55`).
  Hàm chính: `NEUDETLoader.makeT1Loader`.
- `splits/neu_split.json`: Giao thức NEU cố định: lớp theo chữ cái, xáo seed 0, 80/20 mỗi lớp; `validation` làm test.
  Hàm chính: dữ liệu, sha256 `49c23576…`.
- `dataloaders/gc10.py`: Số ảnh theo từng lớp từ file split; nhãn dựng theo số hàng thực lấy (sửa D22); `assert` base/increment, `num_dim` < 130 (D21), số hàng khớp file split. Từ 1/10, bỏ qua `assert num_dim` khi `--cov_prior_kappa` > 0 hoặc `--estimator closed`.
  Hàm chính: `SplitCountIncrementalLoader`, `GC10Loader.makeT1Loader`.
- `splits/gc10_split.json`: Giao thức GC10: thư mục 1–10 → id 0–9, cơ sở 0–4; test ≈20% mỗi lớp **chia theo cuộn thép**; bỏ 24 file trùng; kèm danh sách chẩn đoán.
  Hàm chính: dữ liệu, sha256 `a8a454f8…`.
- `classifier/closed_form.py`: ước lượng dạng đóng cho `--estimator closed`: trung bình mẫu + hiệp phương sai co Ledoit–Wolf; ca biên chốt trước (n_k = 1, n_k = 0, lớp cũ chép nguyên) (E06.03, `c016f29`).
  Hàm chính: `closed_form_params(X, y, classes, prev, num_classes)`.
- `splits/<ds>_cls<m><sel>_img<k>[_r<r>].json` (26 file, 5/10): biến thể ngân sách nhãn cho phase 1: số lớp có nhãn và cách chọn (`first`, `top`, `rand`), số ảnh mỗi lớp (10, 25, 50, 80pct), lần rút r (E02.06, `97c57ee`).
  Hàm chính: dữ liệu, sinh bởi `scripts/make_label_splits.py`.
- `scripts/make_labeled_idx_cub.py` (16/11, `581ae9f`): tính 2500 chỉ số HF mà phase 2 coi là có nhãn trên CUB t5, qua hoán vị D17 (47 shard gộp theo thứ tự chuỗi). Ba phép kiểm: nhãn gộp trùng `hf[P]`; tập hàng trùng tập S0 của `StrictPerClassIncrementalLoader`; 2500 chỉ số khác nhau, xếp theo khối lớp.
  Hàm chính: script, `--check <thư mục đặc trưng>`.
- `splits/cub_labeled_hf_idx.npy` (16/11, `49d5070`): 2500 chỉ số HF, 25 ảnh mỗi lớp 0–99; đầu vào cho việc sửa D9 trên CUB.
  Hàm chính: dữ liệu, sha256 `6dbb1939…`.

---

## 1c. Hạ tầng

Không sửa thuật toán, nhưng **hai file ảnh hưởng số liệu hoặc cách tính số**: `run_phase1_extract.sh`
(32×4 → PATCH 2; `--batch-size 128` quyết định tập có nhãn, D17) và `extract_vbcgcd.py` (M_f proxy, D15).
`run_phase2_cl.sh` viết tường minh mọi tham số, tất cả trùng mặc định `main.py` (kiểm 29/9), nên không đổi số.

**Tạo 14–16/09:**

- `.gitattributes`: quy tắc xuống dòng LF.
- `requirements-lock.txt`: môi trường pip. Commit đầu `fd8316b` là bản hỏng; bản dùng được tạo lại 16/9 (Part D).
- `scripts/exp.py`: chạy mọi run, ghi `config.json`, `INDEX.txt`, entry LOG; `agg` tổng hợp seed.
  Sửa từ 17/9: `cmd_agg`: cảnh báo khi `commit`/`env`/`inputs` khác nhau, có run dirty, trùng seed, hoặc có run bị bỏ qua. 29/9: `--inputs PATH…` hoặc `--inputs -`, trường `inputs_source` (`c4088a1`); `append_log` ghi `(no metrics.json)` và dùng mọi khoá khi không trùng `columns.txt` (`e66e0bd`). Lưu ý: `hash_path` băm cả tên file + nội dung, nên khác sha256 của file (vd. `splits/…npy`: `0a50cc0b…` so với `a11f243b…`).
- `scripts/extract_vbcgcd.py`: trích chỉ số từ log phase 2; M_f là proxy trên tập test S0 (D15).
- `scripts/probe.py`: xem đặc trưng, loader, checkpoint. Xoá 1/11 (`bba8722`); bản cuối ở `114b597`.
  Sửa từ 17/9: đọc đường dẫn từ `runs/inputs.txt`; `ckpt` đọc `runs/<run>/artifacts/saved_models`; thêm repo root vào `sys.path`.
- `scripts/vram_probe.py`: đo VRAM theo batch size. Xoá 1/11 (`bba8722`); bản cuối ở `114b597`.
  Sửa từ 17/9: chỉ đề xuất mốc `OK`, và chọn ước của 128.
- `scripts/check_phase1.py`: 16 kiểm tra sau phase 1.
- `scripts/run_phase1_extract.sh`: wrapper phase 1. **Ảnh hưởng số:** đặt 32×4 (PATCH 2), cố định `--batch-size 128` (D17).
  Sửa từ 17/9: thêm `FT_LAMBDA`, `FT_TAU`, `FT_EPOCHS`, `FT_SEED`, `FT_AUG`, `FT_LABELED_IDX`, `FT_FOLD`; in tất cả ra log; 2 lớp bảo vệ đặt trước `mkdir`; `--seed` lấy từ `FT_SEED`. 29/9: comment ví dụ dùng `--inputs -` (`ad7314f`).
- `scripts/run_phase2_cl.sh`: wrapper phase 2; mọi tham số trùng mặc định `main.py` (kiểm 29/9).
  Sửa từ 17/9: không (tham số mới truyền qua `"$@"`).

**Tạo 17–22/09:**

- `scripts/make_labeled_idx.py` (17/9): Sinh file chỉ số, tự kiểm chứng P là hoán vị và `hf[P]` khớp nhãn trước khi ghi.
  Hàm chính: `main()`.
- `scripts/eval_fold.py` (18/9): Tiêu chí validation trên fold: PCA 384 + k-means/GMM + Hungarian trên lớp mới giả, cộng acc head CE trên tập val.
  Hàm chính: `cluster_acc()`, `main()`.
- `scripts/pick_lambda.py` (19/9): Bản rút gọn, chỉ k-means `n_init=10`, xếp hạng nhanh lưới (λ, τ).
  Hàm chính: `acc()`, `main()`.
- `scripts/queue_eq3.sh` (21/9): Thiết kế 8 so với 8 (đã bỏ, thay bằng lưới 3 seed). **Ngừng dùng**: dòng gọi phase 1 chưa có `--inputs -`, đừng chạy lại.
  Hàm chính: `finished()` + hàng đợi.
- `scripts/queue_grid.sh` (22/9): Hàng đợi lưới (λ, τ); tự bỏ qua bước đã xong; `trap` khôi phục `runs/inputs.txt`. 29/9: gọi phase 1 với `--inputs -` (`ad7314f`).
  Hàm chính: `finished()` + vòng lặp `CELLS` × `SEEDS`.

Bốn file `eval_fold.py`, `pick_lambda.py`, `queue_eq3.sh`, `queue_grid.sh` đã xoá ngày 4/10 (`831386c`); đọc lại bằng `git show 831386c~1:scripts/<tên>`.

**Tạo 30/09 (NEU-DET):**

- `scripts/make_neu_split.py`: Sinh `splits/neu_split.json`, tất định (chạy lại in `unchanged`, cùng sha256).
  Hàm chính: `num()`, `main()`.
- `scripts/run_phase1_neu.sh`: Phase 1 qua `exp.py` (NEU; GC10 truyền `--split`): kiểm GPU, đặt `FT_*` (32×4, seed 42), `HF_HUB_OFFLINE=1` mặc định, gọi `dino-neudet.py`.
  Hàm chính: wrapper.

**Tạo 30/09–01/10 (GC10-DET):**

- `scripts/make_gc10_split.py`: Sinh `splits/gc10_split.json`, tất định; gán cuộn thép cho train/test ở mức toàn cục.
  Hàm chính: `main()`.
- `scripts/cm_summary.py`: Tóm tắt dòng `confusion stage N` của một nhóm run: balanced accuracy, recall từng lớp, nhầm lẫn lớn nhất.
  Hàm chính: `main()`.
- `scripts/feat_probe.py`: Chẩn đoán một bộ đặc trưng: k-means (biết k), silhouette, probe tuyến tính (cận trên).
  Hàm chính: `cacc`, `main()`.

**Tạo 01–02/10 (CT1, CT2):**

- `scripts/test_closed_form.py` (2/10, `c016f29`): unit test cho `classifier/closed_form.py` trên dữ liệu tổng hợp: công thức Σ, trung bình, ca n = 1, xác định dương, khe chưa gặp giữ (0, I), lớp cũ chép nguyên, lớp không có mẫu giữ giá trị cũ.
  Hàm chính: không có hàm; chạy thẳng.
- `scripts/ckpt_vs_data.py` (2/10, `b7cb407`): so tham số S0 đã học với dữ liệu trong cùng không gian PCA (D23). Xoá 1/11 (`bba8722`), `lab/replay.py` làm thay; bản cuối ở `114b597`.
  Hàm chính: chạy thẳng theo tag.

**Tạo 05/10:**

- `scripts/make_label_splits.py` (`97c57ee`): sinh 26 file split ngân sách nhãn (E02.06); `variant.ft_epochs` giữ số bước fine-tune của cấu hình gốc.
  Hàm chính: `main()`.

**Tạo 06–24/10 (`lab/`, QD20, QD21, QD23):**

- `lab/replay.py`: bàn mổ. v1 (6/10, `d912e76`): chạy lại `main.py` từng giai đoạn cho một seed, đúng thứ tự RNG. v2 (16–18/10, `737e8c4`, `a5c9545`, `86ed44d`): bước 04–14 là hàm `sNN(st)`, ảnh chụp theo bước, RNG chung, chuỗi băm; lệnh `run`, `check`, `check-resume`, `check-crn`, `check-hash`. Không sửa code gốc. Chưa dùng số của v2 khi QD21/C1–C3 chưa đạt.
  Hàm chính: `chain()`, `cmd_run()`, các bước `s04_*`…`s14_*`; `replay()` của bản đầu đã xoá 21/10.
  Sửa 20/10 (`3525e86`): `_check_09`, sau mọi hàm của bước 09 kiểm `ids` và `cluster.predict_raw(test1_x)` là số nguyên trong [0, increment), mỗi ảnh một id (id ngoài khoảng không báo lỗi ở các bước 10–13); C4 thêm một trường hợp. QD21/C1–C4 đạt ngày 20/10, nên được dùng số của v2.
  Thêm 1/11 (`53d8603`): `rerun(run, k)` và `_compare` cho notebook 04–14 (QD23): chạy lại bước k từ ảnh chụp của bước trước (bước 04 từ `new_state`), như C2 nhưng chỉ bước k, rồi so từng khoá với ảnh chụp k theo quy tắc của C1 (rời rạc trùng tuyệt đối; số thực lệch ≤ 1e-6 × max|giá trị|; PCA, mô hình cụm bằng `_same`); dừng nếu thiết bị khác thiết bị của ảnh chụp, báo nếu driver GPU khác `config.json`; không ghi file. Nằm ngoài chuỗi băm.
  Sửa 21/10: lệnh `run` dừng khi JAX không chạy trên GPU (A1; `REPLAY_ALLOW_CPU=1` chỉ để thử) hoặc khi chạy tiếp ảnh chụp tạo trên thiết bị khác, và ghi thiết bị vào meta của ảnh chụp (`1253ce6`). Bản đầu (`replay()`, `_reference()`, khối tự kiểm) đã xoá; chữ v2 và mã cũ bỏ khỏi chú thích, docstring, help và dòng log (`replay: …`), trừ một thông báo lỗi trong `new_state` nằm trong chuỗi băm, giữ nguyên để 15 ảnh chụp gốc không cũ (`370cf86`). Từ nay chỉ một bàn mổ (QD21, sửa ngày 2026-10-21).
- `lab/pl_cluster.py` (7/10, `614d12b`, `89e94d4`): bộ đo tầng 1 của bước 09 và phần chạy của E09.04, E09.05; chụp trạng thái RNG ở lối vào `KMeansCluster.fit`.
  Hàm chính: `score()`, `main()`.
  Xoá 21/10 (`71dab1f`), thay bằng `lab/score.py` sau phép hồi quy 15/15 (§4); bản gốc ở commit `89e94d4` và tag `nb-v1`.
- `lab/README.vi.md`: mục lục và quy tắc của `lab/` (QD20).
- `lab/variants.py` (21/10, `092ae8f`): biến thể của các bước của bàn mổ, kèm quy tắc viết ở docstring đầu file; `v09a_overcluster`: k-means với K' > increment cụm trên pool (E09.04, E09.05), ghi `sub_ids`, `sub_cluster`, `sub_inertia`, không ghi `ids`.
  Thêm 10/11 (`709eeca`, E09.07): `multi_init(x, k, m, random_state=None, marks=None)`: m lần khởi tạo k-means++ đơn (`KMeans(n_init=1)`) liên tiếp từ một luồng RNG, giữ nghiệm theo đúng luật của `KMeans.fit` (scikit-learn 1.5.2: lấy lần đầu, thay khi inertia thấp hơn hẳn và phân hoạch khác); `random_state=None` rút RNG toàn cục như bước 09, nên 10 lần đầu chính là `n_init` 10 của pipeline từ cùng trạng thái. `cluster_of(model, k, offset)`: bọc một `KMeans` đã fit thành `KMeansCluster` để bước 10–14 gọi được `predict_raw`. `v09c_multi_init(st, m=1000)`: biến thể `09c`, bước 09 với m lần khởi tạo đơn thay cho `n_init` 10.
  Thêm 11/11 (`440d60c`, QD27, E06.04): hằng `CUB_T5` (cub200, 100 lớp cơ sở + 5 phiên × 20 lớp, `num_dim` 384) và `v04a_cub_t5(st)`: biến thể `04a`, các phiên của giao thức CUB-200 t5 qua `main.Dataloader` (`load_mode` t5, tức bộ chia của `dataloaders/cub200.py`), cắt còn phần bước 05–08 và phiên online đầu cần; không sửa `dataloaders/`.
  Thêm 12/11 (`68285e7`, E09.06, tiền đăng ký `61b9eb9`): `v09b_merge(st, rule="bf", prior=None)`: biến thể `09b`, gộp tham lam các cụm con của `09a` về `increment` cụm, mỗi bước gộp cặp có log Bayes factor NIW lớn nhất (trên đặc trưng PCA của pool); ảnh test vào cụm con có tâm gần nhất (đặc trưng thô) rồi theo bảng gộp; không đọc nhãn pool, không rút RNG. Hàm phụ: `merge_prior(st)` (tiên nghiệm NIW: m0 = trung bình pool, κ0 = `niw.K0`, Ψ0 = off × Σ gộp trong lớp của mọi lớp S0 có nhãn, off chọn bằng leave-one-out trên S0 với tiên nghiệm bỏ-một-lớp, đúng luật `off_lc` của E06.04); `agglomerate(n_sub, k, pair_cost)` (gộp tham lam, cache chi phí theo bộ thành viên); `merge_costs(st, sub_ids, prior)` (luật `bf`; `ward`, `bhatt`, `bhatt_mean` chỉ báo cáo, chỉ chấm tầng 1); `merged_ids`; lớp `MergedCluster` cho bước 10–14.
  Thêm 14/11 (`a4f9578`, E11.01, tiền đăng ký `03e7f86`): `stop_at_step(clf, args)`: thành phần dừng sớm cho một lượt SVI, ép lượt dừng ở bước `args.stop_at` bất kể luật det(Σ); tại bước đó `mngmm` khôi phục trạng thái trước lần cập nhật cuối, như một lần dừng tự nhiên. Từ chối khi `stop_at` ≤ `n_epochs` / 3 (chặn sẵn của `mngmm`). Chỉ dùng cho lượt S của nhánh `var@def` trong E11.01; chuỗi mặc định không gọi.
  Hàm chính: `v04a_cub_t5(st)`, `v09a_overcluster(st, k)`, `v09b_merge(st, rule, prior)`, `v09c_multi_init(st, m)`, `multi_init(x, k, m, random_state, marks)`, `stop_at_step(clf, args)`.
- `lab/score.py` (21/10, `81678ec`): đánh giá tầng 1 (QD18). `score09` cùng code với `pl_cluster.score`; `tier1_09` chạy `v09a_overcluster` với K' cụm từ trạng thái RNG ở lối vào bước 09 trong ảnh chụp của một run, chấm bằng nhãn thật của pool, kèm phép kiểm C1–C3 của E09.04, E09.05; lệnh `python lab/score.py 09 --from <run>`.
  Thêm 1/11 (`b89f1d6`): `tier1_03` cho E03.02, lệnh `python lab/score.py 03` qua `exp.py`: probe theo lớp (logistic regression thường và cân trọng số lớp, k-NN cosine k = 20, logistic regression cân trọng số trên PCA-64 fit ở S0) trên 4 bộ đặc trưng GC10 (`fz`, `ft` r0–r2), học trên vai trò S0, reserve, S1, đo trên test; recall pipeline đọc từ dòng `confusion stage 1` của 5 seed; khoảng tin cậy Wilson; phép kiểm C1, C3, C4 của E03.02. Lệnh `score.py 09` không đổi; `score.py` không nằm trong chuỗi băm của bàn mổ.
  Thêm 4/11 (`3aeb5ae`): sổ mất điểm (QD24, thiết kế ở dòng Sửa của QD24). `ledger(run)` đọc ảnh chụp của một run (từ chối ảnh chụp cũ) và trả đúng/sai từng ảnh của `test1` và `pool` theo cột bước, hiện có 09 (nhiều–một theo lớp chủ trong pool, chỉ để chẩn đoán) và 10 (tên của pipeline); `ledger_summary` gộp theo lớp, kèm luồng mất/được giữa hai cột liền nhau và hàng balanced. Không đổi lệnh nào cũ; nằm ngoài chuỗi băm.
  Thêm 5/11 (`ae132c1`): cột 11 của `ledger` (chỉ tập `test1`, phản thực tế): bộ phân lớp với `params_f` (`_predict` của pipeline, qua `load(run, "11")` và `predict_default`) đặt ảnh vào đúng khe của cụm mang tên lớp đó; `ledger` nay đòi ảnh chụp tươi tới bước 11.
  Thêm 5/11 (`4f8166d`): cột 12 của `ledger` (chỉ tập `pool`): ảnh cũ đúng khi bị bộ lọc của bước 12 loại, ảnh mới đúng khi được giữ và nằm trong cụm mang đúng tên; `ledger` nay đòi ảnh chụp tươi tới bước 12 và kiểm `keep` có đúng một giá trị mỗi ảnh pool.
  Thêm 5/11 (`73e7fae`): cột 14 của `ledger` (bộ phân lớp cuối, `params1`) cho `test1`; tập `test_old` (241 ảnh test cũ) với cột 08 (`params0`), 11 (`params_f`), 14; cột khe được chọn `slot_<k>` (không gian cụm của bảng S1) để tách cũ→mới, mới→cũ. Sổ tính từ một trạng thái sau bước 14 (`ledger_state(st)`; `ledger(run)` = kiểm ảnh chụp tới 14 rồi `load(run, "14")`), để runner của E09.07 dùng đúng hàm này; bộ phân lớp theo `st["predict"]` thay vì luôn `predict_default`. Kiểm gắn trong hàm: cột 14 khớp dòng Novel, Old của bảng S1, cột 08 khớp bảng S0, `test_old_x` = `test0_x`. `ledger_summary` thêm `novel=False` cho lớp cũ. Hồi quy: trên dữ liệu giả, cột 09–12 trùng bản cũ từng giá trị; trên 15 ảnh chụp, trung bình theo FT seed của cột 10, 11 trùng `10-naming`, `11-gauss-f` (`14-eval` §3.1).
  Thêm 11/11 (`aab1178`, E09.07): `tier2_e0907` chạy E09.07 trên một ảnh chụp `gc10_v2_ft_r<r>_s<N>`: các lần rút k (mặc định 0–10), mỗi lần m lần khởi tạo đơn (mặc định 1000) qua `variants.multi_init`; lần rút 0 đặt RNG toàn cục về `rng_in` của bước 09, lần rút k ≥ 1 dùng `RandomState(907000 + 1000 r + 100 N + k)`; nhánh mặc định là nghiệm giữ sau 10 lần khởi tạo đầu; bước 10–14 bằng code mặc định (`_tail_0907`: không ảnh chụp, không rút RNG, SVI từ `PRNGKey(0)`); đại lượng lấy từ sổ mất điểm (`_measures_0907`, qua `ledger_state`); ghi `checks_ok` (C1–C4 của tiền đăng ký). `e0907_summary`: bảng quyết định (Δ_r, Δ, SE_MC, luật của phần 1 với T, B của phần 2), tính lại T, B từ khoá `ref_*` để kiểm.
  Thêm 12/11 (`a5b4ff4`, E06.04): `tier2_e0604` chạy phiên đầu của một chuỗi bằng bốn nhánh trên cùng đặc trưng PCA và tập test: A0 mặc định (`params0` của bước 06), A1 E06.03 (`mngmm._fit_closed`), A2 NIW với Gaussian plug-in, A3 NIW với dự đoán Student-t (`lab/niw.py`); từ chối chạy khi cây bẩn (luật dirty của `exp.py`); ghi `checks_ok` (C1–C3). `e0604_summary`: bảng quyết định cho GC10 và CUB, từ chối đọc khi thiếu run hay `checks_ok` khác 1.
  Thêm 12/11 (`6996b57`, E09.06): `tier2_e0906` chạy E09.06 trên một ảnh chụp: trong mỗi lần rút, nhánh mặc định (`replay.s09_cluster`, `n_init` 10) và biến thể (`v09a_overcluster` K' = 20 rồi `v09b_merge` luật `bf`) cùng xuất phát từ một trạng thái RNG (lần rút 0: `rng_in` của bước 09; k ≥ 1: `RandomState(906000 + 1000 r + 100 N + k)`); ba luật chỉ báo cáo (`RULES_0906`) chỉ chấm tầng 1; ghi `checks_ok` (C1–C5). `e0906_summary`: bảng quyết định, điều kiện bỏ mũi khám phá, độ thuần của các bước gộp (`_purity_0906`, nhãn pool chỉ để chấm), tính lại T, B từ `ref_*`.
  Lệnh mới của `python lab/score.py`: `e0907`, `e0604`, `e0906` (chạy qua `exp.py`, mỗi chuỗi một run) và `e0907-summary`, `e0604-summary`, `e0906-summary` (chỉ đọc).
  Thêm 14/11 (`2917741`, E09.09, tiền đăng ký `cff7c46`, sửa `594e34b`): `tier1_e0909` chạy E09.09 trên một ảnh chụp GC10: 21 lần rút theo luật RNG của E09.06, chỉ tầng 1, không SVI; nhánh `bf0` là đường gộp của `bf5` cắt ở bước đầu có log BF ≤ 0; ghi cổng G1–G3. `e0909_summary`: bảng quyết định từ 25 run, luật áp trên r3, r4; r0–r2 chỉ báo cáo.
  Thêm 14/11 (`d21273a`, E11.01): `tier2_e1101` chạy E11.01 trên một ảnh chụp GC10: lần rút 0–3, ba nhánh `def`, `var` (`09a` + `09b`), `vd` (phân hoạch của `var`, lượt S ép dừng ở bước của `def` bằng `variants.stop_at_step`); ghi cổng G1–G4 và `checks_ok`. `e1101_summary`: bảng quyết định từ 25 run, từ chối đọc khi có run thiếu `checks_ok` = 1.
  Thêm 16/11 (`6d75629`, QD28): `cub_tb`: T, B của CUB từ 15 chuỗi mặc định của mỗi bộ đặc trưng; chỉ đọc `config.json` và `metrics.json`.
  Lệnh mới từ 14/11: `e0909`, `e1101` (qua `exp.py`), `e0909-summary`, `e1101-summary`, `cub-tb` (chỉ đọc).
  Thêm 16/11 (`c3cb144`, E14.01, tiền đăng ký `5c6956f`): `_best_map_1401`, `_rules_1401`, `tier2_e1401`: trên 15 chuỗi CUB của một bộ đặc trưng, bộ phân lớp của ảnh chụp `S<i>_13` dự đoán tập test của phiên, rồi chấm bằng bốn cách ghép khe với lớp (P0 = QD15, P1, P2 = SPE theo phiên, P3 = Hungarian toàn cục); cổng G1–G2. `e1401_summary`: bảng từ hai run.
  Thêm 16/11 (`a1c1d00`, E14.02, tiền đăng ký `8e90cc8`): `_hung_correct_1402`, `_tables_1402`, `tier2_e1402`: bảng đợt lớp × phiên, sổ theo phiên (c09, c10, c11, c14), dòng lỗi; đọc nhãn pool chỉ để chấm c09; cổng G1–G3. `e1402_summary`.
  Thêm 16/11 (`dd876fc`, Sửa đổi của E14.02, `89648a4`): `_tables_1402b`, `tier2_e1402b`: bộ phân lớp của phiên trước và của phiên sau dự đoán cùng ảnh test (bảng 4), xếp khe theo tỉ lệ ảnh cũ của cụm trên pool (bảng 5); cổng G1, G4; `pure` là kết quả. `e1402b_summary`.
  Ba runner trên không chạy SVI, không rút RNG, không sửa `lab/sessions.py` hay `lab/replay.py`; nạp ảnh chụp bằng `sessions.load`. Lệnh mới: `e1401`, `e1402`, `e1402b` với `--prefix cub_chain_ft` hoặc `cub_chain_ft_l25` (qua `exp.py`; `--checks-only` chạy một chuỗi, không ghi), và `e1401-summary`, `e1402-summary`, `e1402b-summary` (chỉ đọc).
  Hàm chính: `score09()`, `tier1_09()`, `wilson()`, `pipeline_recall_03()`, `tier1_03()`, `ledger()`, `ledger_state()`, `ledger_summary()`, `tier2_e0907()`, `e0907_summary()`, `tier2_e0604()`, `e0604_summary()`, `tier2_e0906()`, `e0906_summary()`, `tier1_e0909()`, `e0909_summary()`, `tier2_e1101()`, `e1101_summary()`, `cub_tb()`, `tier2_e1401()`, `e1401_summary()`, `tier2_e1402()`, `e1402_summary()`, `tier2_e1402b()`, `e1402b_summary()`, `main()`.
- `lab/data_io.py` (22/10, `8910925`, `2f6f1e5`): dữ liệu thô cho các notebook `00-data-*` (QD23): bảng ảnh của file split theo thứ tự hàng của file đặc trưng, với vai trò S0, reserve, S1, unused, test (`lab/README.vi.md` §10); hộp lỗi trong XML của một ảnh, cùng quy tắc với `scripts/make_gc10_split.py`; tính lại mục `diagnostics` của file split. Chỉ đọc file split và XML, không cần torch, jax; không run nào dùng file này. NEU-DET thêm 25/10 (`beec7b9`): đường dẫn ảnh như `build()` của `feature_extractor/dino-neudet.py`, vai trò như `dataloaders/neudet.py`, XML tra theo tên file trong cả hai thư mục `annotations` (XML của ảnh train `crazing_240` nằm trong `validation/annotations`).
  Hàm chính: `images(dataset)`, `objects(dataset, path)`, `diagnostics(dataset)`.
- `lab/00-data-gc10.ipynb` (24/10, `1e46674`): notebook dữ liệu GC10-DET (QD23 bước 2), ba mục giao thức, ảnh, nhãn. Đọc file split và XML qua `data_io.py`, và ảnh chụp bước 04 của run `gc10_v2_ft_r0_s4`. Không run nào dùng file này; notebook trong `lab/` không làm run bị dirty (`scripts/exp.py`, 15/10). Sửa 1/11 (`75a7086`): ô Kết quả của cell 0 ghi `1180a1b` như output, thay vì `c4bf75f`.
- `lab/00-data-neu.ipynb` (1/11, `0079819`): notebook dữ liệu NEU-DET (QD23 bước 2), ba mục giao thức, ảnh, nhãn. Đọc file split, ảnh và XML qua `data_io.py`; NEU chưa có ảnh chụp nên không đọc ảnh chụp. Không run nào dùng file này.
- `lab/01-preprocess.ipynb` (1/11, `50767b1`): notebook bước 01 (QD23 bước 3), bốn mục code, resize, chuẩn hoá, kiểm đầu cuối. Đọc `config.json` của 8 run phase 1, code trong `feature_extractor/`, ảnh qua `data_io.py`, và đặc trưng `fz`; tải DINO gốc từ cache Hugging Face để trích lại 128 hàng đầu. Không run nào dùng file này.
- `lab/02-finetune.ipynb` (1/11, `3db9aef`): notebook bước 02 (QD23 bước 3), ba mục thiết lập, toán của BOFT, quá trình học. Đọc `console.log` của 6 run `ft` và checkpoint `ft_ckpt.pt` của GC10 r0–r2; dựng mô hình bằng `load_model()` của `sl_finetuned_model.py` trên CPU, không huấn luyện. Không run nào dùng file này.
- `lab/03-features.ipynb` (1/11, `17f4cc5`): notebook bước 03 (QD23 bước 3), ba mục đặc trưng, `fz` và `ft`, probe theo lớp. Đọc 8 bộ đặc trưng, `config.json`, checkpoint `ft_ckpt.pt`, ảnh qua `data_io.py`; trích lại trên GPU 128 hàng đầu (GC10, NEU) và 455 ảnh test GC10 bằng mô hình dựng từ checkpoint; đọc `metrics.json` của run E03.02 (`gc10_e03_02`). Không run nào dùng file này.
- `lab/04-loader.ipynb` (4/11, `3f6e08e`): notebook bước 04 (QD23 bước 5), ba mục kiểm, code, phiên. Gọi `replay.rerun` cho bước 04 của `gc10_v2_ft_r0_s4`; đọc `snap/04.pkl` của 15 run `gc10_v2_ft_*`; chạy lại loader một lần với `np.random.permutation` được bọc để ghi các lần gọi; chạy `s09_cluster` hai lần (thứ tự gốc và một hoán vị) từ trạng thái RNG của ảnh chụp 09. Không ghi file; không run nào dùng file này.
- `lab/05-pca.ipynb` (4/11, `42c0ca4`): notebook bước 05 (QD23 bước 5, QD24), năm mục kiểm, code, phương sai, lớp mới, 15 ảnh chụp. Gọi `replay.rerun` cho bước 05 của `gc10_v2_ft_r0_s4`; fit lại PCA trên S0 của ảnh chụp (randomized từ trạng thái RNG của ảnh chụp, có và không hoán vị hàng; solver `full`); đọc `snap/04.pkl`, `snap/05.pkl` của 15 run `gc10_v2_ft_*`; nhãn thật của pool chỉ dùng để chấm (§4.1). Không ghi file; không run nào dùng file này.
- `lab/06-gauss-svi.ipynb` (4/11, `4112807`): notebook bước 06 (QD23 bước 5, QD24), bốn mục kiểm, code, params0 so với dữ liệu, 15 ảnh chụp. Gọi `replay.rerun` cho bước 06 của `gc10_v2_ft_r0_s4` (1000 bước SVI trên GPU, chế độ tất định); dựng lại lịch learning rate bằng `optax`; tính log-mật độ bằng numpy (Cholesky) để so `params0` với bộ phân lớp "dấu của tâm"; đọc `snap/04.pkl`, `snap/05.pkl`, `snap/06.pkl` của 15 run `gc10_v2_ft_*`. Không ghi file; không run nào dùng file này.
- `lab/07-early-stop.ipynb` (4/11, `0d6d7ad`): notebook bước 07 (QD23 bước 5, QD24), ba mục kiểm, code, 15 ảnh chụp. Gọi `replay.rerun` cho bước 07 của `gc10_v2_ft_r0_s4`; quét `classifier/mngmm.py` và `main.py` tìm chỗ đọc `scaling_factor`, `early_stop_ratio`; tính ngưỡng dừng từ `params0` (det float32 của JAX); đọc `snap/06.pkl` và `metrics.json` của 15 run `gc10_v2_ft_*`. Không ghi file; không run nào dùng file này.
- `lab/05a-camera.ipynb` (4/11, `75e1bf9`): notebook phân tích cho bước 05 (Q05.03, QD25), làm trước 08; bốn mục dữ liệu, PCA-64, đặc trưng thô, 15 ảnh chụp và `fz`. Đọc `snap/04.pkl`, `snap/05.pkl` của 15 run `gc10_v2_ft_*`, file đặc trưng train của `ft` r0–r2 và `fz`, bảng ảnh của `data_io.py`; gắn mỗi hàng của ảnh chụp với ảnh của nó theo byte; phân tách phương sai bằng hồi quy trên biến giả của lớp, camera, cuộn thép (R² hiệu chỉnh). Chỉ chạy trên CPU. Không ghi file; không run nào dùng file này.
- `lab/08-classifier.ipynb` (4/11, `6cad5fa`): notebook bước 08 (QD23 bước 5, QD24), năm mục kiểm, code, cơ chế, lỗi S0, 15 ảnh chụp. Gọi `replay.rerun` cho bước 08 của `gc10_v2_ft_r0_s4`; nạp trạng thái tới bước 08 bằng `replay.load`; tách log-mật độ của `params0` thành các số hạng bằng numpy float64 và so với `_predict`; gắn hàng của `test0_x` với ảnh theo byte, đọc hộp lỗi bằng `data_io.objects`; đọc `snap/04`, `05`, `06.pkl` của 15 run `gc10_v2_ft_*`. Không ghi file; không run nào dùng file này.
- `lab/landscape.py` (4/11, `1adc3e2`): landscape của k-means ở bước 09 (QD26), dùng chung cho `09-cluster` và `09d-ownership`. Đọc `snap/04.pkl` (`pool_x`) và `snap/09.pkl` (`ids`, `inertia`, `rng_in`) của một run, bỏ khoá `true_*`, từ chối ảnh chụp cũ (`replay._fresh`); chạy `KMeans(n_init=1)` của scikit-learn nhiều lần từ luồng RNG riêng (`zlib.crc32` của tên run) với 1 luồng OpenMP, hoặc `KMeans(n_init=10)` từ `rng_in` với hàng hoán vị. Ghi cache `runs/<run>/landscape/<loại>_<khoá>.npz` (git bỏ qua); khoá = băm của chuỗi băm tới 09, byte của `pool_x`, code của các hàm tạo cache và tham số. Không đổi code của pipeline, không run nào dùng file này.
  - Sửa 4/11 (`62f81a8`): `modes` so một phân hoạch với mọi nghiệm đầu nhóm trong một lần `bincount` (`ari_many`, nhanh hơn 4–5 lần, kết quả trùng); thêm `assign`, `modes_all`; `run_all` tạo tiến trình bằng spawn (fork sau khi OpenMP đã chạy có thể treo). Các hàm trong khoá cache không đổi, nên cache cũ vẫn dùng được.
- `lab/09-cluster.ipynb` (4/11, `8982942`): notebook bước 09, không đọc nhãn (QD26); bốn mục kiểm, code, landscape, cụm × camera/cuộn. Gọi `replay.rerun` cho bước 09 của `gc10_v2_ft_r0_s4`; `landscape.decompose`, `singles` (10.000 lần mỗi ảnh chụp), `permuted` (r0 s4), `modes_all` trên 15 run `gc10_v2_ft_*`; gắn hàng pool với ảnh theo byte, chỉ đọc cột `cam`, `coil` của `data_io.images`. Ghi cache `runs/<run>/landscape/` qua `landscape.py` (git bỏ qua); không run nào dùng file này.
- `lab/09d-ownership.ipynb` (4/11, `abbc3af`; kế hoạch `d282b37`): notebook phân tích có nhãn cho bước 09 (QD26), bảy mục. Cell 0 dừng nếu commit tiền đăng ký E09.07 `1ce3233` không phải tổ tiên của HEAD. Đọc `snap/04.pkl` (`pool_x`, `true_pool_y`) và `snap/09.pkl` (`ids`) của 15 run `gc10_v2_ft_*`, cache landscape qua `landscape.py`, `metrics.json` của `gc10_e09_05_v2_*`; gắn hàng pool với ảnh theo byte (`data_io.images`, `data_io.objects`); gọi `score.score09`, `KMeans` của scikit-learn (Lloyd từ tâm lớp và từ tâm cụm của ảnh chụp khác). Không ghi file; không run nào dùng file này.
- `lab/10-naming.ipynb` (5/11, `b2723cb`): notebook bước 10 (QD23 bước 5, QD24), năm mục kiểm, code, ba cách đặt tên, sổ mất điểm, Q10.03. Gọi `replay.rerun` cho bước 10 của `gc10_v2_ft_r0_s4`; đọc `snap/04.pkl`, `snap/09.pkl`, `snap/10.pkl` của 15 run `gc10_v2_ft_*`; tính lại Hungarian và hai cách đặt tên khác trên tập test (nhãn test chỉ để chấm); gọi `score.ledger`; gắn hàng pool với ảnh theo byte, đọc cột `coil` của `data_io.images`. Không ghi file; không run nào dùng file này.
- `lab/11-gauss-f.ipynb` (5/11, `e46f92d`): notebook bước 11 (QD23 bước 5, QD24), bốn mục kiểm, code, `params_f` so với dữ liệu, cột 11 của sổ mất điểm. Gọi `replay.rerun` cho bước 11 của `gc10_v2_ft_r0_s4` (1000 bước SVI trên GPU, chế độ tất định); tính det float32 bằng JAX như luật dừng; đọc `snap/05.pkl`, `snap/06.pkl`, `snap/09.pkl`, `snap/11.pkl` của 15 run `gc10_v2_ft_*`; gọi `score.ledger` và `_predict` của pipeline qua `replay.load`. Không ghi file; không run nào dùng file này.
- `lab/12-filter.ipynb` (5/11, `c331526`): notebook bước 12 (QD23 bước 5, QD24), bốn mục kiểm, code, bộ lọc so với nhãn thật, cột 12 của sổ mất điểm. Gọi `replay.rerun` cho bước 12 của `gc10_v2_ft_r0_s4`; tính lại khe của bộ phân lớp lượt F trên pool qua `replay.load` và `_predict`; tách log-mật độ thành b, q, ℓ, k bằng numpy float64; đọc `snap/04`, `05`, `09`, `11`, `12.pkl` và `metrics.json` của 15 run `gc10_v2_ft_*`; nhãn thật của pool chỉ để chấm; gọi `score.ledger`. Không ghi file; không run nào dùng file này.
- `lab/13-gauss-s.ipynb` (5/11, `a4b9f7e`): notebook bước 13 (QD23 bước 5, QD24 sửa 2026-11-05), bốn mục kiểm, ngân sách bước, quỹ đạo, bước dừng qua 15 ảnh chụp. Gọi `replay.rerun` cho bước 11 và 13 của `gc10_v2_ft_r0_s4` (SVI trên GPU, chế độ tất định) trong lúc bọc `MNGMMClassifier.calculate_metrics_on_covariances` ở mức class để ghi tham số mỗi bước (trả lại trong `finally`; không sửa file nào); chấm quỹ đạo bằng `_predict`; dựng lại lịch lr bằng `optax`; đọc `snap/04`–`13.pkl` và `metrics.json` của 15 run `gc10_v2_ft_*`. Không ghi file; không run nào dùng file này.
- `lab/14-eval.ipynb` (5/11, `58a698e`): notebook bước 14 (QD23 bước 5, QD24), bốn mục kiểm, code và paper ↔ code (D15, D25), sổ mất điểm ở 14 với T và B, lỗi theo đích và hộp XML, chuỗi theo bước và cách đặt tên khi chấm. Gọi `replay.rerun` cho bước 14 của `gc10_v2_ft_r0_s4` và `score.ledger` cho 15 run `gc10_v2_ft_*`; tính lại log-mật độ bằng float64; đọc `snap/04`–`14.pkl` và `metrics.json` của 15 run, `metrics.json` của 15 run `gc10_e09_05_v2_*`, file `test_features` của 3 bộ đặc trưng `ft`, XML qua `data_io.objects`. Không ghi file; không run nào dùng file này.
- `lab/niw.py` (12/11, `8a7fc23`): tiên nghiệm Normal–Inverse-Wishart liên hợp cho Gaussian theo lớp (QD27; E06.04, dùng lại trong chi phí gộp của `09b`, E09.06). Hậu nghiệm dạng đóng của mọi lớp (`fit`; off = ν0 − d − 1 > 0); mật độ dự đoán Gaussian plug-in của kỳ vọng hậu nghiệm (A2) hoặc Student-t (A3) (`log_density`; `predict` lấy argmax với tiên nghiệm lớp đều, như pipeline); Ψ0 = off × Σ gộp trong lớp, gộp mọi lớp hoặc bỏ lớp đang xét (`prior_cov`, `loo_class`: dạng tiên nghiệm của lớp mới ở E09.08); chọn off bằng leave-one-out (`loo_curve`: cập nhật hạng 2, một Cholesky mỗi ảnh; `choose`: hoà thì lấy off lớn hơn); log hợp lý biên (`log_evidence`, Murphy 2007, eq. 266; docstring ghi "chỉ để kiểm", nhưng `merge_costs` của `09b` dùng nó làm chi phí gộp) và tổng dự đoán tuần tự (`log_sequential`). Hằng `K0` = 0.01 (κ0), `OFFS` = lưới off 2 … 100000. `python lab/niw.py` chạy `_selftest`: ví dụ một chiều; hợp lý biên = tổng dự đoán tuần tự, cả khi n < d; leave-one-out nhanh = tính trực tiếp; off → ∞ cho LDA; luật hoà của `choose`; thời gian ở cỡ S0 của CUB.
  Hàm chính: `class_stats`, `prior_cov`, `fit`, `log_density`, `predict`, `loo_curve`, `choose`, `log_evidence`, `log_sequential`.
- `lab/09b-overcluster-merge.ipynb` (14/11, `cc4212c`): notebook của `09a` + `09b` (B13): phần lợi của `09a` + `09b` qua các bước 10–14, mất điểm lớp cũ theo bước dừng, vết gộp của `rolled_pit` và `crease`. Chỉ đọc run đã commit.
- `lab/sessions.py` (15/11, `d385095`, `4717d81`, `1cc1173`, `18d1844`, `b937d2b`; QD28): bàn mổ nhiều phiên. S0 chạy một lần (04 nạp mọi phiên, 05–08 mặc định); mỗi phiên online gọi nguyên các hàm `s09`–`s14` của `replay.py` trên một khung nhìn phiên (`session_view`); chỉ bước 10 có lớp bọc để nối bảng tên của các phiên. Không sửa `replay.py`, nên 15 ảnh chụp gốc GC10 không cũ đi. Ảnh chụp `04`–`08` rồi `S<i>_<NN>`; chuỗi băm theo (phiên, bước). Lệnh: `run` (qua `exp.py`), `c0`, `check-gc10` (C0), `check` (C1, kèm C5), `check-hash` (C4), và `load`, `rerun` cho notebook. Thêm 17/11 (`4a2967c`, đợt 3b-1, QD28 Sửa 2026-11-17): `var_step` (một bước qua `chain` với spec của riêng bước đó), `var_lut` (nối bảng tên sau bước 10; `rerun` gọi hàm này, `run_session` giữ bản của nó), `var_view_fn`, `var_session` (chạy lại phiên i trong bộ nhớ từ ảnh chụp cha, biến thể qua `--at`, kênh oracle `--view` CHỈ CHẨN ĐOÁN, khoá `sessions` lấy khỏi khung nhìn), lệnh `check-resume` (C2). Tên mới mang tiền tố `var_` để không vào mã băm qua `co_names`. `run --from`, ảnh chụp của biến thể (đợt 3b-2) và C3 làm sau. Các hàm thuộc chuỗi băm (`new_state_sessions`, `s04_sessions`, `_step`, `run_s0`, `session_view`, `run_session`, `_pack`, `_unpack`) không sửa sau chuỗi CUB đầu tiên.
  Hàm chính: `run_s0`, `session_view`, `run_session`, `session_hashes`, `load(run, upto)`, `rerun(run, name)`, `var_session(run, i, start, at, view)`.
- `replay.py` và `pl_cluster.py` nằm ở `scripts/` tới ngày 15/10 (dời ở `dd7a954`); bản cũ ở tag `nb-v1`.

**Sửa file hạ tầng từ 01/10:**

- `scripts/exp.py` (15/10, `dd7a954`): notebook và `.md` trong `lab/` không làm run bị dirty; `lab/*.py` vẫn bị kiểm.
- `requirements-lock.txt` (15/10, `e75889f`): xoá mã escape của terminal ở dòng 1 (pip install từng lỗi ở dòng này).

Ghi chú tiếng Việt: `notes/code-changes.vi.md` (file này), `notes/plan.vi.md`.
Không có file `q6-q8-giai-thich.md` (được nhắc nhưng chưa từng tạo); lý do chọn QD06–QD08 nằm ở D19.

---

## 2. Khác biệt do môi trường, không do sửa code

**BOFT butterfly factor 2 → 1.** `load_model` khai báo `boft_n_butterfly_factor=2`, nhưng peft không build
được extension CUDA `fbd_cuda` nên tự hạ xuống 1 (chỉ báo `UserWarning`). Xác nhận:
`trainable params: 221,184` (factor 2 sẽ là 405,504). **Có ảnh hưởng kết quả.** Chi tiết: A5 trong
`reproducibility-issues.md`.

---

## 3. Tra nhanh các cờ

- `FT_LAMBDA` — mặc định 0, wrapper 0: λ trong Eq. 3; > 0 bật NT-Xent và augmentation 2 view.
- `FT_TAU` — mặc định 0.2, wrapper 0.2: τ của NT-Xent, chỉ dùng khi λ > 0.
- `FT_AUG` — mặc định rỗng, wrapper rỗng: rỗng = theo `FT_LAMBDA`; 1 = bật; 0 = tắt.
- `FT_EPOCHS` — mặc định 10, wrapper 10: số epoch fine-tune.
- `FT_SEED` — mặc định 42, wrapper 42: seed phase 1.
- `FT_BATCH_SIZE` — mặc định **128**, wrapper **32**: batch fine-tune.
- `FT_ACCUM` — mặc định **1**, wrapper **4**: gradient accumulation.
- `FT_LABELED_IDX` — mặc định rỗng, wrapper rỗng: rỗng = `filter(labels < 50)` (upstream); có = `select(idx)`, sửa D9.
- `FT_FOLD` — mặc định rỗng, wrapper rỗng: rỗng = đủ 50 lớp cơ sở; F1–F5 = giữ 10 lớp làm lớp mới giả.
- `--clustering_alg` — mặc định gmm: `gmm` hoặc `kmeans`.
- `--kmeans_n_init` — mặc định 10: số lần khởi tạo k-means.
- `--gmm_n_init` — mặc định 1: số lần khởi tạo GMM (1 = upstream).
- `--hungarian_on` — mặc định train: D13: `train` = upstream (nhãn thật đặt tên cụm); `eval` = thứ tự cụm khi huấn luyện, Hungarian trên tập test khi đánh giá.
- `--print_confusion` — mặc định tắt: chẩn đoán: in ma trận nhầm lẫn test_all (id thật) sau mỗi phiên online.
- `--s0_labels` — mặc định true: `none` = S0 không nhãn (E04.01); bắt buộc kèm `--hungarian_on eval`.
- `--cov_prior_kappa` — mặc định 0: > 0 bật tiên nghiệm co Inverse-Wishart trong SVI (E06.01); 0 = MLE như upstream.
- `--cov_prior_target` — mặc định pooled: đích co của `--cov_prior_kappa`: `pooled` hoặc `sphere` = (tr/d) I.
- `--estimator` — mặc định svi: `closed` = trung bình mẫu + Ledoit–Wolf, không SVI (E06.03).
- `--mask_unseen` — mặc định tắt: chẩn đoán D24: argmax chỉ trên các lớp đã gặp.
- `--oracle_filter` — mặc định tắt: CHỈ CHẨN ĐOÁN, dùng nhãn thật: tách cũ/mới theo nhãn thật trước lượt S.
- `--print_cluster_xtab` — mặc định tắt: chẩn đoán, dùng nhãn thật: in bảng chéo cụm × lớp thật và inertia sau bảng kết quả (E09.03).
- `--dataset neudet --load_mode t1`: NEU-DET b3t1; phải kèm `--base 3 --increment 3 --num_classes 6 --num_dim 128`.
- `--dataset gc10 --load_mode t1`: GC10 b5t1; phải kèm `--base 5 --increment 5 --num_classes 10 --num_dim 64`; đặc trưng đóng băng thêm `--pretrained_model_name dino-vitb16`.

Các run Eq. 3 dùng `FT_BATCH_SIZE=16 FT_ACCUM=8`, vì 32 × 2 view = 7.68 GB > 6.93 GB khả dụng.

**Hai điểm cần biết khi dùng cờ:**
- `FT_SEED=0` **không** truyền được xuống `dino-cifar100.py`: code bỏ qua `--seed` khi giá trị là 0 (D4).
  Vì vậy tránh dùng seed 0 cho phase 1.
- Với `FT_AUG=1` và λ = 0, dòng log in `views=1` dù thực tế forward 2 view, vì trường đó đi theo
  `FT_LAMBDA`. Lỗi hiển thị đã biết, chưa sửa. Bằng chứng thật là dòng `augmentation ON` và thời gian chạy.

---

## 4. Các phép kiểm tra hồi quy đã chạy

- `--gmm_n_init`: `c100_regress_s0` so với `c100_base_s0` → trùng khít 43 chỉ số.
- `FT_SEED`: `c100_seedcheck` so với `c100_con0_probe` → CE 1.1756 trùng, đặc trưng lệch 1.15e-5 (nhiễu GPU).
- `FT_AUG`: `c100_augcheck` so với `c100_con0_probe` → `augmentation OFF`, CE trùng, lệch 1.15e-5.
- `KMeansCluster` *(sanity check, không phải hồi quy: upstream không có k-means)*: `c100_km1` so với `c100_base` (5 seed mỗi bên) → trùng 43 chỉ số. *Việc S0_All trùng nhau ở smoke test không phải bằng chứng: S0 vốn không dùng clustering*.
- `exp.py --inputs` (29/9): `t_regress_old/new/none/cli1/cli2/bad` → không cờ = hành vi cũ (`ac9fb477…`); `-` = `{}`; hash lặp lại được; sai cú pháp in usage, không tạo run.
- `exp.py append_log` (29/9): 3 run thử: không metrics / metrics trùng cột / metrics lạ → `(no metrics.json)`, `S0_All 91.50`, `acc 0.5000`.
- `predict_raw` + `hungarian_map` (29/9): 3 cụm tổng hợp, offset 50 → acc 1.0, trùng đường cũ `predict(..., with_known=True)`.
- `--hungarian_on train` (29/9): `c100_km_hungtrain_s0` so với `c100_km_s0` → trùng khít 46 chỉ số.
- `--print_confusion` (30/9): `neu_km_evhung_d128_cm_s0` so với `neu_km_evhung_d128_s0` → trùng khít mọi chỉ số.
- `make_neu_split.py` (30/9): chạy 2 lần → `unchanged`, cùng sha256.
- `SplitCountIncrementalLoader` (30/9): so với lớp cha trên đặc trưng NEU, cùng seed → mọi mảng trùng khít.
- `dino-neudet.py` tổng quát (30/9): NEU đóng băng chạy lại → max abs diff = 0.
- `make_gc10_split.py` (30/9): chạy 2 lần → `unchanged`, cùng sha256.
- `feat_probe.py` (1/10): đặc trưng GC10 đóng băng 224 → trùng số với phép thử viết tay trước đó.
- `--s0_labels` mặc định (1/10): `gc10_frozen_km_evhung_d64_regress_s0` (commit `b080aad`) so với `gc10_frozen_km_evhung_d64_s0` → `metrics.json` trùng mọi khoá; dòng ma trận nhầm lẫn trùng.
- `--cov_prior_kappa 0` (1/10): `gc10_frozen_km_evhung_d64_regress2_s0` (`f6d7841`) so với cùng run gốc → 5 chỉ số trong LOG trùng (S0_All 77.59, S1_New 52.80, final_All 60.66, M_d 52.80, M_f 9.96).
- sửa lát cắt `fe95908` (1/10): `gc10_frozen_km_evhung_d64_regress3_s0`, chạy khi thay đổi chưa commit (dirty: `classifier/mngmm.py`) → 5 chỉ số trùng.
- `--cov_prior_target` (1/10): code `bd70b17` so với run của code `f6d7841` trên NEU, d 384, κ 64 (`neu_km_evhung_d384_k64_regress_s0`) → **không trùng**: loss trùng tới bước 400, lệch từ bước 500, run thoát lỗi. Nghi do GPU không tất định ở vùng gần suy biến; không kiểm lại vì bỏ tiên nghiệm qua SVI (D23). Đường mặc định κ = 0 đã kiểm ở hai dòng trên.
- `--estimator svi` (2/10): `gc10_frozen_km_evhung_d64_regress4_s0` (`3e45d55`; dirty chỉ vì file chưa track `scripts/ckpt_vs_data.py`) → 5 chỉ số trùng.
- `--mask_unseen`, `--oracle_filter` tắt (2/10): `gc10_frozen_km_evhung_d64_regress5_s0` (`0ea29f0`) → 5 chỉ số trùng.
- `--print_cluster_xtab` tắt (2/10): `gc10_frozen_km_evhung_d64_regress6_s0` (`8aa1c53`) → trùng khít run gốc.
- Sửa ngày 2026-10-20 (kiểm lại trên tag `archive-2026-10-02`): `regress2`, `regress3`, `regress4`, `regress5` trùng run gốc `gc10_frozen_km_evhung_d64_s0` ở cả 13 khoá của `metrics.json`, không chỉ 5 chỉ số như các dòng trên ghi. `regress3` và `regress4` bị dirty, nhưng `regress5` và `regress6` sạch, ở commit mới hơn, cũng trùng, nên đường mặc định của code CT1 đã được kiểm trên cây sạch. Còn thiếu: chưa so tham số; mỗi phép kiểm chỉ 1 seed; phép kiểm `--cov_prior_target` (`bd70b17`) chưa kiểm lại (A6 cho thấy phase 2 trên GPU không lặp lại từng bit, khớp với nghi ngờ ở dòng đó [suy luận]).
- `classifier/closed_form.py` (2/10): `python scripts/test_closed_form.py` → đạt mọi assert.
- `lab/replay.py` v1 (6/10): bảng chéo K' = 5 trùng 5 run `gc10_frozen_km_evhung_d64_head_s*`, inertia lệch tương đối < 1e-6 (E09.04/C1).
- `lab/replay.py` v2 (16–18/10): QD21/C4 đạt; C1–C3 chưa chạy.
- `lab/replay.py` v2 sau `_check_09` (20/10, `3525e86`): C4 đạt. QD21/C1 đạt 15/15: chuỗi mặc định trên 15 run `gc10_v2_ft_r{0,1,2}_s{0–4}` trùng run `main.py` của E02.05 (mọi kết quả rời rạc trùng, tham số lệch dưới 1e-6 × max|giá trị|); C2, C3 đạt trên `gc10_v2_ft_r0_s0`. Output ở `runs/<run>/check_c*.txt` (commit `d62d059`).
- `lab/score.py`, `lab/variants.py` (21/10): `score09` trùng `pl_cluster.score` trên 200 phân hoạch ngẫu nhiên; tầng 1 của E09.05 chạy lại qua bàn mổ (15 run `gc10_e09_05_v2_r{0,1,2}_s{0–4}`, commit `b81b78d`) trùng run `gc10_pl01b_*` ngày 7/10 ở mọi khoá `k*_` (135 khoá mỗi run; inertia lệch tương đối < 1e-6), C1–C3 đạt: 15/15.
- `lab/replay.py` sau phép kiểm GPU và sau khi bỏ bản đầu (21/10, `1253ce6`, `370cf86`): C4 đạt; 15 ảnh chụp gốc vẫn hợp lệ (`_fresh`), tức chuỗi băm không đổi.
- `lab/replay.py` sau khi thêm `rerun` (1/11, `53d8603`): lệnh sửa tự kiểm chuỗi băm h(04)…h(14) không đổi và C4 đạt; 15/15 ảnh chụp gốc vẫn hợp lệ (`_fresh`); `rerun` trên `gc10_v2_ft_r0_s4` cho 11/11 bước trùng từng bit với ảnh chụp, dù driver GPU đã đổi từ 617.42 sang 621.01.
- `lab/data_io.py` (22/10): `images()` khớp file split (S0 764, reserve 193, S1 698, unused 178, test 455; pool 748, test lớp mới 214), thứ tự hàng khớp `build()` của `feature_extractor/dino-neudet.py` (kiểm trên cùng file split khi viết); `diagnostics()` trên XML thật trùng mục `diagnostics` của file split (466/75/6, đúng thứ tự), đọc 2282 file XML trong khoảng 1 s. File mới, không run nào dùng, nên không có phép hồi quy. NEU (25/10, `beec7b9`): phần GC10 cho kết quả không đổi (kiểm khi viết); `images("neudet")` khớp file split (S0 576, reserve 144, S1 576, unused 144, test 360), `diagnostics("neudet")` cho 123/0/0 (`00-data-neu` §1.1, §3.1).
- `lab/landscape.py` (4/11): file mới, không run nào dùng, nên không có phép hồi quy. Kiểm khi viết, trên dữ liệu đồ chơi 748 × 64 với scikit-learn 1.5.2: `ari` trùng `adjusted_rand_score` trên 200 cặp; `decompose` cho nghiệm, inertia và trạng thái RNG sau trùng `KMeans(n_init=10)`; `singles` tính lại cho cùng kết quả từng bit, đọc lại từ cache trùng; trung bình của `instability` bằng 1 − E[Rand index] tính đủ mọi cặp (lệch 8e-10); `run_all` chạy 2 tiến trình.
- `lab/landscape.py` sửa 4/11 (`62f81a8`): `modes` bản mới cho đúng nhóm và nghiệm đầu nhóm như bản cũ trên dữ liệu đồ chơi (3 ngưỡng × 2 bộ, trùng hẳn); `ari_many` trùng `adjusted_rand_score` (lệch dưới 1e-16); cache 15 ảnh chụp tính trước bản sửa vẫn đọc được vì khoá không đổi (`09-cluster` cell 4).
- `lab/variants.py` (`multi_init`, `v09c_multi_init`; `709eeca`) và `lab/score.py` (`tier2_e0907`; `aab1178`): thêm hàm, không sửa thành phần nào của chuỗi băm. C1–C4 của E09.07 đạt ở 15/15 run `gc10_e09_07_r<r>_s<N>`; C1 là phép hồi quy: ở lần rút 0, phân hoạch của nhánh mặc định trùng `ids` của ảnh chụp và số ở bước 14 trùng chuỗi mặc định (rời rạc trùng hẳn, số thực lệch tương đối ≤ 1e-6) (`notes/pl/experiments.vi.md`, E09.07, Kết quả).
- `lab/variants.py` (`v04a_cub_t5`; `440d60c`), `lab/niw.py` (`8a7fc23`), `lab/score.py` (`tier2_e0604`; `a5b4ff4`): `checks_ok` = 1 ở 30/30 run của E06.04. C1 là phép hồi quy: A0 của chuỗi lab trùng `acc0` của ảnh chụp GC10 và S0_All của run `cub_base_ft_r<r>_s<N>` cùng FT seed và seed, nên `v04a_cub_t5` cho lại đúng phiên đầu của `main.py` trên CUB; C2 kiểm `niw.py` trên dữ liệu thật của một chuỗi mỗi dataset; C3: A1 trùng `--estimator closed` (E06.04, Kết quả).
- `lab/variants.py` (`v09b_merge` và các hàm gộp; `68285e7`), `lab/score.py` (`tier2_e0906`; `6996b57`): `checks_ok` = 1 ở 30/30 run của E09.06 (C1, C2 ở 15 run có lần rút 0; C3, C4, C5 ở cả 30). C1 là phép hồi quy của nhánh mặc định; C2: `09a` cho lại số tầng 1 đã lưu của E09.05 v2; C5 tính lại mọi cặp ở mọi bước gộp bằng `niw.log_evidence` (E09.06, Kết quả).
- `feature_extractor/dino-cub200.py` với `FT_LABELED_IDX` (16/11, `4b36e47`): ba lần chạy `FT_EPOCHS=0` (code cũ; code mới không đặt biến; code mới có đặt biến) cho 4 file `.npy` trùng từng bit, `num_rows` 2996 / 2996 / 2500. Không có phép hồi quy từng bit cho phase 1 có fine-tune. Ba run `cub_ft_l25_r0`–`r2` ghi 2500 ảnh và 221,184 tham số huấn luyện được.
- `lab/score.py` (`tier1_e0909`; `2917741`): 25/25 run `gc10_e09_09_r<r>_s<N>` qua G1–G3 (`c76ffcd`). G1 là phép hồi quy: `bf5` trùng số tầng 1 của E09.06 ở mọi lần rút.
- `lab/variants.py` (`stop_at_step`; `a4f9578`) và `lab/score.py` (`tier2_e1101`; `d21273a`): 25/25 run `gc10_e11_01_r<r>_s<N>` qua G1–G4 (`e787ebe`). G1: nhánh `def` và `var` trùng E09.06 của cùng lần rút; G2: ép dừng ở chính bước dừng tự nhiên cho `params1` trùng từng bit; G3: `vd` dừng đúng bước của `def`.
- `lab/sessions.py` (15/11; QD28): C0 đạt ở 15/15 chuỗi GC10 (run `sessions_c0_gc10`, `c0_ok` = 1). C1 đạt 48/48 ở 15/15 chuỗi `cub_chain_ft_r<r>_s<N>` so với mốc `cub_km_evhung_det_ft_r<r>_s<N>` (`bf3f6a6`, `9e110bc`). C5 (trong lệnh `check`) đạt ở 15/15 chuỗi CUB. C4, phần đổi code một bước cho mọi phiên, đạt (lệnh `check-hash`, không ghi run). C2, C3 chưa có (đợt 3b). Run `cub_chain_ft_r2_s3` lần đầu hỏng vì `CUDA_ERROR_UNKNOWN` (exit −6); chạy lại thì qua C1.
- `lab/sessions.py` trên đặc trưng không rò (16/11, `ec522f8`): C1 OK 48/48 ở ba cặp `cub_chain_ft_l25_r<r>_s0` và `cub_km_evhung_det_ft_l25_r<r>_s0`; 3 mốc thay 15 (QD28, Sửa 2026-11-16).
- `lab/score.py` (`cub_tb`, lệnh `cub-tb`; `6d75629`): thêm hàm chỉ đọc `config.json` và `metrics.json`, không sửa thành phần nào của chuỗi băm. Kiểm trong hàm: mỗi bộ đủ 3 FT seed × 5 seed với exit 0, cây sạch, 5 seed của một FT seed chung mã băm đầu vào.
- `lab/score.py` (`tier2_e1401`; `c3cb144`): phần ghép tên thử trên dữ liệu giả ngoài repo (nghiệm vét cạn, ví dụ Hình 2 của bài SPE); trên dữ liệu thật, G1 (P0 tính lại trùng `acc1` của ảnh chụp và `metrics.json`) và G2 đạt ở 30/30 chuỗi (run `cub_e14_01_ft`, `cub_e14_01_ft_l25`, `f1ee1e0`).
- `lab/score.py` (`tier2_e1402`; `a1c1d00`): G1 (All, Old, New, S0test trùng `acc1` và `metrics.json`), G2 (c10 bằng nghiệm tối ưu của Hungarian), G3 đạt ở 30/30 chuỗi (run `cub_e14_02_ft_l25`, `cub_e14_02_ft`, `0706ec8`). Cột c11 không có số tham chiếu.
- `lab/score.py` (`tier2_e1402b`; `dd876fc`): G1 và G4 (bộ phân lớp của phiên trước cho đúng điểm của nó khi chấm trên ảnh của phiên sau; phiên 1 so với `acc0`) đạt ở 30/30 chuỗi (run `cub_e14_02b_ft_l25`, `cub_e14_02b_ft`, `ea43b5e`).
- Ba lần thêm trên không sửa `lab/sessions.py`, `lab/replay.py` hay các file của chuỗi băm; `sessions.load` nạp được cả 30 chuỗi sau mỗi commit.
- `lab/sessions.py` đợt 3b-1 (17/11, `4a2967c`): lệnh sửa tự so `_closure` của 21 hàm (7 hàm runner, `_unpack`, `s04_sessions`, `_step`, `session_hashes`, 11 bước mặc định) trước và sau khi sửa: trùng; ảnh chụp `cub_chain_ft_l25_r0_s0` vẫn hợp lệ (`fresh_sessions`). C2 (`check-resume`) đạt 16/16 trên `cub_chain_ft_l25_r0_s0`, 2179 s (`runs/20261116_021241__cub_chain_ft_l25_r0_s0/check_c2.txt`, `37e44a7`): chạy tiếp từ `S<i>_09` (i = 1–5) và từ bước 10–14 của phiên 1 và phiên 5, mỗi mục một tiến trình mới; RNG sau mỗi bước trùng `rng_in` của ảnh chụp kế tiếp; đối chứng âm (bước 09 của phiên 2 không đặt lại RNG) đổi phân hoạch. Chạy tiếp từ bước 10 trở đi trùng từng bit mọi khoá; chạy tiếp từ bước 09 có một khoá trong ngưỡng 1e-6 × max|giá trị| nhưng không trùng từng bit ở phiên 1, 2, 4 [suy luận: `inertia`, vì `ids` phải trùng hẳn và `cluster` so bằng `_same` ở cả hai mức]. C3 và phần C4 cho biến thể một phiên chưa có (đợt 3b-2).

---

## 5. Danh sách file thay đổi

Không chép danh sách vào đây (chép tay sẽ lệch, xem lỗi "12 file" cũ). Lệnh lấy danh sách:

    git diff --name-status upstream-baseline -- . ':(exclude)runs' ':(exclude)notes'

Kiểm 2026-11-16, sau `6d75629`: **88 dòng = 9 `M` + 79 `A`**; 4 `A` mới: `lab/09b-overcluster-merge.ipynb`, `lab/sessions.py`, `scripts/make_labeled_idx_cub.py`, `splits/cub_labeled_hf_idx.npy`, nên `lab/` có 27 file; 9 `M` không đổi. Lần kiểm 2026-11-13, sau `8a7fc23` (HEAD lúc kiểm: commit 15 run C1 của QD28): 84 dòng = 9 `M` + 75 `A`; `A` mới duy nhất là `lab/niw.py`, nên `lab/` có 25 file; 9 `M` không đổi. Lần kiểm 2026-11-05, sau khi thêm `@torch.no_grad()` vào `feature_extractor/dino-cub200.py` (QD27): 83 dòng = 9 `M` + 74 `A`; 74 `A` không đổi. Lần kiểm trước (2026-11-05, sau khi thêm `lab/14-eval.ipynb`, `58a698e`): 82 dòng = 8 `M` + 74 `A`; trong 74 `A` có 26 file split ngân sách nhãn, `scripts/` có 13 file, `lab/` có 24 file.
Nếu lệnh cho số khác, tài liệu này đã lệch.

---

## 6. Lỗi đã biết

- Trường `views=` in sai khi dùng `FT_AUG` (`sl_finetuned_model.py`, dòng in cuối epoch): **chưa sửa** (quyết định 29/9: chỉ đổi log, không đáng đụng phase 1), xem mục 3.
- `exp.py` hash `runs/inputs.txt` cho cả phase 1 (`exp.py`, `inputs_info`): **đã sửa** 29/9 (`c4088a1`, `ad7314f`). Trường `inputs` của run phase 1 **trước `ad7314f`** là vô nghĩa; không sửa `config.json` cũ vì đó là bản ghi.
- LOG ghi `(no metrics.json yet; copy from runs/INDEX.txt later)` (`exp.py`, `append_log`): **đã sửa** 29/9 (`e66e0bd`); 38 entry cũ đổi ở `98bb890`.
- LOG ghi `result:` trống khi metrics không trùng `columns.txt` (`exp.py`, `append_log`): **đã sửa** 29/9 (`e66e0bd`).
