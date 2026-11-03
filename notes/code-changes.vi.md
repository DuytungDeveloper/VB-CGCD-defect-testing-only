# Thay đổi code so với upstream — VB-CGCD fork

Mốc so sánh **duy nhất**: tag `upstream-baseline` (daihao42/VB-CGCD). Cập nhật 2026-10-19: thêm các thay đổi từ 1/10 tới 18/10; mã theo QD22 (mã cũ: `notes/old-codes.vi.md`); bảng đổi thành danh sách. Cập nhật 2026-10-20: `_check_09` của `lab/replay.py` và kết quả QD21/C1–C4 (§1c, §4); sửa ghi chú về các phép kiểm hồi quy của CT1 (§4). Cập nhật 2026-10-21: `lab/variants.py`, `lab/score.py`; xoá `lab/pl_cluster.py` và bản đầu của bàn mổ; phép kiểm GPU trong `replay.py run` (§1c, §4, §5). Cập nhật 2026-10-22: `lab/data_io.py` (§1c, §4, §5). Cập nhật 2026-10-24: `lab/00-data-gc10.ipynb` (§1c, §5). Cập nhật 2026-11-01: xoá `scripts/probe.py`, `scripts/vram_probe.py`, `scripts/ckpt_vs_data.py`; `lab/data_io.py` đọc NEU-DET; `lab/00-data-neu.ipynb`; sửa ô Kết quả của cell 0 trong `lab/00-data-gc10.ipynb` (§1c, §4, §5); `lab/01-preprocess.ipynb`, `lab/02-finetune.ipynb` (§1c, §5); `lab/score.py` thêm `tier1_03` cho E03.02 (§1c); `lab/03-features.ipynb` (§1c, §5). Cập nhật 2026-11-03: `rerun` trong `lab/replay.py` (§1c, §4).

Nguyên tắc: **mặc định trong code Python = hành vi upstream**. Wrapper phase 1 đặt
`FT_BATCH_SIZE=32 FT_ACCUM=4` vì giới hạn 8 GB; cấu hình này đổi số ở mức nhỏ (PATCH 2).

Cách chia nhóm dựa trên `git diff --name-status upstream-baseline` (xem mục 5), không viết tay.

---

## 1a. Sửa file có sẵn của upstream (8 file)

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

## 1b. File mới, ảnh hưởng kết quả (9 file, cộng 26 file split ngân sách nhãn)

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
  Hàm chính: `v09a_overcluster(st, k)`.
- `lab/score.py` (21/10, `81678ec`): đánh giá tầng 1 (QD18). `score09` cùng code với `pl_cluster.score`; `tier1_09` chạy `v09a_overcluster` với K' cụm từ trạng thái RNG ở lối vào bước 09 trong ảnh chụp của một run, chấm bằng nhãn thật của pool, kèm phép kiểm C1–C3 của E09.04, E09.05; lệnh `python lab/score.py 09 --from <run>`.
  Thêm 1/11 (`b89f1d6`): `tier1_03` cho E03.02, lệnh `python lab/score.py 03` qua `exp.py`: probe theo lớp (logistic regression thường và cân trọng số lớp, k-NN cosine k = 20, logistic regression cân trọng số trên PCA-64 fit ở S0) trên 4 bộ đặc trưng GC10 (`fz`, `ft` r0–r2), học trên vai trò S0, reserve, S1, đo trên test; recall pipeline đọc từ dòng `confusion stage 1` của 5 seed; khoảng tin cậy Wilson; phép kiểm C1, C3, C4 của E03.02. Lệnh `score.py 09` không đổi; `score.py` không nằm trong chuỗi băm của bàn mổ.
  Hàm chính: `score09()`, `tier1_09()`, `wilson()`, `pipeline_recall_03()`, `tier1_03()`, `main()`.
- `lab/data_io.py` (22/10, `8910925`, `2f6f1e5`): dữ liệu thô cho các notebook `00-data-*` (QD23): bảng ảnh của file split theo thứ tự hàng của file đặc trưng, với vai trò S0, reserve, S1, unused, test (`lab/README.vi.md` §10); hộp lỗi trong XML của một ảnh, cùng quy tắc với `scripts/make_gc10_split.py`; tính lại mục `diagnostics` của file split. Chỉ đọc file split và XML, không cần torch, jax; không run nào dùng file này. NEU-DET thêm 25/10 (`beec7b9`): đường dẫn ảnh như `build()` của `feature_extractor/dino-neudet.py`, vai trò như `dataloaders/neudet.py`, XML tra theo tên file trong cả hai thư mục `annotations` (XML của ảnh train `crazing_240` nằm trong `validation/annotations`).
  Hàm chính: `images(dataset)`, `objects(dataset, path)`, `diagnostics(dataset)`.
- `lab/00-data-gc10.ipynb` (24/10, `1e46674`): notebook dữ liệu GC10-DET (QD23 bước 2), ba mục giao thức, ảnh, nhãn. Đọc file split và XML qua `data_io.py`, và ảnh chụp bước 04 của run `gc10_v2_ft_r0_s4`. Không run nào dùng file này; notebook trong `lab/` không làm run bị dirty (`scripts/exp.py`, 15/10). Sửa 1/11 (`75a7086`): ô Kết quả của cell 0 ghi `1180a1b` như output, thay vì `c4bf75f`.
- `lab/00-data-neu.ipynb` (1/11, `0079819`): notebook dữ liệu NEU-DET (QD23 bước 2), ba mục giao thức, ảnh, nhãn. Đọc file split, ảnh và XML qua `data_io.py`; NEU chưa có ảnh chụp nên không đọc ảnh chụp. Không run nào dùng file này.
- `lab/01-preprocess.ipynb` (1/11, `50767b1`): notebook bước 01 (QD23 bước 3), bốn mục code, resize, chuẩn hoá, kiểm đầu cuối. Đọc `config.json` của 8 run phase 1, code trong `feature_extractor/`, ảnh qua `data_io.py`, và đặc trưng `fz`; tải DINO gốc từ cache Hugging Face để trích lại 128 hàng đầu. Không run nào dùng file này.
- `lab/02-finetune.ipynb` (1/11, `3db9aef`): notebook bước 02 (QD23 bước 3), ba mục thiết lập, toán của BOFT, quá trình học. Đọc `console.log` của 6 run `ft` và checkpoint `ft_ckpt.pt` của GC10 r0–r2; dựng mô hình bằng `load_model()` của `sl_finetuned_model.py` trên CPU, không huấn luyện. Không run nào dùng file này.
- `lab/03-features.ipynb` (1/11, `17f4cc5`): notebook bước 03 (QD23 bước 3), ba mục đặc trưng, `fz` và `ft`, probe theo lớp. Đọc 8 bộ đặc trưng, `config.json`, checkpoint `ft_ckpt.pt`, ảnh qua `data_io.py`; trích lại trên GPU 128 hàng đầu (GC10, NEU) và 455 ảnh test GC10 bằng mô hình dựng từ checkpoint; đọc `metrics.json` của run E03.02 (`gc10_e03_02`). Không run nào dùng file này.
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

---

## 5. Danh sách file thay đổi

Không chép danh sách vào đây (chép tay sẽ lệch, xem lỗi "12 file" cũ). Lệnh lấy danh sách:

    git diff --name-status upstream-baseline -- . ':(exclude)runs' ':(exclude)notes'

Kiểm 2026-11-01, sau khi thêm `lab/03-features.ipynb` (`17f4cc5`): **68 dòng = 8 `M` + 60 `A`**; trong 60 `A` có 26 file split ngân sách nhãn, `scripts/` có 13 file, `lab/` có 10 file. Lần kiểm trước (2026-11-01, sau khi thêm `lab/02-finetune.ipynb`): 67 dòng = 8 `M` + 59 `A`.
Nếu lệnh cho số khác, tài liệu này đã lệch.

---

## 6. Lỗi đã biết

- Trường `views=` in sai khi dùng `FT_AUG` (`sl_finetuned_model.py`, dòng in cuối epoch): **chưa sửa** (quyết định 29/9: chỉ đổi log, không đáng đụng phase 1), xem mục 3.
- `exp.py` hash `runs/inputs.txt` cho cả phase 1 (`exp.py`, `inputs_info`): **đã sửa** 29/9 (`c4088a1`, `ad7314f`). Trường `inputs` của run phase 1 **trước `ad7314f`** là vô nghĩa; không sửa `config.json` cũ vì đó là bản ghi.
- LOG ghi `(no metrics.json yet; copy from runs/INDEX.txt later)` (`exp.py`, `append_log`): **đã sửa** 29/9 (`e66e0bd`); 38 entry cũ đổi ở `98bb890`.
- LOG ghi `result:` trống khi metrics không trùng `columns.txt` (`exp.py`, `append_log`): **đã sửa** 29/9 (`e66e0bd`).
