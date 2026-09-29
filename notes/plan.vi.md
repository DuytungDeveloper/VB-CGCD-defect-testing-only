# Theo dõi: biến thể Eq. 3 (contrastive) và Hungarian khi đánh giá

File theo dõi riêng, viết tiếng Việt, cập nhật mỗi phiên.
Chạy song song với hệ thống cũ: `notes/LOG.md` (mỗi run), `notes/paper-code-discrepancies.md`,
`notes/reproducibility-issues.md` (tiếng Anh). File này chỉ ghi **kế hoạch, câu hỏi, quyết định**.

---

## 1. Mục tiêu

1. Cài Eq. 2 + Eq. 3 của paper: `L_ft = (1-λ)·L_ce + λ·L_con`, kèm augmentation (cặp dương).
2. Tìm (λ, τ) bằng validation, **không dùng tập test**.
3. Loại rò rỉ nhãn: Hungarian chỉ dùng khi đánh giá (Q10); fine-tune chỉ trên đúng tập có nhãn của phase 2 (D9).
4. So với baseline `c100_base` (5 seed, `runs/agg_c100_base.txt`).

## 2. Quy tắc

- Mọi thay đổi là **tùy chọn**; mặc định = hành vi upstream (λ = 0, Hungarian trên train, fine-tune 500 ảnh/lớp).
- Phase 1: bật bằng biến môi trường, wrapper phải in giá trị ra log.
- Phase 2: bật bằng tham số dòng lệnh.
- Chọn (λ, τ) **không được** nhìn tập test hoặc nhãn của lớp 50–99.
- **Không đổi batch size lúc trích đặc trưng (128)**: nó quyết định tập có nhãn của phase 2 (D17).
- Commit code trước mỗi run (dirty = no).
- Tag dự kiến: `c100_extract_base400`, `c100_extract_con`, `c100_con_s0..s4`, `c100_base_evhung_s0..s4`.

## 3. Kế hoạch

| Bước | Việc | GPU | Trạng thái |
|---|---|---|---|
| 0 | Xác nhận các điểm rò rỉ; viết D11–D17 | không | đang làm (rò rỉ: xong; D11–D17: chưa) |
| 0b | Tạo `splits/c100_labeled_hf_idx.npy` (20000 chỉ số HF) bằng `scripts/make_labeled_idx.py` | không | **xong** |
| 0c | Đo VRAM bằng `scripts/vram_probe.py` | ~10 phút | **xong** |
| 1 | Cài Eq. 2/3 (mặc định λ = 0) + augmentation 2 view + tùy chọn fine-tune theo file chỉ số; smoke test | vài phút | chưa |
| 2a | Tìm (λ, τ), giai đoạn 1: 7 run trên fold F5 | ~8h | chưa |
| 2b | Giai đoạn 2: λ = 0 + 1 cấu hình, trên F1–F4 (8 run). **Bỏ hẳn nếu λ\* = 0 ở 2a** | ~9h | chưa |
| 2c | Kiểm chứng giới hạn cặp âm: 1 run 32×2 view + checkpointing ở (λ*, τ*), cùng fold F5 | ~1h | chưa |
| 3a | `base400`: phase 1 fine-tune chỉ trên tập có nhãn (`FT_LABELED_IDX`) + 5 seed phase 2 (`scripts/overnight_base400.sh`) | ~4h | chạy qua đêm |
| 3b | Phase 1 `con` với (λ, τ) đã chọn + 5 seed | ~6h + 65 phút | chưa |
| 4 | Phase 2, 5 seed mỗi bộ đặc trưng | ~65 phút/bộ | chưa |
| B | Hungarian khi đánh giá (độc lập, dùng đặc trưng hiện có) | ~65 phút | chưa |

## 4. Quyết định cần chốt

| # | Quyết định | Đề xuất | Chốt |
|---|---|---|---|
| Q1 | Cách validation | 5-fold theo lớp trên lớp 0–49 (F1 = 0–9, …, F5 = 40–49), mỗi fold: 10 lớp mới giả + 40 lớp đã biết giả; thiết kế ghép cặp; 2 giai đoạn (phương án C) | **OK** |
| Q2 | Tiêu chí chọn (λ, τ) | chính: acc phân cụm (PCA 384 fit trên train 40 lớp đã biết giả → GMM k = 10, Hungarian) trên 4000 ảnh của 10 lớp mới giả, trung bình 5 seed GMM. Ràng buộc: acc của head CE trên 3200 ảnh val (40 lớp × 80) không giảm > 1.0 điểm so với λ = 0 | **OK** |
| Q3 | Lưới tìm kiếm | giai đoạn 1 trên F5: λ ∈ {0, 0.25, 0.5, 0.75} ở τ = 0.2 (4 run), rồi τ ∈ {0.07, 0.5} ở λ tốt nhất (2 run), + 1 run lặp λ = 0 để đo nhiễu = **7 run ≈ 6h**. Giai đoạn 2: λ = 0 + **1 cấu hình đã chọn**, trên F1–F4 = 8 run ≈ 9h. Chốt cấu hình trước khi chạy 2b (tránh đa kiểm định). Nếu 2 cấu hình đầu chênh nhau nhỏ hơn nhiễu (đo bằng run lặp λ = 0), chọn cái có λ nhỏ hơn | **OK** |
| Q4 | Ngân sách | **~17h** GPU cho bước 2 (15 run × ~68 phút). Sửa 2 lần: 8h → 18h (phương án C) → 17h (số đo thật từ smoke test) | **OK** |
| Q5 | Vector đưa vào L_con | `pooler_output` (= h của paper), chuẩn hóa L2, **không projection head**; module mới chỉ tạo khi λ > 0 | **OK** |
| Q6 | Dạng loss | NT-Xent (SimCLR), 2N view, đối xứng; Eq. 2 của paper viết thiếu (loss âm nếu cài đúng chữ nghĩa) | **OK** |
| Q7 | Augmentation | RandomResizedCrop(224, scale 0.5–1.0, bicubic) + HorizontalFlip + ColorJitter(0.2,0.2,0.2,0.05) p=0.8; không grayscale/blur; CE trên view 1; chỉ bật khi λ > 0 | **OK** |
| Q8 | Bộ nhớ 8 GB | **16 × 2 view (32 ảnh/forward), FT_ACCUM=8 → batch hiệu dụng 128**, không checkpointing. Đo: 32 ảnh = 4.64 GB OK; 48 = 6.18 GB sát mép; 64 = 7.68 GB OOM. 30 cặp âm mỗi neo | **OK** |
| Q9 | Sửa D9 (fine-tune chỉ trên tập có nhãn của phase 2) | có, cho mọi biến thể phase 1 mới | **OK** |
| Q9b | Cách sửa D9 | lưu file 20000 chỉ số HF (commit vào repo), fine-tune bằng `select(idx)` | **OK** |
| Q1b | Chia 400 ảnh có nhãn của 40 lớp đã biết giả | 320 train / 80 val mỗi lớp, cố định theo thứ tự file chỉ số, giống nhau cho mọi cấu hình | **OK** |
| Q1c | Quy tắc chọn ở giai đoạn 2 | trung bình Δ ghép cặp so với λ = 0 trên 5 fold; kèm số fold thắng (5/5 mới đạt p ≈ 0.03); nếu không cấu hình nào thắng nhất quán → báo cáo "Eq. 3 không giúp" | **OK** |
| Q1d | GMM trên lớp mới giả có trộn mẫu lớp đã biết không (giống S1: 25 mẫu/lớp) | không trộn cho tiêu chí chính; có trộn làm chỉ số phụ (CPU) | **OK: không trộn** |
| Q10 | Hungarian khi đánh giá | huấn luyện gán ID lớp mới theo thứ tự cụm (tất định, không dùng nhãn). Khi đánh giá: Hungarian trên khối 10 ID mới × 10 lớp mới của **tập test phiên đó**, tính **một lần** tại phiên phát hiện rồi **đóng băng** cho các phiên sau. Bật bằng `--hungarian_on eval` (mặc định `train` = upstream) | **OK** |

## 5. Câu hỏi

### Đang mở
- [ ] `utilization.gpu` khi chạy 16×2 view: nếu < 85% thì nút cổ chai là CPU (2 view, num_workers=2)
  → thử num_workers 6, có thể giảm 15–20% mỗi run (tiết kiệm 2–3h).
- [ ] CON tụt dưới ~0.1 sau 2–3 epoch? Nếu có, augmentation quá nhẹ → gradient L_con tắt sớm.
- [ ] Chỉ có 30 cặp âm mỗi neo (giới hạn 8 GB). Nếu Eq. 3 không giúp, không phân biệt được
  "Eq. 3 vô dụng" với "quá ít cặp âm" → bước 2c để kiểm.
- [ ] `clustering_base.evaluate()` (dòng 76–79) có được `main.py` gọi không?
- [ ] VRAM thực tế khi 2 view: có OOM không, checkpointing chậm bao nhiêu?
- [ ] Thứ hạng (λ, τ) sau 3 epoch có giữ nguyên sau 10 epoch không?
- [ ] (λ, τ) tốt trên CIFAR-100 có chuyển được sang NEU-DET / GC10-DET không?
- [ ] Chuyển Hungarian sang đánh giá có đổi kết quả nhiều không? (dự đoán: ít, vì mọi ID lớp mới khởi tạo giống nhau — suy luận)
- [ ] Nếu λ* = 0 (Eq. 3 đúng như paper không giúp): thử biến thể có projection head (SimCLR/GCD) không?
- [ ] D9 ảnh hưởng S1–S5 bao nhiêu? (đo bằng `c100_base` so với `base400`)
- [ ] Thời gian thật mỗi cặp (cấu hình, fold)? Ước lượng ~55 phút (suy luận), cần đo ở smoke test.
- [ ] Nhiễu fine-tune: Δ giữa 2 run λ = 0 cùng fold F5 lớn cỡ nào?

### Đã trả lời
- [x] **File chỉ số tập có nhãn có đúng không?** Có (2026-09-17). `splits/c100_labeled_hf_idx.npy`,
  int64 (20000,), sha256 `a11f243b58653747142556150c8d2522a5d28bd8e3d81c3d4f34563795b3d08e`.
  Script tự kiểm: P hoán vị, `hf[P]` == nhãn `c100_extract`, 400/lớp. Kiểm tra độc lập: đặc trưng
  lấy từ shard theo chỉ số HF == hàng có nhãn của phase 2 (9/9 mẫu, lớp 0/25/49).
- [x] **Nhãn thật đi vào huấn luyện ở đâu?** Chỉ qua Hungarian: `clustering_base._align_clusterId_labelId`
  dòng 44 (`novel_mask`), 47, 54–56 (confusion matrix), 59 (`linear_sum_assignment`). (2026-09-17)
- [x] **Nhãn test có ảnh hưởng early stop không?** Không. `mngmm.run_inference` dòng 118: điều kiện dừng
  chỉ gồm det của Σ và `step`. Acc test (dòng 124, 147) tính sau khi đã quyết định, chỉ để log. (2026-09-17)
- [x] **λ, τ có trong paper không?** Không. Paper cũng không nêu augmentation, projection head. (2026-09-17)
- [x] **Có học λ bằng gradient được không?** Không (nghiệm suy biến). τ học được nhưng tối ưu cho loss,
  không cho khám phá lớp mới → dùng validation. (2026-09-17)
- [x] **Tập 400 ảnh có nhãn có đổi theo seed không?** Không. S0 = 400 hàng đầu mỗi lớp trong `.npy`
  (`dataloader.py:309`), đúng cho seed 0 và 4. (2026-09-17)
- [x] **Mẫu lớp cũ S1–S5 lấy từ đâu?** Từ 100 hàng cuối mỗi lớp = ảnh phase 1 đã fine-tune có nhãn
  (D9 tác động trực tiếp). Pool không bị trừ (`dataloader.py:277` trả `x, y` thay vì `remained`)
  → dùng lại giữa các phiên: 2441 / 2458 mẫu (seed 0 / 4), khớp kỳ vọng 2437.
  Không sửa được: lớp cơ sở cần 125 mẫu > 100 trong pool → ghi D16. (2026-09-17)
- [x] **`merge_npy` sắp file thế nào?** Theo chuỗi: `labels_0, 1, 10, 100, 101...` (`dino-cifar100.py:44-45`).
  → Hàng `.npy` là hoán vị P của thứ tự HF. (2026-09-17)
- [x] **Ánh xạ P (hàng `.npy` → chỉ số HF) có đúng không?** Có, kiểm chứng 3 tầng (2026-09-17):
  1. 391 shard, mỗi shard `labels_b.npy` == `hf[128·b : 128·b+128]` → `shuffle=False` xác nhận bằng dữ liệu.
  2. P là hoán vị, `hf[P]` == đủ 50000 nhãn trong `labels-dino-vitb16-sl.npy`.
  3. Tập có nhãn = 20000 chỉ số HF duy nhất, tất cả nhãn < 50.
- [x] **Sửa D9 kiểu ngây thơ (400 ảnh đầu mỗi lớp theo HF) có đúng không?** Không. Chỉ trùng 15752 / 20000.
  → Sai 4248 ảnh (21%): fine-tune lẫn ảnh của pool không nhãn, bỏ sót ảnh có nhãn. Phải dùng ánh xạ P. (2026-09-17)
- [x] **Tập có nhãn có phụ thuộc batch size lúc trích không?** Có (suy luận từ cơ chế P, được dữ liệu ủng hộ):
  số shard và thứ tự sắp đổi theo batch size → ghi D17. (2026-09-17)

## 6. Ghi chú kỹ thuật (từ đọc code)

- `sl_finetuned_model.py` dòng 67–74 tạo `transform` nhưng **không dùng**. Ảnh đã được transform sẵn
  ở `dino-cifar100.py` dòng 92–108 (`trans_func`: Resize + CenterCrop, không ngẫu nhiên).
  → Augmentation phải cài ở `dino-cifar100.py` cho `finetune_dataset` (dòng 143), không ở `sl_finetuned_model.py`.
- `dino-cifar100.py` dòng 143: `filter(lambda x: x["labels"] < 50)` lấy đủ 500 ảnh/lớp (D9).
  Lọc theo key `labels` sau `with_transform`, nên có thể phải transform cả 50000 ảnh chỉ để lọc (suy luận).
  Cách sửa: `select(idx)` trên dataset gốc, không cần transform để lọc.
- Dataset: `load_dataset("uoft-cs/cifar100", split="train")`, cột nhãn `fine_label`.
- `DataLoader` lúc trích (dòng 113, 161) không có `shuffle` → mặc định False.
- Dòng 84: `torch.manual_seed(args.seed)` khi seed ≠ 0; wrapper truyền `--seed 42`.
- Head phân lớp `Linear(768, 100)` (dòng 145 truyền 100) dù chỉ huấn luyện 50 lớp.
- `vision_transformer`, `DINOHead`, `utils` được import nhưng không dùng.
- InfoNCE lấy cặp âm trong **một lần forward**: gradient accumulation không tăng số cặp âm.
- Cách dựng P: `order = sorted(tên shard)`, `P = concat(arange(128·b, 128·b+128) for b in order)`.
  Tập có nhãn (chỉ số HF) = `concat(P[where(L == k)[:400]] for k in 0..49)`, với `L = hf[P]`.
  Không cần đặc trưng, chỉ cần nhãn HF và batch size 128.

## 7. Discrepancy mới cần ghi vào `paper-code-discrepancies.md`

| Mã | Nội dung |
|---|---|
| D11 | Không có suy luận biến phân: μ, Σ là `numpyro.param`, guide rỗng → MLE, không prior, không KL |
| D12 | "Re-labeling" = lọc mẫu mới, huấn luyện pass S trên nhãn GMM gốc, bỏ mẫu lớp cũ |
| D13 | Hungarian dùng nhãn thật của dữ liệu không nhãn |
| D14 | Luật early stop khác Eq. 11 và Algorithm 1; `--early_stop_ratio` không dùng |
| D15 | M_f là proxy (tập test S0), paper định nghĩa trên D_l |
| D16 | Pool mẫu lớp cũ không bị trừ (`dataloader.py:277`) → dùng lại mẫu giữa các phiên (~2440) |
| D17 | Tập có nhãn = 400 hàng đầu theo thứ tự `.npy` sắp theo chuỗi → phụ thuộc batch size lúc trích |

## 8. Nhật ký phiên

### 2026-09-17
- Baseline sạch xong: `c100_base` 5 seed, S0 91.90 ± 0.07, final 79.86 ± 1.68.
- Lập kế hoạch biến thể. Đóng 2 câu hỏi rò rỉ (nhãn thật chỉ qua Hungarian; nhãn test không ảnh hưởng early stop).
- Q4 OK (8h GPU), Q9 OK (sửa D9).
- Đọc dataloader: tập có nhãn cố định; mẫu lớp cũ lấy từ ảnh phase 1 đã học; pool không bị trừ (D16).
- Kiểm chứng ánh xạ P 3 tầng: đạt. Sửa D9 ngây thơ sai 21% (15752 / 20000) → phải dùng P (D17).
- Chọn phương án C cho Q1 (5-fold theo lớp, ghép cặp, 2 giai đoạn); Q4 sửa thành ~18h.
  Giới hạn đã ghi nhận: fold không độc lập (Bengio & Grandvalet 2004), winner's curse → không báo cáo số validation như kết quả.
- Chốt Q9b (file chỉ số), Q1b (320/80), Q1c (Δ ghép cặp + số fold thắng), Q1d (không trộn).
- Bước 0b xong: file chỉ số, overlap với cách ngây thơ 15752 / 20000 (khớp lần đo trước).
- Q5 chốt: không projection head, theo đúng Eq. 2 (h dùng chung với Eq. 4 và Algorithm 1).
- Lỗi: `splits/*.npy` bị `.gitignore` chặn, commit 2c393c6 KHÔNG chứa file chỉ số → sửa ở commit sau.
- Q6 OK (NT-Xent), Q7 OK (augmentation mức trung bình). Q8 để sau khi đo (bước 0c).
- Giải thích chi tiết Q6–Q8: `notes/q6-q8-giai-thich.md`.
- Đo VRAM (bước 0c): 32 ảnh/forward = 4.64 GB OK; 48 = 6.18 GB sát mép; 64 = 7.68 GB OOM.
  Lưu ý: baseline batch 32 có peak alloc 4.64 GB nhưng nvidia-smi báo 6760 MiB (phần reserved ~2 GB),
  nên mốc "sát mép" coi như không dùng được.
- Chốt Q2, Q3, Q8 (16×2 view, accum 8), Q10. **Mọi quyết định đã chốt → sang bước 1.**
- Lỗi nhỏ trong `vram_probe.py`: đề xuất batch lớn nhất chạy được kể cả mốc RUI RO → đã sửa.
- Chuẩn bị chạy qua đêm bước 3a (`base400`). Lưu ý: 20000 ảnh → ~156 bước tối ưu/epoch (baseline 196);
  chênh lệch này là một phần của việc sửa D9, không tách riêng được. Tiếp theo: chốt Q2/Q3/Q5–Q8 (cần cho bước 1), viết D11–D17.

### 2026-09-18
- Bước 1 xong. Sửa: `run_phase1_extract.sh` (+15: FT_LAMBDA/TAU/EPOCHS/LABELED_IDX/FOLD + 2 guard),
  `ft_data.py` (mới: fold F1–F5, 320/80/400 mỗi lớp), `dino-cifar100.py` (+34: aug 2 view, select theo
  file chỉ số), `sl_finetuned_model.py` (+37: nt_xent, forward 2 view, log CE và CON).
- Kiểm tra: ft_data (3 tập rời nhau, phủ đúng file chỉ số, 320/lớp); nt_xent (trường hợp hoàn hảo
  0.0396 khớp công thức tay, đối xứng, gradient hữu hạn); transform (det tất định, 2 view khác nhau).
- Smoke test `c100_consmoke` (F5, λ = 0.5, 1 epoch): exit 0, 16 PASS, ckpt 888,676.
  CE 1.2269, CON 0.4779 so với mức đoán bừa log(31) = 3.43 → contrastive học ngay epoch 0.
  **26.5 phút cho 1 epoch + trích đặc trưng** → 1 run 3 epoch ≈ 68 phút, không phải 48.
- Sửa Q3/Q4: giai đoạn 2b chỉ chạy 1 cấu hình (8 run), tổng bước 2 ≈ 17h. Lý do: cắt fold sẽ
  giết sức mạnh thống kê (3 fold → p tối thiểu 0.125), cắt quét τ sẽ lẫn "Eq. 3 vô dụng" với
  "τ = 0.2 quá dễ"; cắt cấu hình ở 2b vừa rẻ vừa tránh đa kiểm định.
- Đặc trưng của `c100_consmoke` KHÔNG dùng cho phase 2 (1 epoch, 40 lớp).
- Tiếp theo: kiểm `utilization.gpu`, viết script đánh giá validation (Q2), rồi chạy 2a.