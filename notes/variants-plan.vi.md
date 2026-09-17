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
| 1 | Cài Eq. 2/3 (mặc định λ = 0) + augmentation 2 view + tùy chọn fine-tune theo file chỉ số; smoke test | vài phút | chưa |
| 2a | Tìm (λ, τ), giai đoạn 1: 7 cấu hình + 1 run lặp λ = 0, trên fold F5 | ~7–8h | chưa |
| 2b | Giai đoạn 2: λ = 0 + top-2 cấu hình, trên 4 fold còn lại (F1–F4) | ~11h | chưa |
| 3 | Phase 1 đầy đủ: `base400` (sửa D9) và `con` với (λ, τ) đã chọn | ~3h + ~6h | chưa |
| 4 | Phase 2, 5 seed mỗi bộ đặc trưng | ~65 phút/bộ | chưa |
| B | Hungarian khi đánh giá (độc lập, dùng đặc trưng hiện có) | ~65 phút | chưa |

## 4. Quyết định cần chốt

| # | Quyết định | Đề xuất | Chốt |
|---|---|---|---|
| Q1 | Cách validation | 5-fold theo lớp trên lớp 0–49 (F1 = 0–9, …, F5 = 40–49), mỗi fold: 10 lớp mới giả + 40 lớp đã biết giả; thiết kế ghép cặp; 2 giai đoạn (phương án C) | **OK** |
| Q2 | Tiêu chí chọn (λ, τ) | độ chính xác GMM (k = 10, Hungarian) trên 10 lớp mới giả; ràng buộc: acc 40 lớp không giảm nhiều | ? |
| Q3 | Lưới tìm kiếm | λ ∈ {0, 0.25, 0.5, 0.75} ở τ cố định, rồi τ ∈ {0.07, 0.2, 0.5}; +1 run lặp λ = 0 để đo nhiễu | ? |
| Q4 | Ngân sách | ~18h GPU cho bước 2 (sửa từ 8h, do chọn phương án C) | **OK** |
| Q5 | Vector đưa vào L_con | `pooler_output` (= h của paper), chuẩn hóa L2, **không projection head**; module mới chỉ tạo khi λ > 0 | **OK** |
| Q6 | Dạng loss | NT-Xent (SimCLR), 2 view mỗi ảnh | ? |
| Q7 | Augmentation | RandomResizedCrop + flip + ColorJitter; chỉ khi fine-tune | ? |
| Q8 | Bộ nhớ 8 GB | batch 32 × 2 view + gradient checkpointing (62 cặp âm) | ? |
| Q9 | Sửa D9 (fine-tune chỉ trên tập có nhãn của phase 2) | có, cho mọi biến thể phase 1 mới | **OK** |
| Q9b | Cách sửa D9 | lưu file 20000 chỉ số HF (commit vào repo), fine-tune bằng `select(idx)` | **OK** |
| Q1b | Chia 400 ảnh có nhãn của 40 lớp đã biết giả | 320 train / 80 val mỗi lớp, cố định theo thứ tự file chỉ số, giống nhau cho mọi cấu hình | **OK** |
| Q1c | Quy tắc chọn ở giai đoạn 2 | trung bình Δ ghép cặp so với λ = 0 trên 5 fold; kèm số fold thắng (5/5 mới đạt p ≈ 0.03); nếu không cấu hình nào thắng nhất quán → báo cáo "Eq. 3 không giúp" | **OK** |
| Q1d | GMM trên lớp mới giả có trộn mẫu lớp đã biết không (giống S1: 25 mẫu/lớp) | không trộn cho tiêu chí chính; có trộn làm chỉ số phụ (CPU) | **OK: không trộn** |
| Q10 | Hungarian khi đánh giá: ghép trên tập nào, giữ mapping qua các phiên thế nào | chưa có đề xuất | ? |

## 5. Câu hỏi

### Đang mở
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
- Còn mở: Q2, Q3, Q6–Q8, Q10. Tiếp theo: chốt Q2/Q3/Q5–Q8 (cần cho bước 1), viết D11–D17.