# Theo dõi: biến thể Eq. 3 (contrastive) và Hungarian khi đánh giá

File theo dõi riêng, viết tiếng Việt, cập nhật mỗi phiên.
Chạy song song với hệ thống cũ: `notes/LOG.md` (mỗi run), `notes/paper-code-discrepancies.md`,
`notes/reproducibility-issues.md` (tiếng Anh). File này chỉ ghi **kế hoạch, câu hỏi, quyết định**.

---

## 1. Mục tiêu

1. Cài Eq. 2 + Eq. 3 của paper: `L_ft = (1-λ)·L_ce + λ·L_con`, kèm augmentation (cặp dương).
2. Tìm (λ, τ) bằng validation, **không dùng tập test**.
3. Loại rò rỉ nhãn: Hungarian chỉ dùng khi đánh giá, không dùng trong huấn luyện.
4. So với baseline `c100_base` (5 seed, `runs/agg_c100_base.txt`).

## 2. Quy tắc

- Mọi thay đổi là **tùy chọn**; mặc định = hành vi upstream (λ = 0, Hungarian trên train).
- Phase 1: bật bằng biến môi trường, wrapper phải in giá trị ra log.
- Phase 2: bật bằng tham số dòng lệnh.
- Chọn (λ, τ) **không được** nhìn tập test hoặc nhãn của lớp 50–99.
- Commit code trước mỗi run (dirty = no).
- Tag dự kiến: `c100_extract_base400`, `c100_extract_con`, `c100_con_s0..s4`, `c100_base_evhung_s0..s4`.

## 3. Kế hoạch

| Bước | Việc | GPU | Trạng thái |
|---|---|---|---|
| 0 | Xác nhận các điểm rò rỉ; viết D11–D15 | không | đang làm |
| 1 | Cài Eq. 2/3 (mặc định λ = 0) + augmentation 2 view; smoke test | vài phút | chưa |
| 2 | Tìm (λ, τ) trên validation 40/10 lớp, run ngắn | ~8h | chưa |
| 3 | Phase 1 đầy đủ: `base400` (sửa D9) và `con` với (λ, τ) đã chọn | ~3h + ~6h | chưa |
| 4 | Phase 2, 5 seed mỗi bộ đặc trưng | ~65 phút/bộ | chưa |
| B | Hungarian khi đánh giá (độc lập, dùng đặc trưng hiện có) | ~65 phút | chưa |

## 4. Quyết định cần chốt

| # | Quyết định | Đề xuất | Chốt |
|---|---|---|---|
| Q1 | Cách validation | 40 lớp đã biết giả + 10 lớp mới giả (trong lớp 0–49) | ? |
| Q2 | Tiêu chí chọn (λ, τ) | độ chính xác GMM (k = 10, Hungarian) trên 10 lớp mới giả; ràng buộc: acc 40 lớp không giảm nhiều | ? |
| Q3 | Lưới tìm kiếm | λ ∈ {0, 0.25, 0.5, 0.75} ở τ cố định, rồi τ ∈ {0.07, 0.2, 0.5}; +1 run lặp λ = 0 để đo nhiễu | ? |
| Q4 | Ngân sách | ~8h GPU cho bước 2 | ? |
| Q5 | Vector đưa vào L_con | `pooler_output`, chuẩn hóa L2, không projection head | ? |
| Q6 | Dạng loss | NT-Xent (SimCLR), 2 view mỗi ảnh | ? |
| Q7 | Augmentation | RandomResizedCrop + flip + ColorJitter; chỉ khi fine-tune | ? |
| Q8 | Bộ nhớ 8 GB | batch 32 × 2 view + gradient checkpointing (62 cặp âm) | ? |
| Q9 | Sửa D9 (400 ảnh/lớp khi fine-tune) | có, cho mọi biến thể phase 1 mới | ? |
| Q10 | Hungarian khi đánh giá: ghép trên tập nào, giữ mapping qua các phiên thế nào | chưa có đề xuất | ? |

## 5. Câu hỏi

### Đang mở
- [ ] `clustering_base.evaluate()` (dòng 76–79) có được `main.py` gọi không?
- [ ] VRAM thực tế khi 2 view: có OOM không, checkpointing chậm bao nhiêu?
- [ ] Thứ hạng (λ, τ) sau 3 epoch có giữ nguyên sau 10 epoch không?
- [ ] (λ, τ) tốt trên CIFAR-100 có chuyển được sang NEU-DET / GC10-DET không?
- [ ] Chuyển Hungarian sang đánh giá có đổi kết quả nhiều không? (dự đoán: ít, vì mọi ID lớp mới khởi tạo giống nhau — suy luận)

### Đã trả lời
- [x] **Nhãn thật đi vào huấn luyện ở đâu?** Chỉ qua Hungarian: `clustering_base._align_clusterId_labelId`
  dòng 44 (`novel_mask`), 47, 54–56 (confusion matrix), 59 (`linear_sum_assignment`). (2026-09-17)
- [x] **Nhãn test có ảnh hưởng early stop không?** Không. `mngmm.run_inference` dòng 118: điều kiện dừng
  chỉ gồm det của Σ và `step`. Acc test (dòng 124, 147) tính sau khi đã quyết định, chỉ để log. (2026-09-17)
- [x] **λ, τ có trong paper không?** Không. Paper cũng không nêu augmentation, projection head. (2026-09-17)
- [x] **Có học λ bằng gradient được không?** Không (nghiệm suy biến). τ học được nhưng tối ưu cho loss,
  không cho khám phá lớp mới → dùng validation. (2026-09-17)

## 6. Ghi chú kỹ thuật (từ đọc code)

- `sl_finetuned_model.py` dòng 67–74 tạo `transform` nhưng **không dùng**. Ảnh đã được transform sẵn
  ở `dino-cifar100.py` dòng 92–108 (`trans_func`: Resize + CenterCrop, không ngẫu nhiên).
  → Augmentation phải cài ở `dino-cifar100.py` cho `finetune_dataset` (dòng 143), không ở `sl_finetuned_model.py`.
- `dino-cifar100.py` dòng 143: `filter(lambda x: x["labels"] < 50)` lấy đủ 500 ảnh/lớp (D9).
  Lọc theo key `labels` sau `with_transform`, nên có thể phải transform cả 50000 ảnh chỉ để lọc (suy luận).
- Head phân lớp `Linear(768, 100)` (dòng 145 truyền 100) dù chỉ huấn luyện 50 lớp.
- `vision_transformer`, `DINOHead`, `utils` được import nhưng không dùng.
- InfoNCE lấy cặp âm trong **một lần forward**: gradient accumulation không tăng số cặp âm.

## 7. Nhật ký phiên

### 2026-09-17
- Baseline sạch xong: `c100_base` 5 seed, S0 91.90 ± 0.07, final 79.86 ± 1.68.
- Lập kế hoạch biến thể. Đóng 2 câu hỏi rò rỉ (nhãn thật chỉ qua Hungarian; nhãn test không ảnh hưởng early stop).
- Tiếp theo: chốt Q1–Q10, rồi bước 1.