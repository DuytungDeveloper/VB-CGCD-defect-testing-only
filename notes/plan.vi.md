# Kế hoạch, quyết định, câu hỏi — VB-CGCD fork

Viết tiếng Việt, cập nhật mỗi buổi làm việc. Đây là **nơi duy nhất** chứa câu hỏi mở của repo
(`notes/LOG.md` chỉ trỏ về đây). Không chép số liệu vào file này: trỏ tới entry LOG, `runs/agg_*.txt`
hoặc mã D-xx trong `notes/paper-code-discrepancies.md`. Tên cũ: `notes/variants-plan.vi.md` (đổi 29/9).

---

## 1. Mục tiêu

1. **GĐ1 — Eq. 2/3 (contrastive fine-tuning): đã đóng 27/9.** Kết luận ở D19.
2. **GĐ2 — loại hai rò rỉ nhãn trên CIFAR-100: đã đóng 30/9.** D13 sửa và đo (Q10b); D9 sửa, đo 1 lần,
   chưa kết luận, không lặp lại (phương án B). **Giao thức sạch cho mọi run từ đây:**
   `--hungarian_on eval` + fine-tune chỉ trên tập có nhãn.
3. **GĐ3 — dữ liệu công nghiệp (đang làm).** NEU-DET chạm trần ở mọi mức nhãn: chỉ dùng kiểm tra tính đúng đắn
   và ca quên inclusion. **GC10-DET là benchmark chính** (D21, D22).
4. **Hướng chung (chốt 30/9):** đồ án và paper là một mạch — VB-CGCD cải tiến cho khám phá lỗi liên tục,
   có **trục lượng nhãn** (đầy đủ → giữa → không nhãn). Việc tiếp: S0 không nhãn, lặp FT seed, ablation dừng sớm.

## 2. Quy tắc

- Mọi thay đổi là **tùy chọn**; mặc định = hành vi upstream (λ = 0, Hungarian trên train, fine-tune 500 ảnh/lớp).
- Phase 1: bật bằng biến môi trường, wrapper phải in giá trị ra log.
- Phase 2: bật bằng tham số dòng lệnh.
- Chọn (λ, τ) **không được** nhìn tập test hoặc nhãn của lớp 50–99.
- **Không đổi batch size lúc trích đặc trưng (128)**: nó quyết định tập có nhãn của phase 2 (D17).
- Commit code trước mỗi run (dirty = no).
- Tag GĐ2 (đã chạy): `c100_km_evhung_s0..s4` (D13); `c100_extract_base400` + `c100_km_evhung_base400_s0..s4` (D9).
- Từ GĐ3: mọi run phase 2 dùng `--hungarian_on eval`; phase 1 fine-tune chỉ trên tập có nhãn (sửa D9).
- Phase 1 luôn gọi qua `exp.py run TAG --inputs - -- …` (phase 1 không đọc `runs/inputs.txt`).

## 3. Kế hoạch

Cập nhật 29/9. Không ghi số liệu ở đây; xem mã tham chiếu.

| Bước | Việc | Trạng thái |
|---|---|---|
| 0 | Xác nhận các điểm rò rỉ; viết D11–D17 | xong (17/9) |
| 0b | File chỉ số tập có nhãn (`scripts/make_labeled_idx.py`) | xong (17/9) |
| 0c | Đo VRAM (`scripts/vram_probe.py`) | xong (17/9) |
| 1 | Cài Eq. 2/3, augmentation 2 view, fine-tune theo file chỉ số | xong (18/9) |
| 2a | Lưới (λ, τ) trên fold F5, 3 epoch | xong (20/9); không dự đoán được kết quả full-scale (D19) |
| 2b | Giai đoạn 2 trên F1–F4 | bỏ, thay bằng lưới full-scale 3 seed phase 1 mỗi ô (D19). *Suy luận từ LOG: không có run F1–F4* |
| 2c | Kiểm giới hạn 30 cặp âm | chưa làm; vẫn là giới hạn mở của D19. *Suy luận từ LOG* |
| 3a | `base400`: fine-tune chỉ trên tập có nhãn (sửa D9) | xong 30/9; 1 phase 1, chưa kết luận (D9) |
| 3b, 4 | Phase 1 `con` + 5 seed phase 2 | thay bằng lưới full-scale, xong 27/9 (D19) |
| B | Hungarian khi đánh giá (sửa D13) | xong 29/9 (Q10b, D13) |
| N1 | NEU-DET: split, phase 1, loader, baseline b3t1 5 seed | xong 30/9 (`num_dim` 128, D21) |
| N2 | NEU: ablation đồng xuất hiện | chưa |
| G1 | GC10: kiểm tra dữ liệu, split theo cuộn, loader theo số ảnh từng lớp (D22) | xong 30/9 |
| G2 | GC10: baseline đóng băng; GMM, 224×448, chuẩn hoá camera | xong 1/10 (cả ba bị loại) |
| G3 | GC10: pipeline fine-tune đầy đủ | xong 1/10 (một lần phase 1) |
| G4 | GC10: S0 không nhãn | chưa |

## 4. Quyết định

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

**Đã thay thế (ghi 29/9; các dòng trên giữ làm lịch sử):**
- Q2: tiêu chí chính chuyển từ GMM sang k-means `n_init=10` ngày 19/9 (D18, `scripts/eval_fold.py`).
- Q3/Q4: bỏ giai đoạn 2b trên F1–F4; thay bằng lưới full-scale với 3 seed phase 1 mỗi ô (D19).
- Q7: augmentation không còn gắn với λ > 0; cờ `FT_AUG` tách hai thứ (nhánh `augonly`, D19).
- Q10 → **Q10b** (29/9): ánh xạ lấy từ dự đoán của cụm trên tập test, không từ classifier; không phải
  sửa `run()`. Trung bình không đáng kể nhưng mỗi lần lệch tốn khoảng 1% New (D13). Xem lại trên NEU.

## 5. Câu hỏi
Danh sách duy nhất của repo. Câu đã trả lời ghi kèm nơi có bằng chứng; không chép số liệu.

### GĐ2 (CIFAR-100): đã đóng 30/9
- [x] **`clustering_base.evaluate()` có được gọi?** Không; code chết (grep toàn repo, 29/9).
- [x] **Hungarian khi đánh giá (D13) đổi bao nhiêu?** Trung bình ≈ 0; lệch ở 1/25 phiên (phiên có cụm gộp). Chi tiết D13.
- [x] **D9 (`base400`) ảnh hưởng bao nhiêu?** Tăng nhẹ nhưng chưa kết luận (1 phase 1, một phần do seed 1); M_f giảm chỉ vì S0 thấp hơn. Chi tiết D9. Không lặp lại (phương án B).
- [x] **Khoảng cách S1_New so với paper có do D13?** Không; S1_New không đổi.

### Mang sang, không chặn GĐ3
- [ ] Khoảng cách S1_New so với paper. Manh mối: S1_New gắn với độ cân bằng cụm ở S1 (note LOG `c100_km_evhung_base400_s4`, D9).
- [ ] D9 với n ≥ 3, kèm đối chứng "400 ảnh ngẫu nhiên" để tách rò rỉ khỏi kích thước dữ liệu. Chỉ khi cần số cho báo cáo.
- [ ] Cố định Σ = I ở phase 2 thì còn bao nhiêu? Mahalanobis có thực sự đóng góp? (D11, D20.4, Bảng 15)
- [ ] logdet của lớp mới tăng hay giảm theo từng bước? Cần log logdet theo bước (Hình 2b so với D14).
- [ ] `--scaling-factor 1.2` tác động vào đâu trong luật dừng D14?
- [ ] Code có tái lập Bảng 5 "w/o fine-tuning" (S0 = 88.54)? Cần wrapper `c100_extract_noft`.
- [ ] D21: công thức CUB của upstream có n = 25 < d = 384; số CUB của paper đến từ cấu hình nào?
- [ ] `exp.py agg`: cột `paper` cần tra theo bộ dữ liệu (hiện luôn là số CIFAR).

### GĐ3 (NEU-DET): đã trả lời 30/9
- [x] **Dữ liệu?** NEU-DET có split sẵn, nhãn theo thư mục, ảnh xám lưu RGB (3 kênh giống nhau). Khoảng 7% ảnh chứa nhiều loại lỗi: giữ nhãn thư mục (giao thức chuẩn).
- [x] **Chia lớp?** b3t1 theo chữ cái, k = 3 (tránh k = 1).
- [x] **Chia train/test, số mẫu?** `splits/neu_split.json` (80/20 mỗi lớp, seed 0); `validation` làm test.
- [x] **Tiền xử lý, backbone?** Như CIFAR: DINO ViT-B/16 + BOFT, fine-tune chỉ trên ảnh có nhãn.
- [x] **`num_dim` 384 có ổn?** Không: Σ sụp, NaN ở S0 (D21). Dùng 128.
- [x] **Fine-tune trên 576 ảnh có quá khớp?** CE train về gần 0 nhưng S0 test vẫn tuyệt đối: không thấy hại.
- [x] **DINO tách lỗi thép tốt đến đâu?** Probe hoàn hảo trên 6 lớp; khám phá lớp mới chạm trần, giữ lớp cũ thì không (note LOG `neu_km_evhung_d128_s4`).
- [x] **Ánh xạ Hungarian có mơ hồ trên NEU?** Không đo trực tiếp (chỉ chạy eval), nhưng S1_New tuyệt đối nên không mơ hồ; Q10b hay Q10 không quan trọng trên NEU.
- [x] **Quên do covariance rộng (b) hay đồng xuất hiện (c)?** Chủ yếu (c): mọi ảnh bị quên là inclusion, phần lớn sang scratches (note LOG `neu_km_evhung_d128_cm_s0`); (b) có thể góp phần nhỏ.

### GĐ3 GC10: đã trả lời 30/9–1/10
- [x] **Benchmark chính?** GC10-DET: khám phá không nhãn thấp xa cận trên có giám sát, còn NEU chạm trần ở mọi mức nhãn (note LOG `gc10_extract_frozen`, `neu_extract_frozen`).
- [x] **Chia dữ liệu?** Theo cuộn thép, để khung hình gần trùng không rơi vào cả train lẫn test; bỏ ảnh trùng điểm ảnh ở hai thư mục (`splits/gc10_split.json`).
- [x] **Loader cho dữ liệu mất cân bằng?** Số ảnh theo từng lớp lấy từ file split (D22).
- [x] **k-means hay GMM?** Như nhau về balanced novel; nút thắt là đặc trưng (note LOG `gc10_frozen_gmm10_evhung_d64_s4`).
- [x] **Độ phân giải cao hơn?** Giúp proxy nhưng làm hỏng pipeline; giữ 224 (note LOG `gc10_fr224x448_km_evhung_d64_s4`).
- [x] **Chuẩn hoá theo camera?** Camera có mặt trong đặc trưng, nhưng trừ trung bình không vượt nhiễu k-means (chẩn đoán 10 seed, không lưu run).
- [x] **Fine-tune bằng nhãn lớp cơ sở?** Đánh đổi: giữ lớp cũ tốt hơn, khám phá lớp mới kém hơn (note LOG `gc10_ft_km_evhung_d64_s4`).

### Đang mở — GĐ3
- [ ] S0 không nhãn: điểm thứ ba của trục lượng nhãn.
- [ ] Đánh đổi ổn định–thích nghi có vững qua các FT seed không (ít nhất 2 seed nữa)?
- [ ] Dừng sớm: không kích hoạt trên NEU, kích hoạt giữa chừng trên GC10 fine-tune (D14). Tắt dừng sớm thì khám phá lớp mới có khá hơn không?
- [ ] Tách yếu tố trộn pooler/CLS: trích CLS từ mô hình đã fine-tune.
- [ ] Ước lượng k trên GC10 (silhouette thất bại): giao thức không biết trước k.
- [ ] Proxy giống pipeline hơn: PCA fit trên lớp cơ sở, 64 chiều.
- [ ] NEU: ablation đồng xuất hiện (oracle), và vì sao patches → pitted_surface không gây quên.
- [ ] Đưa balanced accuracy vào chỉ số chính (hiện chỉ có qua `cm_summary`).

### Đang mở — phân tích, ưu tiên thấp
- [ ] Một cụm có hút hết mẫu lớp cũ ở S1 không?
- [ ] Trong khoảng 4000 mẫu được lọc là "mới" mỗi phiên, bao nhiêu dương tính giả / âm tính giả?
- [ ] New acc có thấp hơn khi `cluster_max_min` cao?
- [ ] S4_New ổn định, S5_New không: có đánh đổi giữa hai phiên? (r = −0.55, n = 5, không có ý nghĩa)
- [ ] S0_All tăng theo số bước SVI thế nào?
- [ ] Giới hạn D19: 30 cặp âm mỗi neo, không projection head; biến thể SimCLR có head chưa thử.

### Đã trả lời
- [x] **Số của paper có nằm trong khoảng 5 seed?** Có với chỉ số cuối; không với S1_All, S1_New, S2_Old (`runs/agg_c100_base.txt`, LOG `c100_base_s4`).
- [x] **Đặc trưng trích lại có giống hệt bản lưu trữ?** Không, nhưng phase 2 cho số giống hệt (LOG `c100_extract`).
- [x] **PATCH 5 có hoạt động?** Có (LOG `c100_extract`).
- [x] **Tham số lớp cũ có đứng yên qua các phiên?** Có với các lớp đã dò (D12).
- [x] **Σ học được gần I hay gần covariance thật?** Gần I (D11); Hình 5 của paper cũng vậy (D20.4). Chưa so với covariance thực nghiệm.
- [x] **Clustering hay số lần khởi tạo quyết định kết quả?** Số lần khởi tạo quyết định độ chính xác, thuật toán quyết định độ ổn định (D18).
- [x] **Vì sao bước dừng của pass S giảm dần qua các phiên?** Giải thích khớp dữ liệu ở D14 (suy luận; chưa log det theo bước).
- [x] **M_f có đúng định nghĩa trên D_l?** Không, là proxy trên tập test S0 (D15).
- [x] **Eq. 3 có giúp?** Không cải thiện chỉ số liên tục; lợi ích nhỏ ở S0 đến từ augmentation (D19).
- [x] **Nhiễu phase 1 bao nhiêu?** Đo bằng `FT_SEED` 42–46 (D19, mục "statistical unit").
- [x] **CON có tụt về ~0? VRAM khi 2 view? Thời gian mỗi run?** D19 và LOG.
- [x] **Thứ hạng (λ, τ) sau 3 epoch có giữ ở 10 epoch?** Không; lưới F5 không dự đoán được full-scale (D19, limits).
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

### Bỏ (hết liên quan sau D19)
- `utilization.gpu` khi chạy 2 view; (λ, τ) có chuyển sang NEU không; nhiễu λ = 0 trên F5 (thay bằng `FT_SEED`).

## 6. Ghi chú đọc code (17/9)

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

## 7. Nhật ký

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
- Giải thích chi tiết Q6–Q8: file `notes/q6-q8-giai-thich.md` chưa từng được tạo; lý do Q6–Q8 xem D19.
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

### 2026-09-19 → 2026-09-27 (tóm tắt; chi tiết trong LOG, D18, D19)
- 19/9: thiết kế 2×2 clustering (D18); k-means `n_init=10` là cấu hình tốt nhất (`c100_km`); tiêu chí validation chuyển sang k-means.
- 19–20/9: lưới F5 (bước 2a) xong; không dự đoán được kết quả full-scale.
- 20–27/9: `FT_SEED` 42–46 đo nhiễu phase 1; lưới full-scale (λ, τ); nhánh `augonly` tách augmentation khỏi L_con.
  Kết luận ở D19. **GĐ1 đóng.**

### 2026-09-29
- Rà soát tài liệu và hạ tầng: `exp.py --inputs` (`c4088a1`, `ad7314f`); LOG `(no metrics.json)` (`e66e0bd`, `98bb890`);
  xoá `scripts/__init__.py` (`98d690f`); viết lại `code-changes.vi.md` (`fab822c`); đổi tên file này (`98ce817`);
  README (`65eb142`); D9/D13 status + D20 (`fd9969c`); phạm vi Part B (`96b8ec7`).
- Quyết định: câu hỏi mở chỉ ở file này. Thứ tự: GĐ2 (D13, D9) trên CIFAR-100 trước NEU-DET.

### 2026-09-30
- D13: so ghép cặp theo seed xong; chốt Q10b (D13).
- D9: `c100_extract_base400` + 5 seed phase 2; 1 phase 1, chưa kết luận (D9, note LOG `c100_km_evhung_base400_s4`).
- Quyết định (phương án B): không lặp lại D9; đóng GĐ2; sang GĐ3 (NEU-DET) với giao thức sạch.

### 2026-09-30 (GĐ3)
- NEU-DET: split cố định, phase 1 (chỉ ảnh có nhãn), loader b3t1, baseline 5 seed.
- `num_dim` 384 → NaN ở S0 (D21); dùng 128. Dừng sớm không kích hoạt trên NEU (D14).
- Quên tập trung ở inclusion (ma trận nhầm lẫn): giả thuyết đồng xuất hiện (c) được ủng hộ.
- Đợt tài liệu GĐ2 chưa commit làm mọi run NEU bị `dirty` (chỉ README); đã commit và push. `console.log` không vào git (.gitignore:212): bằng chứng chép vào note LOG và D21; log sao lưu ở `E:/AI_Storage/backups`.

### 2026-10-01 (GĐ3 GC10)
- GC10-DET làm benchmark chính: kiểm tra dữ liệu (lệch lớp, ảnh đa lỗi, nhãn lỗi, ảnh trùng, cuộn thép), split theo cuộn, loader theo số ảnh từng lớp (D22).
- Chẩn đoán đặc trưng: GMM, 224×448 và chuẩn hoá camera đều bị loại; nút thắt là hình học của không gian đặc trưng.
- Luật dừng sớm chạy ở hai nhánh det tuỳ thang đo đặc trưng (D14).
- Trục lượng nhãn: điểm đầy đủ (fine-tune) và điểm giữa (đóng băng) đã xong; fine-tune đổi khả năng khám phá lấy độ bền. Còn điểm không nhãn.
- Bài học: proxy k-means dự đoán sai pipeline (224×448); một lần chạy k-means đơn lẻ gây ảo tưởng (chuẩn hoá camera).
