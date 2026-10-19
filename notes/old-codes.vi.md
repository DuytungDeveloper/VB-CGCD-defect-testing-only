# Tra mã cũ

Hệ mã hiện hành: QD22 (`notes/pl/README.vi.md` §3), từ 2026-10-18. Từ ngày này mọi file `.md` trong repo chỉ dùng mã mới.
Mã cũ còn ở ba nơi: commit message, chú thích trong code (mục 5), bản `.md` cũ trong lịch sử git. Gặp mã cũ ở đó thì tra file này.

- Hai đợt mã cũ:
  - trước 2026-10-15: `PL-Q<n>`, `PL-E<nn>`, `PL-QD<n>`, `M1`–`M13`, `Q1`–`Q10b` của `plan.vi.md`, mã của CT1 và CT2;
  - 2026-10-15 → 2026-10-17 (QD20): đã có dạng `Q<bước>.<nn>`, `E<bước>.<nn>` nhưng số khác bây giờ; file này ghi kèm "(15–17/10)".
- Mỗi dòng: mã mới = mã cũ: nội dung ngắn [trạng thái ngày 2026-10-18]. Dòng ghi tên file thay cho mã cũ là mục chưa có mã, chuyển vào từ file đó.

## 1. Câu hỏi (`notes/pl/questions.vi.md`)

Đánh số lại một lần ngày 2026-10-18: trong mỗi bước, câu còn mở trước, câu đã xong hoặc bỏ sau; mỗi nhóm theo ngày mở.

- Q00.01 = `plan.vi.md` §5: NEU, ablation đồng xuất hiện [mở]
- Q00.02 = Q00.01 (15–17/10) = PL-Q6: GC10 khó vì mất cân bằng hay vì lỗi nhỏ [mở]
- Q02.01 = `plan.vi.md` §5: D9 với ít nhất 3 lần phase 1 [mở]
- Q02.02 = `plan.vi.md` §5: tái lập Bảng 5 "w/o fine-tuning" [mở]
- Q02.03 = `plan.vi.md` §5: giới hạn của E02.02 (30 cặp âm, không projection head) [tạm dừng]
- Q02.04 = `plan.vi.md` §5: đánh đổi ổn định–thích nghi qua các FT seed [xong]
- Q02.05 = `questions.vi.md`, mục bỏ: EMA teacher, consistency [bỏ]
- Q03.01 = `plan.vi.md` §5: tách pooler và CLS [mở]
- Q03.02 = Q3 của CT2: crease, rolled_pit có tách được khỏi waist_folding không [mở]
- Q03.03 = Q03.01 (15–17/10) = PL-Q5: đặc trưng mức vùng lỗi [mở]
- Q04.01 = `questions.vi.md`, mục tạm dừng: confirmation bias qua nhiều phiên [tạm dừng]
- Q04.02 = `plan.vi.md` §5: S0 không nhãn [xong]
- Q05.01 = `plan.vi.md` §5: D21, số CUB của paper đến từ cấu hình nào [mở]
- Q05.02 = Q4 của CT2: phép chiếu học từ nhãn lớp cũ (LDA, làm trắng trong lớp) [mở]
- Q06.01 = `plan.vi.md` §5: cố định Σ = I ở phase 2 [mở]
- Q06.02 = `plan.vi.md` §5: S0_All tăng theo số bước SVI [mở]
- Q06.03 = H2 của CT1: điều chuẩn mạnh hơn có cứu được CIFAR không [mở]
- Q06.04 = `questions.vi.md`, mục tạm dừng: hiệp phương sai low-rank/sparse [tạm dừng]
- Q07.01 = `plan.vi.md` §5: logdet của lớp mới theo từng bước [mở]
- Q07.02 = `plan.vi.md` §5: tác động của `--scaling-factor 1.2` [mở]
- Q07.03 = `plan.vi.md` §5: tắt dừng sớm [mở]
- Q08.01 = `improvement-1.vi.md` §7: D24, paper có chỉ dự đoán trên lớp đã gặp không [mở]
- Q08.02 = Q08.01 (15–17/10) = PL-Q8: inclusion bị kéo về lớp cũ [mở]
- Q09.01 = `plan.vi.md` §5: khoảng cách S1_New so với paper (CIFAR) [mở]
- Q09.02 = `plan.vi.md` §5: ước lượng k trên GC10 [mở]
- Q09.03 = `plan.vi.md` §5: New và `cluster_max_min` [mở]
- Q09.04 = Q09.02 (15–17/10) = PL-Q2: OT với biên nới lỏng [mở]
- Q09.05 = Q09.03 (15–17/10) = PL-Q4: nhãn tối thiểu cho lớp mới [mở]
- Q09.06 = Q09.04 (15–17/10) = PL-Q7: phương pháp phân cụm khác k-means [mở]
- Q09.07 = `plan.vi.md` §5: k-means hay GMM [bỏ]
- Q09.08 = `plan.vi.md` §5: proxy giống pipeline hơn [xong]
- Q09.09 = `plan.vi.md` §5: một cụm có hút hết ảnh cũ không [xong]
- Q09.10 = `improvement-1.vi.md` §7: khám phá lớp mới có phụ thuộc bộ phân loại không [xong]
- Q09.11 = Q09.01 (15–17/10) = PL-Q1: ở K' = 20, lớp hiếm có cụm riêng không [xong]
- Q10.01 = Q1c của CT2: đổi tên cụm [xong]
- Q10.02 = `questions.vi.md`, mục bỏ: ánh xạ nhãn giả động [bỏ]
- Q12.01 = `plan.vi.md` §5: dương tính giả, âm tính giả của bộ lọc (CIFAR) [mở]
- Q12.02 = `improvement-1.vi.md` §7: bộ lọc không dùng nhãn [mở]
- Q12.03 = Q12.01 (15–17/10) = PL-Q3: re-labeling đúng như paper (D12) [mở]
- Q14.01 = `plan.vi.md` §5: cột paper của `exp.py agg` theo bộ dữ liệu [mở]
- Q14.02 = `plan.vi.md` §5: balanced accuracy vào chỉ số chính [mở]
- Q14.03 = `plan.vi.md` §5: S4_New và S5_New (CIFAR) [mở]
- Q14.04 = Q14.01 (15–17/10) = PL-Q9: tầng 1 và tầng 2 lệch ở khâu nào [mở]

## 2. Thí nghiệm (`notes/pl/experiments.vi.md`)

Đánh số theo thứ tự thời gian trong mỗi bước; gồm cả thí nghiệm của GĐ1–GĐ3, CT1, CT2.

- E01.01 = G2 của `plan.vi.md` §3: GC10 đóng băng, ảnh 224×448
- E02.01 = bước 2a của `plan.vi.md` §3: lưới (λ, τ) trên fold F5
- E02.02 = bước 3b, 4 của `plan.vi.md` §3 (thay bước 2b): Eq. 3 full-scale, nhiễu FT_SEED, augonly
- E02.03 = bước 3a của `plan.vi.md` §3: sửa D9, `base400`
- E02.04 = G3 của `plan.vi.md` §3: GC10 fine-tune so với đóng băng, FT seed 42, 0, 1
- E02.05 = "nhóm A": cấu hình paper chuẩn hoá, FT seed r0–r2, GC10 và NEU
- E02.06 = "label budget (B)", "label budget (C)" trong LOG: ngân sách nhãn ở phase 1
- E03.01 = G2 của `plan.vi.md` §3: GC10, chuẩn hoá đặc trưng theo camera
- E04.01 = G4 của `plan.vi.md` §3: GC10, S0 không nhãn
- E05.01 = N1 của `plan.vi.md` §3: NEU, `num_dim` 384 so với 128
- E06.01 = CT1 v1, v2: tiên nghiệm Inverse-Wishart qua SVI
- E06.02 = thí nghiệm A của CT1: tăng ngân sách bước (lr ×10, ×100)
- E06.03 = B của CT1 (B1, B2 là hai bản): ước lượng dạng đóng
- E09.01 = D18: bảng 2×2 phân cụm trên CIFAR-100
- E09.02 = G2 của `plan.vi.md` §3: GC10 đóng băng, GMM thay k-means
- E09.03 = Q1 của CT2 (= "H2-Q1" trong LOG): bảng chéo cụm × lớp
- E09.04 = E09.01 (15–17/10) = PL-E01: phân cụm thừa, `fz`
- E09.05 = E09.02 (15–17/10) = PL-E01b: phân cụm thừa, `ft`
- E09.06 = E09.03 (15–17/10) = PL-E02: gộp cụm con
- E10.01 = B của `plan.vi.md` §3 (D13): Hungarian khi đánh giá

## 3. Quyết định

Một chuỗi cho cả repo, theo thời gian. QD01–QD15 ở `notes/plan.vi.md` §4; từ QD16 ở `notes/pl/README.vi.md` §5.

- QD01 = Q1: cách validation (5-fold theo lớp, ghép cặp, 2 giai đoạn)
- QD02 = Q2: tiêu chí chọn (λ, τ)
- QD03 = Q3: lưới tìm kiếm
- QD04 = Q4: ngân sách GPU
- QD05 = Q5: vector đưa vào L_con
- QD06 = Q6: dạng loss (NT-Xent)
- QD07 = Q7: augmentation
- QD08 = Q8: bộ nhớ 8 GB
- QD09 = Q9: sửa D9
- QD10 = Q9b: cách sửa D9
- QD11 = Q1b: chia 400 ảnh có nhãn
- QD12 = Q1c: quy tắc chọn ở giai đoạn 2
- QD13 = Q1d: GMM trên lớp mới giả không trộn mẫu lớp đã biết
- QD14 = Q10: Hungarian khi đánh giá
- QD15 = Q10b: ánh xạ lấy từ dự đoán của cụm trên tập test
- QD16 = PL-QD1: trọng tâm nhãn giả, nhãn thật tối thiểu
- QD17 = PL-QD2: `.md` là bản chuẩn
- QD18 = PL-QD3: notebook ba lớp, thứ tự làm
- QD19 = PL-QD4: mốc so sánh là cấu hình paper
- QD20 = PL-QD5: bố cục `lab/`, mã theo bước, khung notebook
- QD21 = PL-QD6: bàn mổ v2
- QD22: hệ mã này (mới)

## 4. Mã cục bộ và tên gọi khác

- I1–I3 (phép kiểm toàn vẹn) → C1–C3 của thí nghiệm chứa nó: E09.03, E09.04, E09.05.
- C1–C4 của PL-QD6 → QD21/C1–C4.
- P1–P4 (dự đoán của E09.03) → E09.03/P1–P4. Giả thuyết A, B của E09.03 → "giả thuyết sức chứa", "giả thuyết chồng lấn". R0, R1 giữ nguyên (tên hai mode phân hoạch trong E09.03).
- H1, H2: trong LOG, H2 là hướng 2 (→ CT2; "H2-Q1" → E09.03). Trong CT1 là hai giả thuyết: H1 = khe lớp chưa gặp (→ D24, Q08.01; chỉ còn ở chú thích `classifier/mngmm.py`, cờ `--mask_unseen`) [suy luận từ chú thích đó và commit `0ea29f0`]; H2 = n ≈ d (`improvement-1.vi.md`, → Q06.03).
- Tên dòng việc: cải tiến 1 → CT1; cải tiến 2, hướng 2 → CT2; "cải tiến B" → E06.03.
- "run E1" → các run `gc10_frozen_km_evhung_d64_head_s0`–`s4` (2026-10-04).
- "nhóm A" → 30 run phase 2 của E02.05.
- "phương án C" (QD01) và "phương án B" (2026-09-30, không lặp lại D9) → bỏ chữ cái, ghi bằng lời.
- Bước kế hoạch của `plan.vi.md` §3: 2a → E02.01; 2b → bỏ (thay bằng E02.02); 2c → Q02.03; 3a → E02.03; 3b, 4 → E02.02; B → E10.01; N1 → E05.01; N2 → Q00.01; G2 → E09.02, E01.01, E03.01; G3 → E02.04; G4 → E04.01. Các bước 0, 0b, 0c, 1, G1 là việc chuẩn bị, không có mã.
- Mã module M1–M13 → bước: M1 → 00 · M2 = 01 · M3 = 02 · M4 = 03 · M5 = 04 · M6 = 05 · M7 = 06 (và 11, 13) · M8 = 07 · M9 = 08 (và 12, 14) · M10 = 09 · M11 = 10 · M12 = 12 · M13 = 14.

## 5. Mã cũ còn trong code

QD22 chỉ sửa `.md`; chú thích, docstring và help của cờ trong code giữ nguyên (kiểm bằng grep ngày 2026-10-19):
- `main.py`: "improvement 1" (= CT1); "(v1)", "(v2)" ở help của `--cov_prior_target` (= hai bản của E06.01); "(B)", "# B" (= E06.03).
- `classifier/mngmm.py`: "improvement 1" (= CT1); "v1", "v2" (= hai bản của E06.01); "B:" (= E06.03); "diagnostic H1" (= D24).
- `classifier/closed_form.py`: "improvement 1 redesign, B" (= CT1, E06.03).
- `scripts/test_closed_form.py`: B1 (= bản 1 của E06.03, bộ ước lượng), cả dòng in cuối "B1 ok".
- `feature_extractor/dino-cifar100.py`: Q7 (= QD07). `feature_extractor/ft_data.py`: Q1b (= QD11). `feature_extractor/sl_finetuned_model.py`: Q6 (= QD06).
- `lab/pl_cluster.py`: PL-E01 (= E09.04), PL-E01b (= E09.05), I1, I3 (= C1, C3), M10 (= bước 09). File này nghỉ khi đã có `score.py` và bàn mổ v2 qua QD21/C1–C4.
- `lab/replay.py`: PL-QD6 (= QD21); "the E1 runs" ở phần v1 (= run `gc10_frozen_km_evhung_d64_head_s*`). C1–C4 trong docstring đã đúng nghĩa QD21/C1–C4. Sửa khi đụng tới file này lần tới.
- Các file code khác chỉ dùng D<n>, A<n>, PATCH <n>.
