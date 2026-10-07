# Thí nghiệm của dòng PL

Mỗi thí nghiệm một mục: tiền đăng ký (không sửa sau khi commit), rồi Kết quả, Đánh đổi, Kết luận. Quy trình ở `README.vi.md` §4.

## PL-E01 — Phân cụm thừa: lớp hiếm có cụm riêng không? (PL-Q1)

### Tiền đăng ký (2026-10-05; commit: e02f0af)

- Câu hỏi (PL-Q1): trên đặc trưng DINO đóng băng của GC10, nếu k-means có nhiều cụm hơn số lớp mới (K' > 5), crease hoặc rolled_pit có sở hữu một cụm chứa ít nhất một nửa số ảnh của nó không?
- Làm rõ PL-Q1: "chiếm đa số" hiểu là lớp đông nhất trong cụm (điều kiện sở hữu, như Q1 ở `notes/improvement-2.vi.md`).
- Giả thuyết: lỗi ở Q1 (crease, rolled_pit nằm trong cụm của waist_folding; silk_spot chia 3 cụm) là do thiếu sức chứa (uniform effect của k-means; Xiong, Wu & Chen 2009, IEEE TSMC-B), không phải do đặc trưng không tách được các lớp này.
- Dữ liệu: pool S1 đúng như pipeline, 748 ảnh (silk 420, inclusion 135, waist 94, crease 32, rolled 17, ảnh cũ 50); đặc trưng `gc10_fz_cls5first_img80pct`, 768 chiều thô như bước phân cụm của pipeline.
- Cách làm: script offline gọi lại loader → PCA của S0 → k-means đúng thứ tự RNG như `main.py` (PCA ở S0 dùng solver randomized nên cũng rút RNG; PCA chỉ gọi để giữ thứ tự, phân cụm vẫn trên 768 chiều). Với mỗi seed 0–9, chạy k-means (`n_init` 10) với K' ∈ {5, 10, 15, 20, 30}, mỗi K' bắt đầu từ cùng một trạng thái RNG. Nhãn thật chỉ dùng để chấm.
- Đại lượng (n_ck = số ảnh lớp c trong cụm k, từ bảng chéo `contingency_matrix`):
  - c sở hữu k nếu c là lớp đông nhất trong k (tính cả 5 lớp cũ).
  - best(c) = max n_ck / n_c trên các cụm c sở hữu; bằng 0 nếu c không sở hữu cụm nào.
  - Kèm theo: số cụm c sở hữu, tổng recall trên các cụm c sở hữu, độ thuần n_ck / n_k của cụm tốt nhất (purity; Manning, Raghavan & Schütze 2008, §16.3), inertia, CV cỡ cụm.
  - Độ phân giải: một ảnh rolled = 5.9 điểm recall (100/17), một ảnh crease = 3.1 điểm (100/32). Ngưỡng 50% nghĩa là rolled ≥ 9 ảnh, crease ≥ 16 ảnh.
- Phép kiểm toàn vẹn; chỉ đọc kết quả K' > 5 khi cả ba đạt:
  - I1: ở K' = 5, seed 0–4, inertia trùng khít inertia mà các run `gc10_frozen_km_evhung_d64_head_s*` in ra (792075.8125, 789687.4375, 790486.1875, 788922.1875, 790444.375). Nếu trượt: dừng, chẩn đoán thứ tự RNG.
  - I2: pool đúng 748 ảnh với số ảnh từng lớp như trên.
  - I3 (đối chứng dương): ở K' = 5, waist_folding có best ≥ 50% ở mọi seed (Q1: 92/94 ảnh trong một cụm).
- Dự đoán [suy luận, xác suất chủ quan khoảng 60%]: ở K' = 20, ít nhất một trong hai lớp đạt; rolled_pit dễ đạt hơn crease (probe tuyến tính: rolled 67, crease 45).
- Quy tắc quyết định (chốt ở K' = 20; các K' khác chỉ để mô tả):
  - Một lớp đạt nếu best ≥ 50% ở ít nhất 6/10 seed.
  - Sống: crease hoặc rolled_pit đạt → PL-E02, gộp cụm con theo khoảng cách Bhattacharyya, ngưỡng hiệu chỉnh trên lớp cơ sở có nhãn.
  - Loại: cả hai không đạt → bỏ hướng phân cụm thừa trên đặc trưng CLS đóng băng; chuyển sang câu hỏi đặc trưng (phase 1) hoặc Active GCD (PL-Q4).
- Ngoài phạm vi: đặc trưng fine-tune (làm sau nếu cần).
- Tag: `gc10_pl01_s<N>`, N = 0–9; mỗi run ghi mọi K' vào `metrics.json`.

### Sửa đổi (2026-10-06, quyết định 2026-10-05; trước khi chạy; phần tiền đăng ký ở trên giữ nguyên)

- I1 đổi thành: ở K' = 5, seed 0–4, bảng chéo lớp × cụm (contingency table) của bản phát lại (`scripts/replay.py`) trùng khít bảng mà các run `gc10_frozen_km_evhung_d64_head_s*` in ra; inertia chỉ cần lệch tương đối < 1e-6 so với các giá trị ở I1 cũ.
- Lý do: `scripts/replay.py` cho bảng chéo trùng khít ở cả 5 seed, nhưng inertia lệch đúng 1 ULP float32 (0.0625 ở độ lớn khoảng 790 nghìn); seed bị lệch đổi theo thứ tự import thư viện (lần đầu seed 1–3, sau khi đổi thứ tự import là seed 0). [suy luận] Bit cuối của inertia phụ thuộc thứ tự cộng số dấu phẩy động (Goldberg 1991), không phụ thuộc luồng RNG.
- Tiêu chí mới đủ cho I1: mọi đại lượng quyết định của PL-E01 (sở hữu, best, độ thuần) là hàm của bảng chéo. Bảng trùng chưa chắc phân hoạch trùng từng ảnh (đổi chỗ hai ảnh cùng lớp giữa hai cụm cho cùng bảng), nhưng điều đó không đổi kết quả PL-E01.
- Nguồn tham chiếu: bảng chéo chỉ có trong `console.log` của các run trên (không vào git); phần có trong git là inertia ở I1 cũ và cỡ cụm từng seed (`notes/improvement-2.vi.md`, Q1). Vì vậy `metrics.json` của PL-E01 phải ghi bảng chéo K' = 5 của bản phát lại.
- Sửa đổi làm trước khi có bất kỳ dữ liệu nào của PL-E01 (K' > 5 chưa chạy).

### Kết quả (2026-10-07; run `gc10_pl01_s0`–`s9`, commit 89e94d4)

- Cài đặt: `scripts/pl_cluster.py` gọi `scripts/replay.py` (`stop="cluster"`, nên SVI ở S0 vẫn chạy) và chụp trạng thái RNG ở lối vào `KMeansCluster.fit`; mọi K' bắt đầu từ trạng thái đó. Thứ tự rút RNG như tiền đăng ký; I1 xác nhận.
- Toàn vẹn: I1 (seed 0–4: bảng chéo trùng run E1, inertia lệch tương đối < 1e-6; seed 5–9: phân hoạch K' = 5 trùng pipeline), I2, I3 đạt ở cả 10 seed.
- Quyết định ở K' = 20, số ảnh trong cụm tốt nhất theo seed 0–9:
  - rolled_pit: 13, 0, 0, 0, 0, 0, 0, 11, 0, 0 → đạt 2/10 seed.
  - crease: 8, 15, 13, 13, 0, 14, 0, 8, 16, 13 → đạt 1/10 seed.
  - Cả hai không đạt: LOẠI.
- Kèm theo ở K' = 20 (trung bình 10 seed): rolled sở hữu 0.2 cụm, độ thuần trung vị của cụm tốt nhất 0.25 (2 seed có cụm); crease best 10.0/32, tổng trên các cụm sở hữu 11.6/32, độ thuần trung vị 0.41.
- Mô tả theo K' = 5, 10, 15, 20, 30 (số seed đạt): crease 0, 7, 4, 1, 1; rolled 0, 0, 2, 2, 0. Balanced accuracy ghép một–một trên pool: 0.366, 0.407, 0.391, 0.342, 0.282.
- Dự đoán (ít nhất một lớp đạt; rolled dễ hơn crease): sai.

### Đánh đổi

- Chốt ở K' = 20 tránh vườn rẽ nhánh, nhưng bỏ qua việc crease đạt nhiều nhất ở K' = 10; số đó chỉ để mô tả.
- best(c) chỉ tính cụm lớn nhất, nên lớp bị chẻ thành nhiều cụm nhỏ bị đánh giá thấp; bước gộp có thể nhặt lại.

### Kết luận

- LOẠI theo quy tắc: trên đặc trưng CLS đóng băng, phân cụm thừa không cho rolled_pit hay crease cụm riêng; khi có, cụm đó lẫn nhiều lớp khác. Đây là bối cảnh phụ (PL-QD4); kết luận về hướng lấy theo PL-E01b.

## PL-E01b — Phân cụm thừa trên cấu hình paper (PL-Q1, PL-QD4)

### Tiền đăng ký (2026-10-07; commit: 885e2dc)

- Câu hỏi (PL-Q1, trên cấu hình paper): với đặc trưng fine-tune trên tập có nhãn, nếu k-means có K' > 5 cụm, crease hoặc rolled_pit có sở hữu một cụm chứa ít nhất một nửa số ảnh của nó không?
- Vì sao có thí nghiệm này: PL-QD4 đặt mốc so sánh là cấu hình paper; PL-E01 (`fz`) là bối cảnh phụ. Kết luận về hướng phân cụm thừa lấy theo PL-E01b.
- Giả thuyết: như PL-E01 (thiếu sức chứa do hiệu ứng đồng đều; Xiong, Wu & Chen 2009). Ở `ft` r0 seed 4 với K = 5, 27/32 ảnh crease và 16/17 ảnh rolled nằm chung cụm do waist sở hữu (`notes/nb/04_s1_discovery.ipynb`).
- Dữ liệu: pool S1 đúng như pipeline (748 ảnh, thành phần như PL-E01); đặc trưng `gc10_ft_cls5first_img80pct_r0`, `_r1`, `_r2` (`dino-vitb16-sl`, 768 chiều, đúng đầu vào của bước phân cụm).
- Cách làm: với mỗi FT seed r ∈ {0, 1, 2} và mỗi seed 0–9, `scripts/replay.py` (`stop="cluster"`) chạy lại pipeline tới hết M10; trạng thái RNG ở lối vào `KMeansCluster.fit` được chụp từ bên ngoài (`notes/nb/12_cluster.ipynb` cell 8). Với mỗi K' ∈ {5, 10, 15, 20, 30}, khôi phục trạng thái đó rồi chạy `KMeansCluster` (`n_init` 10). Nhãn thật chỉ dùng để chấm.
- Đại lượng: như PL-E01 (sở hữu, best(c), các đại lượng kèm theo), tính bằng `scripts/pl_cluster.py`. Ngưỡng 50% nghĩa là rolled ≥ 9/17 ảnh, crease ≥ 16/32 ảnh.
- Phép kiểm toàn vẹn; chỉ đọc kết quả K' > 5 khi cả ba đạt:
  - I1: ở K' = 5, phân hoạch chạy lại từ trạng thái đã chụp trùng phân hoạch của pipeline trong cùng lần phát lại; với seed 0–4, cỡ cụm (sắp xếp) còn phải trùng cỡ cụm in trong `console.log` của run `gc10_ft_cls5first_img80pct_r<r>_km_evhung_d64_s<N>` (nhóm A, 2026-10-05).
  - I2: pool đúng 748 ảnh với thành phần như PL-E01.
  - I3 (đối chứng dương): ở K' = 5, waist_folding có best ≥ 50% ở mọi cặp (FT seed, seed).
- Dự đoán [suy luận, xác suất chủ quan khoảng 50%]: ở K' = 20, ít nhất một trong hai lớp đạt. Không đoán lớp nào dễ hơn: trên `ft` cả hai cùng nằm trong cụm của waist.
- Quy tắc quyết định (chốt ở K' = 20; các K' khác chỉ để mô tả):
  - Trong một FT seed, một lớp đạt nếu best ≥ 50% ở ít nhất 6/10 seed.
  - Một lớp đạt PL-E01b nếu đạt ở ít nhất 2/3 FT seed.
  - Sống: crease hoặc rolled_pit đạt → PL-E02 (gộp cụm con), làm trên cấu hình paper.
  - Loại: cả hai không đạt → bỏ hướng phân cụm thừa cho cấu hình paper, dù PL-E01 cho kết quả nào; chuyển sang PL-Q7 (phương pháp khác ở M10), PL-Q5 (đặc trưng) hoặc PL-Q4.
- Ngoài phạm vi: phương pháp phân cụm khác (PL-Q7); bước gộp (PL-E02).
- Tag: `gc10_pl01b_r<r>_s<N>`, r = 0–2, N = 0–9 (30 run); mỗi run ghi mọi K' vào `metrics.json`.

### Kết quả (2026-10-07; run `gc10_pl01b_r0_s0`–`gc10_pl01b_r2_s9`, commit 89e94d4)

- Toàn vẹn: I1 (phân hoạch K' = 5 trùng pipeline; seed 0–4: cỡ cụm trùng run nhóm A), I2, I3 đạt ở cả 30 run.
- Quyết định ở K' = 20, số ảnh trong cụm tốt nhất theo seed 0–9:
  - rolled_pit r0: 13, 0, 11, 12, 12, 0, 0, 0, 12, 10 (6/10); r1: 13, 13, 14, 0, 0, 13, 11, 12, 12, 14 (8/10); r2: 11, 0, 8, 13, 0, 12, 12, 8, 0, 14 (5/10) → đạt 2/3 FT seed.
  - crease r0: 19, 19, 19, 19, 20, 15, 20, 14, 17, 19 (8/10); r1: 14, 13, 15, 14, 26, 15, 15, 15, 14, 11 (1/10); r2: 14, 16, 15, 15, 15, 16, 23, 19, 23, 20 (6/10) → đạt 2/3 FT seed.
  - Cả hai đạt: SỐNG.
- Kèm theo ở K' = 20 (trung bình 10 seed, r0/r1/r2): độ thuần trung vị của cụm tốt nhất: rolled 0.48/0.40/0.46, crease 0.75/0.71/0.62. Riêng crease r1: best 15.2/32 nhưng tổng trên các cụm sở hữu 20.2/32 (trung bình 1.5 cụm).
- Mô tả theo K' = 5, 10, 15, 20, 30 (số lần đạt trên 30 run): rolled 0, 2, 3, 19, 21; crease 15, 12, 19, 15, 11.
- Balanced accuracy ghép một–một trên pool ở K' = 5 (trung bình 10 seed): r0/r1/r2 0.497/0.491/0.392, so với `fz` 0.366. Cùng cấu hình, balanced novel cuối pipeline trên test (nhóm A, 5 seed) là 39.4/29.2/36.2, so với `fz` 37.6: tính trung bình, tầng 1 xếp `ft` trên `fz`, tầng 2 xếp ngược lại (PL-Q9).
- Dự đoán (ít nhất một lớp đạt): đúng.

### Đánh đổi

- Biên mỏng: rolled r0 và crease r2 đạt đúng 6/10. Kết luận SỐNG chịu được một seed đổi kết quả, không chịu được hai seed đổi đúng ở hai ô này. crease r1 trượt sát ngưỡng (9/10 seed có 13–15 ảnh).
- Cụm của rolled lẫn nhiều (độ thuần trung vị 0.40–0.48), nên bước gộp có nguy cơ nuốt lại.
- Khi chưa gộp, phân cụm thừa làm giảm balanced accuracy một–một trên pool (r0: 0.497 ở K' = 5, 0.413 ở K' = 20), nên bước gộp là bắt buộc.
- Kết luận ràng buộc theo `ft` đúng với cấu hình paper, nhưng thừa hưởng nhiễu giữa các FT seed (crease r1).

### Kết luận

- SỐNG theo quy tắc, ràng buộc cho cấu hình paper (PL-QD4): với K' = 20, sức chứa đủ để rolled_pit và crease có cụm riêng ở 2/3 FT seed. Đây chỉ là điều kiện cần.
- Bước tiếp: PL-E02 (gộp cụm con), tiền đăng ký riêng. K' và ngưỡng gộp phải chốt trước hoặc hiệu chỉnh trên lớp cơ sở; không chọn theo các số mô tả ở trên.
