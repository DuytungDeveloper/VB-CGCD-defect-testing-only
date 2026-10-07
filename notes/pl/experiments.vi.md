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
