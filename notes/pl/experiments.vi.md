# Thí nghiệm của dòng PL

Mỗi thí nghiệm một mục: tiền đăng ký (không sửa sau khi commit), rồi Kết quả, Đánh đổi, Kết luận. Quy trình ở `README.vi.md` §4.

## PL-E01 — Phân cụm thừa: lớp hiếm có cụm riêng không? (PL-Q1)

### Tiền đăng ký (2026-10-05; commit: điền sau khi push)

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
