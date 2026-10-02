# Cải tiến 2: khám phá lớp mới trên GC10

Giả thuyết, tiền đăng ký, kết quả và đánh đổi của hướng 2. Câu hỏi mở vẫn nằm ở `plan.vi.md`.

## Q1: tiền đăng ký (2/10, trước mọi run có `--print_cluster_xtab`)

Câu hỏi: balanced novel kẹt quanh 37% vì k-means thiếu sức chứa (A: đặc trưng gom được lớp, nhưng
5 cụm gần đều cỡ không đủ chỗ) hay vì các lớp chồng lấn trong đặc trưng (B)?

Dữ liệu: bảng chéo cụm × lớp thật trên pool train S1, 748 ảnh = 698 ảnh lớp mới + 50 ảnh cũ
(10 ảnh × 5 lớp). Cấu hình `gc10_frozen_km_evhung_d64`, seed 0–4, chạy lại bằng `rerun` + cờ mới.
Id lớp: silk 5, inclusion 6, rolled 7, crease 8, waist 9; lớp cũ 0–4 cộng thành một hàng "cũ".

Đại lượng: số ảnh của lớp c trong cụm chứa lớp đó nhiều nhất. Ngưỡng = số nhỏ nhất có P ≤ 1%
dưới null (giữ kích thước cụm của từng seed, xáo nhãn 20 000 lần).

Chế độ: R1 = s1–s4 (gần như cùng một phân hoạch) là một quyết định, đạt khi ≥ 3/4 seed vượt
ngưỡng. R0 = s0, báo riêng.

Dự đoán (ngưỡng R1 / R0, mức nghiêng):
- P1 crease và rolled đều tụ: crease ≥ 19/32 / ≥ 15/32, rolled ≥ 12/17 / ≥ 10/17. Nghiêng ~55%.
- P2 inclusion tụ: ≥ 61/135 / ≥ 46/135. Nghiêng ~35%.
- P3 có ≥ 1 cụm với silk ≥ 80% (dưới null P ≈ 0). Nghiêng ~75%.
- P4 50 ảnh cũ tụ: ≥ 27/50 / ≥ 21/50. Nghiêng ~30%.
- Đối chứng dương, waist tụ: ≥ 45/94 / ≥ 34/94. Phải đúng.

Toàn vẹn (trượt thì dừng, không đọc bảng):
- I1: tổng cột bảng train = kích thước cụm trong log của run gốc cùng seed.
- I2: metrics.json của run có cờ trùng khít run gốc ở 5/5 seed.
- I3: tổng hàng = n_use (silk 420, inclusion 135, waist 94, crease 32, rolled 17), 10 mỗi lớp cũ.

Quy tắc quyết định:
- P1 đúng: lớp hiếm theo A, thử phân cụm cho phép cụm lệch cỡ (phân cụm thừa + gộp, ...).
  P1 sai: B, ưu tiên Q3/Q4. Chỉ một trong hai lớp tụ: kết luận riêng từng lớp.
- P2 đúng mà recall test của inclusion vẫn thấp: mất ở khâu sau k-means, xem bộ lọc pass F.
  P2 sai: inclusion chồng lấn silk, Q3.
- P3 sai: phân hoạch gần như không bám theo lớp, nghiêng mạnh về B.
- P4 đúng: ảnh cũ chiếm một cụm, lọc ảnh cũ trước khi phân cụm là hướng rẻ (paper §3.2.2,
  nguồn lỗi (1)).
