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

## Q1: kết quả (2/10)

- Run: 1 run hồi quy `20261002_172051__…_regress6_s0` (cờ tắt) và 5 run `…_xtab_s0`–`s4` (cờ bật); cấu hình `gc10_frozen_km_evhung_d64`, SVI, S0 có nhãn.
- Pool S1: 748 ảnh, gồm 698 ảnh lớp mới (silk 420, inclusion 135, waist 94, crease 32, rolled 17) và 50 ảnh cũ. k-means chạy trên đặc trưng thô 768 chiều của toàn pool.

### Toàn vẹn

- `regress6_s0` trùng khít run gốc; 5 run bật cờ trùng 5 run gốc (I2). I1, I2, I3 đạt 5/5 seed.
- Đối chứng dương đạt: 92/94 ảnh waist nằm trong một cụm.

### Kích thước cụm

- s0 [181, 181, 148, 133, 105]; s1 [263, 163, 124, 105, 93]; s2 [268, 159, 124, 104, 93]; s3 [263, 169, 127, 105, 84]; s4 [267, 163, 123, 105, 90].
- CV: lớp thật 1.17; cụm s0 0.22; cụm s1–s4 0.46–0.48. s1–s4 gần như cùng một phân hoạch (mode R1); s0 là mode R0.

### Chấm dự đoán (ngưỡng: P ≤ 1% dưới null hoán vị)

- R1: P1 đúng 4/4 (crease 21–22 ≥ 19; rolled 14–15 ≥ 12). P2 đúng theo định nghĩa (71–75 ≥ 61) nhưng không hợp lệ (xem lỗi đo). P3 đúng: có hai cụm silk 97.6–99.0%. P4 sai (20–25 < 27).
- R0: P1 sai (rolled 9 < 10; crease 15 vừa chạm ngưỡng). P2 và P3 đúng. P4 đúng sát ngưỡng (22 ≥ 21).

### Lỗi đo (construct validity)

- "Tụ" chỉ đếm số ảnh trong một cụm, không kiểm cụm đó thuộc lớp nào. Cụm chứa nhiều inclusion nhất là cụm của silk; cụm chứa nhiều crease/rolled nhất là cụm của waist.
- Quyết định 2/10: không định nghĩa lại sau khi thấy dữ liệu; quy tắc của P2 không áp dụng, quy tắc của P1 vẫn áp dụng. Lần sau "tụ" phải kèm điều kiện sở hữu: lớp đó là lớp đông nhất trong cụm.

### Cơ chế (s4 đại diện R1; tên là tên ghép Hungarian lúc đánh giá)

- Cụm 267: silk 175, inclusion 75, ảnh cũ 13 → tên silk.
- Cụm 123: silk 121 (98%) → tên rolled_pit. Cụm 105: silk 104 (99%) → tên crease.
- Cụm 163: waist 92, ảnh cũ 24, crease 22, rolled 14, inclusion 10 → tên waist_folding. Trên test, 5/6 ảnh rolled rơi vào cụm này.
- Cụm 90: inclusion 49, silk 19, ảnh cũ 12, crease 7 → tên inclusion.
- Silk chiếm 3/5 cụm (s0: 4/5). Hai cụm silk thừa lấy tên rolled và crease, nên recall hai lớp này bằng 0. Đổi tên không cứu được, vì một cụm chỉ mang một tên (Q1c đóng).
- Ảnh cũ rải theo độ giống, không tụ: punching/welding vào cụm waist, water/oil vào cụm silk chính, crescent vào cụm inclusion.
- s0 so với R1: inertia 792 076 so với 788 922–790 486; New 52.80 so với 56.07. Địa hình hàm mục tiêu rất phẳng (hạn chế "nhạy với khởi tạo" mà paper tự nêu).

### Kết luận

- Lớp hiếm bị nuốt vì không sở hữu được cụm nào. Đòn bẩy có thể là thêm sức chứa (phân cụm thừa rồi gộp) hoặc đổi hình học; Q1 không phân biệt được hai đòn bẩy này.
- Lọc ảnh cũ trước khi phân cụm không phải đòn bẩy chính (P4 sai).
- Câu tiếp theo chuyển sang dòng PL: PL-Q1 trong `notes/pl/questions.vi.md`.
