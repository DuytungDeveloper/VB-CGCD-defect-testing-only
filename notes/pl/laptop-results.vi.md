# Bảng số liệu laptop (CUB và GC10)

Bảng tổng hợp các kết quả chính đã xác nhận trên laptop, để đem ra khi cần (báo thầy, viết luận văn). Mỗi số chép từ mục Kết quả đã commit trong `notes/pl/experiments.vi.md`; số tính tay ghi "tính tay". Cập nhật mỗi khi có Kết quả mới.

> **Nhắc nhở:** đây là số laptop. Theo QD27, số đưa vào luận văn phải do server chạy lại; việc chạy server do Rito tự làm. File này chỉ ghi nhắc, không lên kế hoạch server.

Cập nhật lần cuối: 2026-11-21, sau E09.13 (`1517208`).

## 1. CUB t5 (đặc trưng không rò `cub_ft_l25`, PCA-384, K biết trước)

- Đại lượng: M_d là độ chính xác lớp mới (New) trung bình 5 phiên; Old là độ chính xác lớp cũ trung bình 5 phiên; S5_S0test là độ chính xác trên lớp gốc ở phiên 5. Mọi nhánh chấm bằng đặt tên P1. Mỗi giá trị là trung bình các chuỗi (5 chuỗi mỗi FT seed).
- P: pipeline tái lập code gốc (MLE). NL: hậu nghiệm NIW + lọc lề LOO (E12.03). FTR1: lọc NL rồi gom lại (E09.12).

| Nhóm FT seed | Vai trò | M_d P | M_d NL | M_d FTR1 | ΔM_d FTR1 − NL | ΔM_d FTR1 − P | ΔOld FTR1 − NL | ΔOld FTR1 − P | ΔS5_S0test FTR1 − P | Nguồn |
|---|---|---|---|---|---|---|---|---|---|---|
| r0–r2 (15 chuỗi) | dò hướng | 57.707 | 60.330 | 69.575 | +9.245 | +11.868 (tính tay) | +3.554 | +4.918 | +1.982 | E09.12 |
| r3, r4 (10 chuỗi) | xác nhận | 58.213 | 60.908 | 69.876 | +8.968 | +11.663 (tính tay) | +3.469 | +4.827 | +1.850 | E09.12 |
| r5 (5 chuỗi) | lặp lại | 57.259 | 59.752 | 68.821 | +9.069 | +11.562 (tính tay) | +3.417 | +5.212 | +2.745 | E09.13 |

- Trạng thái: FTR1 − NL XÁC NHẬN (E09.12, r3, r4) và LẶP LẠI (E09.13, r5); không đổi lớp cũ lấy lớp mới (ΔOld dương ở mọi nhóm).
- NL − P (ΔM_d): +2.623 (r0–r2), +2.695 (r3, r4), +2.493 (r5).
- Mức trần oracle Npre (gom chỉ trên ảnh mới, không lọc): M_d 73.204 (r0–r2), 73.726 (r3, r4).
- Cơ chế (E09.13): lợi của FTR1 không đến từ số khe dùng được (FTR1e ép cùng số khe rỗng với NL vẫn giữ toàn bộ lợi, S = 1.040).
- Câu được phép báo: "Trên CUB t5 (số laptop), lọc NL rồi gom lại (hậu nghiệm Bayes liên hợp NIW + lề LOO) vượt NL khoảng 9 điểm M_d và pipeline tái lập khoảng 11–12 điểm, không đổi lớp cũ lấy lớp mới; ổn định trên 6 FT seed." Không gọi là "VB" (D26).

## 2. GC10 t1 (`gc10_v2_ft`, 25 ảnh chụp mỗi nhóm; balanced novel theo QD25)

| Nhóm FT seed | Vai trò | balanced novel P | balanced novel N | balanced novel NL | Δ balanced NL − P | Old P | Old NL | Nguồn |
|---|---|---|---|---|---|---|---|---|
| r0–r2 | dò hướng | 0.3494 | 0.3911 | 0.3815 | +0.0322 | 81.5768 | 82.7939 | E12.04 |
| r3, r4 | xác nhận | 0.3055 | 0.3027 | 0.2895 | −0.0160 | 76.5975 | 82.6971 | E12.04 |

- Trạng thái: NL và N KHÔNG XÁC NHẬN trên GC10 về balanced novel; không báo như cải tiến. NIW giữ lớp cũ tốt hơn pipeline (hậu kiểm, chưa đăng ký): ΔOld NL − P +6.0996 ở r3, r4.
- FTR1 trên GC10: chưa chạy.

## 3. Ghi chú

- Mọi số là số laptop (WSL, một luồng BLAS cho các runner `lab/`), do lệnh in ra; xem từng mục Kết quả để biết run và commit.
- Không so số giữa hai máy (QD27).
