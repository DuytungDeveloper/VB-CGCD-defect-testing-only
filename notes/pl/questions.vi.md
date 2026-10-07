# Câu hỏi của dòng PL

Nơi duy nhất chứa câu hỏi mở của dòng PL. Trạng thái ghi ở đầu mỗi mục: `mở`, `đang làm`, `chờ <việc>`, `đóng`, `bỏ`.
Khi đóng, ghi kết luận một dòng ngay dưới câu hỏi, trỏ tới bằng chứng (PL-E hoặc entry LOG). Không chép số liệu.

## Phiên

- (Đầu buổi: chép các câu định làm. Cuối buổi: trả lời ngắn từng câu, câu chưa xong mang sang, rồi xoá các dòng trong khối này.)

## Đang mở

- PL-Q1 [đóng] Ở số cụm K' = 20, crease hoặc rolled_pit có cụm riêng không (lớp đó chiếm đa số trong cụm, recall ≥ 50%)? Nếu không, hướng phân cụm thừa rồi gộp bị loại.
  - Kết luận (2026-10-07): trên cấu hình paper, có (PL-E01b SỐNG); trên `fz`, không (PL-E01 LOẠI). Bước tiếp: PL-E02.
- PL-Q2 [mở] Gán nhãn bằng optimal transport với biên nới lỏng có thắng k-means không?
- PL-Q3 [mở] Re-labeling đúng như paper (D12) có tốt hơn cách code hiện đang chỉ lọc không?
- PL-Q4 [mở] Nhãn thật tối thiểu có được dùng cho lớp mới (như setting Active GCD), hay chỉ cho lớp cơ sở?
- PL-Q5 [mở] Đặc trưng mức vùng lỗi, không dùng nhãn (patch token, bản đồ bất thường), có giúp khám phá các lớp lỗi nhỏ (inclusion, rolled_pit, crease) không? Hộp XML của lớp mới chỉ được dùng làm cận trên (oracle). Nguồn: `notes/nb/01_data.ipynb` C2, D.
- PL-Q6 [mở] GC10 khó hơn NEU vì mất cân bằng hay vì lỗi nhỏ? Kiểm bằng pool S1 của NEU dựng theo tỉ lệ lớp mới của GC10. Nguồn: `notes/nb/01_data.ipynb` B1, E1.
- PL-Q7 [mở] Ở M10 (phân cụm), phương pháp nào cho nhãn giả tốt hơn k-means theo bộ đo tầng 1 (ứng viên: GMM trên PCA-64 hoặc covariance chéo, phân cụm phổ trên đồ thị kNN, HDBSCAN), và thứ hạng có giữ ở tầng 2 không? Siêu tham số không chỉnh trên pool có nhãn thật của lớp mới. Nguồn: PL-QD3.
- PL-Q8 [mở] Vì sao bộ phân lớp 10 slot đẩy nhiều ảnh inclusion sang lớp cũ, và bộ lọc cũng loại một phần ảnh inclusion của pool? Giả thuyết: nhiễu nhãn và đồng xuất hiện với water_spot (`01_data` C1), lỗi nhỏ (PL-Q5), hay tham số Gaussian gần khởi tạo (D23). Nguồn: `notes/nb/04_s1_discovery.ipynb` (cell 7, 9; một seed).
- PL-Q9 [mở] Trên cấu hình paper, vì sao balanced accuracy của phân hoạch pool (tầng 1) cao hơn nhiều so với balanced novel cuối pipeline trên test (tầng 2), nhất là ở FT seed r1, trong khi trên `fz` hai con số gần nhau? Mất ở khâu nào: chuyển từ pool sang test, đặt tên cụm, hay bộ phân lớp 10 slot? Ảnh hưởng: tầng 1 có dùng được để sàng lọc phương pháp không (PL-QD3). Nguồn: PL-E01b (Kết quả).

## Đã bỏ hoặc tạm dừng

- Ánh xạ nhãn giả động (dynamic pseudo-label mapping) [bỏ]: pipeline chỉ phân cụm một lần mỗi phiên, không có gì để ánh xạ lại.
- EMA teacher, consistency [bỏ]: cần huấn luyện mạng và augmentation ảnh, không hợp với đặc trưng đóng băng tính sẵn.
- Confirmation bias qua nhiều phiên [tạm dừng]: GC10 và NEU chỉ có một phiên online nên chưa quan sát được.
- Hiệp phương sai low-rank/sparse [tạm dừng]: chỉ chạm giới hạn ước lượng Σ khi ít mẫu, không chạm lỗi phân cụm.

## Nhật ký

- 2026-10-04: mở dòng PL; gieo PL-Q1–Q4 sau khảo sát tài liệu ngày 2026-10-03.

- 2026-10-05: thêm PL-Q5, PL-Q6 từ `notes/nb/01_data.ipynb`.

- 2026-10-06: thêm PL-Q7; ghi PL-QD3 (notebook ba lớp, thứ tự làm mới).

- 2026-10-06: thêm PL-Q8 từ `notes/nb/04_s1_discovery.ipynb`.

- 2026-10-07: tiền đăng ký PL-E01b (phân cụm thừa trên cấu hình paper, PL-QD4).

- 2026-10-07: PL-E01 LOẠI (`fz`), PL-E01b SỐNG (`ft`); đóng PL-Q1; thêm PL-Q9.
