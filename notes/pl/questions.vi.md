# Câu hỏi của dòng PL

Nơi duy nhất chứa câu hỏi mở của dòng PL. Trạng thái ghi ở đầu mỗi mục: `mở`, `đang làm`, `chờ <việc>`, `đóng`, `bỏ`.
Khi đóng, ghi kết luận một dòng ngay dưới câu hỏi, trỏ tới bằng chứng (PL-E hoặc entry LOG). Không chép số liệu.

## Phiên

- (Đầu buổi: chép các câu định làm. Cuối buổi: trả lời ngắn từng câu, câu chưa xong mang sang, rồi xoá các dòng trong khối này.)

## Đang mở

- PL-Q1 [mở] Ở số cụm K' = 20, crease hoặc rolled_pit có cụm riêng không (lớp đó chiếm đa số trong cụm, recall ≥ 50%)? Nếu không, hướng phân cụm thừa rồi gộp bị loại.
- PL-Q2 [mở] Gán nhãn bằng optimal transport với biên nới lỏng có thắng k-means không?
- PL-Q3 [mở] Re-labeling đúng như paper (D12) có tốt hơn cách code hiện đang chỉ lọc không?
- PL-Q4 [mở] Nhãn thật tối thiểu có được dùng cho lớp mới (như setting Active GCD), hay chỉ cho lớp cơ sở?

## Đã bỏ hoặc tạm dừng

- Ánh xạ nhãn giả động (dynamic pseudo-label mapping) [bỏ]: pipeline chỉ phân cụm một lần mỗi phiên, không có gì để ánh xạ lại.
- EMA teacher, consistency [bỏ]: cần huấn luyện mạng và augmentation ảnh, không hợp với đặc trưng đóng băng tính sẵn.
- Confirmation bias qua nhiều phiên [tạm dừng]: GC10 và NEU chỉ có một phiên online nên chưa quan sát được.
- Hiệp phương sai low-rank/sparse [tạm dừng]: chỉ chạm giới hạn ước lượng Σ khi ít mẫu, không chạm lỗi phân cụm.

## Nhật ký

- 2026-10-04: mở dòng PL; gieo PL-Q1–Q4 sau khảo sát tài liệu ngày 2026-10-03.
