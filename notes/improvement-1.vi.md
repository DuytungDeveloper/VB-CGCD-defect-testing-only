# Cải tiến 1: từ tiên nghiệm co qua SVI đến ước lượng dạng đóng (B)

Trạng thái: **đã chốt 2026-10-02**. B giữ dạng cờ opt-in (`--estimator closed`), không làm mặc định.
Commit: f6d7841 (v1), bd70b17 (v2), fe95908 (sửa lỗi lát cắt), c016f29 (B1), 3e45d55 (B2),
b7cb407 (script đo), 0ea29f0 (cờ chẩn đoán `--mask_unseen`, `--oracle_filter`).

## 1. Tóm tắt
1. Mục tiêu ban đầu: giữ Σ ổn định khi lớp có ít mẫu (n ≤ d, D21) bằng tiên nghiệm co Inverse-Wishart, tối ưu qua SVI.
2. Qua SVI, tiên nghiệm gần như vô hiệu, vì bộ phân loại gần như không rời khởi tạo (D23). Có lỗi code: ở S1+, tiên nghiệm áp lên lớp cũ (đã sửa ở fe95908).
3. Thí nghiệm A (lr ×10, ×100) xác nhận D23 bằng can thiệp.
4. B thay SVI bằng ước lượng dạng đóng. Theo tiêu chí chốt trước: đạt 1, 2a, 3; **trượt 2b** (GC10 có nhãn, balanced novel −1.6 < −1).
5. B giúp GC10 và NEU (nhận diện, giữ lớp cũ, ổn định), làm CIFAR tụt mạnh, và không giúp khám phá lớp mới.

## 2. Công thức, suy dẫn, nguồn
Với mỗi lớp k của giai đoạn hiện tại (n_k mẫu, không gian PCA d chiều):
μ̂_k = x̄_k;  Ŝ_k = (1/n_k) Σᵢ (xᵢ − x̄_k)(xᵢ − x̄_k)ᵀ;  ν_k = tr(Ŝ_k)/d;  Σ̂_k = (1 − α_k) Ŝ_k + α_k ν_k I,
với α_k là hệ số co giải tích Ledoit–Wolf (`sklearn.covariance.LedoitWolf`).

Suy dẫn (MAP dưới tiên nghiệm Inverse-Wishart, Ψ = κ ν_k I):
1. Log-likelihood: −(n/2) log|Σ| − ½ tr(Sₙ Σ⁻¹), với Sₙ = n Ŝ.
2. Cộng log-tiên nghiệm: −((ν+d+1)/2) log|Σ| − ½ tr(Ψ Σ⁻¹).
3. Đạo hàm theo Σ rồi cho bằng 0: Σ = (Sₙ + Ψ)/(n + ν + d + 1).
4. Đặt κ = ν + d + 1, Ψ = κ ν_k I: Σ = (1 − α) Ŝ + α ν_k I, với α = κ/(n + κ), tương đương κ̂ = nα/(1 − α).

Ledoit–Wolf ước lượng α từ dữ liệu, tức ước lượng độ mạnh tiên nghiệm (empirical Bayes). Ví dụ: n = 150, α = 0.3 → κ̂ ≈ 64.
Phép co nâng trị riêng nhỏ nhất: λᵢ' = (1 − α)λᵢ + αν. Trên GC10: 0.09 → 1.43.
Trường hợp biên (chốt trước): n_k = 1 → ν gộp × I; n_k = 0 ở pass S → giữ ước lượng pass F; lớp cũ chép nguyên (D12).
Nguồn: Ledoit & Wolf (2004, J. Multivariate Anal.); Friedman (1989, RDA, JASA); Murphy (2012, §4.6);
Marchenko & Pastur (1967): MLE kém điều kiện khi n ≈ d.
B nhắm đúng vào hạn chế (2) mà paper tự nêu ở Mục 5 (ước lượng hiệp phương sai khi ít mẫu).

## 3. Kết quả
### 3.1 A: tăng ngân sách bước (GC10 đóng băng, S0 có nhãn, 5 seed)
| lr | S0 | Old | New | final | M_f | NaN | r_μ (s0) | Trị riêng Σ (s0) |
|---|---|---|---|---|---|---|---|---|
| ×1 | 78.67 | 68.55 | 55.42 | 62.37 | 10.12 | 0/5 | 0.996 | [0.88, 1.33] |
| ×10 | 79.25 | 74.61 | 57.94 | 66.77 | 4.65 | 0/5 | 0.970 | [0.18, 2.54] |
| ×100 | 79.83 | — | — | — | — | 2/5 ở S1 | 0.811 | [0.11, 5.82] |

Dữ liệu: trị riêng nhỏ nhất 0.09, trung bình 14.1–19.5. Dòng Step 0 (sau 1 bước Adam) cho cùng độ chính xác ở mọi lr: quyết định do mẫu dấu của gradient tại khởi tạo.

### 3.2 B (5 seed, ghép cặp; phân hoạch k-means trùng khít giữa hai bộ ước lượng)
| Cấu hình | S0 | Old | New | final | M_f | Balanced cơ sở / mới |
|---|---|---|---|---|---|---|
| GC10 có nhãn, SVI | 78.67 | 68.55 | 55.42 | 62.37 | 10.12 | 70.0 / 37.6 |
| GC10 có nhãn, closed | 81.99 | 76.18 | 56.73 | 67.03 | 5.81 | 77.9 / 36.0 |
| GC10 không nhãn, SVI | 59.00 | 54.69 | 50.56 ± 10.14 | 52.75 ± 3.80 | 4.31 | 58.0 / 36.3 |
| GC10 không nhãn, closed | 59.92 | 59.34 | 56.36 ± 2.98 | 57.93 ± 0.85 | 0.58 | 63.0 / 36.1 |
| NEU d128, SVI | 100 | 89.33 | 100 | 94.67 | 10.67 | — |
| NEU d128, closed | 98.56 | 99.67 | 100 | 99.83 | −1.11 | — |
| NEU d384, SVI | NaN 5/5 | | | | | — |
| NEU d384, closed | 97.78 | 95.44 | 100 | 97.72 | 2.33 | — |
| CIFAR, SVI | 91.90 | 82.34 (S5) | 76.28 (S5) | 81.74 | 6.61 | — |
| CIFAR, closed | 84.40 | 69.85 (S5) | 77.88 (S5) | 70.65 | 13.54 | — |
| Paper "w/o SVI†" | 88.92 | | | 68.90 | 17.10 | — |

### 3.3 Chẩn đoán CIFAR (seed 0; oracle dùng nhãn thật, CHỈ để chẩn đoán)
| | S0 | Old S5 | New S5 | final | M_f | kept S1–S5 |
|---|---|---|---|---|---|---|
| SVI | 91.84 | 82.24 | 77.00 | 81.72 | 6.64 | 3955–4027 |
| closed | 84.40 | 69.81 | 78.50 | 70.68 | 13.80 | 4567–5062 |
| + mask | 87.60 | 69.81 | 78.50 | 70.68 | 17.00 | không đổi |
| + oracle | 84.40 | 74.43 | 75.20 | 74.51 | 7.48 | 4000 |
| + cả hai | 87.60 | 74.43 | 75.20 | 74.51 | 10.68 | 4000 |

Phân rã khoảng cách final (11.04 điểm): lọt mẫu cũ 3.83 (35%); khe chưa gặp 0 (chỉ tác động lên S0: +3.20);
còn lại 7.21 (65%) do chính các Gaussian dạng đóng [SUY LUẬN, theo phép loại trừ].

## 4. Cơ chế đã xác lập
1. D23: SVI dừng ở lân cận khởi tạo; quyết định do mẫu dấu của gradient đầu tiên.
2. Bộ lọc mới/cũ (D12) hoạt động nhờ pass F yếu. Gaussian dạng đóng fit đúng cụm đã lẫn mẫu cũ nên không lọc được chúng
   (CIFAR: ít nhất 45–50% mẫu cũ lọt; GC10: ít nhất 16/50).
3. Ứng viên D24: code lấy argmax trên cả các khe lớp chưa huấn luyện (μ = 0, Σ = I). Vô hại với SVI, nhưng làm closed mất 3.2 điểm S0 trên CIFAR.
   Chờ đối chiếu công thức dự đoán trong paper.
4. H2 (chưa kiểm trực tiếp): khi n ≈ d (CIFAR 400 so với 384), Gaussian riêng từng lớp với α ≈ 0.11 phân biệt kém hơn bộ gần khởi tạo.
5. GC10 không nhãn, seed 1: New 32.7 với SVI, 52.8 với closed, trên cùng phân hoạch. Cú sụp do bộ phân loại, không do phân cụm.

## 5. Đánh đổi quan sát được
- Ổn định ↔ dẻo: GC10 Old +7.6, nhưng crease 11% → 0% (bị welding_line hút).
- Độ phủ lớp mới ↔ giữ lớp cũ: mẫu cũ lọt vào làm Gaussian lớp mới rộng ra (CIFAR: New 78.5 / Old 69.8; khi sạch: 75.2 / 74.4).
- Khớp ↔ lọc: mô hình khớp tốt thì lọc kém (pass F).
- Ngân sách ↔ ổn định số: ×10 tốt hơn, ×100 sinh NaN.
- Thiên lệch ↔ phương sai: α của Ledoit–Wolf tối ưu theo sai số Frobenius, không theo độ phân biệt.

## 6. Giới hạn
- Benchmark chính chỉ có GC10, một lần trích đặc trưng, 5 seed phase 2.
- Chẩn đoán CIFAR chỉ chạy seed 0 (S0 của CIFAR tất định; các phiên sau có std khoảng 0.2).
- Oracle dùng nhãn thật nên chỉ có giá trị chẩn đoán. H2 suy ra bằng loại trừ. D24 chưa đối chiếu paper.
- Tiêu chí 2b trượt vì một lớp hiếm (crease, 11 ảnh test mỗi seed). Độ phân giải thô, nhưng phán quyết giữ nguyên.

## 7. Câu hỏi mở (mang sang)
- H2: điều chuẩn mạnh hơn (Σ gộp, tức LDA) có cứu được CIFAR không? Chọn trên CIFAR, xác nhận trên GC10 một lần.
- Một bộ lọc không dùng nhãn và không dựa vào pass F yếu.
- D24: paper có chỉ dự đoán trên các lớp đã gặp không?
- Khám phá lớp mới (balanced novel khoảng 36%) không phụ thuộc bộ phân loại → quay lại hình học đặc trưng và phân cụm.
