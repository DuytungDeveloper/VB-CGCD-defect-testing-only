# Câu hỏi

Nơi duy nhất chứa câu hỏi của repo (QD22). Trạng thái ghi ở đầu mỗi câu: `mở`, `đang làm`, `chờ <việc>`, `tạm dừng`, `xong`, `bỏ`.
Khi đóng, ghi kết luận một dòng ngay dưới câu hỏi, trỏ tới bằng chứng (mã E hoặc entry LOG), rồi chuyển câu xuống mục "Đã xong hoặc bỏ", giữ số. Không chép số liệu.
Mã theo bước; câu mới lấy số kế tiếp trong bước. Mã cũ: `notes/old-codes.vi.md`. Notebook cũ (`01_data`, `04_s1_discovery`) nằm ở tag `nb-v1`.

## Phiên

- (Đầu buổi: chép các câu định làm. Cuối buổi: trả lời ngắn từng câu, câu chưa xong mang sang, rồi xoá các dòng trong khối này.)

## Đang mở

### 00 Dữ liệu

- Q00.01 [mở] NEU: ablation đồng xuất hiện (oracle), và vì sao patches → pitted_surface không gây quên? Nguồn mô tả: `00-data-neu` §3.3 (37 ảnh pitted_surface có hộp patches; 90/125 cặp đi chung là ảnh lớp mới chứa hộp lớp cơ sở).
- Q00.02 [mở] GC10 khó hơn NEU vì những yếu tố nào? Hai bộ khác nhau cùng lúc ở mức lệch lớp, cỡ lỗi, chỗ tập trung ảnh nhiều lỗi và thống kê độ sáng theo lớp (`00-data-neu` Summary 5), nên so hai bộ với nhau không quy được nguyên nhân. Phải can thiệp trong một bộ, ví dụ dựng pool S1 của NEU theo tỉ lệ lớp mới của GC10 (tách riêng mất cân bằng); NEU đang ở trần (lớp mới 100%, `01_data` E), nên kết quả "không giảm" khó đọc. Nguồn: `00-data-gc10` §1.3 và §3.5; `00-data-neu` §1.3 (cân bằng tuyệt đối) và §3.4 (lỗi nhỏ nhất theo ảnh là 34.7 patch, so với inclusion của GC10 là 1.0) (cũ: `01_data`, mục B1 và E1). Sửa câu chữ 2026-11-01: câu cũ "vì mất cân bằng hay vì lỗi nhỏ" ngầm cho rằng chỉ có hai nguyên nhân.

### 02 Fine-tune

- Q02.01 [mở] D9 với ít nhất 3 lần phase 1, kèm đối chứng "400 ảnh ngẫu nhiên" để tách rò rỉ khỏi kích thước dữ liệu. Chỉ làm khi cần số cho báo cáo. Nguồn: E02.03.
- Q02.02 [mở] Code có tái lập Bảng 5 "w/o fine-tuning" (S0 = 88.54) không? Cần wrapper `c100_extract_noft`.
- Q02.03 [tạm dừng] Giới hạn của E02.02: 30 cặp âm mỗi neo, không projection head; bản SimCLR có head chưa thử. Tạm dừng vì GĐ1 đã đóng.
- Q02.06 [mở] Số epoch fine-tune trên GC10 chưa được kiểm: sau 10 epoch, cross-entropy huấn luyện còn 0.151–0.156 và vẫn giảm khoảng 10% mỗi epoch; tham số BOFT lớn nhất chạm mốc 60 bước × lr (`02-finetune` §2.3, §3.1). Fine-tune lâu hơn làm đặc trưng và khám phá lớp mới tốt hơn, hay chỉ khớp quá (overfit)? Cần run phase 1; hiện không có tập kiểm định (held-out). Nguồn: `02-finetune` (`3db9aef`).

### 03 Đặc trưng

- Q03.01 [mở] Tách yếu tố trộn pooler/CLS (D8): trích CLS từ mô hình đã fine-tune.
- Q03.02 [mở] crease và rolled_pit có tách được khỏi waist_folding trong đặc trưng không? Kiểm bằng probe cân trọng số lớp trên ba lớp này. Nguồn: CT2.
  - 2026-11-01: E03.02 (probe 10 lớp, cân trọng số, trên `ft` r0–r2): crease đọc được 5–8/11 ảnh test, rolled_pit 3/6; với crease, phần lớn điểm mất ở khâu khám phá (D ở 2/3 bộ), rolled_pit chưa xác định do chỉ 6 ảnh test. Câu này chưa đóng: chưa đo riêng ba lớp crease, rolled_pit, waist_folding.
- Q03.03 [mở] Đặc trưng mức vùng lỗi, không dùng nhãn (patch token, bản đồ bất thường), có giúp khám phá các lớp lỗi nhỏ (inclusion, rolled_pit, crease) không? Hộp XML của lớp mới chỉ được dùng làm cận trên (oracle). Nguồn: `00-data-gc10` §2.5 và §3.5 (cũ: `01_data`, mục C2 và D).
  - 2026-10-24: tính theo ảnh, chỉ inclusion thật sự nhỏ; rolled_pit và crease to ngang các lớp cũ water_spot, crescent_gap, nhưng là hai lớp hiếm nhất của S1 (`00-data-gc10` §3.5). [SUY LUẬN] Câu này có thể thu hẹp về inclusion; với rolled_pit và crease, số ảnh ít là một ứng viên khác (xem cả Q03.02).

### 04 Nạp dữ liệu theo phiên

- Q04.01 [tạm dừng] Confirmation bias qua nhiều phiên: GC10 và NEU chỉ có một phiên online nên chưa quan sát được.
- Q04.03 [mở] Nhãn thật tối thiểu ở phase 2 (QD16): loader GC10 lấy nhãn S0 theo `n_use` của `splits/gc10_split.json`, bất kể phase 1 fine-tune trên file split ngân sách nhãn nào, và bàn mổ v2 khoá `s0_labels=true`; nên hiện nay "nhãn tối thiểu" chỉ tác động lúc fine-tune. Nếu chạy phase 2 cho E02.06 như hiện tại, số đo chỉ phản ánh ngân sách nhãn lúc fine-tune. Phase 2 có cần dùng cùng tập nhãn với phase 1 không, và cài thế nào (cờ ở loader hay biến thể bước 04)? Chưa làm (quyết định 2026-10-20). Nguồn: E02.06, QD16.

### 05 PCA

- Q05.02 [mở] Phép chiếu học từ nhãn lớp cũ (LDA, hoặc làm trắng trong lớp) có tách được các lớp chồng lấn không? Đây là hướng sửa biểu diễn nếu giả thuyết chồng lấn của E09.03 đúng. Nguồn: CT2.
  - 2026-11-04: không gian con PCA của S0 giữ 0.84–0.94 khoảng cách giữa các tâm lớp mới, thấp nhất ở ba cặp của crease, rolled_pit, waist_folding (0.842–0.857), và giữ 0.53–0.69 phương sai trong lớp (`05-pca` §4.1, đo Euclid). [SUY LUẬN] Một phép chiếu học từ lớp cũ có thể còn ít thấy hướng phân biệt ba lớp này hơn PCA; phép kiểm phải đo riêng ba cặp này.

### 06 Gaussian theo lớp (SVI)

- Q06.01 [mở] Cố định Σ = I ở phase 2 thì còn bao nhiêu? Mahalanobis có thực sự đóng góp? (D11, D20.4, Bảng 15 của paper)
- Q06.02 [mở] S0_All tăng theo số bước SVI thế nào? Một phần đã thấy ở E06.02.
- Q06.03 [mở] Khi n ≈ d (CIFAR: 400 so với 384), điều chuẩn mạnh hơn (Σ gộp, tức LDA) có cứu được ước lượng dạng đóng không? Chọn trên CIFAR, xác nhận trên GC10 một lần. Nguồn: E06.03.
- Q06.04 [tạm dừng] Hiệp phương sai low-rank/sparse: chỉ chạm giới hạn ước lượng Σ khi ít mẫu, không chạm lỗi phân cụm.
- Q06.06 [mở] D21: vì sao NEU d384 (n = 192 < d = 384, tính tay) sụp ở S0 còn CUB (n = 25 < d = 384) thì không, khi cùng cờ optimizer và luật dừng? [SUY LUẬN] Giả thuyết: n < d làm MLE của Σ suy biến (điều kiện cần); số ảnh cùng lớp mỗi batch quyết định gradient của Σ có cùng hướng qua các bước hay không, tức Adam có kịp đẩy Σ tới suy biến trong 1000 bước không (mỗi batch 128: NEU khoảng 42.7 ảnh mỗi lớp, CUB khoảng 1.28, tính tay). Phép thử, chưa duyệt, NEU ngoài phạm vi QD27: S0 của NEU với `--batch-size 8`, và với `--num_dim 128`; giả thuyết đúng thì cả hai hết sụp. Nguồn: Q05.01, D21.

### 07 Dừng sớm

- Q07.03 [mở] Tắt dừng sớm thì khám phá lớp mới có khá hơn không? Dừng sớm không kích hoạt trên NEU, kích hoạt giữa chừng trên GC10 fine-tune (D14). Biến thể dự kiến: `07a-no-early-stop`.
  - 2026-11-05 (`13-gauss-s` §3.2, `a4b9f7e`; nhìn trước một phần, một ảnh chụp): ở r0 s4, lượt F (không lọc, đủ 1000 bước) so với chính nó ở bước 525: balanced novel +2.0 điểm, S1_New +2.3, S1_Old −3.7 (9 ảnh test cũ). Tiền đăng ký `07a-no-early-stop` phải ghi đã thấy các số này.

### 08 Bộ phân lớp

- Q08.01 [mở] D24: paper có chỉ dự đoán trên các lớp đã gặp không? Code lấy argmax trên cả các khe chưa huấn luyện. Nguồn: E06.03.
  - 2026-10-20: ưu tiên thấp cho dòng PL. Ở phiên S1 của GC10 và NEU (t1), mọi khe đều đã gặp (`n_seen` = `num_classes`, `classifier/mngmm.py` dòng 216), nên `--mask_unseen` không đổi gì ở S1, kể cả bộ lọc ở bước 12; cờ này chỉ chạm bảng S0 và M_f proxy.
  - 2026-11-04: với SVI trên GC10 `ft`, không ảnh test S0 nào vào khe 5–9 ở cả 15 ảnh chụp, vì khe đã học có số hạng bậc hai dương với ảnh của lớp mình (`08-classifier` §3.2, §5.1, `6cad5fa`). Câu về paper chưa trả lời được từ văn bản; D24 giữ CANDIDATE.
- Q08.02 [mở] Vì sao bộ phân lớp 10 slot đẩy nhiều ảnh inclusion sang lớp cũ, và bộ lọc cũng loại một phần ảnh inclusion của pool? Giả thuyết: nhiễu nhãn và đồng xuất hiện với water_spot (`00-data-gc10` §3.2 và §3.4; cũ: `01_data`, mục C1), lỗi nhỏ (Q03.03), hay tham số Gaussian gần khởi tạo (D23). Nguồn: `04_s1_discovery` (cell 7, 9; một seed).
  - 2026-10-24: inclusion là nhãn lớp mới kém tin cậy nhất, và một phần ảnh inclusion có hộp water_spot, một lớp cũ (`00-data-gc10` §3.2, §3.4). [SUY LUẬN] Một phần ảnh bị đẩy sang water_spot có thể đúng theo nội dung ảnh; chưa đối chiếu với chính các ảnh bị đẩy.
  - 2026-11-04: ở S0, quyết định của bộ phân lớp do số hạng bậc hai q = −½ xᵀ(Σ⁻¹ − I)x chi phối, và Σ − I lớn dần theo số bước Adam (`08-classifier` §3.1–3.3). [SUY LUẬN] Nếu khe mới ở S1 học ít bước hơn khe cũ, q của chúng nhỏ hơn và chúng thua khi tranh ảnh inclusion với khe cũ; kiểm ở 12, 14 bằng cùng phép tách.
  - 2026-11-05: ở lượt F (`11-gauss-f` §4.2, `e46f92d`), trong các ảnh test lớp mới đúng ở bước 10 mà bộ phân lớp của lượt F đoán sai, 71–81% sang một khe mới khác và chỉ 19–29% sang khe cũ; khe cũ cướp chủ yếu inclusion → oil_spot (35/51 ảnh trên 15 ảnh chụp), không phải water_spot. [SUY LUẬN] Phần ảnh inclusion bị đẩy sang lớp cũ cần đo lại ở 14, trên bộ phân lớp cuối.
  - 2026-11-05 (`12-filter` §3.1, `c331526`): trên pool, bộ lọc loại 4–14/135 ảnh inclusion; cộng 15 ảnh chụp, ảnh inclusion bị loại vào khe water 80, oil 32, punching 16. Trên pool khe nhận chính là water_spot, trên test (lượt F) là oil_spot. [SUY LUẬN] Khớp với 23/135 ảnh inclusion của S1 có hộp water_spot (`00-data-gc10` §3.4); một phần ảnh bị loại có thể đúng theo nội dung ảnh.
  - 2026-11-05 (`14-eval` §3.2, `58a698e`): ở bộ phân lớp cuối, cộng 15 ảnh chụp, 411 lỗi mới→cũ, 289 là inclusion (→ oil_spot 197, → water_spot 59, → punching_hole 28). Chỉ 33/411 ảnh có hộp của lớp đích; 0/59 ảnh inclusion→water có hộp water_spot, nên giả thuyết đồng xuất hiện bị bác ở bước 14. [SUY LUẬN] Hướng còn lại: lỗi nhỏ (inclusion và oil_spot cùng dưới 1 patch, `00` §3.5) cộng khe mới dừng ở khoảng 50% ngân sách (D23); chưa kiểm bằng phép tách q.

### 09 Phân cụm

- Q09.01 [mở] Khoảng cách S1_New so với paper trên CIFAR-100. Manh mối: S1_New gắn với độ cân bằng cụm ở S1 (note LOG `c100_km_evhung_base400_s4`, D9).
- Q09.02 [mở] Ước lượng k trên GC10 (silhouette thất bại): giao thức không biết trước k.
  - 2026-11-14 (`09b-overcluster-merge` (`cc4212c`) §5.3): trong phép gộp của `09b`, dấu của log Bayes factor tách khá rõ bước gộp cùng lớp chủ khỏi bước gộp khác lớp chủ. [SUY LUẬN] Đây là một ứng viên ước lượng số cụm không cần biết k; xem Q09.14.
  - 2026-11-14 (E09.09, `150a2fd`): dừng gộp ở log Bayes factor ≤ 0 cho K̂ trung vị 11–13, nhưng K̂ dịch theo off (off × 0.5: 14–17; off × 2: 8–9). Dấu của log Bayes factor không ước lượng được k nếu không chỉnh tiên nghiệm.
- Q09.03 [mở] New có thấp hơn khi `cluster_max_min` cao không?
- Q09.04 [mở] Gán nhãn bằng optimal transport với biên nới lỏng có thắng k-means không?
- Q09.05 [mở] Nhãn thật tối thiểu có được dùng cho lớp mới (như setting Active GCD), hay chỉ cho lớp cơ sở? (QD16)
- Q09.06 [mở] Ở bước 09, phương pháp nào cho nhãn giả tốt hơn k-means theo bộ đo tầng 1 (ứng viên: GMM trên PCA-64 hoặc covariance chéo, phân cụm phổ trên đồ thị kNN, HDBSCAN), và thứ hạng có giữ ở tầng 2 không? Siêu tham số không chỉnh trên pool có nhãn thật của lớp mới. Nguồn: QD18.
  - 2026-11-01: E03.02: probe k-NN (k = 20, cosine) gần như không đọc được lớp hiếm (crease 0/11 ở cả 4 bộ, rolled_pit 1–2/6) dù probe tuyến tính đọc được. [SUY LUẬN] Phương pháp dựa trên láng giềng gần (phân cụm phổ trên kNN, HDBSCAN) cần được kiểm riêng cho lớp hiếm trước khi xếp hạng.
- Q09.15 [mở] Gộp các cụm thừa về k lớp mà không dùng nhãn: có tiêu chí nào tách "hai mode của cùng một lớp" (silk, inclusion theo camera hoặc cuộn) khỏi "hai lớp khác nhau" (crease, rolled_pit) tốt hơn Bayes factor NIW không? Bayes factor coi các mode của inclusion khác nhau hơn crease so với rolled_pit. Ứng viên: tiên nghiệm Dirichlet trên trọng số thành phần (E09.08), đặc trưng đã khử camera/cuộn. Nguồn: E09.09, Kết luận; E09.06, Ghi chú 2026-11-14, §5.3.

### 12 Bộ lọc cũ/mới

- Q12.01 [mở] Trên CIFAR, trong khoảng 4000 mẫu mỗi phiên bị lọc là "mới", bao nhiêu là dương tính giả, bao nhiêu âm tính giả?
- Q12.02 [mở] Một bộ lọc không dùng nhãn và không dựa vào pass F yếu. Nguồn: E06.03.
  - 2026-11-05 (`12-filter` §3.2, `c331526`): bộ lọc SVI quyết định bằng số hạng bậc hai q ở 98.4–100% ảnh pool; bộ lọc chỉ dùng ℓ (hướng của μ) lọt ×1.38 ảnh cũ và loại ×2.6 ảnh mới, nên giả thuyết "khuôn không mang vị trí nên lọc được" không được ủng hộ. Bộ lọc hiện tại đã loại 39–48/50 ảnh cũ; một bộ lọc mới chỉ còn chỗ ở 2–11 ảnh cũ lọt và 5–16 ảnh mới bị loại mỗi ảnh chụp.
- Q12.03 [mở] Re-labeling đúng như paper (D12) có tốt hơn cách code hiện đang chỉ lọc không?
  - 2026-11-05 (`12-filter` §2.1, `c331526`; không dùng nhãn thật): bộ phân lớp của lượt F trùng id cụm ở 89.6–92.6% pool; trong ảnh được giữ, chỉ 7–21 ảnh nằm ở khe mới khác. Re-labeling của paper (Algorithm 1 dòng 34–39) đổi ít nhãn lớp mới; khác biệt chính là 46–61 ảnh bị loại mà paper gán cho lớp cũ (D12, ý b). Chưa chấm bằng nhãn thật (`lab/README.vi.md` §9): muốn chấm thì tiền đăng ký `12a-relabel` trước.
- Q12.04 [mở] Vì sao bộ lọc trên cụm đã gộp (`09a` + `09b`) giữ thêm 17–88 ảnh mới mang đúng tên và 0–4 ảnh cũ mỗi lần rút, và phần mất ở bộ lọc cộng lượt S lớn hơn nhánh mặc định 0.024? Giả thuyết [SUY LUẬN]: cụm gộp rộng hơn nên ranh giới chấp nhận của bộ lọc rộng hơn. Cách kiểm: mở rộng (2) của E11.01 (bộ lọc dùng nhãn thật, tách phần do bộ lọc khỏi phần của lượt S); cần tiền đăng ký mới vì điều kiện của mở rộng (2) không xảy ra. Nguồn: E11.01 (`2596f28`).

### 14 Đánh giá

- Q14.01 [mở] `exp.py agg`: cột `paper` cần tra theo bộ dữ liệu (hiện luôn là số CIFAR).
- Q14.03 [mở] S4_New ổn định, S5_New không: có đánh đổi giữa hai phiên? (r = −0.55, n = 5, không có ý nghĩa; E09.01)

## Đã xong hoặc bỏ

### 02 Fine-tune

- Q02.04 [xong] Đánh đổi ổn định–thích nghi có vững qua các FT seed không?
  - Kết luận (2026-10-01): giữ lớp cũ tốt hơn ở cả 3 FT seed; khám phá lớp mới thì không ổn định (E02.04).
- Q02.05 [bỏ] EMA teacher, consistency: cần huấn luyện mạng và augmentation ảnh, không hợp với đặc trưng đóng băng tính sẵn.

### 04 Nạp dữ liệu theo phiên

- Q04.02 [xong] S0 không nhãn: điểm thứ ba của trục lượng nhãn.
  - Kết luận (2026-10-01): chạy được; khám phá lớp mới không cần nhãn S0 (E04.01). Đầu thấp của trục lượng nhãn sau đó đổi thành nhãn thật tối thiểu (QD16).

### 05 PCA

- Q05.01 [xong] D21: công thức CUB của upstream có n = 25 < d = 384; số CUB của paper đến từ cấu hình nào?
  - 2026-11-13: dự đoán sụp của D21 sai. Lưới `cub_base_ft` (15 run, `50d72f3`, agg `c251cd7`) chạy đủ, không run nào sụp, dù dùng cùng cờ optimizer và luật dừng với 5 run NEU d384 đã sụp (tag `archive-2026-10-02`); chi tiết ở dòng measured 2026-11-13 của D21. Câu hỏi còn mở ở dạng mới: vì sao NEU sụp còn CUB thì không, dù n/d của CUB thấp hơn. Chưa có cơ chế đã kiểm.
  - Kết luận (2026-11-13): số CUB của paper đến từ cấu hình của code upstream: lưới `cub_base_ft` tái lập S0 và S5 của paper mà không sụp dù n < d (`50d72f3`, agg `c251cd7`; D21, dòng measured 2026-11-13). Câu tiếp: Q06.06.
- Q05.03 [xong] Các thành phần PCA đầu của S0 mã hoá camera (và cuộn thép) hay lớp lỗi? Đo η² (tỉ số tương quan, correlation ratio) của từng thành phần theo camera và theo lớp. Nếu camera chiếm các thành phần đầu, Gaussian ở 06–14 có thể gom ảnh theo camera; phân cụm ở 09 chạy trên đặc trưng thô nên cần đo riêng. E03.01 đã loại chuẩn hoá camera bằng trừ trung bình, nên câu này hỏi tín hiệu camera nằm ở đâu, không thử lại cách sửa đó. Làm ở `05a-camera` (QD24). Nguồn: `00-data-gc10` §2.2–2.4 (Cramér's V giữa lớp và camera 0.31).
  - Kết luận (2026-11-04): ở S0, các thành phần đầu mang lớp; với lớp mới, camera mang nhiều phương sai hơn lớp ở cả PCA-64 và đặc trưng thô, ở cả 3 FT seed; trong silk_spot, cuộn thép và camera cùng giải thích phần lớn phương sai (`05a-camera`, `75e1bf9`). Câu tiếp: Q09.13.

### 06 Gaussian theo lớp (SVI)

- Q06.05 [xong] Vì sao Σ học ở S0 có det < 1 (det trung bình của lớp cũ 0.3707 ở r0 s4, notebook cũ `04_s1_discovery`, tag `nb-v1`) trong khi 24–32/64 hướng của mỗi lớp cũ có phương sai trong lớp lớn hơn 1 (`05-pca` §3.2)? Ứng viên [SUY LUẬN]: với Σ = I + E nhỏ, log det ≈ tr E − ½‖E‖²_F, và Adam đẩy cả 4032 phần tử ngoài đường chéo (D23), nên phần ngoài đường chéo kéo log det xuống. Kiểm ở `06-gauss-svi`: tách log det của `params0` thành tr E và ‖E‖²_F. Ảnh hưởng: nhánh của luật dừng (D14) và cách đọc 07.
  - Kết luận (2026-11-04): hai phần cùng kéo log det xuống: phần ngoài đường chéo của E (chiếm phần lớn ‖E‖²_F, có ở mọi lớp cũ) và vết âm của E; thang đo của dữ liệu một mình không quyết định nhánh det của luật dừng (`06-gauss-svi` §3.2, `4112807`).

### 07 Dừng sớm

- Q07.01 [xong] logdet của lớp mới tăng hay giảm theo từng bước? Cần log logdet theo bước (Hình 2b của paper so với D14).
  - Kết luận (2026-11-05): trên GC10 `ft`, giảm đơn điệu từ 1 ở cả lượt F và lượt S, nhanh nhất ở các cụm silk chặt, ngược chiều Hình 2b của paper; lượt F cắt ngưỡng S ở bước 541 và det cũ ở bước 740 (`13-gauss-s` §3.1, `a4b9f7e`).
- Q07.02 [xong] `--scaling-factor 1.2` tác động vào đâu trong luật dừng (D14)?
  - Kết luận (2026-11-04): không tác động vào phép tính nào; `self.scaling_factor` chỉ được gán, `main.py` chỉ in và truyền cờ đi; hệ số của luật dừng tính từ số lớp (`07-early-stop` §2.1, `0d6d7ad`).
- Q07.04 [xong] Lớp cũ mất thêm ở nhánh `09a` + `09b` là do lượt S dừng muộn hơn, hay do chính phân hoạch? Ở r0, r1 hai nhánh giữ số ảnh test cũ như nhau khi cùng bước dừng; ở r2 bước dừng chỉ giải thích khoảng một nửa và hai nhánh không có vùng bước dừng chung. Cách kiểm: phản thực tế, chạy lượt S của phân hoạch `09a` + `09b` nhưng dừng ở bước của nhánh mặc định, trên một mẫu con các lần rút; đo cả phần lớp mới mất khi dời bước dừng. Nguồn: E09.06, Ghi chú 2026-11-14; `09b-overcluster-merge` (`cc4212c`) §4.
  - Kết luận (2026-11-15): bước dừng giải thích (E11.01, `2596f28`): ép lượt S của `09a` + `09b` dừng ở bước của nhánh mặc định cho D_old = +0.67 ảnh (< B = 1.48 ảnh). Dừng muộn khoảng 49 bước đổi 1.41 ảnh cũ lấy 0.013 balanced novel (tính tay); r4 lệch xu hướng (+3.25 ảnh). Luật dừng là núm đánh đổi cũ/mới; E09.08 so các nhánh ở cùng mức ảnh cũ.

### 08 Bộ phân lớp

- Q08.03 [xong] Nhãn mơ hồ ở ba lớp cũ quanh mối hàn (punching_hole, welding_line, crescent_gap): độ chính xác trên ảnh test chỉ có một loại lỗi khác bao nhiêu so với trên ảnh nhiều loại lỗi, và trong các ảnh nhiều lỗi bị đoán sai, bao nhiêu ảnh được gán cho một lớp có hộp trong chính ảnh đó? So trong từng lớp, vì ảnh nhiều lỗi dồn ở ba lớp này; water_spot và oil_spot có quá ít ảnh test nhiều lỗi. Nguồn: `00-data-gc10` §3.2, §3.4, §3.6.
  - Kết luận (2026-11-04): ảnh nhiều lỗi chỉ làm hại welding_line (đúng 10/19 ảnh nhiều lỗi so với 28/35 ảnh một lỗi); 17/35 ảnh S0 bị đoán sai được gán cho một lớp có hộp lỗi trong chính ảnh, 23/35 ảnh sai nằm trong ba lớp quanh mối hàn (welding_line → punching_hole: 7/7 ảnh có hộp punching_hole); trần khi tính các ảnh này là đúng ≈ 223/241 = 92.5%; camera không phải nguồn lỗi chính. Hệ quả: S0_All và S1_Old có một trần do nhãn; nên báo thêm độ chính xác trên ảnh test một lỗi, nhãn thư mục trùng XML, làm đại lượng phụ ở 14 (`08-classifier` §4.1, `6cad5fa`).

### 09 Phân cụm

- Q09.07 [bỏ] k-means hay GMM? Phép so cũ (E09.02) không kiểm được giả thuyết sức chứa, vì GMM full ở n_k < d tự ưu tiên cụm đều; câu được làm tiếp bằng E09.03, rồi thay bằng Q09.06. CORRECTION ở note LOG `gc10_frozen_gmm10_evhung_d64_s4`.
- Q09.08 [xong] Proxy giống pipeline hơn (PCA fit trên lớp cơ sở, 64 chiều).
  - Kết luận (2026-10-06): bàn mổ chạy lại đúng pipeline (`scripts/replay.py`, nay `lab/replay.py`), nên không cần proxy.
- Q09.09 [xong] Một cụm có hút hết mẫu lớp cũ ở S1 không?
  - Kết luận (2026-10-02): không; ảnh cũ rải theo độ giống (E09.03/P4).
- Q09.10 [xong] Khám phá lớp mới có phụ thuộc bộ phân loại không?
  - Kết luận (2026-10-02): không; balanced novel khoảng 36% với cả SVI và dạng đóng (E06.03). Chuyển sang hình học đặc trưng và phân cụm (CT2).
- Q09.11 [xong] Ở số cụm K' = 20, crease hoặc rolled_pit có cụm riêng không (lớp đó chiếm đa số trong cụm, recall ≥ 50%)? Nếu không, hướng phân cụm thừa rồi gộp bị loại.
  - Kết luận (2026-10-07): trên cấu hình paper, có (E09.05, SỐNG); trên `fz`, không (E09.04, LOẠI). Bước tiếp: E09.06.
- Q09.12 [xong] k-means ở bước 09 (`n_init` 10) có ổn định theo thứ tự hàng của pool không? Ở r0 seed 4, cùng trạng thái RNG, chỉ đổi thứ tự hàng (do bước 04 xáo theo seed): ARI 0.73, inertia 82926 thành 82757, balanced accuracy lớp mới trên pool 0.32 thành 0.56, crease có cụm riêng (`04-loader` §3.3; một hoán vị). Cần đo trên 15 ảnh chụp × nhiều hoán vị: mức dao động; inertia thấp hơn có đi cùng balanced accuracy cao hơn không; tăng `n_init`, hoặc chọn nghiệm inertia thấp nhất qua nhiều thứ tự, có giúp không. Nếu dao động lớn, nó cũng ảnh hưởng cách đọc mọi kết quả theo seed (Q14.03) và thiết kế E09.06. Cấu hình nào thuộc thí nghiệm thì tiền đăng ký trước khi chạy trên pool lớp mới. Nguồn: `04-loader` (`3f6e08e`).
  - Kết luận (2026-11-11): best-of-10 không ổn định theo khởi tạo (`09-cluster`); trong cùng ảnh chụp, inertia thấp hơn thường đi với balanced novel cao hơn, nhưng thay `n_init` 10 bằng best-of-1000 không đủ lớn ở tầng 2 (E09.07, KHÔNG ĐỦ LỚN). Giữ `n_init` 10; E09.06 chạy trên k-means mặc định.
  - 2026-11-04 (QD26): với cùng trạng thái RNG, đổi thứ tự hàng cho cùng phân phối nghiệm như rút khởi tạo mới (k-means++ chọn tâm theo chỉ số hàng; Lloyd không phụ thuộc thứ tự; đã thử trên dữ liệu đồ chơi), nên câu hỏi là độ ổn định của best-of-10 theo khởi tạo, và ARI 0.73 ở trên là một lần rút. Pool khác nhau giữa các seed (50 ảnh reserve), nên inertia chỉ so được trong cùng pool. Phần không nhãn: `09-cluster`; phần có nhãn: E09.07 (tiền đăng ký 2026-11-04).
  - Sửa (2026-11-04, `09-cluster` §3.4b, `8982942`): ghi chú trên sai một phần. Với trạng thái RNG cố định, đổi thứ tự hàng không cho cùng phân phối với rút khởi tạo mới (KS D của inertia 0.087–0.089 so với mức nền 0.056–0.059 ở r0 s4): `searchsorted` ở các mức tích luỹ cố định không còn là lấy mẫu theo D². Nó vẫn tác động qua cơ chế chọn tâm và cho dao động cùng cỡ ở cấp nhóm nghiệm; khi cả thứ tự hàng lẫn RNG cùng đổi (giữa các seed), dao động đúng là của rút khởi tạo mới.
  - 2026-11-04 (`09-cluster`, `8982942`), phần không nhãn: trên cùng một pool, hai lần chạy pipeline có E[ARI] 0.75–0.90 và hiếm khi ra cùng phân hoạch (0.005–0.038); pipeline ở nhóm nghiệm sâu nhất (ARI ≥ 0.95) ở 7/15 ảnh chụp; các seed của một FT seed rút từ gần như cùng một landscape (E[ARI] giữa hai seed thấp hơn trong một seed 0.007–0.020), nên dao động giữa các seed ở bước 09 chủ yếu là may rủi khởi tạo. Phần có nhãn: E09.07.
  - 2026-11-04 (`09d-ownership` §7.1, `abbc3af`), một ca có nhãn: trên cùng pool r0, cấu hình cho crease cụm riêng có inertia cao hơn 0.23–0.35% nhưng balanced accuracy tầng 1 0.56–0.57 so với 0.32–0.33. Không thay E09.07.
- Q09.13 [xong] k-means ở bước 09 có chia silk_spot (và inclusion) theo camera hoặc cuộn thép không? Với 698 ảnh lớp mới, trong đặc trưng thô, phần chỉ do camera lớn hơn phần chỉ do lớp; camera giải thích 0.43 phương sai trong silk_spot, cuộn và camera cùng giải thích 0.61 (`05a-camera` §3.1, §4.1). Kiểm ở `09-cluster` bằng bảng cụm × camera và cụm × cuộn trên 15 ảnh chụp (camera, cuộn lấy từ tên file, không cần nhãn lớp). Nếu đúng, đây là một cơ chế của tranh cụm ở K = 5 (QD24) và là đầu vào cho thiết kế E09.06, E09.07. Nguồn: `05a-camera` (`75e1bf9`).
  - 2026-11-04 (`09-cluster` §4.3, `8982942`), phần không nhãn: trên toàn pool, id cụm liên hệ mạnh với camera (Cramér's V 0.50–0.65) và với cuộn (0.71–0.78), vượt xa mức nền; chưa tách được khỏi lớp. Phép đo trong từng lớp ở `09d-ownership`.
  - Kết luận (2026-11-04): có, cho silk_spot và inclusion. Trong silk, camera và cuộn cùng giải thích 0.87–0.92 thành viên cụm (inclusion 0.66–0.71), và mỗi yếu tố vẫn có ý nghĩa khi giữ yếu tố kia cố định ở 5/5 ảnh chụp của mọi FT seed; các cụm con của silk về thực chất là nhóm camera/cuộn. Đối chứng: rolled không; crease ở r1, r2 bám cuộn. Đây là một cơ chế của tranh cụm ở K = 5 (QD24) và là đầu vào cho E09.06 (`09d-ownership` §6, `abbc3af`).
- Q09.14 [xong] Dừng phép gộp của `09b` khi log Bayes factor đổi dấu, thay vì ép về k = 5, có giữ được cụm của lớp hiếm (rolled_pit, crease) mà không trả giá quá nhiều ở silk không? Chấm bằng Hungarian một–một như QD15, cụm thừa tính là sai. Điểm 0 phụ thuộc tiên nghiệm (off) và ngưỡng được nghĩ ra sau khi đã xem nhãn pool của GC10, nên phải tiền đăng ký trước khi tính; inclusion gộp vào silk với log Bayes factor dương nên không được cứu bằng cách này. Giao thức: k = 5 vẫn là dòng chính, nhánh k tự do báo cáo riêng. Nguồn: E09.06, Ghi chú 2026-11-14; `09b-overcluster-merge` (`cc4212c`) §5.
  - Kết luận (2026-11-14): KHÔNG RÕ (E09.09, `150a2fd`): Δ ≈ 0 trên r3, r4. Khi không bị ép về k = 5, phép gộp giữ được cụm con của rolled_pit ở 108/110 lần, nhưng lớp đông (inclusion) bị tách và mất điểm ở Hungarian một–một; điểm 0 dịch theo off, nên không phải điểm dừng tự nhiên. Câu tiếp: Q09.15.

### 10 Đặt tên cụm

- Q10.01 [xong] Đổi tên cụm có cứu được recall của lớp hiếm không?
  - Kết luận (2026-10-02): không; một cụm chỉ mang một tên, và hai cụm silk thừa lấy tên rolled_pit và crease (E09.03).
- Q10.02 [bỏ] Ánh xạ nhãn giả động (dynamic pseudo-label mapping): pipeline chỉ phân cụm một lần mỗi phiên, không có gì để ánh xạ lại.
- Q10.03 [xong] Cụm bám theo cuộn thép có co lại trên tập test không? GC10 chia train/test theo cuộn (không chung cuộn nào), và silk_spot có 420 ảnh pool từ 16 cuộn. Nếu k-means chia silk_spot theo cuộn (Q09.13), phần ảnh test lớp mới rơi vào cụm đó (`cluster.predict_raw(test1_x)`) sẽ nhỏ hơn phần ảnh pool lớp mới của cụm. Đo ở `10-naming`: tỷ lệ mỗi cụm trên 698 ảnh pool lớp mới so với trên 214 ảnh test lớp mới, kèm mức tập trung theo cuộn của cụm. Nếu đúng, đây là một nguồn chênh giữa tầng 1 và tầng 2 (Q14.04). Nguồn: QD26.
  - 2026-11-04 (`09d-ownership` §6, `abbc3af`): cụm con của silk là nhóm camera/cuộn (camera và cuộn giải thích 0.87–0.92 thành viên cụm trong silk); cách chia crease ở r1, r2 bám cuộn, không bám camera. Đo ở `10-naming` như trên.
  - Kết luận (2026-11-05): đúng với silk: phân bố ảnh silk qua 5 cụm khác giữa pool và test ở 15/15 ảnh chụp, và ô silk dồn vào ít cuộn co lại trên test (Spearman giữa số cuộn hiệu dụng và độ co +0.45 đến +0.77 theo FT seed); ở r0 s4 hai cụm silk mang tên rolled, crease đều từ một cuộn. inclusion chỉ dịch ở r1 và không theo cuộn (`10-naming` §5, `b2723cb`).

### 11 Lượt F

- Q11.01 [xong] Vì sao các bước 11–14 lấy của nhánh `09a` + `09b` nhiều điểm hơn của nhánh mặc định, ở cả ba FT seed? Giả thuyết [SUY LUẬN]: cụm sau gộp là hợp của nhiều cụm con, một Gaussian mỗi cụm ở lượt F và lượt S khớp kém. Cần cột 11 của sổ mất điểm cho nhánh `09a` + `09b` (chưa có trong `metrics.json` của E09.06, phải chạy lại SVI) để tách lượt F khỏi bộ lọc và lượt S. Ảnh hưởng: E09.08 có nên giữ nhiều thành phần cho mỗi lớp không. Nguồn: E09.06, Ghi chú 2026-11-14; `09b-overcluster-merge` (`cc4212c`) §3.1.
  - Kết luận (2026-11-15): theo quy tắc là lượt F, nhưng sát nút: L_F ≥ L_S ở 4/5 FT seed, r4 chỉ hơn 0.003, r2 ngược dấu (E11.01, `2596f28`). Phần mất thêm so với nhánh mặc định lại nằm nhiều hơn ở bộ lọc và lượt S (+0.024, so với +0.014 ở lượt F), và bộ lọc của `09a` + `09b` giữ thêm cả ảnh mới đúng tên lẫn ảnh cũ. Giả thuyết "một Gaussian mỗi cụm khớp kém" chỉ được ủng hộ một phần. Câu tiếp: Q12.04.

### 14 Đánh giá

- Q14.02 [xong] Đưa balanced accuracy vào chỉ số chính (hiện chỉ có qua `cm_summary`).
  - Kết luận (2026-11-04): balanced novel ở tầng 2 là đại lượng chính, S1_Old là ràng buộc lớp cũ (QD25); tính từ ảnh chụp bằng sổ mất điểm (10–14).
- Q14.04 [xong] Trên cấu hình paper, vì sao balanced accuracy của phân hoạch pool (tầng 1) cao hơn nhiều so với balanced novel cuối pipeline trên test (tầng 2), nhất là ở FT seed r1, trong khi trên `fz` hai con số gần nhau? Mất ở khâu nào: chuyển từ pool sang test, đặt tên cụm, hay bộ phân lớp 10 slot? Ảnh hưởng: tầng 1 có dùng được để sàng lọc phương pháp không (QD18). Nguồn: E09.05, mục Kết quả và Ghi chú (2026-10-20): so trên cùng seed 0–4, r0 và r1 bằng nhau ở tầng 1 nhưng cách xa nhau ở tầng 2.
  - 2026-11-01: E03.02 thêm mốc biểu diễn: probe `lr_bal` (có nhãn) đạt balanced novel 0.61 / 0.68 / 0.71 trên `ft` r0 / r1 / r2, cuối pipeline 0.39 / 0.29 / 0.36. Đặc trưng không phải chỗ mất chính; chỗ mất nằm ở các bước 09–14.
  - 2026-11-05: ở mức cụm (k-means trên test), r0 hơn r1 7.2 điểm balanced theo tên của pipeline nhưng 1.6 theo cách đặt tên tối ưu cho balanced; kiểm chéo hai nửa, cách đó vẫn hơn tên của pipeline 6.7 điểm ở r1 (cụm do crease sở hữu trong pool mang tên inclusion ở r1 s0–s2). [SUY LUẬN] Cách đặt tên cân theo số ảnh là một phần chỗ mất của r1; kiểm ở 14 bằng dự đoán của bộ phân lớp (`10-naming` §3, `b2723cb`).
  - 2026-11-05: nếu dừng pipeline sau lượt F (cột 11 của sổ mất điểm, phản thực tế), balanced mức cụm giảm 3.7 / 7.5 / 1.9 điểm (r0 / r1 / r2) so với cột 10; r1 mất nhiều nhất (114 ảnh trên 5 ảnh chụp, chủ yếu do chia lại giữa các khe mới). Cột 11 (0.421 / 0.311 / 0.386) đã gần mức cuối pipeline của E03.02 (`11-gauss-f` §4, `e46f92d`).
  - 2026-11-05: bộ lọc không phải chỗ mất. Trên pool, balanced đổi dưới 1 điểm từ cột 10 sang 12, độ sạch nhãn của tập học chỉ tăng +1.7 đến +3.1 điểm, và 80% ảnh mới bị loại nằm ở cụm sai tên. Trước bộ lọc, chỉ 27–45% nhãn mà lượt F học là đúng tên (`12-filter` §4.1, `c331526`). [SUY LUẬN] Nhiễu nhãn của lượt S đến chủ yếu từ bước 09–10.
  - 2026-11-05: từ lượt F tới cuối pipeline ở r0 s4, lớp mới mất ít (balanced 0.380 → 0.368). Dừng sớm cắt khe mới còn 54% ngân sách, giữ 9 ảnh test cũ, mất 5 ảnh test mới; ở cùng ngân sách, bộ lọc không đổi lớp cũ (`13-gauss-s` §3.2, `a4b9f7e`). Chỗ mất lớn của lớp mới vẫn ở 09–11.
  - 2026-11-05 (`14-eval` §4, `58a698e`): trên 15 chuỗi mặc định, balanced theo bước (tầng 1 → 10 → 11 → 14): r0 0.469 → 0.458 → 0.421 → 0.394; r1 0.469 → 0.386 → 0.311 → 0.292; r2 0.399 → 0.405 → 0.386 → 0.362. Trên dự đoán của bộ phân lớp cuối, đặt tên theo chính dự đoán đó (QD14, kiểm chéo hai nửa) hơn tên của pipeline 6.2 điểm ở r1, −0.3 ở r0 và r2.
  - Kết luận (2026-11-05): r1 bằng r0 ở tầng 1, thua 10.2 điểm ở cuối: 7.2 ở khâu pool→test và đặt tên, 3.8 ở lượt F; 12–14 làm mọi FT seed mất 2–3 điểm như nhau. Khoảng 6 điểm là do cách đặt tên khi chấm (QD15, cân theo số ảnh), không do cái mô hình học. Đặc trưng không phải chỗ mất chính (E03.02). Tầng 1 không xếp hạng được FT seed theo tầng 2, nên sàng lọc ở tầng 1 (QD18) phải kèm cột 10 trên tập test.

## Nhật ký

- 2026-10-04: mở dòng PL; gieo Q09.11, Q09.04, Q12.03, Q09.05 sau khảo sát tài liệu ngày 2026-10-03.
- 2026-10-05: thêm Q03.03, Q00.02 từ `notes/nb/01_data.ipynb`.
- 2026-10-06: thêm Q09.06; ghi QD18 (notebook ba lớp, thứ tự làm mới).
- 2026-10-06: thêm Q08.02 từ `notes/nb/04_s1_discovery.ipynb`.
- 2026-10-07: tiền đăng ký E09.05 (phân cụm thừa trên cấu hình paper, QD19).
- 2026-10-07: E09.04 LOẠI (`fz`), E09.05 SỐNG (`ft`); đóng Q09.11; thêm Q14.04.
- 2026-10-15: đổi sang mã theo bước (QD20); xếp lại theo thứ tự bước; nguồn notebook trỏ tới tag `nb-v1`.
- 2026-10-18: hệ mã mới (QD22): chuyển vào mọi câu còn mở của `plan.vi.md` §5 và `improvement-1.vi.md` §7, cùng ba câu của CT2; đánh số lại một lần, câu còn mở lên trước.
- 2026-10-20: thêm Q04.03 (nhãn tối thiểu ở phase 2, chưa làm); Q08.01 hạ ưu tiên cho dòng PL; Q14.04 thêm nguồn (E09.05, Ghi chú 2026-10-20).
- 2026-10-24: `00-data-gc10` (`1e46674`) thay `01_data` làm nguồn của Q00.02, Q03.03, Q08.02, kèm phát hiện mới; thêm Q08.03 (nhãn mơ hồ quanh mối hàn).
- 2026-11-01: `00-data-neu` (`0079819`) làm nguồn phía NEU của Q00.01, Q00.02; sửa câu chữ Q00.02 (nhiều yếu tố khác nhau cùng lúc, cần can thiệp trong một bộ).
- 2026-11-01: thêm Q02.06 (số epoch fine-tune trên GC10) từ `02-finetune` (`3db9aef`).
- 2026-11-01: E03.02 xong (`d467c6b`): ghi kết quả vào Q03.02, Q14.04, Q09.06.
- 2026-11-04: thêm Q09.12 (k-means không ổn định theo thứ tự hàng của pool) từ `04-loader` (`3f6e08e`).
- 2026-11-04: thêm Q05.03 (camera, làm ở `05a`) và Q06.05 (vì sao det Σ < 1); ghi chú vào Q05.02; từ `05-pca` (`42c0ca4`).
- 2026-11-04: đóng Q06.05 bằng `06-gauss-svi` (`4112807`).
- 2026-11-04: đóng Q07.02 bằng `07-early-stop` (`0d6d7ad`).
- 2026-11-04: QD25 đóng Q14.02 (đại lượng chính); `05a-camera` (`75e1bf9`) đóng Q05.03 và thêm Q09.13.
- 2026-11-04: `08-classifier` (`6cad5fa`) đóng Q08.03; ghi chú vào Q08.01, Q08.02.
- 2026-11-04: QD26: ghi chú Q09.12 (đổi thứ tự hàng là rút khởi tạo mới; E09.07 đã tiền đăng ký); thêm Q10.03.
- 2026-11-04: `09-cluster` (`8982942`): sửa ghi chú Q09.12 về thứ tự hàng; ghi phần không nhãn của Q09.12, Q09.13.
- 2026-11-04: `09d-ownership` (`abbc3af`, kế hoạch `d282b37`) đóng Q09.13; ghi chú vào Q09.12, Q10.03.
- 2026-11-05: `10-naming` (`b2723cb`) đóng Q10.03; ghi chú Q14.04.
- 2026-11-05: `11-gauss-f` (`e46f92d`): ghi chú Q08.02, Q14.04.
- 2026-11-05: `12-filter` (`c331526`): ghi chú Q08.02, Q12.02, Q12.03, Q14.04.
- 2026-11-05: `13-gauss-s` (`a4b9f7e`): đóng Q07.01; ghi chú Q07.03, Q14.04.
- 2026-11-05: `14-eval` (`58a698e`): đóng Q14.04; ghi chú Q08.02.
- 2026-11-11: E09.07 (`0e630ad`) đóng Q09.12.
- 2026-11-13: đóng Q05.01 (cấu hình upstream tái lập số CUB của paper, `50d72f3`; D21 measured); thêm Q06.06 (cơ chế sụp của NEU d384).
- 2026-11-14: `09b-overcluster-merge` (`cc4212c`), B13: thêm Q07.04 (bước dừng và lớp cũ), Q09.14 (dừng gộp theo dấu của log Bayes factor), Q11.01 (bước 11–14 trên cụm đã gộp); ghi chú Q09.02.
- 2026-11-14: E09.09 (`150a2fd`): đóng Q09.14 (KHÔNG RÕ); thêm Q09.15 (gộp cụm thừa không dùng nhãn); ghi chú Q09.02.
- 2026-11-15: E11.01 (`2596f28`): đóng Q07.04 (bước dừng giải thích), Q11.01 (lượt F, sát nút); thêm Q12.04 (bộ lọc trên cụm đã gộp).
