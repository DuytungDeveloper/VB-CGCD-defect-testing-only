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

- Q05.01 [mở] D21: công thức CUB của upstream có n = 25 < d = 384; số CUB của paper đến từ cấu hình nào?
- Q05.02 [mở] Phép chiếu học từ nhãn lớp cũ (LDA, hoặc làm trắng trong lớp) có tách được các lớp chồng lấn không? Đây là hướng sửa biểu diễn nếu giả thuyết chồng lấn của E09.03 đúng. Nguồn: CT2.
  - 2026-11-04: không gian con PCA của S0 giữ 0.84–0.94 khoảng cách giữa các tâm lớp mới, thấp nhất ở ba cặp của crease, rolled_pit, waist_folding (0.842–0.857), và giữ 0.53–0.69 phương sai trong lớp (`05-pca` §4.1, đo Euclid). [SUY LUẬN] Một phép chiếu học từ lớp cũ có thể còn ít thấy hướng phân biệt ba lớp này hơn PCA; phép kiểm phải đo riêng ba cặp này.

### 06 Gaussian theo lớp (SVI)

- Q06.01 [mở] Cố định Σ = I ở phase 2 thì còn bao nhiêu? Mahalanobis có thực sự đóng góp? (D11, D20.4, Bảng 15 của paper)
- Q06.02 [mở] S0_All tăng theo số bước SVI thế nào? Một phần đã thấy ở E06.02.
- Q06.03 [mở] Khi n ≈ d (CIFAR: 400 so với 384), điều chuẩn mạnh hơn (Σ gộp, tức LDA) có cứu được ước lượng dạng đóng không? Chọn trên CIFAR, xác nhận trên GC10 một lần. Nguồn: E06.03.
- Q06.04 [tạm dừng] Hiệp phương sai low-rank/sparse: chỉ chạm giới hạn ước lượng Σ khi ít mẫu, không chạm lỗi phân cụm.

### 07 Dừng sớm

- Q07.01 [mở] logdet của lớp mới tăng hay giảm theo từng bước? Cần log logdet theo bước (Hình 2b của paper so với D14).
- Q07.03 [mở] Tắt dừng sớm thì khám phá lớp mới có khá hơn không? Dừng sớm không kích hoạt trên NEU, kích hoạt giữa chừng trên GC10 fine-tune (D14). Biến thể dự kiến: `07a-no-early-stop`.

### 08 Bộ phân lớp

- Q08.01 [mở] D24: paper có chỉ dự đoán trên các lớp đã gặp không? Code lấy argmax trên cả các khe chưa huấn luyện. Nguồn: E06.03.
  - 2026-10-20: ưu tiên thấp cho dòng PL. Ở phiên S1 của GC10 và NEU (t1), mọi khe đều đã gặp (`n_seen` = `num_classes`, `classifier/mngmm.py` dòng 216), nên `--mask_unseen` không đổi gì ở S1, kể cả bộ lọc ở bước 12; cờ này chỉ chạm bảng S0 và M_f proxy.
  - 2026-11-04: với SVI trên GC10 `ft`, không ảnh test S0 nào vào khe 5–9 ở cả 15 ảnh chụp, vì khe đã học có số hạng bậc hai dương với ảnh của lớp mình (`08-classifier` §3.2, §5.1, `6cad5fa`). Câu về paper chưa trả lời được từ văn bản; D24 giữ CANDIDATE.
- Q08.02 [mở] Vì sao bộ phân lớp 10 slot đẩy nhiều ảnh inclusion sang lớp cũ, và bộ lọc cũng loại một phần ảnh inclusion của pool? Giả thuyết: nhiễu nhãn và đồng xuất hiện với water_spot (`00-data-gc10` §3.2 và §3.4; cũ: `01_data`, mục C1), lỗi nhỏ (Q03.03), hay tham số Gaussian gần khởi tạo (D23). Nguồn: `04_s1_discovery` (cell 7, 9; một seed).
  - 2026-10-24: inclusion là nhãn lớp mới kém tin cậy nhất, và một phần ảnh inclusion có hộp water_spot, một lớp cũ (`00-data-gc10` §3.2, §3.4). [SUY LUẬN] Một phần ảnh bị đẩy sang water_spot có thể đúng theo nội dung ảnh; chưa đối chiếu với chính các ảnh bị đẩy.
  - 2026-11-04: ở S0, quyết định của bộ phân lớp do số hạng bậc hai q = −½ xᵀ(Σ⁻¹ − I)x chi phối, và Σ − I lớn dần theo số bước Adam (`08-classifier` §3.1–3.3). [SUY LUẬN] Nếu khe mới ở S1 học ít bước hơn khe cũ, q của chúng nhỏ hơn và chúng thua khi tranh ảnh inclusion với khe cũ; kiểm ở 12, 14 bằng cùng phép tách.

### 09 Phân cụm

- Q09.01 [mở] Khoảng cách S1_New so với paper trên CIFAR-100. Manh mối: S1_New gắn với độ cân bằng cụm ở S1 (note LOG `c100_km_evhung_base400_s4`, D9).
- Q09.02 [mở] Ước lượng k trên GC10 (silhouette thất bại): giao thức không biết trước k.
- Q09.03 [mở] New có thấp hơn khi `cluster_max_min` cao không?
- Q09.04 [mở] Gán nhãn bằng optimal transport với biên nới lỏng có thắng k-means không?
- Q09.05 [mở] Nhãn thật tối thiểu có được dùng cho lớp mới (như setting Active GCD), hay chỉ cho lớp cơ sở? (QD16)
- Q09.06 [mở] Ở bước 09, phương pháp nào cho nhãn giả tốt hơn k-means theo bộ đo tầng 1 (ứng viên: GMM trên PCA-64 hoặc covariance chéo, phân cụm phổ trên đồ thị kNN, HDBSCAN), và thứ hạng có giữ ở tầng 2 không? Siêu tham số không chỉnh trên pool có nhãn thật của lớp mới. Nguồn: QD18.
  - 2026-11-01: E03.02: probe k-NN (k = 20, cosine) gần như không đọc được lớp hiếm (crease 0/11 ở cả 4 bộ, rolled_pit 1–2/6) dù probe tuyến tính đọc được. [SUY LUẬN] Phương pháp dựa trên láng giềng gần (phân cụm phổ trên kNN, HDBSCAN) cần được kiểm riêng cho lớp hiếm trước khi xếp hạng.
- Q09.12 [mở] k-means ở bước 09 (`n_init` 10) có ổn định theo thứ tự hàng của pool không? Ở r0 seed 4, cùng trạng thái RNG, chỉ đổi thứ tự hàng (do bước 04 xáo theo seed): ARI 0.73, inertia 82926 thành 82757, balanced accuracy lớp mới trên pool 0.32 thành 0.56, crease có cụm riêng (`04-loader` §3.3; một hoán vị). Cần đo trên 15 ảnh chụp × nhiều hoán vị: mức dao động; inertia thấp hơn có đi cùng balanced accuracy cao hơn không; tăng `n_init`, hoặc chọn nghiệm inertia thấp nhất qua nhiều thứ tự, có giúp không. Nếu dao động lớn, nó cũng ảnh hưởng cách đọc mọi kết quả theo seed (Q14.03) và thiết kế E09.06. Cấu hình nào thuộc thí nghiệm thì tiền đăng ký trước khi chạy trên pool lớp mới. Nguồn: `04-loader` (`3f6e08e`).
  - 2026-11-04 (QD26): với cùng trạng thái RNG, đổi thứ tự hàng cho cùng phân phối nghiệm như rút khởi tạo mới (k-means++ chọn tâm theo chỉ số hàng; Lloyd không phụ thuộc thứ tự; đã thử trên dữ liệu đồ chơi), nên câu hỏi là độ ổn định của best-of-10 theo khởi tạo, và ARI 0.73 ở trên là một lần rút. Pool khác nhau giữa các seed (50 ảnh reserve), nên inertia chỉ so được trong cùng pool. Phần không nhãn: `09-cluster`; phần có nhãn: E09.07 (tiền đăng ký 2026-11-04).
- Q09.13 [mở] k-means ở bước 09 có chia silk_spot (và inclusion) theo camera hoặc cuộn thép không? Với 698 ảnh lớp mới, trong đặc trưng thô, phần chỉ do camera lớn hơn phần chỉ do lớp; camera giải thích 0.43 phương sai trong silk_spot, cuộn và camera cùng giải thích 0.61 (`05a-camera` §3.1, §4.1). Kiểm ở `09-cluster` bằng bảng cụm × camera và cụm × cuộn trên 15 ảnh chụp (camera, cuộn lấy từ tên file, không cần nhãn lớp). Nếu đúng, đây là một cơ chế của tranh cụm ở K = 5 (QD24) và là đầu vào cho thiết kế E09.06, E09.07. Nguồn: `05a-camera` (`75e1bf9`).

### 10 Đặt tên cụm

- Q10.03 [mở] Cụm bám theo cuộn thép có co lại trên tập test không? GC10 chia train/test theo cuộn (không chung cuộn nào), và silk_spot có 420 ảnh pool từ 16 cuộn. Nếu k-means chia silk_spot theo cuộn (Q09.13), phần ảnh test lớp mới rơi vào cụm đó (`cluster.predict_raw(test1_x)`) sẽ nhỏ hơn phần ảnh pool lớp mới của cụm. Đo ở `10-naming`: tỷ lệ mỗi cụm trên 698 ảnh pool lớp mới so với trên 214 ảnh test lớp mới, kèm mức tập trung theo cuộn của cụm. Nếu đúng, đây là một nguồn chênh giữa tầng 1 và tầng 2 (Q14.04). Nguồn: QD26.

### 12 Bộ lọc cũ/mới

- Q12.01 [mở] Trên CIFAR, trong khoảng 4000 mẫu mỗi phiên bị lọc là "mới", bao nhiêu là dương tính giả, bao nhiêu âm tính giả?
- Q12.02 [mở] Một bộ lọc không dùng nhãn và không dựa vào pass F yếu. Nguồn: E06.03.
- Q12.03 [mở] Re-labeling đúng như paper (D12) có tốt hơn cách code hiện đang chỉ lọc không?

### 14 Đánh giá

- Q14.01 [mở] `exp.py agg`: cột `paper` cần tra theo bộ dữ liệu (hiện luôn là số CIFAR).
- Q14.03 [mở] S4_New ổn định, S5_New không: có đánh đổi giữa hai phiên? (r = −0.55, n = 5, không có ý nghĩa; E09.01)
- Q14.04 [mở] Trên cấu hình paper, vì sao balanced accuracy của phân hoạch pool (tầng 1) cao hơn nhiều so với balanced novel cuối pipeline trên test (tầng 2), nhất là ở FT seed r1, trong khi trên `fz` hai con số gần nhau? Mất ở khâu nào: chuyển từ pool sang test, đặt tên cụm, hay bộ phân lớp 10 slot? Ảnh hưởng: tầng 1 có dùng được để sàng lọc phương pháp không (QD18). Nguồn: E09.05, mục Kết quả và Ghi chú (2026-10-20): so trên cùng seed 0–4, r0 và r1 bằng nhau ở tầng 1 nhưng cách xa nhau ở tầng 2.
  - 2026-11-01: E03.02 thêm mốc biểu diễn: probe `lr_bal` (có nhãn) đạt balanced novel 0.61 / 0.68 / 0.71 trên `ft` r0 / r1 / r2, cuối pipeline 0.39 / 0.29 / 0.36. Đặc trưng không phải chỗ mất chính; chỗ mất nằm ở các bước 09–14.

## Đã xong hoặc bỏ

### 02 Fine-tune

- Q02.04 [xong] Đánh đổi ổn định–thích nghi có vững qua các FT seed không?
  - Kết luận (2026-10-01): giữ lớp cũ tốt hơn ở cả 3 FT seed; khám phá lớp mới thì không ổn định (E02.04).
- Q02.05 [bỏ] EMA teacher, consistency: cần huấn luyện mạng và augmentation ảnh, không hợp với đặc trưng đóng băng tính sẵn.

### 04 Nạp dữ liệu theo phiên

- Q04.02 [xong] S0 không nhãn: điểm thứ ba của trục lượng nhãn.
  - Kết luận (2026-10-01): chạy được; khám phá lớp mới không cần nhãn S0 (E04.01). Đầu thấp của trục lượng nhãn sau đó đổi thành nhãn thật tối thiểu (QD16).

### 05 PCA

- Q05.03 [xong] Các thành phần PCA đầu của S0 mã hoá camera (và cuộn thép) hay lớp lỗi? Đo η² (tỉ số tương quan, correlation ratio) của từng thành phần theo camera và theo lớp. Nếu camera chiếm các thành phần đầu, Gaussian ở 06–14 có thể gom ảnh theo camera; phân cụm ở 09 chạy trên đặc trưng thô nên cần đo riêng. E03.01 đã loại chuẩn hoá camera bằng trừ trung bình, nên câu này hỏi tín hiệu camera nằm ở đâu, không thử lại cách sửa đó. Làm ở `05a-camera` (QD24). Nguồn: `00-data-gc10` §2.2–2.4 (Cramér's V giữa lớp và camera 0.31).
  - Kết luận (2026-11-04): ở S0, các thành phần đầu mang lớp; với lớp mới, camera mang nhiều phương sai hơn lớp ở cả PCA-64 và đặc trưng thô, ở cả 3 FT seed; trong silk_spot, cuộn thép và camera cùng giải thích phần lớn phương sai (`05a-camera`, `75e1bf9`). Câu tiếp: Q09.13.

### 06 Gaussian theo lớp (SVI)

- Q06.05 [xong] Vì sao Σ học ở S0 có det < 1 (det trung bình của lớp cũ 0.3707 ở r0 s4, notebook cũ `04_s1_discovery`, tag `nb-v1`) trong khi 24–32/64 hướng của mỗi lớp cũ có phương sai trong lớp lớn hơn 1 (`05-pca` §3.2)? Ứng viên [SUY LUẬN]: với Σ = I + E nhỏ, log det ≈ tr E − ½‖E‖²_F, và Adam đẩy cả 4032 phần tử ngoài đường chéo (D23), nên phần ngoài đường chéo kéo log det xuống. Kiểm ở `06-gauss-svi`: tách log det của `params0` thành tr E và ‖E‖²_F. Ảnh hưởng: nhánh của luật dừng (D14) và cách đọc 07.
  - Kết luận (2026-11-04): hai phần cùng kéo log det xuống: phần ngoài đường chéo của E (chiếm phần lớn ‖E‖²_F, có ở mọi lớp cũ) và vết âm của E; thang đo của dữ liệu một mình không quyết định nhánh det của luật dừng (`06-gauss-svi` §3.2, `4112807`).

### 07 Dừng sớm

- Q07.02 [xong] `--scaling-factor 1.2` tác động vào đâu trong luật dừng (D14)?
  - Kết luận (2026-11-04): không tác động vào phép tính nào; `self.scaling_factor` chỉ được gán, `main.py` chỉ in và truyền cờ đi; hệ số của luật dừng tính từ số lớp (`07-early-stop` §2.1, `0d6d7ad`).

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

### 10 Đặt tên cụm

- Q10.01 [xong] Đổi tên cụm có cứu được recall của lớp hiếm không?
  - Kết luận (2026-10-02): không; một cụm chỉ mang một tên, và hai cụm silk thừa lấy tên rolled_pit và crease (E09.03).
- Q10.02 [bỏ] Ánh xạ nhãn giả động (dynamic pseudo-label mapping): pipeline chỉ phân cụm một lần mỗi phiên, không có gì để ánh xạ lại.

### 14 Đánh giá

- Q14.02 [xong] Đưa balanced accuracy vào chỉ số chính (hiện chỉ có qua `cm_summary`).
  - Kết luận (2026-11-04): balanced novel ở tầng 2 là đại lượng chính, S1_Old là ràng buộc lớp cũ (QD25); tính từ ảnh chụp bằng sổ mất điểm (10–14).

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
