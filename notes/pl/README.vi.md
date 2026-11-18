# Dòng nghiên cứu nhãn giả (PL)

Cổng vào của dòng PL: quy ước ghi chép, hệ mã của cả repo và các quyết định đã chốt.
Trạng thái câu hỏi không ghi ở đây: xem `questions.vi.md`.

## 1. Phạm vi

- Mở ngày 2026-10-03, khi các dòng cũ (GĐ1–GĐ3, CT1, CT2) được chốt hoặc tạm dừng; xem dòng đầu `notes/plan.vi.md`.
- Trọng tâm: chất lượng nhãn giả (pseudo-label) trong VB-CGCD, với nhãn thật tối thiểu (QD16).

## 2. File

- `README.vi.md` (file này): quy ước, hệ mã và quyết định; ít khi sửa.
- `questions.vi.md`: nơi duy nhất chứa câu hỏi của repo, kể cả câu chuyển từ các dòng cũ. Thứ tự: khối phiên, câu còn mở, câu xong hoặc bỏ, nhật ký. Câu nào đóng thì ghi kết luận ngay tại câu đó. Không chép số liệu, chỉ trỏ tới mã thí nghiệm hoặc entry LOG.
- `experiments.vi.md`: danh mục mọi thí nghiệm (mục 0), rồi mỗi thí nghiệm của dòng PL một mục gồm tiền đăng ký, kết quả, đánh đổi. Số liệu được phép ở đây.
- Dùng chung toàn repo: `notes/LOG.md` (exp.py tự ghi mọi run), `notes/code-changes.vi.md` (mỗi thứ mới trong code: mục đích, hàm cốt lõi, mặc định, phép kiểm hồi quy), `notes/paper-code-discrepancies.md` (mã D), `notes/old-codes.vi.md` (tra mã cũ).
- Trạng thái của một mục chỉ sửa ở nơi chứa nó; nơi khác chỉ trỏ bằng mã.

## 3. Mã, trạng thái và tag (QD22)

- Mỗi họ mã một nghĩa và một file gốc; nơi khác chỉ trỏ bằng mã. Mã không đánh lại, không dùng lại; mục bỏ ghi "bỏ" kèm lý do, không xoá. Mã cũ chỉ còn ở `notes/old-codes.vi.md`.
- Bước `00`–`14`: 00 là dữ liệu, 01–14 là các bước của mô hình theo thứ tự chạy; biến thể và phần thêm là `NNx` (`lab/README.vi.md`).
- Câu hỏi `Q<bước>.<nn>` (`questions.vi.md`): câu mới lấy số kế tiếp trong bước. Danh sách để câu còn mở ở trên; câu xong hoặc bỏ chuyển xuống mục cuối, giữ số.
- Thí nghiệm `E<bước>.<nn>` (`experiments.vi.md`): số theo thứ tự thời gian trong mỗi bước, kể cả thí nghiệm của các dòng cũ.
- Quyết định `QD<nn>`: một chuỗi cho cả repo, theo thời gian. QD01–QD15 ở `notes/plan.vi.md` §4, từ QD16 ở mục 5 file này.
- Khác biệt paper–code `D<n>` (`notes/paper-code-discrepancies.md`); lỗi môi trường `A<n>` và bản vá `PATCH <n>` (`notes/reproducibility-issues.md`).
- Mã cục bộ (phép kiểm `C<n>`, dự đoán `P<n>`, tiêu chí) chỉ có nghĩa trong mục chứa nó; trích ở nơi khác thì ghi kèm mã chủ, ví dụ `E09.05/C1`, `QD21/C2`.
- Dòng việc là tên, không phải mã: GĐ1 (Eq. 2/3, 17–27/9), GĐ2 (hai rò rỉ nhãn trên CIFAR-100, 28–30/9), GĐ3 (dữ liệu công nghiệp, 30/9–2/10), CT1 (ước lượng hiệp phương sai, 1–2/10), CT2 (khám phá lớp mới trên GC10, 2–4/10), PL (từ 3/10).
- Nhóm run: gọi bằng tag hoặc bằng mã thí nghiệm, không đặt chữ cái riêng.
- Trạng thái câu hỏi: mở · đang làm · chờ <việc> · tạm dừng · xong · bỏ. Thí nghiệm: chưa tiền đăng ký · đã tiền đăng ký · đang chạy · xong (kèm SỐNG hoặc LOẠI khi có quy tắc quyết định) · bỏ.
- Tag run mới:
  - thí nghiệm: `<dataset>_e<bước>_<nn>[_<biến thể>]_s<N>`, ví dụ `gc10_e09_06_r0_s3`, để `exp.py agg gc10_e09_06_r0` gom đúng các seed;
  - chuỗi mặc định của bàn mổ: `<dataset>_chain_<đặc trưng>_s<N>`, ví dụ `gc10_chain_ft_r0_s0` (từ 2026-10-21; trước đó `<dataset>_v2_<đặc trưng>_s<N>`, như 15 run gốc `gc10_v2_ft_r<r>_s<N>`).
  Tag cũ giữ nguyên.
- Trong `notes/LOG.md`, dòng `why:` bắt đầu bằng mã thí nghiệm hoặc quyết định, ví dụ `E09.06: ...`, `QD21/C1: ...`.

## 4. Thí nghiệm

1. Lấy mã E kế tiếp trong bước, thêm một dòng vào danh mục (mục 0 của `experiments.vi.md`), rồi viết mục tiền đăng ký: câu hỏi, giả thuyết, đại lượng (công thức kèm nguồn), dự đoán, phép kiểm toàn vẹn, quy tắc quyết định, điều kiện bỏ ý tưởng.
2. Commit và push; ghi hash của commit tiền đăng ký vào mục.
3. Chạy qua `exp.py run`, kể cả script offline (script ghi số vào `$EXP_RUN_DIR/metrics.json`, vì `console.log` không vào git). Commit một run gồm thư mục run, `notes/LOG.md` và `runs/INDEX.txt`, vì `exp.py` viết lại `INDEX.txt` sau mỗi run (E03.02 từng sót file này, sửa ở `77bb06a`).
4. Chỉ thêm Kết quả, Đánh đổi, Kết luận; không sửa phần tiền đăng ký. Thay đổi làm sau khi đã thấy dữ liệu ghi riêng, kèm ngày và lý do. Ngoại lệ duy nhất: đổi mã theo QD22; bản gốc kiểm được bằng hash commit ghi trong mục.
5. Cập nhật câu hỏi tương ứng trong `questions.vi.md`.

Quy tắc chung của repo (commit trước khi chạy, Hungarian chỉ lúc đánh giá, mặc định = upstream) giữ nguyên theo `notes/plan.vi.md` §2.

## 5. Quyết định

QD01–QD15 (các dòng cũ) ở `notes/plan.vi.md` §4.

### QD16 — Trọng tâm nhãn giả, nhãn thật tối thiểu (2026-10-03)

- Nội dung: dòng mới tập trung vào chất lượng nhãn giả; đầu thấp của trục lượng nhãn là nhãn thật tối thiểu, không phải 0.
- Thay cho: đích ngày 2026-09-30 "chạy được hoàn toàn không nhãn, kể cả lớp cơ sở".
- Lý do: theo đề nghị nhận ngày 2026-10-03, vì trong nhà máy luôn có một ít nhãn thật.
- Bằng chứng nền: nhãn S0 gần như không đổi khả năng khám phá lớp mới (E04.01; bảng ở `notes/improvement-1.vi.md` §3.2), nên lượng nhãn và chất lượng nhãn giả là hai câu hỏi tách rời.
- Chưa chốt: nhãn tối thiểu có được dùng cho lớp mới không (Q09.05).

### QD17 — Ghi chép: `.md` là bản chuẩn (2026-10-04)

- `.md` trong repo là bản chuẩn cho cả nhóm. Sổ giấy cá nhân (mục B/E/I theo ngày) chỉ là bản tra cứu khi bị hỏi.
- Khi giấy và `.md` lệch nhau, `.md` đúng. `.md` phải tự đủ, không trỏ sang sổ giấy.
- Công cụ khác (Obsidian, tablet) không chứa số liệu hay trạng thái dự án.
- Xem lại cách ghi sau hai tuần dùng thử.
- Sửa ngày 2026-11-13 (hết hai tuần dùng thử): sổ giấy thành nhật ký khoa học, mỗi lần ghi một khối có ngày với năm mục Hỏi / Đoán / Thấy / Nghĩ / Ý, thay cho mục B/E/I. Đoán ghi trước khi thấy kết quả; ghi sau thì nói rõ là ghi sau. Phần còn lại giữ nguyên: `.md` vẫn là bản chuẩn.

### QD18 — Notebook theo ba lớp; thứ tự làm mới (2026-10-06)

- Trạng thái: thay bằng QD20 (2026-10-15).
- Nội dung: `notes/nb/` gồm bàn mổ (`scripts/replay.py`), atlas 00–04 theo luồng và khoang 10–13 theo nhóm module; mã module và thứ tự chạy định nghĩa ở `notes/nb/README.vi.md` (nay là các bước 00–14).
- Thứ tự làm (tên notebook cũ, tag `nb-v1`): 04 (mỏng) → `12_cluster` (Học, Mổ; chốt bộ đo tầng 1) → lưu trạng thái RNG trước bước 09, script tầng 1, E09.04 → `12_cluster` (Cấy) → 02, 03 (kèm đếm D24) → khoang khác khi cần → 00 cuối cùng.
- Thay cho: thứ tự chốt ngày 2026-10-05: 00 (phần bản đồ) → 04 → E09.04 → 02 → 03 → 00 (bảng nghẽn).
- Lý do: phần bản đồ của 00 không đổi quyết định thí nghiệm nào, vì E09.04 đã tiền đăng ký; trang tổng quan nên viết một lần ở cuối; 04 và `12_cluster` dựng đúng các mảnh E09.04 cần.
- Đánh giá module thay được: tầng 1 chấm đầu ra của module bằng nhãn thật (chỉ để chấm) để lọc nhanh; tầng 2 chạy cả pipeline. Thứ hạng ở tầng 1 phải được xác nhận ở tầng 2.

### QD19 — Mốc so sánh là cấu hình paper (fine-tune); `fz` là bối cảnh phụ (2026-10-06)

- Nội dung: mốc so sánh của dòng PL là cấu hình paper trên GC10, fine-tune trên tập có nhãn (`gc10_ft_cls5first_img80pct_r0`–`r2`, mỗi FT seed 5 seed phase 2; run của E02.05). Đặc trưng không fine-tune (`fz`) là bối cảnh phụ để hiểu cơ chế, không dùng để kết luận cải tiến.
- Thay cho: quyết định ngày 2026-10-05 (bối cảnh chính `fz`, cấu hình paper là bối cảnh thứ hai).
- Lý do: cải tiến phải đo so với VB-CGCD như công bố; fine-tune offline là một thành phần cốt lõi của paper (Bảng 1). Thứ tự giữ đúng: tái lập (CIFAR-100 khớp paper ở chỉ số cuối; GC10 cấu hình paper tái lập được theo `FT_SEED`), rồi mới thay đổi và so sánh.
- Bằng chứng nền: E02.04 (fine-tune giữ lớp cũ tốt hơn, ổn định qua FT seed; khám phá không ổn định; note LOG `gc10_ftseed0_km_evhung_d64_s4`).
- Hệ quả: so sánh trên `ft` ghép cặp theo FT seed (r0–r2); atlas dùng `ft` r0, seed 4 (mode đa số). E09.04 giữ tiền đăng ký trên `fz`; thí nghiệm tương ứng trên `ft` (E09.05) tiền đăng ký riêng, và hướng phân cụm thừa xét theo kết quả trên `ft`.
- Lưu ý: `fz` và `ft` khác nhau ở hai yếu tố cùng lúc (có fine-tune hay không; CLS hay pooler, D8); fine-tune trong code chỉ có cross-entropy (D3).
- Sửa ngày 2026-11-04: "mode đa số" ở dòng Hệ quả tính theo S1_New: ở r0, seed 0, 1, 3, 4 có S1_New 56.5–57.5, seed 2 có 49.1 (notebook cũ `04_s1_discovery`, tag `nb-v1`). Theo tầng 1 ở bước 09, seed 4 thuộc nhóm thiểu số: crease có cụm riêng ở seed 0–2, không có ở seed 3, 4 (QD24).

### QD20 — Bố cục `lab/`, mã theo bước, khung notebook (2026-10-15)

- Nội dung:
  - Notebook và bàn mổ nằm thẳng trong thư mục gốc `lab/`, không có thư mục con. Mục lục và quy tắc notebook ở `lab/README.vi.md`.
  - Bước 00 là dữ liệu: mỗi dataset một notebook `00-data-<dataset>.ipynb`, được phép dài. Mô hình chia thành 14 bước 01–14 theo thứ tự chạy, thay cho mã module cũ (tra ở `notes/old-codes.vi.md`).
  - Biến thể và phần thêm mang chữ cái của bước gốc, có ba loại: thay thế, chèn sau, phân tích. Số 00–14 không bao giờ đổi.
  - Mã câu hỏi và thí nghiệm theo bước (`Q<bước>.<nn>`, `E<bước>.<nn>`). Cách đánh số và chỗ ghi mã cũ: sửa ở QD22.
  - "Chạy độc lập": sửa bước k thì không chạy lại 01…k−1, chạy lại toàn bộ k+1…14. Đạt được khi có bàn mổ v2 (`lab/replay.py` nâng tại chỗ).
  - Kế hoạch file `.py`: `replay.py` (v2), `variants.py`, `score.py`, `data_io.py`; `pl_cluster.py` nghỉ khi đã có `score.py` và v2.
  - Notebook có khung cố định: cell đầu (tiêu đề; Câu hỏi, Vị trí, Bối cảnh, Kết luận), Markdown Trước và Sau mỗi cell code, cuối là Summary (EN). Không làm notebook tổng quan 00.
- Thay cho: QD18 (bố cục ba lớp trong `notes/nb/` và thứ tự làm); quyết định ngày 2026-10-05 không dời file code, nay có ngoại lệ cho `replay.py` và `pl_cluster.py`.
- Lý do: tách hẳn dữ liệu và mô hình; đánh số theo thứ tự chạy để cải tiến gắn ngay tại bước của nó và so được với chuỗi mặc định; một hệ mã đồng bộ cho bước, notebook, câu hỏi và thí nghiệm.
- Bản cũ: tag `nb-v1` (`notes/nb/`, `scripts/replay.py`, `scripts/pl_cluster.py`); dời ở commit `dd7a954`.
- Hệ quả: tiền đăng ký E09.04, E09.05 trỏ tới đường dẫn cũ, chạy lại bằng commit `89e94d4` hoặc tag `nb-v1`; `exp.py` không kiểm dirty notebook và `.md` trong `lab/`, vẫn kiểm `lab/*.py`.
- Sửa ngày 2026-10-24: khung notebook gọn lại (`lab/README.vi.md` §9, commit `c4bf75f`, chỉnh tiếp cùng ngày ở `1871141`). Cell đầu chỉ còn tiêu đề và dòng Câu hỏi; notebook chữ cái thêm dòng Vị trí. Đầu vào do cell in ra; kết luận nằm ở Summary (EN) cuối notebook. Markdown trước mỗi cell code là một tiêu đề đánh số, rồi nêu ngắn gọn mục đích hoặc câu hỏi và cách làm, không ghi nhãn; có công thức thì thêm dòng Cách làm. Markdown sau mỗi cell chỉ còn Kết quả và Ý nghĩa (bỏ Tiếp), viết khi cell đã chạy, từ output thật. Trước khi commit, kiểm: số chạy 1..N, không cell lỗi, mỗi cell code kẹp giữa hai ô Markdown, có Summary (EN). Lý do: bốn dòng ở cell đầu và ba nhãn trước mỗi cell lặp lại code, output và Summary (EN); tiêu đề chia mục và chi tiết cấp thấp để trong chú thích code theo Rule et al. 2019 (PLOS Comput Biol, quy tắc 1 và 3); Tiếp lặp lại tiêu đề của cell sau; kết luận viết trước khi chạy dễ thành số dự đoán thay vì số đo.

### QD21 — Bàn mổ v2: bước là hàm, ảnh chụp theo bước, RNG chung (2026-10-15)

- Nội dung:
  - `lab/replay.py` nâng tại chỗ lên v2. Phạm vi: t1, `--hungarian_on eval`, S0 có nhãn; GC10 và NEU.
  - Mỗi bước 04–14 là một hàm `sNN(st) -> dict`: đọc trạng thái `st`, trả các khoá mới; Vào/Ra ghi trong docstring. Bước mặc định gọi đúng hàm của pipeline, theo thứ tự của `main.py` và `MNGMMClassifier.train()`.
  - 06, 07, 08 trả thêm thành phần (cách ước lượng, luật dừng sớm, cách phân lớp) để 11–14 dùng lại, nên biến thể ở 06–08 áp dụng cả xuống 11–14. Thành phần cắm bằng cách gán đè phương thức trên đối tượng `clf`, không sửa code gốc.
  - Nhãn thật của pool và của mọi tập test nằm trong khoá `true_*`. Hàm biến thể nhận `st` đã bỏ các khoá này; chỉ các bước chấm (s08, s10, s14) và `score.py` đọc chúng. Bước mặc định 06/11/13 vẫn nhận nhãn test như `main.py`, chỉ để in log.
  - Ảnh chụp sau mỗi bước: `runs/<run_id>/snap/NN.pkl` gồm đầu ra, trạng thái RNG, commit, mã băm. Không vào git (`*.pkl` đã bị bỏ qua); chỉ dùng trên máy đã tạo ra nó.
  - Chuỗi mặc định chạy từ đầu một lần cho mỗi (bộ đặc trưng, seed); biến thể chạy tiếp từ ảnh chụp đó (`--from`, `--start`, `--at`).
  - RNG chung (common random numbers): ở lối vào 04, 05, 09 đặt lại trạng thái RNG của chuỗi mặc định.
  - GPU chạy ở chế độ tất định: `new_state` đặt `XLA_FLAGS=--xla_gpu_deterministic_ops=true` trước khi nạp jax, nên hai lần chạy cùng đầu vào trùng từng bit.
  - Phát hiện ảnh chụp cũ theo từng bước, bằng chuỗi băm: h(k) = băm(h(k−1), AST của hàm ở bước k và các hàm trong `lab/` mà nó gọi, tham số biến thể, file pipeline dùng lần đầu ở bước k). AST bỏ chú thích và docstring. File pipeline: 04 `main.py`, `dataloaders/`, `utils/`; 05 `classifier/`; 09 `clustering/`. h(04) băm thêm cấu hình, đặc trưng, file split, phiên bản các gói tính toán và code của bộ chạy. Sửa bước k chỉ làm cũ ảnh chụp từ k trở đi. Không chạy tiếp từ run gốc bị dirty.
  - Run qua `exp.py`: console log cùng định dạng `main.py`; `metrics.json` do `scripts/extract_vbcgcd.py` tạo.
  - Phép kiểm, phải qua trước khi dùng:
    - C1: chuỗi mặc định khớp 15 run `gc10_ft_cls5first_img80pct_r{0,1,2}_km_evhung_d64_s{0–4}` ở từng ảnh chụp: trùng tuyệt đối các kết quả rời rạc (bảng S0 và S1, bảng chéo, `pi_test`, bước dừng F/S, số ảnh giữ, kích thước cụm, ma trận nhầm lẫn S1); tham số S0 và S1 lệch không quá 1e-6 × max|giá trị|; inertia lệch tương đối dưới 1e-6. 15 run này cũng là ảnh chụp gốc cho các biến thể.
    - C2: với 1 seed, chạy tiếp từ mỗi k = 05…14 ra đúng như chạy từ đầu.
    - C3: biến thể giả ở 05 rút thêm RNG; phân hoạch ở 09 không đổi.
    - C4: sửa giả bước k (trong bộ nhớ) chỉ đổi h(k)…h(14); sửa chú thích không đổi mã băm nào.
- Thay cho: không. Cụ thể hoá "chạy độc lập" của QD20. Chọn ngày 2026-10-15: phát hiện cũ theo từng bước (không theo file), C1 trên cả 15 run.
- Lý do: chạy độc lập (QD20); hiệu số giữa biến thể và mặc định không lẫn nhiễu RNG (Law & Kelton); chỉ tạo lại phần bị ảnh hưởng, như `make` (Feldman 1979) nhưng so nội dung thay vì thời gian.
- Hệ quả: số của biến thể đo bằng v2 có thể lệch bản cài thẳng vào `main.py` chỉ do đường đi của RNG; biến thể được chọn làm mặc định thì đo lại. `pl_cluster.py` nghỉ khi v2 qua C1–C4; bản v1 ở tag `nb-v1`.
- Sửa ngày 2026-10-15, sau lệnh kiểm phần 1, trước khi có ảnh chụp nào: `main.py` trên GPU không lặp lại từng bit (`notes/reproducibility-issues.md` A6), nên C1 so tham số theo ngưỡng; thêm chế độ tất định cho v2.
- Sửa ngày 2026-10-17, trước khi có ảnh chụp nào:
  - C2 chạy mỗi bước k trong một tiến trình mới, như khi một run biến thể nạp ảnh chụp. Trên máy test (CPU), chạy liền 10 lần trong một tiến trình làm XLA hết bộ nhớ JIT.
  - C3 thêm đối chứng: chạy lại k-means ở 09 với RNG bị dịch phải ra kết quả khác. Không khác thì C3 không thể trượt, nên báo INCOMPLETE thay vì OK. Kiểm này hợp lệ vì k-means đọc đặc trưng gốc của pool, không đọc PCA của 05.
- Sửa ngày 2026-10-20, trước khi có ảnh chụp nào:
  - Bộ chạy kiểm đầu ra của bước 09 (`_check_09`, commit `3525e86`): sau mọi hàm của bước 09 và trước khi ghi ảnh chụp, `ids` của pool và id mà mô hình cụm gán cho tập test lớp mới (`cluster.predict_raw(test1_x)`) phải là số nguyên trong [0, increment), mỗi ảnh một id. Lý do: id ngoài khoảng không làm các bước sau báo lỗi (`confusion_matrix` ở 10 bỏ qua chúng; phép lấy chỉ số của JAX ở 11 và 13 kẹp chúng về khe cuối), nên một biến thể sai ở 09 cho số sai mà không báo. C4 thêm một trường hợp cho hàm này (cũ từ 04).
  - Quy tắc khi C1 trượt, chốt trước khi chạy:
    - 04, 05, 09, 10 chạy trên CPU và không phụ thuộc kết quả GPU (SVI chỉ dùng RNG của JAX, không rút RNG toàn cục của numpy). Kết quả rời rạc (bảng chéo của phân hoạch, `pi_test`) phải trùng tuyệt đối; inertia lệch tương đối dưới 1e-6, vì bit cuối của nó đổi theo thứ tự cộng (E09.04, sửa đổi 2026-10-06). Lệch ở nhóm này là lỗi bàn mổ (thứ tự RNG hoặc đầu vào): dừng và chẩn đoán.
    - 06–08 và 11–14 phụ thuộc SVI trên GPU. Tham số lệch quá ngưỡng của C1: lỗi bàn mổ. Tham số trong ngưỡng mà kết quả rời rạc lệch (bảng S0 và S1, số ảnh giữ, bước dừng, ma trận nhầm lẫn): chạy lại run tham chiếu của cặp đó một lần bằng `main.py` ở chế độ tất định (`XLA_FLAGS=--xla_gpu_deterministic_ops=true`, cùng chế độ với v2), rồi chạy `check` giữa run v2 và run mới này. Trùng mọi mục: bàn mổ đúng, chênh lệch ban đầu là nhiễu A6, cặp đó tính là đạt và ghi lại. Khác: lỗi bàn mổ.
    - Không dùng "chạy lại `main.py` hai lần ở chế độ thường": v2 cộng theo thứ tự của chế độ tất định, khác mọi lần chạy thường [suy luận từ A6], nên việc `main.py` không tự lật không chứng minh v2 sai; hai lần không lật cũng ít thông tin (nếu xác suất lật mỗi lần là 0.3 thì xác suất không lật cả hai lần là 0.49).
  - Kiểm trước khi chạy: môi trường `893900df6375` chỉ khác môi trường của 15 run tham chiếu (`f70b1aeb7805`) ở 21 gói Jupyter; đủ log và tham số của 15 run tham chiếu; mã băm 3 bộ đặc trưng trùng ngày 2026-10-05; JAX chạy trên GPU; driver 617.42 (run tham chiếu: 616.92; A6: với 617.42, mọi kết quả in ra của r0 seed 0 vẫn trùng).
- Hiện trạng (2026-10-20): đã cài ở các commit `737e8c4`, `a5c9545`, `86ed44d`, `3525e86`; C1–C4 đạt, nên được dùng số của bàn mổ. C1: cả 15 run `gc10_v2_ft_r<r>_s<N>` (r = 0–2, N = 0–4) trùng run của E02.05 ở mọi mục, không phải dùng tới quy tắc khi C1 trượt; trong 15 cặp, nhiễu GPU của `main.py` (A6) không làm lật kết quả rời rạc nào. C2 và C3 chạy trên `gc10_v2_ft_r0_s0`. Output ở `runs/<run>/check_c*.txt` (commit `d62d059`). Ảnh chụp của 15 run này là ảnh chụp gốc cho các biến thể; chúng chỉ có trên laptop, vì `*.pkl` không vào git.
- Lịch sử git (2026-10-20): trước khi push, message của 4 commit sau `6bbf204` được viết lại để bỏ hai dòng ghi công tự động ở cuối; cây thư mục của từng commit không đổi, chỉ hash đổi (hash mới = hash cũ): `3525e86` = `7a6291b`, `9a07e5b` = `67b927d`, `d62d059` = `02f9c9c`, `8ea7e40` = `2f033b5`. `config.json` của 15 run và ảnh chụp vẫn ghi hash cũ `67b927d` (bản ghi, không sửa), tức commit `9a07e5b`.
- Sửa ngày 2026-10-21: chỉ còn một bàn mổ. Bản đầu (`replay()`, gọi là v1) đã xoá khỏi `lab/replay.py` (commit `370cf86`), cùng `lab/pl_cluster.py` (`71dab1f`), sau khi `lab/score.py` tái lập tầng 1 của E09.05 ở 15/15 cặp (`b81b78d`); bản gốc ở tag `nb-v1`. Từ nay gọi là "bàn mổ", không gọi v1, v2. Chữ "v2" chỉ còn ở tag của các run cũ (`gc10_v2_ft_*`, `gc10_e09_05_v2_*`), ở bản ghi viết trước ngày này, và ở một thông báo lỗi trong `new_state`: chuỗi đó nằm trong chuỗi băm, sửa thì 15 ảnh chụp gốc thành cũ, nên giữ nguyên. Lệnh `run` dừng khi JAX không chạy trên GPU và ghi thiết bị vào meta của ảnh chụp (`1253ce6`).

### QD22 — Hệ mã thống nhất cho cả repo (2026-10-18)

- Nội dung:
  - Các họ mã, trạng thái và tag như mục 3.
  - Đánh số lại một lần: câu hỏi theo bước, câu còn mở trước, câu xong hoặc bỏ sau; thí nghiệm theo thời gian, kể cả GĐ1–GĐ3, CT1, CT2; quyết định thành một chuỗi QD01–QD22.
  - Mọi `.md` viết lại theo mã mới, kể cả bản ghi (tiền đăng ký, `notes/LOG.md`, `notes/plan.vi.md`). Mã cũ chỉ còn ở `notes/old-codes.vi.md`.
- Thay cho: phần mã của QD20 (mã cũ ghi trong ngoặc ở mọi file); quy tắc không sửa dòng `why:` cũ trong LOG.
- Lý do: một chữ đang mang nhiều nghĩa (Q, I, C, A, B, E1, H2); có mã treo (D24 chưa có trong danh sách D); một thứ có hai tên (cải tiến 2 = hướng 2); dòng tra mã cũ trong ngoặc làm rối các file chính.
- Hệ quả: commit message và chú thích trong code giữ mã cũ, tra ở `notes/old-codes.vi.md`. Từ nay mã không đánh lại, không dùng lại.
- Sửa ngày 2026-10-21: tag chuỗi mặc định của bàn mổ đổi thành `<dataset>_chain_<đặc trưng>_s<N>` (mục 3), vì không còn v1, v2 (QD21, sửa ngày 2026-10-21). Tag cũ `<dataset>_v2_<đặc trưng>_s<N>` giữ nguyên.

### QD23 — Dựng nền `lab/` trước cải tiến (2026-10-22)

- Nội dung:
  - Dựng đủ notebook nền theo thứ tự pipeline, xong notebook này mới sang notebook sau: `lab/data_io.py`, `00-data-gc10`, `00-data-neu`; rồi `01`–`03` (đọc đặc trưng và file của phase 1, không chạy lại); rồi `04`–`14` trên ảnh chụp `gc10_v2_ft_r0_s4` (r0 seed 4, như QD19).
  - Code dùng chung giải thích một lần ở bước đầu dùng nó (SVI ở 06, bộ phân lớp ở 08); các bước dùng lại (11, 13; 12, 14) trỏ về đó.
  - Mỗi notebook đối chiếu số với một nguồn đã kiểm: notebook dữ liệu với file split (số ảnh từng lớp, mục `diagnostics`) và với đầu vào bước 04 (pool 748 ảnh, test lớp mới 214 ảnh); `01`–`03` với mã băm đặc trưng mà các run đã ghi (`config.json`, mục `inputs`); `04`–`14` chạy lại bước đó từ ảnh chụp của bước trước và phải trùng ảnh chụp (ảnh chụp đã trùng run `main.py`, QD21/C1).
  - Cải tiến (14a, E09.06 và các biến thể khác) làm sau khi xong nền.
- Thay cho: thứ tự chốt ngày 2026-10-20 (14a trên 15 ảnh chụp gốc → tiền đăng ký E09.06 → notebook 14a, 09b); quy tắc "notebook chỉ được tạo khi bước đó có câu hỏi hoặc thí nghiệm đang làm" ở `lab/README.vi.md` §1, nay chỉ áp cho notebook chữ cái.
- Lý do: `lab/` là chỗ mổ xẻ model để hiểu toán và thử cải tiến; cần hiểu đủ từng bước, trên số đã đối chiếu, trước khi sửa bước nào.
- Hệ quả: bàn mổ đã chốt cho bước 04–14 trên GC10 `ft` (QD21); còn thiếu `lab/data_io.py` cho bước 00; NEU và `fz` ở bước 04–14 chưa có ảnh chụp, làm khi một notebook cần. E09.06 vẫn là thí nghiệm kế tiếp của dòng PL, làm sau khi xong nền.

### QD24 — Dựng notebook 05–14: ảnh chụp minh hoạ, sổ mất điểm, Q09.12 (2026-11-04)

- Nội dung:
  - Ảnh chụp: hình và giải thích dùng `gc10_v2_ft_r0_s4`. Từ bước 09, cell nào cần cơ chế của một cụm crease thì dùng `gc10_v2_ft_r0_s2`, chốt trước khi viết, không chọn lại theo kết quả. Số để kết luận tính trên cả 15 ảnh chụp `gc10_v2_ft_r<r>_s<N>`; mỗi minh hoạ ghi ảnh chụp đó đứng ở đâu trong 15. Kết luận về biểu diễn lấy FT seed làm đơn vị (n = 3), không lấy 15 ảnh chụp.
  - Sổ mất điểm (loss ledger): bảng lớp × bước trên 15 ảnh chụp, mỗi ô là số ảnh còn đúng sau bước đó; đúng/sai lưu theo từng ảnh. Một hàm trong `lab/score.py`, viết ở notebook 10, bước đầu tiên mà cụm có tên; 11–14 mỗi notebook thêm cột. 14a chỉ đọc sổ.
  - Q09.12 tách đôi: notebook 09 chỉ mô tả phần không dùng nhãn (phân bố inertia qua nhiều lần khởi tạo, ARI giữa các nghiệm, nghiệm của pipeline đứng ở đâu). Phần chấm chất lượng là E09.07 ("khởi tạo nhiều lần, giữ inertia thấp nhất"), tiền đăng ký với đại lượng chính chốt trước; làm sau khi xong nền, trước E09.06.
  - Notebook 07 viết đúng thứ tự: luật dừng, hai nhánh và ngưỡng của lượt F, lượt S tính từ `params0`; 11 kiểm lượt F, 13 kiểm lượt S và vẽ quỹ đạo det theo bước.
  - Notebook 05: bảy cell code; phần camera sang `05a`. Phần lớp mới trong không gian con S0 chỉ đo phương sai trong lớp và khoảng cách giữa các tâm lớp, không tính phần dư của từng ảnh hay chỉ số tách cũ/mới nào, để Q12.02 còn sạch.
  - Tính mới: cải tiến ở giai đoạn này làm từng bước nhỏ, không cần đột phá; sau đó đọc thêm vài paper có hướng mới để vá những chỗ gây hại mà sổ mất điểm chỉ ra.
- Thay cho: quy tắc tạm "Ý 6" trong file bàn giao ngày 2026-11-01 (chưa vào repo); trong QD23, câu "`04`–`14` trên ảnh chụp `gc10_v2_ft_r0_s4`" (nay: minh hoạ trên ảnh chụp đó, kết luận trên 15 ảnh chụp) và vai trò của 14a (nay: đọc sổ mất điểm).
- Lý do:
  - Ở K = 5, các lớp mới tranh nhau một số cụm cố định. Qua 15 ảnh chụp, waist_folding luôn sở hữu 1 cụm, rolled_pit không cụm nào, silk_spot 2–3 cụm; khi silk_spot chiếm 3 cụm (12/15), cụm còn lại về crease (4) hoặc inclusion (8); cả crease lẫn inclusion có cụm chỉ ở r0 s0, s1 (khoá `k5_owned_*` trong `metrics.json` của `gc10_e09_05_v2_*`). Một ảnh chụp không đại diện được cho cả 15. r0 s2 là đối chứng sát nhất của r0 s4: cùng silk_spot 3 cụm, cụm còn lại về crease thay vì inclusion.
  - Chấm "khởi tạo nhiều lần" bằng nhãn thật ngay trong notebook là nhìn trước kết quả của một ứng viên cải tiến (`lab/README.vi.md` §9). E09.06 cũng chạy k-means, nên phải đứng trên k-means đã ổn định, để phần thắng do gộp cụm không lẫn với phần thắng do tối ưu.
  - Ngưỡng dừng chỉ cần `params0` và code, như notebook cũ `04_s1_discovery` (tag `nb-v1`) đã tính; viết 07 sau 13 thì 13 phải giải thích luật dừng trước.
  - Sổ mất điểm cho mọi bước một đại lượng chung. Ở bước 09 cụm chưa có tên, nên sổ bắt đầu ở 10.
- Hệ quả: câu hỏi camera vào `questions.vi.md` khi viết 05; E09.07 vào danh mục thí nghiệm và rủi ro về đại lượng chính vào Q09.12 khi viết 09; `lab/README.vi.md` §3 và §9 sửa theo quyết định này.
- Sửa (2026-11-04, `10-naming` §4, trước khi viết hàm): thiết kế sổ mất điểm. Hàm `ledger` của `lab/score.py` lưu đúng/sai từng ảnh, mỗi cột một bước; `ledger_summary` báo theo lớp (mỗi cột cho một balanced novel) và theo luồng đúng→sai, sai→đúng giữa hai cột liền nhau, vì đúng/sai không đi một chiều qua các bước. Ba tập hàng: test lớp mới (cột 09, 10, 11, 14; cột 11 là phản thực tế: bộ phân lớp của lượt F trên test), test lớp cũ (cột 08, 14; ở 14 tách lỗi cũ→mới khỏi cũ→cũ), pool (cột 09, 10, 12; ở 12 ảnh cũ đúng khi bị loại, ảnh lớp mới đúng khi được giữ và nằm trong cụm mang đúng tên). Cột 09: cụm của ảnh có lớp chủ trong pool là lớp của ảnh (nhiều–một, chỉ để chẩn đoán); cột 10: tên của pipeline (QD15). Lý do: một cột chỉ đếm ảnh còn đúng thì giấu ảnh được bước sau sửa lại, và không cho balanced novel theo từng bước.
- Sửa (2026-11-05, đầu phiên, trước `12-filter`): (a) notebook 13 vẽ thêm Hình 2 của paper trên GC10: độ chính xác lớp cũ, lớp mới, toàn bộ và det trung bình theo bước, cho lượt F (tới bước 1000) và lượt S (tới điểm dừng), dùng chung wrapper ở mức class của quỹ đạo det; (b) sổ mất điểm: tập test lớp cũ có thêm cột 11 phản thực tế (bộ phân lớp với `params_f`), thêm ở 14 cùng các cột 08, 14. Lý do: kết luận "cột 11 đã gần cuối pipeline nên dồn cải tiến vào 09–11" chỉ nhìn lớp mới, trong khi 12–13 giữ phía ổn định (luật dừng cắt lượt S còn 48–64% ngân sách bước); hiệu thuần 2–3 điểm balanced chỉ cỡ một ảnh rolled; E06.03 (`fz`) đổi bộ phân lớp mà balanced novel không đổi. Hình 2 thử thẳng lập luận P1–P3 và Eq. 9–11. Đường cong lượt F sau bước dừng của lượt S là nhìn trước một phần Q07.03: nếu sau này tiền đăng ký `07a-no-early-stop` thì ghi đã thấy. Câu "cải tiến dồn vào bước nào" hoãn tới sau 14.
- Sửa (2026-11-05, sau `14-eval`, `58a698e`): bỏ `14a-loss-by-step`; chuỗi mất điểm theo bước (Q14.04) nằm ở `14-eval` §4.1, vì 14a chỉ đọc sổ và một cell ở 14 đã đủ. Sổ thêm cột khe được chọn (`slot_08`, `slot_11`, `slot_14`) để tách cũ→mới khỏi cũ→cũ và mới→cũ khỏi mới→mới, và hàm `ledger_state(st)` để E09.07 đo bằng đúng hàm của sổ (`73e7fae`). Câu "cải tiến dồn vào bước nào" chưa quyết; số cho câu đó ở `14-eval` §3.1, §4.1.

### QD25 — Đại lượng chính của dòng PL; `05a` trước 08 (2026-11-04)

- Nội dung:
  - Đại lượng chính (Q14.02) của mọi thí nghiệm cải tiến, từ E09.07: balanced novel ở tầng 2, tức trung bình recall của 5 lớp mới trên 214 ảnh test lớp mới, với tên cụm đặt như chuỗi mặc định (QD15). So biến thể với mặc định theo cặp (cùng FT seed, cùng seed, RNG chung): lấy hiệu trung bình trong từng FT seed, rồi trung bình của 3 FT seed. FT seed là đơn vị (n = 3).
  - Ràng buộc lớp cũ: S1_Old (241 ảnh test lớp cũ, cuối phiên S1) không được giảm quá một biên chốt trong tiền đăng ký. Dùng mức S1_Old, không dùng M_f: M_f là hiệu, nên giảm được chỉ bằng cách hạ S0_All (D9).
  - Chỉ báo cáo, không dùng để quyết định: S1_New (với t1 bằng M_d), S1_All, M_f, recall từng lớp kèm khoảng Wilson, balanced accuracy 10 lớp. Đại lượng cơ chế (sổ mất điểm, tầng 1) chốt riêng trong từng tiền đăng ký.
  - Ngưỡng của quy tắc quyết định và biên của S1_Old chốt trong từng tiền đăng ký, dựa trên dao động giữa các seed của chuỗi mặc định (đo ở 14), trước mọi run biến thể.
  - `05a-camera` (Q05.03) làm ngay sau 05, trước 08. Phạm vi: tín hiệu camera và cuộn thép trong không gian PCA-64 (dùng ở 06–14) và trong đặc trưng thô (dùng ở 09), tách khỏi tín hiệu lớp bằng phân tách phương sai (variance partitioning). FT seed là đơn vị; `fz` là bối cảnh phụ ở phần đặc trưng thô. Chỉ mô tả: không chạy cách sửa nào (E03.01), không tính chỉ số tách cũ/mới nào (QD24, Q12.02).
- Thay cho: Q14.02 còn mở; QD24 chưa chốt thời điểm làm `05a`.
- Lý do:
  - Câu hỏi nghiên cứu là khám phá loại lỗi, nên mỗi lớp mới phải nặng như nhau (balanced accuracy; Brodersen et al. 2010, ICPR). S1_New nặng theo số ảnh: silk_spot chiếm 124/214 ảnh test lớp mới, và ở r0 s4, 88/122 ảnh lớp mới đoán đúng là silk_spot (notebook cũ `04_s1_discovery`, cell 9, tag `nb-v1`). Khi đổi cụm inclusion lấy cụm crease (r0 s2 so với s3, s4; QD24), S1_New cân hai lớp theo 43 so với 11 ảnh test, còn balanced novel cân 1 so với 1.
  - Bước dừng của lượt S đi theo phân hoạch ở 09 (`07-early-stop` §3.1), nên một cải tiến ở 09 có thể đổi cả kết quả lớp cũ. Một đại lượng gộp cũ và mới sẽ giấu đánh đổi ổn định–thích nghi; đại lượng chính cộng một ràng buộc riêng cho lớp cũ thì không.
  - Giá phải trả: balanced novel nhiễu (1 ảnh rolled_pit = 3.3 điểm, 1 ảnh crease = 1.8 điểm), nên phải so theo cặp và đặt ngưỡng theo dao động đã đo.
  - Camera gắn với lớp: Cramér's V 0.31; punching_hole 100% từ 3 camera, waist_folding 89% từ 2 camera, còn silk_spot, inclusion, crease trải đủ 8 camera (`00-data-gc10` §2.3). Biết trước tín hiệu camera nằm ở đâu thì đọc đúng hơn các lỗi của 08 và các cụm của 09.
- Hệ quả: Q14.02 đóng bằng QD25; balanced novel được tính từ ảnh chụp bằng sổ mất điểm (10–14). Tiền đăng ký E09.07, E09.06 dùng QD25. `lab/README.vi.md` §3 ghi `05a` làm trước 08.
- Sửa ngày 2026-11-13, sau E09.06 (`b06586c`), trước tiền đăng ký E09.08:
  - Trên GC10 t1, T = 0.0273 và B = 0.615 (`14-eval` §3.3, chốt ở E09.07 phần 2) giữ cố định làm cỡ hiệu ứng tối thiểu, không tính lại khi mặc định của dòng PL đổi sang `09a` + `09b` (E09.06). Lý do: vạch trôi theo từng biến thể được nhận thì các thí nghiệm không còn so được với nhau; σ ước lượng từ 3 FT seed vốn rộng (khoảng tin cậy 95% 0.0196–0.0451, E09.07), tính lại chỉ thêm nhiễu vào vạch. Tiền đăng ký in kèm SD giữa các seed của mặc định mới để kiểm. CUB tính T, B riêng (QD28).
  - Hiệu tích luỹ so với upstream đo trên lần rút mới, bằng một nhánh upstream trong E09.08, không dùng lại các lần rút 0–20 của E09.06: lần rút đã dùng để quyết định cho ước lượng thiên lạc quan. Câu "đã đạt +2T" chỉ dùng sau T2 hoặc khi có số server.

### QD26 — Bước 09: notebook không nhãn và notebook có nhãn, landscape dùng chung, E09.07 lặp phép so trong ảnh chụp; notebook được chạy lâu (2026-11-04)

- Nội dung:
  - Bước 09 có hai notebook, mỗi notebook một câu hỏi chính (`lab/README.vi.md` §9):
    - `09-cluster`: không đọc nhãn lớp hay vai trò ảnh; chỉ đọc camera và cuộn từ tên file. Kiểm bước 09 (`rerun`; tách `n_init` 10 thành 10 lần khởi tạo đơn từ `rng_in`), code, landscape của k-means (Q09.12, phần không nhãn), phần không nhãn của Q09.13.
    - `09d-ownership` (phân tích, `lab/README.vi.md` §4 loại 3): dùng nhãn để mô tả nghiệm của pipeline và đo trần của mục tiêu k-means: thành phần và sở hữu cụm, hiệu ứng đồng đều, sổ Ward (chi phí gộp hai cụm n_a·n_b/(n_a + n_b)·‖μ_a − μ_b‖²; Ward 1963), nghiệm oracle (Lloyd từ tâm lớp; ba loại a/b/c chốt trong notebook trước khi chạy), độ bất ổn từng ảnh theo lớp, Q09.13 trong từng lớp, cụm crease ở r0 s2 (QD24). Không chấm bằng nhãn các cực tiểu khác của k-means (việc của E09.07). Cell 0 dừng nếu commit tiền đăng ký E09.07 không phải tổ tiên của HEAD.
  - Landscape (`lab/landscape.py`, dùng chung cho 09 và 09d): mỗi ảnh chụp 10.000 lần khởi tạo đơn k-means++ từ một luồng RNG riêng (không phải `rng_in`), 1 luồng OpenMP mỗi tiến trình; ở r0 s4 thêm 1.000 lần `n_init` 10 với `rng_in` cố định và hàng hoán vị ngẫu nhiên. Nhóm nghiệm: các phân hoạch có ARI ≥ 0.95 với nghiệm đầu nhóm, xét theo inertia tăng dần; báo thêm ngưỡng 0.90 và 0.99, kết luận phải đứng ở cả ba. Độ phủ: khối lượng chưa thấy Good–Turing n₁/N trên nhóm nghiệm (Good 1953). Kết quả lưu cache trên máy (`runs/<run>/landscape/`, git bỏ qua), khoá theo băm của ảnh chụp, code và tham số; khoá sai thì tính lại. Không chạy qua `exp.py`.
  - E09.07 tiền đăng ký trước `09-cluster`, chạy sau 14: mỗi ảnh chụp 11 lần rút lồng nhau (k = 0 dùng `rng_in`), so nghiệm best-of-10 (mặc định) với best-of-1000 (biến thể) qua bước 10–14; đại lượng là trung bình theo lần rút của hiệu theo cặp. Số chính thức ra từ một hàm của `lab/score.py` chạy qua `exp.py` (các lần rút chạy trong cùng tiến trình), không qua `replay.py run --at`, vì mỗi ảnh chụp cần nhiều lần rút; phép kiểm E09.07/C1 nối hàm này với chuỗi mặc định. Chi tiết: `experiments.vi.md`, mục E09.07.
  - Cách so của QD25, cho E09.07: so kỳ vọng qua nhiều lần rút trong cùng ảnh chụp (cùng FT seed, cùng seed, RNG chung) thay vì một lần chạy; phép so một lần chạy (k = 0) báo cáo kèm. Thí nghiệm khác ở bước 09 chọn cách so trong tiền đăng ký của mình.
  - `lab/README.vi.md` §9: bỏ giới hạn 5 phút của notebook mô hình; cell chạy lâu in thời gian chạy; giữ "một câu hỏi chính" và Restart & Run All trước commit (đọc cache có khoá đúng được tính là chạy lại).
  - Q09.12: với cùng trạng thái RNG, đổi thứ tự hàng cho cùng phân phối nghiệm như rút khởi tạo mới, nên câu hỏi là độ ổn định của best-of-10 theo khởi tạo. Thêm Q10.03 (cụm bám theo cuộn trên tập test).
  - Chữ cái theo thứ tự tạo (`lab/README.vi.md` §4): `09c-multi-init` (E09.07, hàm `v09c_multi_init`), `09d-ownership`; `09a`, `09b` vẫn là phân cụm thừa và bước gộp (E09.06).
- Thay cho: trong QD24, "notebook 09 chỉ mô tả phần không dùng nhãn" (nay: `09-cluster` không nhãn, `09d-ownership` có nhãn, sau tiền đăng ký E09.07); trong QD25, phép so một lần chạy cho E09.07; giới hạn 5 phút ở `lab/README.vi.md` §9.
- Lý do:
  - Thiết kế chốt trước khi có dữ liệu thì phân tích sau không làm lệch được kiểm định (phân tích mù, Klein & Roodman 2005; tiền đăng ký, Nosek et al. 2018). Tiền đăng ký E09.07 trước `09-cluster` để không số nào của landscape hay của `09d` lọt vào thiết kế.
  - Phép so một lần chạy ở mỗi ảnh chụp dính may rủi khởi tạo của cả hai nhánh. Trung bình 11 lần rút giảm độ lệch chuẩn của phần nhiễu đó khoảng √11 ≈ 3.3 lần mà không cần thêm FT seed; hai nhánh lồng nhau dùng chung RNG. Phép so một lần chạy là trường hợp một lần rút.
  - Pool khác nhau giữa các seed (50 ảnh reserve bốc theo seed), nên inertia chỉ so được trong cùng một pool; so sánh giữa các seed làm trên ảnh chung.
  - Thử trên dữ liệu đồ chơi với scikit-learn 1.5.2 (2026-11-04, ngoài repo): `KMeans(n_init=10)` cho đúng nghiệm được giữ của 10 lần `n_init=1` liên tiếp từ cùng trạng thái RNG toàn cục (20/20 bộ 748 × 768: nhãn, trạng thái RNG sau, inertia trùng); hàng hoán vị với trạng thái RNG cố định cho cùng phân phối inertia (KS D = 0.019, p = 0.65) và ARI với nghiệm tốt nhất (D = 0.028, p = 0.19) như luồng RNG mới (mức nền D = 0.022 và 0.021). Lý do trong code: `check_random_state(None)` trả RNG toàn cục; k-means++ chọn tâm theo chỉ số hàng (`random_state.choice`, `searchsorted` trên tổng tích luỹ); Lloyd không phụ thuộc thứ tự hàng.
  - GC10 chia train/test theo cuộn: 347 cuộn train, 120 cuộn test, không chung cuộn nào; cả 8 camera có ở hai phía; silk_spot có 420 ảnh pool từ 16 cuộn. Cụm bám theo cuộn không chuyển được sang test, nên Q09.13 tách camera với cuộn, và Q10.03 đo ở 10.
  - `09d` cần landscape của 09: phần tính nằm trong `.py` dùng chung (§9: bộ đo viết một lần), có cache để `09d` không tính lại. Notebook được chạy lâu theo quyết định của Rito (2026-11-04).
- Hệ quả: `experiments.vi.md` thêm E09.07 (tiền đăng ký phần 1); `questions.vi.md` ghi chú Q09.12, thêm Q10.03; `lab/README.vi.md` §9 bỏ giới hạn 5 phút; `lab/landscape.py` viết trước `09-cluster`; mục lục `lab/README.vi.md` §3 thêm `09c`, `09d` khi có notebook.
- Sửa (2026-11-04, sau `09-cluster` §3.4b, `8982942`): ý "đổi thứ tự hàng cho cùng phân phối nghiệm như rút khởi tạo mới" (mục Nội dung, Q09.12) và phép thử đồ chơi ở mục Lý do không đúng khi trạng thái RNG cố định. Với u cố định, `searchsorted` chọn tâm ở các mức tích luỹ cố định, không lấy mẫu theo D² (ví dụ 7 điểm, trọng số 10, 5, 1, 1, 1, 1, 1: u = 0.5 chọn điểm nặng nhất 0.86 lần thay vì 0.50); trên r0 s4, KS D của inertia 0.087–0.089 so với mức nền 0.056–0.059. Phần còn đúng: khi thứ tự hàng và RNG cùng đổi (giữa các seed), việc chọn tâm là lấy mẫu theo D². Thiết kế E09.07 không dùng hoán vị hàng nên không đổi.

### QD27 — Định vị paper, CUB-200 là dataset thứ hai, VB thật làm xương sống; phase 1 CUB; laptop dò hướng, server ra số (2026-11-05)

- Nội dung:
  - Định vị: paper là một phương pháp C-GCD chung, tạm giữ tên VB-CGCD. Benchmark: GC10 t1 (ca công nghiệp khó, benchmark chính của `lab/`), CUB-200 t5 (dataset thứ hai), CIFAR-100 t5 (hiệu chuẩn server với paper). Dataset của paper không cần là dữ liệu công nghiệp. Trục lượng nhãn (QD16) thành hướng dự bị, cùng Tiny-ImageNet.
  - "VB" phải có thật: μ_k, Σ_k là biến ẩn có hậu nghiệm, đúng Eq. 7 của paper; code hiện ước lượng MLE với guide rỗng (D11) và gần như không rời khởi tạo (D23). Xương sống gồm hai biến thể; mỗi biến thể được tiền đăng ký và trình suy dẫn kèm nguồn trước khi viết code:
    - `06b-vb-s0` (E06.04): S0 có nhãn; tiên nghiệm Normal–Inverse-Wishart liên hợp, hậu nghiệm dạng đóng, phân lớp bằng phân phối dự đoán Student-t (Bayesian QDA; Murphy 2012 §4.6; Srivastava, Gupta & Frigyik 2007, JMLR). Siêu tham số của tiên nghiệm chọn bằng empirical Bayes trên S0, không nhìn test.
    - `09e-vbgmm` (E09.08, trải 09–13): phiên không nhãn bằng VB-GMM (Bishop 2006 §10.2; Corduneanu & Bishop 2001): trách nhiệm mềm thay nhãn cứng; trọng số Dirichlet là nút khối xác suất giữa nhóm cũ và nhóm mới; thành phần cũ nhận hậu nghiệm S0 làm tiên nghiệm; Σ của thành phần mới có tiên nghiệm phân cấp lấy từ các lớp cũ. Biến thể thay lượt F, bộ lọc (D12), lượt S và luật dừng (D14). Cập nhật CAVI dạng đóng; SVI (Hoffman, Blei, Wang & Paisley 2013, JMLR) là bản ngẫu nhiên của cùng các cập nhật, cùng điểm bất động. Chốt khi tiền đăng ký E09.08: thành phần cũ cố định hay cập nhật; E12.01 và `12a` còn là thí nghiệm riêng hay chỉ là ablation; cách đặt tên cụm khi chấm nhánh VB (QD15 đặt tên theo k-means).
  - Thứ tự: E09.07 (GC10) song song với phase 1 CUB; rồi E06.04 → E09.06 (`09b`, cũng là khởi tạo cho `09e`) → E09.08. Đưa vào fork bằng cờ opt-in, mặc định = upstream, kiểm hồi quy trùng số; bảng chính chạy trên server.
  - Phase 1 CUB theo cấu hình upstream (`feature_extractor/dino-cub200.py`, dataset HF `cassiekang/cub200_dataset`), thêm một sửa không đổi đầu ra: `@torch.no_grad()` khi trích đặc trưng (commit "dino-cub200.py: no_grad at extraction ...", như `dino-cifar100.py`). Lệnh:

    ```bash
    python scripts/exp.py run cub_fz_up --inputs - -- env FT_SEED=0 bash -c 'O="features/$(basename "$EXP_RUN_DIR")"; mkdir -p "$O" && python -u feature_extractor/dino-cub200.py --output_dir "$O" --batch-size 128 --seed 0'
    for r in 0 1 2; do python scripts/exp.py run cub_ft_up_r$r --inputs - -- env FT_SEED=$r FT_EPOCHS=10 FT_BATCH_SIZE=32 FT_ACCUM=4 bash -c 'O="features/$(basename "$EXP_RUN_DIR")"; mkdir -p "$O" && FT_CKPT_PATH="$O/ft_ckpt.pt" python -u feature_extractor/dino-cub200.py --finetuned --output_dir "$O" --batch-size 128 --seed "$FT_SEED"'; done
    ```

    Giữ `--batch-size 128` (D17). FT seed 0, 1, 2: paper ghi seed 0 (D4), luật ≥ 3 FT seed. Run trên laptop ngày 2026-11-05: `20261105_183044__cub_fz_up`, `20261105_183328__cub_ft_up_r0`, `20261105_185533__cub_ft_up_r1`, `20261105_191720__cub_ft_up_r2`; run `20261105_182726__cub_fz_up` hỏng trước khi trích (torch.hub bị GitHub API trả 403, giới hạn lượt).
  - Lưu ý khi chạy phase 1 CUB trên máy khác:
    - torch.hub gọi GitHub API để kiểm tra fork khi chưa có cache; bị 403 thì clone `https://github.com/facebookresearch/dino` vào `$(python -c "import torch;print(torch.hub.get_dir())")/facebookresearch_dino_main` rồi chạy lại.
    - `scripts/check_phase1.py` viết cứng cho CIFAR-100; không dùng cho CUB.
    - Ghi dòng `trainable params` trong log phase 1: số này cho biết BOFT chạy factor nào (A5). Chỉ ghi, không loại run theo số này.
    - Nhãn dựng bằng `np.unique` trên chuỗi "A photo of a ...", nên 100 lớp cơ sở là 100 tên đầu theo thứ tự chữ cái, không phải mã 001–100 của CUB. Code không ghim revision của dataset HF; ghi revision lúc tải.
    - Fine-tune dùng mọi ảnh train của 100 lớp cơ sở, trong khi bộ nạp t5 chỉ coi `num_labeled` = 2500 ảnh là có nhãn: cùng loại rò như D9. Giữ nguyên để tái lập; bản không rò làm sau, theo cách `FT_LABELED_IDX` của CIFAR-100.
    - Trước khi đọc kết quả t5: đếm các lớp có ít ảnh train hơn `num_labeled/base + num_known_inc` (D16, D22).
  - Vai trò hai máy: laptop dò hướng cải tiến; server ra số chính thức, kết luận cuối trên tập xác nhận đã khoá. Không so số giữa hai máy: mặc định và biến thể chạy trên cùng máy, cùng bộ đặc trưng; server chạy lại baseline của mình. Một cải tiến chỉ chuyển cho server khi hiệu vượt rõ ngưỡng B qua đủ các chuỗi và đạt phép kiểm cơ chế đã chốt trước; siêu tham số chọn lại trên server bằng cùng quy tắc.
  - CUB vào `lab/` bằng một biến thể bước 04 trong `lab/variants.py`, không sửa `dataloaders/`: mã băm của bước 04 phủ cả thư mục đó, sửa sẽ làm cũ mọi ảnh chụp GC10.
- Thay cho: định vị ngày 2026-09-30 (khám phá loại lỗi liên tục, có trục lượng nhãn; chưa có mã QD), nay là phương pháp C-GCD chung với trục lượng nhãn là dự bị; mục 5 (dataset) của doc chiến lược ngày 2026-11-05 trong Project.
- Lý do:
  - Mục 5 của paper tự nêu hai hạn chế: phụ thuộc chất lượng phân cụm ban đầu (gợi ý "adaptive mixture models") và ước lượng hiệp phương sai khi ít mẫu (gợi ý "hierarchical Bayesian priors"). VB-GMM với tiên nghiệm phân cấp nhắm thẳng vào cả hai.
  - CUB là nơi hạn chế (2) nặng nhất: bỏ SVI làm CUB rơi còn 27.06 ở S5 (paper Bảng 5), và khoảng cách giữa M_d và UpperBound lớn nhất trong bốn dataset (56.46 so với 79.65, paper Bảng 3).
  - E06.03 cho thấy ước lượng điểm, nhãn cứng và bộ lọc là chưa đủ: Gaussian dạng đóng khớp đúng cụm đã lẫn ảnh cũ nên bộ lọc để lọt ít nhất 16/50 ảnh cũ trên GC10 (`notes/improvement-1.vi.md` §4). Hậu nghiệm đầy đủ cộng trách nhiệm mềm là bước tiếp theo có nguyên lý.
  - GC10 chỉ có 5 lớp mới và balanced novel nhiễu (QD25), nên kết luận về phương pháp cần thêm một dataset chuẩn.
  - E06.03 đổi dấu giữa các bộ đặc trưng (GC10 và NEU tăng, CIFAR-100 giảm), nên độ lớn của một cải tiến phụ thuộc bộ đặc trưng; vì vậy không so số giữa hai máy.
- Hệ quả: danh mục `experiments.vi.md` thêm E06.04, E09.08 khi tiền đăng ký; `lab/README.vi.md` §3 thêm `06b`, `09e` khi có notebook; `notes/code-changes.vi.md` ghi sửa `no_grad` của `dino-cub200.py`; doc chiến lược trong Project cập nhật mục 5 theo QD này.

### QD28 — Bàn mổ nhiều phiên (giao thức t5) cho CUB-200 và CIFAR-100; đại lượng quyết định trên CUB; bản CUB không rò (2026-11-13)

- Nội dung:
  - File mới `lab/sessions.py`. Không sửa `chain`, `new_state` và các hàm bước của `lab/replay.py`, nên 15 ảnh chụp gốc GC10 không cũ. Phạm vi: t5 (CUB-200: 100 lớp cơ sở + 5 phiên × 20 lớp; CIFAR-100: 50 + 5 × 10); t1 là trường hợp riêng một phiên, chỉ dùng cho C0. Giao thức như QD21 (k-means, `--hungarian_on eval`, S0 có nhãn, mngmm, SVI); riêng `load_mode` nhận t1 hoặc t5, qua `new_state_sessions` của file mới.
  - Bước. Phần S0 chạy một lần: 04 nạp mọi phiên (thay `v04a_cub_t5`), 05 fit PCA trên S0 rồi chiếu mọi tập của mọi phiên, 06–08 mặc định. Mỗi phiên i = 1…5 chạy nguyên các hàm `s09`–`s14` trên một khung nhìn phiên (session view):
    - `offset` = base + (i−1)·increment; `pool_x`, `z_pool`, `true_pool_y` lấy từ pool của phiên i; `test1_x`, `z_test1`, `true_test1_y` từ test của các lớp mới ở phiên i; `test_old_x`, `test_all_x` (và `true_*`) từ test của mọi lớp trước phiên i và của mọi lớp tới phiên i; `test0_x`, `true_test0_y` là test S0, không đổi.
    - `params0` = mô hình sau phiên i−1 (S0 khi i = 1).
    - `lut` của phiên i = `lut` của phiên i−1 cộng ánh xạ của phiên i, như `to_cluster` của `main.py`; chỉ bước 10 có hàm bọc làm việc này.
    - Khung nhìn thêm khoá `session` và `seen` cho biến thể cần biết đang ở phiên nào.
  - Ảnh chụp: `snap/04.pkl` … `08.pkl`, rồi `S<i>_<NN>.pkl` với i = 1–5, NN = 09–14. Mảng có trục đầu là khe lớp chỉ lưu các khe khác mô hình trước đó; các khe chưa gặp ở S0 lưu dạng khởi tạo. Lý do: một `class_covs` đủ 200 khe × 384² ở float32 nặng 118 MB, 20 khe nặng 11.8 MB; lưu đủ khoảng 1.3 GB mỗi chuỗi CUB, lưu phần khác khoảng 0.25 GB (tính tay).
  - Chuỗi băm theo (phiên, bước): h(S_i.k) = băm(h(bước trước), mã phiên, code của bước k như QD21). Biến thể ở bước k áp cho mọi phiên, nên làm cũ ảnh chụp từ S1.k trở đi; biến thể chỉ cho một phiên viết `--at S<i>.<NN>=...`.
  - RNG chung: đặt lại ở lối vào 04, 05 và S_i.09 của mọi phiên. [suy luận từ code] Phân hoạch ở 09 chỉ phụ thuộc đặc trưng gốc của pool và RNG toàn cục; SVI chỉ dùng RNG của JAX và 10–14 không rút RNG toàn cục, nên nhiễu GPU (A6) và biến thể ở 06–08, 11–14 không đổi phân hoạch của các phiên sau.
  - GPU chạy chế độ tất định như QD21. Run qua `exp.py`; console log cùng định dạng `main.py`; `metrics.json` do `scripts/extract_vbcgcd.py` tạo (bảng S0–S5, M_d, M_f proxy).
  - Lệnh: `run`, `check` (C1), `check-gc10` (C0), `check-resume` (C2), `check-crn` (C3), `check-hash` (C4); `load`, `rerun` cho notebook. Tag chuỗi mặc định: `cub_chain_ft_r<r>_s<N>` (CIFAR-100: `c100_chain_...`).
  - Phép kiểm, phải qua trước khi dùng số:
    - C0: chạy với t1 trên 15 ảnh chụp `gc10_v2_ft_r<r>_s<N>` cho lại mọi khoá của mọi ảnh chụp (kết quả rời rạc trùng hẳn; số thực lệch không quá 1e-6 × max|giá trị|).
    - C1: 15 chuỗi CUB khớp 15 run mốc ở mọi phiên: `metrics.json` trùng (bảng S0–S5, tỉ lệ cỡ cụm, số ảnh giữ, bước dừng F/S, M_d, M_f); `pi_test`, bảng chéo và ma trận nhầm lẫn của từng phiên trùng hẳn; inertia lệch tương đối dưới 1e-6; tham số của mỗi phiên lệch không quá 1e-6 × max|giá trị|.
    - C2: trên `cub_chain_ft_r0_s0`, chạy lại từng bước (35 bước, mỗi bước một tiến trình mới) ra đúng ảnh chụp; thêm một lần chạy tiếp từ S3.09 tới hết.
    - C3: biến thể giả ở 05 rút thêm RNG; phân hoạch của cả 5 phiên không đổi. Đối chứng: RNG bị dịch phải đổi phân hoạch, không đổi thì báo INCOMPLETE.
    - C4: sửa giả bước k của phiên i (trong bộ nhớ) chỉ đổi mã băm từ S_i.k trở đi; sửa chú thích không đổi mã băm nào.
    - C5: trong chuỗi mặc định, khe của các lớp có trước phiên i giữ nguyên từng bit qua phiên i, và khe chưa gặp ở S0 giữ nguyên khởi tạo. [suy luận] Nhãn của lượt F và S chỉ trỏ vào khe của phiên i, nên gradient ở khe khác bằng 0, và Adam với gradient 0 cho bước cập nhật bằng 0.
  - Mốc của C1: 15 run mới bằng `main.py` ở chế độ tất định (`XLA_FLAGS=--xla_gpu_deterministic_ops=true`), cấu hình như lưới `cub_km_evhung_ft` cộng `--print_cluster_xtab --print_confusion`; tag `cub_km_evhung_det_ft_r<r>_s<N>`.
    - Lý do: 15 run `cub_km_evhung_ft` chạy chế độ thường và không in bảng chéo, inertia, ma trận nhầm lẫn; qua nhiều phiên, nhiễu A6 còn lan qua bước dừng sang mô hình khởi đầu của phiên sau.
    - Mốc và bàn mổ cùng chạy chế độ tất định, nên mọi lệch ở C1 phải chẩn đoán và ghi nguyên nhân trước khi dùng số. Lệch ở 04, 05, 09, 10 của phiên nào là lỗi bàn mổ (như quy tắc của QD21, sửa ngày 2026-10-20).
    - So `metrics.json` của 15 mốc mới với 15 run `cub_km_evhung_ft` cho biết A6 lật bao nhiêu kết quả trên CUB t5; ghi lại, không dùng để quyết định.
  - Đại lượng quyết định trên CUB, chốt trước E09.08 (như QD25 cho GC10):
    - Chính: M_d, trung bình độ chính xác lớp mới của 5 phiên, cụm đặt tên bằng Hungarian lúc chấm.
    - Ràng buộc: S5_S0test (độ chính xác trên test S0 cuối phiên 5; M_f proxy = S0_S0test − S5_S0test) không giảm quá B.
    - Báo cáo phụ: S5_All (chỉ số của paper), S5_Old, S<i>_New của từng phiên, M_f.
    - T, B = SD gộp giữa các seed trong mỗi FT seed của 15 chuỗi CUB, tính bằng code sau khi C1 đạt (như `14-eval` §3.3). Đơn vị: FT seed (n = 3).
  - CIFAR-100: chỉ khai báo preset (50 + 5 × 10, `num_labeled` 20000, 400 ảnh mới mỗi lớp mỗi phiên, 25 ảnh đã biết mỗi lớp); C1 chạy trên server khi có đặc trưng và mốc.
- Thay cho: runner dựng phiên đầu bên trong `tier2_e0604` (phương án P1, 2026-11-11), cho mọi việc CUB về sau. Không thay QD21: GC10 vẫn dùng `lab/replay.py`.
- Lý do:
  - E09.08 trên CUB cần chuỗi nhiều phiên; so cặp với RNG chung bỏ nhiễu k-means khỏi hiệu số (Law & Kelton). Trên 15 run `cub_km_evhung_ft`, SD gộp giữa các seed là 1.20 điểm với M_d và 2.5 điểm với S1_New (lệnh chỉ đọc ngày 2026-11-13; số chính thức tính lại sau C1).
  - Cùng hàm bước và cùng `lab/variants.py` cho GC10 và CUB, nên biến thể viết một lần; C0 chứng minh một phiên của bộ chạy mới tương đương bộ chạy cũ.
  - Bộ chạy riêng không làm cũ 15 ảnh chụp gốc GC10.
  - Đại lượng trên CUB chốt trước khi có kết quả E09.08, để không chọn đại lượng sau khi thấy số.
- Hệ quả:
  - `lab/README.vi.md` §5 thêm `lab/sessions.py`; `notes/code-changes.vi.md` ghi file mới và kết quả C0–C5.
  - Bảng chính giữ benchmark t5 của paper. Kết luận trên CUB ghi "đo dưới giao thức upstream": bàn mổ giữ D9 (ảnh cũ trong pool đã được fine-tune thấy kèm nhãn), D16 (ảnh cũ lặp lại qua các phiên: mỗi lớp còn 4–5 ảnh sau 25 ảnh có nhãn (tính tay), trong khi giao thức đòi 5 ảnh mỗi phiên), và cấu hình `km_evhung`, khác cấu hình của paper (GMM, Hungarian lúc train, D13). Số đặt cạnh bảng của paper do server chạy bằng `main.py`.
  - Bản CUB không rò làm trước khi kết luận E09.08 trên CUB: fine-tune đúng 2500 ảnh S0 theo cách `FT_LABELED_IDX` của CIFAR-100, kèm assert tập ảnh fine-tune trùng tập S0 của bộ nạp; báo cáo cả hai bộ đặc trưng.
  - D16: E09.08 giữ thành phần cũ cố định làm mặc định; không cập nhật lớp cũ từ ảnh trong pool, trừ khi xử lý được việc cùng một ảnh lặp lại qua các phiên. Giao thức không lặp ảnh là một benchmark khác, chỉ làm nếu cần, ở phụ lục.
  - D22 trên CUB (việc "đếm trước khi đọc t5" của QD27): đếm ngày 2026-11-13 trên `cub_ft_up_r0` bằng lệnh chỉ đọc: 5994 ảnh train, mỗi lớp 29–30 ảnh, không lớp nào dưới 25 ảnh, nên nhãn dựng theo số đếm cố định không lệch; 6 lớp có 29 ảnh, nếu là lớp cơ sở thì mỗi phiên chỉ có 4 ảnh đã biết (tính tay).
  - GC10 không dính D9: phase 1 (`feature_extractor/dino-neudet.py` đọc `splits/gc10_split.json`) chỉ fine-tune trên `n_use` ảnh đầu của mỗi lớp cơ sở, đúng tập S0 có nhãn.
  - Ảnh CUB trùng ImageNet, tập pretrain của DINO (không nhãn): trang của bộ dữ liệu (Caltech) có cảnh báo; một đối chiếu bằng MD5 tìm 23 cặp trùng với tập test, danh sách của nhóm Spatial Transformer Networks có 22 (Guo Pei, 2016). Khi có danh sách, báo cáo kết quả CUB cả có và không có các ảnh này.
  - Sổ mất điểm (`score.ledger_state`) và notebook đang giả định t1 (assert tập test cũ của phiên 1 trùng test S0); bản theo phiên làm sau QD28.
- Hiện trạng (2026-11-13): thiết kế đã duyệt; chưa có code.
- Sửa ngày 2026-11-13, sau khi có 15 run mốc C1 (`9f57983`), trước khi có code `lab/sessions.py` và trước mọi số E09.08 trên CUB:
  - A6 trên CUB t5: 15 run mốc trùng hẳn 15 run `cub_km_evhung_ft` ở cả 46 khoá `metrics.json` (`notes/reproducibility-issues.md` A6, `ca844f6`): cờ tất định không lật kết quả in ra nào trên CUB t5.
  - Báo cáo phụ thêm balanced M_d (trung bình theo phiên của balanced accuracy lớp mới), chỉ để báo cáo; M_d vẫn là đại lượng chính. Test CUB gần cân (ít nhất 11 ảnh mỗi lớp) nên hai số gần nhau; balanced M_d chặn ảnh hưởng của lớp ít ảnh và cùng kiểu với thước của GC10.
  - Bản CUB không rò, chi tiết hoá dòng Hệ quả ở trên: (1) thêm `FT_LABELED_IDX` vào `feature_extractor/dino-cub200.py` như `dino-cifar100.py` (hiện chỉ CIFAR có), chỉ mục 25 ảnh mỗi lớp tính qua hoán vị D17 (shard gộp theo thứ tự chuỗi); kiểm hồi quy: không đặt biến thì đặc trưng trùng từng bit bản cũ, đặt biến thì tập ảnh fine-tune trùng tập S0 của bộ nạp; thiết kế trình duyệt trước khi viết code; (2) phase 1 lại r0–r2; (3) 15 chuỗi mặc định trên đặc trưng mới, T, B của CUB tính trên 15 chuỗi này (bộ dùng để quyết định E09.08), số trên bộ có rò chỉ báo cáo; (4) một chuỗi đủ S0–S5 của bàn mổ so `main.py` theo tiêu chí C1; (5) ghi cỡ rò ở cấp phiên (S0_All, M_d, S5_S0test, số ảnh giữ), vì ảnh rò chính là ảnh cũ trong pool của mọi phiên (D16).
  - Trên CUB, mặc định của bước 09 vẫn là k-means upstream; `09a` + `09b` trên CUB và cách chọn K' hoãn tới sau kết quả E09.08 trên GC10. Lý do [suy luận]: gần nửa pool CUB là ảnh cũ và bộ lọc giữ gần hết; `09b` gộp mọi cụm con về cụm mới nên không đánh vào lỗi này, còn thành phần cũ của E09.08 thì có. Nếu sau đó vẫn làm, K' chốt bằng quy tắc không dùng nhãn trước khi chạy (tỉ lệ K'/K = 4, hoặc số ảnh mỗi cụm con như GC10); không chọn K' bằng tầng 1 trên chính các phiên sẽ đo.
- Sửa ngày 2026-11-15, sau commit 1–2 của `lab/sessions.py` (`d385095`, `4717d81`), trước mọi số E09.08 trên CUB:
  - Code chia hai đợt. Đợt 3a: `check` (C1, kèm C5), `check-gc10` (C0 trên 15 chuỗi, qua `exp.py`), `check-hash` (C4, phần đổi code của một bước, áp cho mọi phiên), `rerun`. Đợt 3b, làm sau khi thiết kế E09.08 được duyệt: chạy biến thể từ ảnh chụp cha (`--from`, `--start`, `--at`, kể cả biến thể chỉ cho một phiên), C2, C3, và phần C4 cho biến thể một phiên.
  - Lý do: chưa biết E09.08 (trải 09–13, có thể giữ trạng thái giữa các phiên) cắm vào bàn mổ thế nào; viết phần chạy biến thể trước thiết kế dễ phải viết lại. C2, C3 kiểm đúng phần đó nên đi cùng đợt 3b.
  - Hệ quả: T, B của CUB tính từ 15 chuỗi mặc định sau khi C0, C1, C4, C5 qua; chuỗi mặc định không đi qua phần mà C2, C3 kiểm. C2, C3 phải qua trước khi so bất kỳ biến thể nào trên CUB.
  - Kết quả thử: C0 trên `gc10_v2_ft_r0_s0` đạt 34/34 khoá, cả sau commit 1 lẫn commit 2 (lệnh `c0`, không ghi run); chuỗi `cub_chain_ft_r0_s0` (`bf3f6a6`) trùng 46/46 khoá `metrics.json` với `cub_km_evhung_det_ft_r0_s0`, chạy 1062 s.
  - Dung lượng ảnh chụp: 407.6 MiB mỗi chuỗi CUB (ảnh chụp 04 chiếm 170.1 MiB vì giữ tập của mọi phiên), không phải khoảng 0.25 GB như ước tính ở trên; 15 chuỗi khoảng 6 GB (tính tay).
  - Nhãn phiên trong log: `s10_naming` in "hungarian stage 1:" ở mọi phiên; log của phiên i được ghi lại thành "hungarian stage i:", như `main.py` in. Đây là chuỗi duy nhất phải đổi: tiền tố lượt F, S chỉ đi vào TensorBoard writer.
- Sửa ngày 2026-11-16, sau bản CUB không rò và `cub-tb` (`6d75629`), trước tiền đăng ký và mọi số E09.08 trên CUB:
  - Bản CUB không rò đã xong (D9, D17 trong `notes/paper-code-discrepancies.md`): `FT_LABELED_IDX=splits/cub_labeled_hf_idx.npy` (`4b36e47`), phase 1 `cub_ft_l25_r0`–`r2`, 15 chuỗi mặc định `cub_chain_ft_l25_r<r>_s<N>`.
  - C1 trên đặc trưng mới dùng 3 mốc `cub_km_evhung_det_ft_l25_r<r>_s0` (seed 0 của mỗi FT seed) thay 15; luật chốt trước khi chạy: một cặp lệch thì chạy đủ 15. Kết quả: 3/3 cặp OK 48/48. Lý do: `lab/sessions.py` đã qua C1 ở 15/15 chuỗi trên bộ cũ; bộ mới chỉ đổi đầu vào.
  - T, B của CUB, từ `cub_chain_ft_l25` (`python lab/score.py cub-tb`): T = 0.93462, B = 0.37623 điểm; chốt T = 0.935 và B = 0.376, giữ cố định làm cỡ hiệu ứng tối thiểu trên CUB t5, không tính lại khi mặc định đổi (như T, B của GC10). Trên bộ có rò `cub_chain_ft`: T = 1.20128, B = 0.32895, chỉ để đối chiếu.
  - T, B đo nhiễu giữa các seed của chuỗi khi đặc trưng cố định: dùng cho so cặp biến thể trên cùng ảnh chụp. Không dùng cho hiệu giữa hai bộ đặc trưng; thước của hiệu đó là SD của trung bình theo FT seed (n = 3).
  - Hiệu `cub_chain_ft_l25` trừ `cub_chain_ft`: M_d −0.197, S5_S0test −0.542, S5_All −0.399, S5_Old −0.235, M_f +0.183 điểm. Đây là hiệu của hai bộ đặc trưng, không phải cỡ rò: bộ mới còn fine-tune trên ít ảnh hơn (2500 so với 2996); FT seed của hai bộ không ghép cặp được, không làm kiểm định. [suy luận] M_d nằm trong độ phân tán giữa FT seed (0.629 và 0.367); S5_S0test thấp hơn ở cả ba FT seed (độ phân tán 0.219 và 0.249).
  - Thay dòng Hệ quả "đo dưới giao thức upstream: bàn mổ giữ D9": từ đây bộ dùng để quyết định trên CUB là `cub_chain_ft_l25`, D9 đã gỡ; D16 và cấu hình `km_evhung` giữ nguyên. Mọi kết luận trên CUB ghi tên bộ đặc trưng; số lớp cũ (S5_S0test, S5_Old) không chuyển giữa hai bộ.
- Sửa ngày 2026-11-17, sau khi Rito duyệt thiết kế 3b (đêm 16/11 rạng sáng 17/11), trước khi có code 3b và trước tiền đăng ký thang oracle:
  - Đợt 3b lên trước thiết kế E09.08 (dòng Sửa 2026-11-15 đặt nó sau): thang oracle cần chạy biến thể ở một phiên của chuỗi CUB. Chia hai commit. 3b-1: chạy một phiên từ ảnh chụp cha trong bộ nhớ và C2. 3b-2: `run --from`, ảnh chụp của biến thể, mã băm có biến thể, đọc phần thiếu từ run cha, phần C4 cho biến thể một phiên. C3 và chạy cộng dồn sang phiên sau vẫn để sau thiết kế E09.08.
  - Cú pháp: `--start S<i>_<NN> --at S<i>_<NN>=f1+f2`, như tên ảnh chụp; thay cách viết `S<i>.<NN>` ở trên. Biến thể chạy từ bước `start` của phiên i tới `S<i>_14` rồi dừng. RNG toàn cục đặt về `rng_in` của ảnh chụp cha ở lối vào `S<i>_09`.
  - Hàm mới, đều ngoài chuỗi băm (tên dự kiến): `var_step` gọi `chain` với spec của riêng một bước, nên dùng lại vòng item, `_check_out`, `_check_09`; `var_lut` nối bảng tên của bước 10 (`rerun` chuyển sang gọi hàm này, `run_session` giữ bản của nó); `var_session` nạp ảnh chụp cha, dựng khung nhìn, chạy các bước.
  - Kênh oracle `--view f`, CHỈ CHẨN ĐOÁN: f nhận khung nhìn còn `true_*`, được thay `pool_x`, `z_pool`, `true_pool_y` và thêm khoá `oracle_*`; biến thể qua `--at` đọc được `oracle_*`. Lý do: `chain` bỏ `true_*` trước khi gọi biến thể và `_check_out` cấm biến thể ghi khoá của bước khác, nên nhánh oracle không đi qua `--at` được; sửa `chain` thì mọi ảnh chụp thành cũ. Hàm oracle nằm ở `lab/oracle.py`; meta của run ghi `oracle` = 1; lệnh summary không trộn run oracle với run phương pháp; code của f vào mã băm của `S<i>_09`.
  - Khoá `sessions` (tập và nhãn thật của mọi phiên) được lấy khỏi `st` trước khi dựng khung nhìn cho biến thể: `session_view` chép mọi khoá của `st`, và `load` gắn `sessions` vào `st`.
  - Giữ chuỗi băm: không sửa 8 hàm đã nêu ở dòng Sửa 2026-11-15, các hằng chúng đọc, `lab/replay.py`, và các thư mục băm cả file (`main.py`, `dataloaders/`, `utils/`, `classifier/`, `clustering/`). Tên mới ở mức module của `lab/sessions.py` mang tiền tố `var_`: `_closure` duyệt `co_names`, gồm cả tên thuộc tính, nên một tên mới trùng tên thuộc tính mà hàm bị băm dùng sẽ vào mã băm. Mỗi lệnh sửa `lab/sessions.py` tự so `_closure` của các hàm runner, `s04_sessions` và 11 bước mặc định trước và sau khi sửa. Không cài hay nâng gói trong env `vbcgcd`: `_base` băm phiên bản gói.
  - C2, thay mô tả ở trên; chạy trên `cub_chain_ft_l25_r0_s0`, mỗi mục một tiến trình mới, qua đúng đường của biến thể với `--at` rỗng: (a) chạy tiếp từ `S<i>_09` cho i = 1–5 ra đúng `S<i>_09`–`S<i>_14` (rời rạc trùng hẳn, số thực lệch không quá 1e-6 × max|giá trị|; in thêm có trùng từng bit không); (b) chạy tiếp từ 10, 11, 12, 13, 14 ở phiên 1 và phiên 5; (c) sau mỗi bước, RNG toàn cục trùng `rng_in` của ảnh chụp kế tiếp; (d) khung nhìn giao cho biến thể không có `sessions`, không có `true_*`; (e) đối chứng: bước 09 của phiên 2 không đặt lại RNG phải cho `ids` khác, không khác thì báo INCOMPLETE. Bỏ phần "35 bước, mỗi bước một tiến trình" (`rerun` đã làm, và không đi qua đường của biến thể) và phần "chạy tiếp từ S3.09 tới hết" (3b không chạy cộng dồn).
  - Chưa kiểm: `replay._metrics` với log chỉ có một phiên; run biến thể có thể phải tự ghi `metrics.json` từ `acc1`.
- Sửa ngày 2026-11-18, trước tiền đăng ký E14.07 và E09.08: bước D (thang oracle trên pipeline, chiến lược 2026-11-16 sửa tối §3, ngoài repo) không chạy.
  - Lý do: E14.05 và E14.06 cho thấy pipeline ở chế độ mean gần 0 (D23); mức nâng của một nhánh oracle đo trên pipeline không chuyển sang mô hình mà E09.08 dùng.
  - Các câu của D đo trên mô hình N: bộ lọc thật là Nc (E14.06, E12.02); bộ định tuyến lúc test là g20 (E14.03, E14.06); phân hoạch thật là Nlab (E14.07).
  - Nhánh "bỏ ảnh cũ trước bước 09" không đo: nó đổi phân hoạch k-means của mọi phiên và cần đợt 3b-2; để sau E09.08 nếu cần.
  - E09.08 quyết định trên CUB, khởi tạo bằng k-means mặc định của bước 09, không bằng `09a` + `09b` (thay câu tương ứng của QD27, theo dòng Sửa 2026-11-13 ở trên); GC10 chỉ báo cáo.
