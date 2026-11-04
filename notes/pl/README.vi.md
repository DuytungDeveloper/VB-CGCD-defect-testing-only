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
