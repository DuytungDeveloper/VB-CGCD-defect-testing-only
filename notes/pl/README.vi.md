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
  - chuỗi mặc định của bàn mổ v2: `<dataset>_v2_<đặc trưng>_s<N>`, ví dụ `gc10_v2_ft_r0_s0`.
  Tag cũ giữ nguyên.
- Trong `notes/LOG.md`, dòng `why:` bắt đầu bằng mã thí nghiệm hoặc quyết định, ví dụ `E09.06: ...`, `QD21/C1: ...`.

## 4. Thí nghiệm

1. Lấy mã E kế tiếp trong bước, thêm một dòng vào danh mục (mục 0 của `experiments.vi.md`), rồi viết mục tiền đăng ký: câu hỏi, giả thuyết, đại lượng (công thức kèm nguồn), dự đoán, phép kiểm toàn vẹn, quy tắc quyết định, điều kiện bỏ ý tưởng.
2. Commit và push; ghi hash của commit tiền đăng ký vào mục.
3. Chạy qua `exp.py run`, kể cả script offline (script ghi số vào `$EXP_RUN_DIR/metrics.json`, vì `console.log` không vào git).
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
- Hiện trạng (2026-10-20): đã cài ở các commit `737e8c4`, `a5c9545`, `86ed44d`, `3525e86`; C1–C4 đạt, nên được dùng số của v2. C1: cả 15 run `gc10_v2_ft_r<r>_s<N>` (r = 0–2, N = 0–4) trùng run của E02.05 ở mọi mục, không phải dùng tới quy tắc khi C1 trượt; trong 15 cặp, nhiễu GPU của `main.py` (A6) không làm lật kết quả rời rạc nào. C2 và C3 chạy trên `gc10_v2_ft_r0_s0`. Output ở `runs/<run>/check_c*.txt` (commit `d62d059`). Ảnh chụp của 15 run này là ảnh chụp gốc cho các biến thể; chúng chỉ có trên laptop, vì `*.pkl` không vào git.
- Lịch sử git (2026-10-20): trước khi push, message của 4 commit sau `6bbf204` được viết lại để bỏ hai dòng ghi công tự động ở cuối; cây thư mục của từng commit không đổi, chỉ hash đổi (hash mới = hash cũ): `3525e86` = `7a6291b`, `9a07e5b` = `67b927d`, `d62d059` = `02f9c9c`, `8ea7e40` = `2f033b5`. `config.json` của 15 run và ảnh chụp vẫn ghi hash cũ `67b927d` (bản ghi, không sửa), tức commit `9a07e5b`.

### QD22 — Hệ mã thống nhất cho cả repo (2026-10-18)

- Nội dung:
  - Các họ mã, trạng thái và tag như mục 3.
  - Đánh số lại một lần: câu hỏi theo bước, câu còn mở trước, câu xong hoặc bỏ sau; thí nghiệm theo thời gian, kể cả GĐ1–GĐ3, CT1, CT2; quyết định thành một chuỗi QD01–QD22.
  - Mọi `.md` viết lại theo mã mới, kể cả bản ghi (tiền đăng ký, `notes/LOG.md`, `notes/plan.vi.md`). Mã cũ chỉ còn ở `notes/old-codes.vi.md`.
- Thay cho: phần mã của QD20 (mã cũ ghi trong ngoặc ở mọi file); quy tắc không sửa dòng `why:` cũ trong LOG.
- Lý do: một chữ đang mang nhiều nghĩa (Q, I, C, A, B, E1, H2); có mã treo (D24 chưa có trong danh sách D); một thứ có hai tên (cải tiến 2 = hướng 2); dòng tra mã cũ trong ngoặc làm rối các file chính.
- Hệ quả: commit message và chú thích trong code giữ mã cũ, tra ở `notes/old-codes.vi.md`. Từ nay mã không đánh lại, không dùng lại.
