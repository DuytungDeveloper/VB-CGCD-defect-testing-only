# Dòng nghiên cứu nhãn giả (PL)

Cổng vào của dòng PL: quy ước ghi chép và các quyết định đã chốt.
Trạng thái câu hỏi không ghi ở đây: xem `questions.vi.md`.

## 1. Phạm vi

- Mở ngày 2026-10-03, khi dòng cũ (GĐ1–GĐ3, cải tiến 1, hướng 2) được chốt hoặc tạm dừng; xem dòng đầu `notes/plan.vi.md`.
- Trọng tâm: chất lượng nhãn giả (pseudo-label) trong VB-CGCD, với nhãn thật tối thiểu (PL-QD1).

## 2. File

- `README.vi.md` (file này): quy ước và quyết định; ít khi sửa.
- `questions.vi.md`: nơi duy nhất chứa câu hỏi mở của dòng PL. Khối phiên ở đầu, nhật ký ở cuối. Câu nào đóng thì ghi kết luận ngay tại câu đó. Không chép số liệu, chỉ trỏ tới mã thí nghiệm hoặc entry LOG.
- `experiments.vi.md` (tạo khi có thí nghiệm đầu tiên): mỗi thí nghiệm một mục gồm tiền đăng ký, kết quả, đánh đổi. Số liệu được phép ở đây.
- Dùng chung toàn repo: `notes/LOG.md` (exp.py tự ghi mọi run), `notes/code-changes.vi.md` (mỗi thứ mới trong code: mục đích, hàm cốt lõi, mặc định, phép kiểm hồi quy), `notes/paper-code-discrepancies.md` (D-list).
- Trạng thái của một mục chỉ sửa ở nơi chứa nó; nơi khác chỉ trỏ bằng mã.

## 3. Mã và tag

- Từ 2026-10-15 (PL-QD5), mã theo bước: câu hỏi `Q<bước>.<nn>`, thí nghiệm `E<bước>.<nn>`, phép kiểm trong một thí nghiệm `C1, C2…`. Bước 00 là dữ liệu; 01–14 là các bước của mô hình (`lab/README.vi.md`). Quyết định vẫn là `PL-QD<n>`.
- Không dùng lại mã; mục bị bỏ ghi "bỏ" kèm lý do, không xoá.
- Mã cũ (`PL-Q<n>`, `PL-E<nn>`) giữ trong ngoặc, ví dụ `Q09.01 (= PL-Q1)`; bảng đối chiếu ở `lab/README.vi.md` mục 8.
- Tag run mới: `<dataset>_e<bước>_<nn>[_<biến thể>]_s<N>`, ví dụ `gc10_e09_03_r0_s3`, để `exp.py agg gc10_e09_03_r0` gom đúng các seed. Tag cũ (`gc10_pl01_…`) giữ nguyên.
- Trong `notes/LOG.md`, dòng `why:` bắt đầu bằng mã thí nghiệm, ví dụ `E09.03: ...`; dòng cũ (`PL-E01: ...`) không sửa.

## 4. Thí nghiệm

1. Viết mục tiền đăng ký trong `experiments.vi.md`: câu hỏi, giả thuyết, đại lượng (công thức kèm nguồn), dự đoán, phép kiểm toàn vẹn, quy tắc quyết định, điều kiện bỏ ý tưởng.
2. Commit và push; ghi hash của commit tiền đăng ký vào mục.
3. Chạy qua `exp.py run`, kể cả script offline (script ghi số vào `$EXP_RUN_DIR/metrics.json`, vì `console.log` không vào git).
4. Chỉ thêm Kết quả, Đánh đổi, Kết luận; không sửa phần tiền đăng ký. Thay đổi làm sau khi đã thấy dữ liệu ghi riêng, kèm ngày và lý do.
5. Cập nhật câu hỏi tương ứng trong `questions.vi.md`.

Quy tắc chung của repo (commit trước khi chạy, Hungarian chỉ lúc đánh giá, mặc định = upstream) giữ nguyên theo `notes/plan.vi.md`.

## 5. Quyết định

### PL-QD1 — Trọng tâm nhãn giả, nhãn thật tối thiểu (2026-10-03)

- Nội dung: dòng mới tập trung vào chất lượng nhãn giả; đầu thấp của trục lượng nhãn là nhãn thật tối thiểu, không phải 0.
- Thay cho: đích ngày 2026-09-30 "chạy được hoàn toàn không nhãn, kể cả lớp cơ sở".
- Lý do: theo đề nghị nhận ngày 2026-10-03, vì trong nhà máy luôn có một ít nhãn thật.
- Bằng chứng nền: nhãn S0 gần như không đổi khả năng khám phá lớp mới (`notes/improvement-1.vi.md` §3.2), nên lượng nhãn và chất lượng nhãn giả là hai câu hỏi tách rời.
- Chưa chốt: nhãn tối thiểu có được dùng cho lớp mới không (PL-Q4).

### PL-QD2 — Ghi chép: `.md` là bản chuẩn (2026-10-04)

- `.md` trong repo là bản chuẩn cho cả nhóm. Sổ giấy cá nhân (mục B/E/I theo ngày) chỉ là bản tra cứu khi bị hỏi.
- Khi giấy và `.md` lệch nhau, `.md` đúng. `.md` phải tự đủ, không trỏ sang sổ giấy.
- Công cụ khác (Obsidian, tablet) không chứa số liệu hay trạng thái dự án.
- Xem lại cách ghi sau hai tuần dùng thử.

### PL-QD3 — Notebook theo ba lớp; thứ tự làm mới (2026-10-06)

- Nội dung: `notes/nb/` gồm bàn mổ (`scripts/replay.py`), atlas 00–04 theo luồng và khoang 10–13 theo nhóm module; mã M1–M13 và thứ tự chạy định nghĩa ở `notes/nb/README.vi.md`.
- Thứ tự làm: 04 (mỏng) → `12_cluster` (Học, Mổ; chốt bộ đo tầng 1) → lưu trạng thái RNG trước M10, script tầng 1, PL-E01 → `12_cluster` (Cấy) → 02, 03 (kèm đếm D24) → khoang khác khi cần → 00 cuối cùng.
- Thay cho: thứ tự chốt ngày 2026-10-05: 00 (phần bản đồ) → 04 → PL-E01 → 02 → 03 → 00 (bảng nghẽn).
- Lý do: phần bản đồ của 00 không đổi quyết định thí nghiệm nào, vì PL-E01 đã tiền đăng ký; trang tổng quan nên viết một lần ở cuối; 04 và `12_cluster` dựng đúng các mảnh PL-E01 cần.
- Đánh giá module thay được: tầng 1 chấm đầu ra của module bằng nhãn thật (chỉ để chấm) để lọc nhanh; tầng 2 chạy cả pipeline. Thứ hạng ở tầng 1 phải được xác nhận ở tầng 2.

### PL-QD4 — Mốc so sánh là cấu hình paper (fine-tune); `fz` là bối cảnh phụ (2026-10-06)

- Nội dung: mốc so sánh của dòng PL là cấu hình paper trên GC10, fine-tune trên tập có nhãn (`gc10_ft_cls5first_img80pct_r0`–`r2`, mỗi FT seed 5 seed phase 2; 30 run ngày 2026-10-05). Đặc trưng không fine-tune (`fz`) là bối cảnh phụ để hiểu cơ chế, không dùng để kết luận cải tiến.
- Thay cho: quyết định ngày 2026-10-05 (bối cảnh chính `fz`, cấu hình paper là bối cảnh thứ hai).
- Lý do: cải tiến phải đo so với VB-CGCD như công bố; fine-tune offline là một thành phần cốt lõi của paper (Bảng 1). Thứ tự giữ đúng: tái lập (CIFAR-100 khớp paper ở chỉ số cuối; GC10 cấu hình paper tái lập được theo `FT_SEED`), rồi mới thay đổi và so sánh.
- Bằng chứng nền: note LOG `gc10_ftseed0_km_evhung_d64_s4` (fine-tune giữ lớp cũ tốt hơn, ổn định qua FT seed; khám phá không ổn định).
- Hệ quả: so sánh trên `ft` ghép cặp theo FT seed (r0–r2); atlas dùng `ft` r0, seed 4 (mode đa số). PL-E01 giữ tiền đăng ký trên `fz`; thí nghiệm tương ứng trên `ft` tiền đăng ký riêng, và hướng phân cụm thừa xét theo kết quả trên `ft`.
- Lưu ý: `fz` và `ft` khác nhau ở hai yếu tố cùng lúc (có fine-tune hay không; CLS hay pooler, D8); fine-tune trong code chỉ có cross-entropy (D3).

### PL-QD5 — Bố cục `lab/`, mã theo bước, khung notebook (2026-10-15)

- Nội dung:
  - Notebook và bàn mổ nằm thẳng trong thư mục gốc `lab/`, không có thư mục con. Mục lục, bảng đối chiếu mã và quy tắc notebook ở `lab/README.vi.md`.
  - Bước 00 là dữ liệu: mỗi dataset một notebook `00-data-<dataset>.ipynb`, được phép dài. Mô hình chia thành 14 bước 01–14 theo thứ tự chạy, thay cho M1–M13 (mã M giữ làm tra cứu).
  - Biến thể và phần thêm mang chữ cái của bước gốc, có ba loại: thay thế, chèn sau, phân tích. Số 00–14 không bao giờ đổi.
  - Mã câu hỏi và thí nghiệm theo bước (`Q<bước>.<nn>`, `E<bước>.<nn>`); mã cũ giữ trong ngoặc.
  - "Chạy độc lập": sửa bước k thì không chạy lại 01…k−1, chạy lại toàn bộ k+1…14. Đạt được khi có bàn mổ v2 (`lab/replay.py` nâng tại chỗ).
  - Kế hoạch file `.py`: `replay.py` (v2), `variants.py`, `score.py`, `data_io.py`; `pl_cluster.py` nghỉ khi đã có `score.py` và v2.
  - Notebook có khung cố định: cell đầu (tiêu đề; Câu hỏi, Vị trí, Bối cảnh, Kết luận), Markdown Trước và Sau mỗi cell code, cuối là Summary (EN). Không làm notebook tổng quan 00.
- Thay cho: PL-QD3 (bố cục ba lớp trong `notes/nb/` và thứ tự làm); quyết định ngày 2026-10-05 không dời file code, nay có ngoại lệ cho `replay.py` và `pl_cluster.py`.
- Lý do: tách hẳn dữ liệu và mô hình; đánh số theo thứ tự chạy để cải tiến gắn ngay tại bước của nó và so được với chuỗi mặc định; một hệ mã đồng bộ cho bước, notebook, câu hỏi và thí nghiệm.
- Bản cũ: tag `nb-v1` (`notes/nb/`, `scripts/replay.py`, `scripts/pl_cluster.py`); dời ở commit `dd7a954`.
- Hệ quả: tiền đăng ký E09.01, E09.02 trỏ tới đường dẫn cũ, chạy lại bằng commit `89e94d4` hoặc tag `nb-v1`; `exp.py` không kiểm dirty notebook và `.md` trong `lab/`, vẫn kiểm `lab/*.py`.

### PL-QD6 — Bàn mổ v2: bước là hàm, ảnh chụp theo bước, RNG chung (2026-10-15)

- Nội dung:
  - `lab/replay.py` nâng tại chỗ lên v2. Phạm vi: t1, `--hungarian_on eval`, S0 có nhãn; GC10 và NEU.
  - Mỗi bước 04–14 là một hàm `sNN(st) -> dict`: đọc trạng thái `st`, trả các khoá mới; Vào/Ra ghi trong docstring. Bước mặc định gọi đúng hàm của pipeline, theo thứ tự của `main.py` và `MNGMMClassifier.train()`.
  - 06, 07, 08 trả thêm thành phần (cách ước lượng, luật dừng sớm, cách phân lớp) để 11–14 dùng lại, nên biến thể ở 06–08 áp dụng cả xuống 11–14. Thành phần cắm bằng cách gán đè phương thức trên đối tượng `clf`, không sửa code gốc.
  - Nhãn thật của pool và của mọi tập test nằm trong khoá `true_*`. Hàm biến thể nhận `st` đã bỏ các khoá này; chỉ s10, s14 và `score.py` đọc chúng. Bước mặc định 06/11/13 vẫn nhận nhãn test như `main.py`, chỉ để in log.
  - Ảnh chụp sau mỗi bước: `runs/<run_id>/snap/NN.pkl` gồm đầu ra, trạng thái RNG, commit, mã băm. Không vào git; chỉ dùng trên máy đã tạo ra nó.
  - Chuỗi mặc định chạy từ đầu một lần cho mỗi (bộ đặc trưng, seed); biến thể chạy tiếp từ ảnh chụp đó (`--from`, `--start`, `--at`).
  - RNG chung (common random numbers): ở lối vào 04, 05, 09 đặt lại trạng thái RNG của chuỗi mặc định.
  - Phát hiện ảnh chụp cũ theo từng bước, bằng chuỗi băm: h(k) = băm(h(k−1), AST của hàm ở bước k và các hàm trong `lab/` mà nó gọi, tham số biến thể, file pipeline dùng lần đầu ở bước k). AST bỏ chú thích và docstring. File pipeline: 04 `main.py`, `dataloaders/`, `utils/`; 05 `classifier/`; 09 `clustering/`. h(04) băm thêm cấu hình, đặc trưng, file split, phiên bản các gói tính toán và code của bộ chạy. Sửa bước k chỉ làm cũ ảnh chụp từ k trở đi. Không chạy tiếp từ run gốc bị dirty.
  - Run qua `exp.py`: console log cùng định dạng `main.py`; `metrics.json` do `scripts/extract_vbcgcd.py` tạo.
  - Phép kiểm, phải qua trước khi dùng:
    - C1: chuỗi mặc định khớp 15 run `gc10_ft_cls5first_img80pct_r{0,1,2}_km_evhung_d64_s{0–4}` ở từng ảnh chụp (tham số S0 và S1, bảng S0 và S1, bảng chéo và inertia, `pi_test`, bước dừng F/S, số ảnh giữ). 15 run này cũng là ảnh chụp gốc cho các biến thể.
    - C2: với 1 seed, chạy tiếp từ mỗi k = 05…14 ra đúng như chạy từ đầu.
    - C3: biến thể giả ở 05 rút thêm RNG; phân hoạch ở 09 không đổi.
    - C4: sửa giả bước k (trong bộ nhớ) chỉ đổi h(k)…h(14); sửa chú thích không đổi mã băm nào.
- Thay cho: không. Cụ thể hoá "chạy độc lập" của PL-QD5. Chọn ngày 2026-10-15: phát hiện cũ theo từng bước (không theo file), C1 trên cả 15 run.
- Lý do: chạy độc lập (PL-QD5); hiệu số giữa biến thể và mặc định không lẫn nhiễu RNG (Law & Kelton); chỉ tạo lại phần bị ảnh hưởng, như `make` (Feldman 1979) nhưng so nội dung thay vì thời gian.
- Hệ quả: số của biến thể đo bằng v2 có thể lệch bản cài thẳng vào `main.py` chỉ do đường đi của RNG; biến thể được chọn làm mặc định thì đo lại. `pl_cluster.py` nghỉ khi v2 qua C1–C4; bản v1 ở tag `nb-v1`.
