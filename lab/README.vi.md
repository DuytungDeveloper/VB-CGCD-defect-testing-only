# lab — bàn thí nghiệm

Nơi mổ xẻ dữ liệu và mô hình VB-CGCD theo từng bước, cấy thử phương án thay thế, và học các thuật toán trong và ngoài model.
Số liệu chính thức không nằm ở đây: chúng đến từ các run qua `scripts/exp.py` (`notes/LOG.md`, `notes/pl/experiments.vi.md`); notebook chỉ trỏ tới bằng mã.
Quyết định: bố cục QD20, bàn mổ QD21, hệ mã QD22, dựng nền QD23 (`notes/pl/README.vi.md` §5). Bố cục cũ (`notes/nb/`, atlas 00–04, khoang 10–13) nằm ở tag `nb-v1`; đối chiếu ở mục 8.

## 1. Bố cục

Mọi file nằm thẳng trong `lab/`, không có thư mục con. Số ở đầu tên file cho thứ tự pipeline.

- `00-data-<dataset>.ipynb`: phân tích dữ liệu, mỗi dataset một notebook (bước 00).
- `NN-<tên>.ipynb`: bước gốc NN của mô hình (01–14), theo thứ tự chạy.
- `NNx-<tên>.ipynb`: biến thể hoặc phần thêm của bước NN (x = a, b, c… theo thứ tự tạo).
- `*.py`: bàn mổ và các hàm dùng chung (mục 5).
- `README.vi.md`: file này.

Notebook nền 00–14 dựng đủ trước, theo thứ tự (QD23); notebook chữ cái chỉ tạo khi có câu hỏi hoặc thí nghiệm đang làm.

## 2. Mục lục: dữ liệu (bước 00)

- `00-data-gc10.ipynb` [có, commit `1e46674`]: GC10-DET, ba mục
  1. giao thức: vai trò ảnh, dựng lại file split, mức lệch lớp, chia train/test theo cuộn thép, ảnh trùng tên, thứ tự hàng của file đặc trưng, đầu vào bước 04 lấy từ ảnh chụp;
  2. ảnh: kích thước, độ sáng theo camera và lớp, bảng lớp × camera, cuộn thép, ảnh mẫu ở đầu vào 224×224;
  3. nhãn: cờ chất lượng nhãn theo lớp và vai trò, bảng hộp lỗi, loại lỗi đi chung, cỡ vùng lỗi tính bằng patch, độ cao của punching_hole và crescent_gap so với welding_line.
  Nguồn cũ: `01_data` mục A–D (tag `nb-v1`).
- `00-data-neu.ipynb` [có, commit `0079819`]: NEU-DET, ba mục
  1. giao thức: vai trò ảnh, dựng lại file split, số ảnh theo lớp và vai trò (cân bằng), ảnh trùng điểm ảnh giữa train và test, thứ tự hàng của file đặc trưng;
  2. ảnh: kích thước, kênh màu và độ sáng theo lớp, ảnh mẫu kèm khung crop 224;
  3. nhãn: cờ chất lượng nhãn theo lớp và vai trò, bảng hộp lỗi, loại lỗi đi chung, cỡ vùng lỗi tính bằng patch sau crop.
  Nguồn cũ: `01_data` mục E.
- Dataset mới: thêm `00-data-<tên>.ipynb`.

## 3. Mục lục: mô hình (bước 01–14)

Bước 01–03 chạy một lần cho mỗi bộ đặc trưng; bước 04–14 chạy cho mỗi seed. Mặc định của mọi bước hiện là code gốc với cấu hình đã chốt cho GC10/NEU (k-means `n_init` 10, Hungarian chỉ khi chấm, d = 64/128).

1. `01-preprocess` [có, commit `50767b1`]: tiền xử lý ảnh, bốn mục
   1. code: tham số phase 1 của 8 run (GC10, NEU × `fz`, `ft` r0–r2), đoạn transform; ảnh lúc fine-tune đi qua cùng transform, không augmentation;
   2. resize: GC10 co bicubic 9.14 × 4.46 lần (vạch mảnh mất tương phản), NEU phóng 1.28 lần rồi cắt giữa 224;
   3. chuẩn hoá: ảnh xám thành ba kênh, thống kê đầu vào S0 so với ImageNet;
   4. kiểm đầu cuối: trích lại batch đầu bằng DINO gốc, trùng từng bit với đặc trưng `fz` (GC10, NEU).
2. `02-finetune` [có, commit `3db9aef`]: fine-tune backbone, ba mục
   1. thiết lập: siêu tham số và log của 6 run `ft`; mô hình dựng lại bằng `load_model()`: BOFT bọc 24 `output.dense`, butterfly factor 1 (A5), pooler đóng băng;
   2. toán của BOFT: tính tay W′ = diag(s) · W · Mᵀ với khối Cayley 4 × 4, trùng peft; thay đổi theo lớp; ba FT seed cùng độ lớn, khác hướng ở lớp sâu;
   3. quá trình học: cross-entropy theo epoch, GC10 chưa hội tụ ở epoch 10.
3. `03-features` [có, commit `17f4cc5`]: trích đặc trưng, ba mục
   1. đặc trưng: mã băm 8 bộ trùng `config.json`; kiểm đầu cuối `ft` (trích lại từ checkpoint, trùng từng bit); pooler = tanh của một phép chiếu ngẫu nhiên, gần như giữ khoảng cách (D8);
   2. `fz` và `ft`: fine-tune dịch mọi ảnh (cosine khoảng 0.75); ba FT seed có hình học gần nhau (Spearman 0.93);
   3. probe theo lớp (E03.02, run `gc10_e03_02`, bộ đo `score.py` `tier1_03`): đặc trưng đọc được gần gấp đôi pipeline; crease mất ở khám phá, rolled_pit chưa xác định.
4. `04-loader` [chưa có]: nạp dữ liệu theo phiên (tập S0, pool, các tập test).
5. `05-pca` [chưa có]: giảm chiều bằng PCA, fit ở S0.
6. `06-gauss-svi` [chưa có]: Gaussian theo lớp, học bằng SVI ở S0. Code dùng lại ở 11, 13.
7. `07-early-stop` [chưa có]: luật dừng sớm theo det Σ; chỉ chạy ở phiên online.
8. `08-classifier` [chưa có]: bộ phân lớp, đánh giá S0. Code dùng lại ở 12, 14.
9. `09-cluster` [chưa có]: phân cụm pool. Nguồn cũ: `12_cluster`.
10. `10-naming` [chưa có]: đặt tên cụm bằng Hungarian trên tập test, chỉ dùng khi chấm.
11. `11-gauss-f` [chưa có]: lượt F, SVI trên nhãn cụm.
12. `12-filter` [chưa có]: lọc cũ/mới, giữ ảnh bị đoán là lớp mới.
13. `13-gauss-s` [chưa có]: lượt S, SVI lại trên ảnh được giữ, khởi tạo từ tham số S0.
14. `14-eval` [chưa có]: đánh giá S1 (All/Old/New/S0test, M_f, M_d).

Biến thể và phần thêm:
- `06a-closed-form` [có trong code, chưa có notebook]: E06.03, `--estimator closed`; đã đóng ngày 2/10.
- `09a-overcluster` [có hàm `v09a_overcluster`, chưa có notebook]: phân cụm thừa K' = 20; E09.04, E09.05. Nguồn cũ: `12b_cluster`.
- `09b-overcluster-merge` [chưa có]: gộp cụm con; E09.06, chưa tiền đăng ký. Nguồn cũ: phần Bhattacharyya của `12b_cluster`.
- `14a-loss-by-step` [chưa có]: phân tích, mất điểm ở khâu nào; Q14.04. Nguồn cũ: `04_s1_discovery`.
- Dự kiến: `07a-no-early-stop` (Q07.03), `12a-relabel` (Q12.03).

`04_s1_discovery` (tag `nb-v1`) đi xuyên các bước 09–14; nội dung chia vào notebook của từng bước khi làm tới bước đó.

## 4. Quy tắc biến thể

- Số 00–14 không bao giờ đổi. Chữ cái gắn vào bước gốc gần nhất phía trước, theo thứ tự tạo.
- Ba loại:
  1. thay thế bước gốc, kết quả đổi (ví dụ `06a-closed-form` thay SVI);
  2. chèn thêm ngay sau bước gốc, kết quả đổi (ví dụ `09b-overcluster-merge` chạy sau phân cụm);
  3. phân tích thêm cho bước gốc, kết quả không đổi (ví dụ `14a-loss-by-step`).
- Dòng **Vị trí** ở cell đầu của notebook chữ cái ghi ba điều: loại (thay thế, chèn, phân tích); vị trí trong thứ tự chạy; dựa trên notebook nào (khung ở mục 9).
- Biến thể dựa trên một biến thể khác lấy chữ kế tiếp và ghi nguồn; không lồng chữ (`09c` ghi "dựa trên 09b", không đặt `09b1`).
- Biến thể được chọn làm mặc định mới vẫn giữ chữ cái của nó; mục lục đánh dấu bản nào đang là mặc định.
- Code dùng chung (SVI ở 06, 11, 13; bộ phân lớp ở 08, 12, 14): cải tiến một bước là hàm mới của riêng bước đó. Sửa thẳng code chung thì coi như sửa ở bước sớm nhất dùng nó.
- Mỗi biến thể có hai phần: hàm `vNNx_<tên>` trong `variants.py`, chạy qua `exp.py` bằng `replay.py run --from <run> --start NN --at NN=vNNx_<tên>` (ra số chính thức); và notebook `NNx-<tên>.ipynb` để giải thích và mổ.
- Biến thể ở bước 09: sau mọi hàm của bước, `ids` (pool) và `cluster.predict_raw(test1_x)` phải là số nguyên trong [0, increment), mỗi ảnh một id; bộ chạy kiểm điều này và dừng nếu sai (`_check_09`, QD21). Kết quả trung gian có số cụm khác, như cụm con của `09a`, ghi vào khoá riêng; hàm chạy cuối cùng của bước mới ghi `ids` và `cluster`.
- Biến thể gọi hàm của `lab/` bằng tên đã import (`from replay import fit_svi`), không qua module (`replay.fit_svi`); thành phần (component) nêu bằng `f.__name__`, không bằng chuỗi gõ tay. Lý do: chuỗi băm chỉ lần theo tên, nên code gọi theo cách khác có thể đổi mà ảnh chụp không bị báo cũ (đã kiểm ngày 2026-10-20).

## 5. File `.py`

Hiện có:
- `replay.py`: bàn mổ (QD21; commit `737e8c4`, `a5c9545`, `86ed44d`, `3525e86`, `1253ce6`, `370cf86`). Mỗi bước 04–14 là một hàm `sNN(st)`; ảnh chụp ở `runs/<run>/snap/NN.pkl`. Chạy qua `exp.py`: `run --features <F> --seed <N>` (chuỗi mặc định), `run --from <run> --start NN --at NN=<hàm>` (biến thể). Phép kiểm QD21/C1–C4: `check` (C1), `check-resume` (C2), `check-crn` (C3), `check-hash` (C4); cả bốn đạt ngày 2026-10-20. Ảnh chụp gốc cho biến thể: 15 run `gc10_v2_ft_r<r>_s<N>` (chỉ có trên laptop). Lệnh `run` dừng nếu JAX không chạy trên GPU và ghi thiết bị vào meta của ảnh chụp. Bản đầu (hàm `replay()` cho notebook) đã xoá ngày 2026-10-21; bản gốc ở tag `nb-v1`. Hàm `rerun(run, k)` (commit `53d8603`, 2026-11-01) cho notebook 04–14: chạy lại bước k từ ảnh chụp của bước trước (bước 04 từ `new_state`), so từng khoá với ảnh chụp k theo quy tắc của C1 (rời rạc trùng tuyệt đối; số thực lệch ≤ 1e-6 × max|giá trị|), dừng nếu thiết bị khác thiết bị của ảnh chụp, báo nếu driver GPU khác `config.json`; không ghi file. Thử ngày 2026-11-01 trên `gc10_v2_ft_r0_s4`: 11/11 bước OK, mọi khoá trùng từng bit, dù driver đã đổi từ 617.42 sang 621.01.
- `variants.py`: biến thể, mỗi hàm mang mã bước (`v09a_overcluster`; sau thêm `v09b_merge`…), cùng đầu vào/đầu ra với bước gốc; không nhận nhãn thật của pool. `replay.py run --at` tìm hàm ở đây. Quy tắc viết biến thể: docstring đầu file và mục 4.
- `score.py`: bộ đo tầng 1, mỗi bước một hàm (`score09`; sau thêm `score10`, `score12`); `tier1_09` chạy K' cụm từ ảnh chụp của một run, lệnh `python lab/score.py 09 --from <run>` qua `exp.py`. Ngoài các lệnh kiểm của `replay.py`, đây là nơi duy nhất trong `lab/` đọc nhãn thật của pool, chỉ để chấm. Thay `pl_cluster.py` (xoá ngày 2026-10-21) sau khi tái lập tầng 1 của E09.05 ở 15/15 cặp (run `gc10_e09_05_v2_*`, commit `b81b78d`).
- `data_io.py`: dữ liệu thô cho các notebook `00-data-*` (QD23; commit `8910925`, `2f6f1e5`). `images(dataset)`: bảng ảnh của file split, mỗi ảnh một dòng, theo thứ tự hàng của file đặc trưng; vai trò S0, reserve, S1, unused, test (mục 10). `objects(dataset, path)`: hộp lỗi trong XML của một ảnh, cùng quy tắc với `scripts/make_gc10_split.py`. `diagnostics(dataset)`: tính lại mục `diagnostics` của file split, trùng 466/75/6; đọc 2282 file XML trong khoảng 1 s nên không cần bộ nhớ đệm. Chỉ đọc file split và XML, không cần torch, jax. NEU thêm ở commit `beec7b9`: đường dẫn ảnh như `build()` của `feature_extractor/dino-neudet.py`, vai trò như `dataloaders/neudet.py`, XML tra theo tên file trong cả hai thư mục `annotations` (XML của ảnh train `crazing_240` nằm trong `validation/annotations`); `diagnostics("neudet")` cho 123/0/0. Không đặt tên `datasets.py` (sẽ che gói `datasets` của HuggingFace).

Quy tắc:
- Không sửa code gốc (`main.py`, `classifier/`, `clustering/`); các file này chỉ gọi lại hàm của pipeline.
- Notebook import từ `.py`, không có chiều ngược lại.
- Mọi `.py` trong `lab/` bị kiểm dirty: sửa xong phải commit trước khi chạy run. Notebook và `.md` trong `lab/` không bị kiểm.
- Từ khi có 15 ảnh chụp gốc (2026-10-20), không sửa phần bộ chạy nằm trong chuỗi băm: `chain`, `new_state` và mọi thứ chúng gọi tới (`_check_out`, `_fn`, `_base`, `step_hashes`, `_closure`, `_check_09`, các hằng như `STEPS`, `OWNS`, `PROTOCOL`). Sửa phần này làm cũ mọi ảnh chụp từ 04, phải chạy lại 15 run. Các lệnh `cmd_*`, `_fresh`, `load`, `snapshots`, `_confusion`, `_same`, `rerun`, `_compare` nằm ngoài chuỗi băm (đã kiểm bằng `step_hashes` ngày 2026-10-20 và 2026-11-01): sửa chúng không làm cũ ảnh chụp.
  Danh sách đủ 41 thành phần trong chuỗi băm (khoá `file:tên` của `_closure`; `:=` là hằng), tính lại ngày 2026-11-01 từ `_closure(chain)`, `_closure(new_state)` và `_closure` của 11 hàm bước mặc định:
  - hằng: `DEFAULT`, `DETERMINISTIC`, `FILES`, `OWNS`, `PACKAGES`, `PROTOCOL`, `RNG_STEPS`, `STEPS`;
  - bộ chạy: `chain`, `new_state`, `_Tee`, `_base`, `_check_09`, `_check_out`, `_closure`, `_file_hash`, `_fn`, `_sha_file`, `_src_hash`, `_xla_flags`, `step_hashes`;
  - thành phần dùng chung: `fit_svi`, `stop_default`, `predict_default`, `_clf`, `_component`, `_jnp`, `_np`, `_rel`, `_table`;
  - 11 bước: `s04_loader`, `s05_pca`, `s06_gauss_svi`, `s07_early_stop`, `s08_classifier`, `s09_cluster`, `s10_naming`, `s11_gauss_f`, `s12_filter`, `s13_gauss_s`, `s14_eval`.
  Lệnh tính lại: `python -c "import sys; sys.path[:0]=['lab','.']; import replay as r; k=set(); [k.update(r._closure(f)) for f in (r.chain, r.new_state, *(r._fn(r.DEFAULT[s]) for s in r.STEPS))]; print(len(k), sorted(k))"`.

## 6. Chạy độc lập (mục tiêu của bàn mổ)

- Định nghĩa: sửa bước k thì đầu ra của 01…k−1 giữ nguyên, không chạy lại; chạy lại toàn bộ k+1…14.
- 01–03: đạt (đặc trưng lưu sẵn trong `features/`).
- 04–14: bàn mổ đã cài (QD21): mỗi bước là một hàm có đầu vào/đầu ra rõ; ảnh chụp đầu ra mỗi bước kèm commit, trạng thái RNG và mã băm; RNG chung ở lối vào 04, 05, 09; chuỗi băm theo bước báo ảnh chụp nào đã cũ. QD21/C1–C4 đạt ngày 2026-10-20 (mục 5).
- Kết luận ở bước sau gắn với phiên bản các bước phía trên: ghi rõ "đo trên …". Đổi mặc định ở bước k thì tạo lại ảnh chụp từ k và kiểm lại kết luận của các bước sau.

## 7. Hệ mã

- Hệ mã của cả repo (bước, câu hỏi, thí nghiệm, quyết định, mã cục bộ, trạng thái, tag run): QD22, `notes/pl/README.vi.md` §3. Mã cũ, kể cả mã module của bố cục cũ: `notes/old-codes.vi.md`.
- Trong `lab/`: tên notebook mang số bước (`09-cluster`, `09a-overcluster`); hàm biến thể mang mã bước (`v09a_overcluster`); cell đầu notebook ghi mã Q/E mà nó phục vụ.
- Câu hỏi chỉ ghi ở `notes/pl/questions.vi.md`, thí nghiệm ở `notes/pl/experiments.vi.md`; notebook chỉ trỏ tới bằng mã.

## 8. Bố cục cũ (tag `nb-v1`)

Notebook cũ (đọc bằng `git show nb-v1:notes/nb/<tên>`):
- `01_data` → `00-data-gc10` (mục A–D), `00-data-neu` (mục E).
- `04_s1_discovery` → các bước 09–14; phần mất điểm theo khâu → `14a-loss-by-step`.
- `12_cluster` → `09-cluster`.
- `12b_cluster` → `09a-overcluster`; phần Bhattacharyya → `09b-overcluster-merge`.
- Notebook tổng quan 00 của bố cục cũ: không làm.

Đường dẫn cũ → mới:
- `scripts/replay.py` → `lab/replay.py`; `scripts/pl_cluster.py` → `lab/pl_cluster.py` (xoá 2026-10-21, thay bằng `lab/score.py`); `notes/nb/README.vi.md` → `lab/README.vi.md`.
- Tiền đăng ký của E09.04, E09.05 trỏ tới đường dẫn cũ: chạy lại đúng lệnh cũ bằng commit `89e94d4` hoặc tag `nb-v1`.

Cảnh báo trùng số: `12_cluster` cũ (khoang 12 = phân cụm) khác hẳn `12-filter` mới.

## 9. Quy tắc notebook

- Khung mỗi notebook:
  - Cell đầu (Markdown, trước cell 0): tiêu đề trùng tên file, ví dụ `# 09a-overcluster — Phân cụm thừa K' = 20`, rồi một dòng **Câu hỏi** (kèm mã Q/E nếu có). Notebook chữ cái thêm dòng **Vị trí** (mục 4). Đầu vào do cell in ra (cell 0 và cell nạp dữ liệu đầu tiên); kết luận nằm ở Summary (EN) cuối notebook.
  - Mỗi cell code kẹp giữa hai cell Markdown:
    - Trước: một tiêu đề đánh số, ví dụ `### 1.2 Dựng lại file split`, rồi nêu ngắn gọn mục đích hoặc câu hỏi và cách làm, không ghi nhãn như "Câu hỏi:", "Giải thích:". Cell có công thức hay thuật toán mới thêm dòng **Cách làm** (công thức kèm nguồn). Vào, ra, shape ghi ở chú thích code. Nguồn: Rule et al. 2019, PLOS Comput Biol 15: e1007007, quy tắc 1 và 3.
    - Sau, hai gạch: **Kết quả** (số chính, chép đúng từ output); **Ý nghĩa** (nói gì cho câu hỏi, khớp hay trái dự đoán; phần diễn giải ghi [SUY LUẬN]). Không có gạch Tiếp. Chỉ viết ô Sau khi cell đã chạy, từ output thật.
  - Notebook dữ liệu: tiêu đề mục `## 1. Giao thức`, `## 2. Ảnh`, `## 3. Nhãn`.
  - Cuối notebook: "Summary (EN)".
- Cách viết Markdown: cho người chưa đọc code; dùng số cụ thể, không diễn giải lại code; thuật ngữ lần đầu kèm tiếng Anh (mục 10). Mỗi cell code làm một việc: tiêu đề không gói được việc của cell thì tách cell. Cell thiết lập (cell 0, import) cũng có Sau, một dòng kiểm tra, ví dụ "dirty phải là False". Chạy lại mà số đổi thì sửa Sau.
- Code, chú thích code, tên biến, nhãn trục, tiêu đề và chú giải hình: tiếng Anh. Markdown: tiếng Việt, kèm thuật ngữ tiếng Anh.
- Notebook mô hình ngắn, một câu hỏi chính, Restart & Run All không quá 5 phút; dài thì tách thành chữ cái mới. Notebook dữ liệu được phép dài.
- Cell code đầu tiên (cell 0) in commit và trạng thái dirty, rồi đặt `XLA_FLAGS` ở chế độ GPU tất định (deterministic) trước mọi lần import jax, vì `new_state` của bàn mổ dừng nếu jax đã được import mà thiếu cờ này (mẫu ở mục 11). Cell nạp dữ liệu đầu tiên in đầu vào thật của notebook: run và đường dẫn đặc trưng trong ảnh chụp, hoặc file split kèm sha256. Trước khi commit: Restart & Run All, rồi kiểm năm điều: số chạy của các cell code là 1, 2, …, N; không cell nào lỗi; mỗi cell code kẹp giữa hai ô Markdown; có Summary (EN); commit mà cell 0 in ra trùng commit ghi trong ô Sau của nó (thêm 2026-11-01, sau lỗi ở `00-data-gc10`, sửa ở `75a7086`). Commit kèm output, hình nhỏ.
- Không huấn luyện nặng trong notebook: đọc đặc trưng có sẵn; tối đa chạy SVI cho 1 seed.
- Bộ đo viết một lần trong `.py`, notebook import lại. Xem code trong model: in đúng hàm bằng `inspect.getsource` cạnh đầu vào/đầu ra của nó, không chép code.
- Đánh giá hai tầng: tầng 1 chấm đầu ra của một bước bằng nhãn thật (chỉ để chấm), để lọc nhanh; tầng 2 chạy cả pipeline qua `exp.py`, ít nhất 5 seed. Thứ hạng ở tầng 1 phải được xác nhận ở tầng 2.
- Bối cảnh mặc định: cấu hình paper (`ft`); `fz` là bối cảnh phụ (QD19).
- Không nhìn trước dữ liệu: cấu hình thuộc một thí nghiệm sẽ tiền đăng ký không chạy trên pool lớp mới của GC10 trước khi tiền đăng ký; muốn thử thì dùng dữ liệu đồ chơi hoặc mô phỏng khám phá trên lớp cơ sở.
- Cuối mỗi notebook có "Summary (EN)": 3–5 phát biểu dạng kết quả, mỗi câu kèm số và nguồn (mã E, run trong LOG, hoặc mục của notebook như §3.4); phần diễn giải ghi [inference].

## 10. Thuật ngữ (Vi — En)

- lớp cơ sở — base class; lớp mới — novel class; phiên — session
- nhãn giả — pseudo-label; khám phá lớp mới — novel-class discovery
- đặc trưng đóng băng / fine-tune — frozen / fine-tuned features
- cụm — cluster; sở hữu cụm — cluster ownership (lớp đông nhất trong cụm)
- bảng chéo — contingency table; đặt tên cụm — cluster naming (Hungarian matching)
- bộ lọc cũ/mới — old/new filter; quên — forgetting
- hiệu ứng đồng đều — uniform effect (of k-means)
- ngân sách nhãn — label budget; tiền đăng ký — pre-registration
- bàn mổ — replay harness (`lab/replay.py`); bước gốc — base step; biến thể — variant
- cấy ghép — graft; ảnh chụp — snapshot; chạy độc lập — independent step runs
- RNG chung — common random numbers; chuỗi băm — hash chain (báo ảnh chụp cũ)
- đánh giá tầng 1 / tầng 2 — tier-1 (step output) / tier-2 (full pipeline) evaluation
- pool (`pool_x`) — 748 ảnh huấn luyện của phiên S1 = vai trò S1 (698) + 50 ảnh bốc từ reserve; bản ghi cũ viết "pool S1"; khác `pool_x` trong `dataloaders/gc10.py` (= reserve lúc bốc)
- vai trò ảnh — image role (cột `role` của `data_io.images`): S0, S1 = phiên mà file split đưa ảnh vào; reserve = phần còn lại của lớp cơ sở, mỗi seed bốc `NUM_KNOWN` ảnh mỗi lớp (GC10: 10) vào phiên S1; unused = phần còn lại của lớp mới; test

## 11. Cell 0 (mẫu)

Chép nguyên khối dưới vào cell code đầu tiên của mỗi notebook:

    # cell 0: provenance (run first)
    import os, subprocess, sys
    from pathlib import Path
    ROOT = Path.cwd().resolve()
    while not (ROOT / "main.py").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    os.chdir(ROOT); sys.path[:0] = [str(ROOT), str(ROOT / "lab")]
    import replay; os.environ["XLA_FLAGS"] = replay._xla_flags()   # GPU deterministic mode before any jax import
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    print("commit", git("rev-parse", "--short", "HEAD"),
          "| code dirty", bool(git("status", "--porcelain", "--", ".", ":(exclude)runs", ":(exclude)notes",
                                   ":(exclude)lab/*.ipynb", ":(exclude)lab/*.md")),
          "| XLA_FLAGS", os.environ["XLA_FLAGS"])
