# lab — bàn thí nghiệm

Nơi mổ xẻ dữ liệu và mô hình VB-CGCD theo từng bước, cấy thử phương án thay thế, và học các thuật toán trong và ngoài model.
Số liệu chính thức không nằm ở đây: chúng đến từ các run qua `scripts/exp.py` (`notes/LOG.md`, `notes/pl/experiments.vi.md`); notebook chỉ trỏ tới bằng mã.
Quyết định: bố cục QD20, bàn mổ v2 QD21, hệ mã QD22 (`notes/pl/README.vi.md` §5). Bố cục cũ (`notes/nb/`, atlas 00–04, khoang 10–13) nằm ở tag `nb-v1`; đối chiếu ở mục 8.

## 1. Bố cục

Mọi file nằm thẳng trong `lab/`, không có thư mục con. Số ở đầu tên file cho thứ tự pipeline.

- `00-data-<dataset>.ipynb`: phân tích dữ liệu, mỗi dataset một notebook (bước 00).
- `NN-<tên>.ipynb`: bước gốc NN của mô hình (01–14), theo thứ tự chạy.
- `NNx-<tên>.ipynb`: biến thể hoặc phần thêm của bước NN (x = a, b, c… theo thứ tự tạo).
- `*.py`: bàn mổ và các hàm dùng chung (mục 5).
- `README.vi.md`: file này.

Notebook chỉ được tạo khi bước đó có câu hỏi hoặc thí nghiệm đang làm.

## 2. Mục lục: dữ liệu (bước 00)

- `00-data-gc10.ipynb` [chưa có]: GC10-DET, chia mục theo thứ tự
  1. giao thức: lớp và số ảnh, chia train/test theo cuộn thép, S0/S1, mức lệch lớp, ngân sách nhãn, ảnh trùng;
  2. ảnh: kích thước, độ sáng theo lớp, ảnh mẫu, ảnh hưởng của camera và cuộn thép;
  3. nhãn và hình học vết lỗi: nhãn thư mục lệch XML, ảnh nhiều lỗi, diện tích và vị trí hộp, số patch.
  Nguồn cũ: `01_data` mục A–D (tag `nb-v1`).
- `00-data-neu.ipynb` [chưa có]: NEU-DET, cùng ba mục. Nguồn cũ: `01_data` mục E.
- Dataset mới: thêm `00-data-<tên>.ipynb`.

## 3. Mục lục: mô hình (bước 01–14)

Bước 01–03 chạy một lần cho mỗi bộ đặc trưng; bước 04–14 chạy cho mỗi seed. Mặc định của mọi bước hiện là code gốc với cấu hình đã chốt cho GC10/NEU (k-means `n_init` 10, Hungarian chỉ khi chấm, d = 64/128).

1. `01-preprocess` [chưa có]: tiền xử lý ảnh.
2. `02-finetune` [chưa có]: fine-tune backbone.
3. `03-features` [chưa có]: trích đặc trưng, CLS hay pooler.
4. `04-loader` [chưa có]: nạp dữ liệu theo phiên (tập S0, pool S1, các tập test).
5. `05-pca` [chưa có]: giảm chiều bằng PCA, fit ở S0.
6. `06-gauss-svi` [chưa có]: Gaussian theo lớp, học bằng SVI ở S0. Code dùng lại ở 11, 13.
7. `07-early-stop` [chưa có]: luật dừng sớm theo det Σ; chỉ chạy ở phiên online.
8. `08-classifier` [chưa có]: bộ phân lớp, đánh giá S0. Code dùng lại ở 12, 14.
9. `09-cluster` [chưa có]: phân cụm pool S1. Nguồn cũ: `12_cluster`.
10. `10-naming` [chưa có]: đặt tên cụm bằng Hungarian trên tập test, chỉ dùng khi chấm.
11. `11-gauss-f` [chưa có]: lượt F, SVI trên nhãn cụm.
12. `12-filter` [chưa có]: lọc cũ/mới, giữ ảnh bị đoán là lớp mới.
13. `13-gauss-s` [chưa có]: lượt S, SVI lại trên ảnh được giữ, khởi tạo từ tham số S0.
14. `14-eval` [chưa có]: đánh giá S1 (All/Old/New/S0test, M_f, M_d).

Biến thể và phần thêm:
- `06a-closed-form` [có trong code, chưa có notebook]: E06.03, `--estimator closed`; đã đóng ngày 2/10.
- `09a-overcluster` [chưa có]: phân cụm thừa K' = 20; E09.04, E09.05. Nguồn cũ: `12b_cluster`.
- `09b-overcluster-merge` [chưa có]: gộp cụm con; E09.06, chưa tiền đăng ký. Nguồn cũ: phần Bhattacharyya của `12b_cluster`.
- `14a-loss-by-step` [chưa có]: phân tích, mất điểm ở khâu nào; Q14.04. Nguồn cũ: `04_s1_discovery`.
- Dự kiến: `07a-no-early-stop` (Q07.03), `12a-relabel` (Q12.03).

`04_s1_discovery` (tag `nb-v1`) đi xuyên các bước 09–14; nội dung chia vào từng bước khi có bàn mổ v2.

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

## 5. File `.py`

Hiện có:
- `replay.py`: bàn mổ, hai phần.
  - v2 (QD21; commit `737e8c4`, `a5c9545`, `86ed44d`, `3525e86`): mỗi bước 04–14 là một hàm `sNN(st)`; ảnh chụp ở `runs/<run>/snap/NN.pkl`. Chạy qua `exp.py`: `run --features <F> --seed <N>` (chuỗi mặc định), `run --from <run> --start NN --at NN=<hàm>` (biến thể). Phép kiểm QD21/C1–C4: `check` (C1), `check-resume` (C2), `check-crn` (C3), `check-hash` (C4). Cả bốn đạt ngày 2026-10-20, nên được dùng số của v2; ảnh chụp gốc cho biến thể là 15 run `gc10_v2_ft_r<r>_s<N>` (chỉ có trên laptop).
  - v1: hàm `replay()` (notebook và `pl_cluster.py` dùng), tự kiểm `python lab/replay.py [--full] <seed…>`. Bỏ cùng lúc với `pl_cluster.py`; bản gốc ở tag `nb-v1`.
- `pl_cluster.py`: bộ đo tầng 1 của bước 09 (`score()`) và phần chạy cho E09.04, E09.05 (`--ref-tag` bắt buộc).

Kế hoạch (tạo khi cần):
- `variants.py`: biến thể, mỗi hàm mang mã bước (`v09a_overcluster`, `v09b_merge`…), cùng đầu vào/đầu ra với bước gốc; không nhận nhãn thật của pool. `replay.py run --at` tìm hàm ở đây.
- `score.py`: bộ đo tầng 1, mỗi bước một hàm (`score09` = `pl_cluster.score`; sau thêm `score10`, `score12`). Ngoài các lệnh kiểm của `replay.py`, đây là nơi duy nhất trong `lab/` đọc nhãn thật của pool, chỉ để chấm.
- `data_io.py`: nạp split, ảnh và hộp XML cho các notebook `00-data-*`, cùng quy tắc với `scripts/make_gc10_split.py`; kiểm bằng mục `diagnostics` của file split. Không đặt tên `datasets.py` (sẽ che gói `datasets` của HuggingFace).
- `pl_cluster.py` nghỉ khi đã có `score.py` và v2 qua QD21/C1–C4.

Quy tắc:
- Không sửa code gốc (`main.py`, `classifier/`, `clustering/`); các file này chỉ gọi lại hàm của pipeline.
- Notebook import từ `.py`, không có chiều ngược lại.
- Mọi `.py` trong `lab/` bị kiểm dirty: sửa xong phải commit trước khi chạy run. Notebook và `.md` trong `lab/` không bị kiểm.
- Từ khi có 15 ảnh chụp gốc (2026-10-20), không sửa phần bộ chạy nằm trong chuỗi băm: `chain`, `new_state` và mọi thứ chúng gọi tới (`_check_out`, `_fn`, `_base`, `step_hashes`, `_closure`, `_check_09`, các hằng như `STEPS`, `OWNS`, `PROTOCOL`). Sửa phần này làm cũ mọi ảnh chụp từ 04, phải chạy lại 15 run. Các lệnh `cmd_*`, `_fresh`, `load`, `snapshots`, `_confusion`, `_same` và phần v1 nằm ngoài chuỗi băm (đã kiểm bằng `step_hashes` ngày 2026-10-20): sửa chúng không làm cũ ảnh chụp.

## 6. Chạy độc lập (mục tiêu của bàn mổ v2)

- Định nghĩa: sửa bước k thì đầu ra của 01…k−1 giữ nguyên, không chạy lại; chạy lại toàn bộ k+1…14.
- 01–03: đạt (đặc trưng lưu sẵn trong `features/`).
- 04–14: bàn mổ v2 đã cài (QD21): mỗi bước là một hàm có đầu vào/đầu ra rõ; ảnh chụp đầu ra mỗi bước kèm commit, trạng thái RNG và mã băm; RNG chung ở lối vào 04, 05, 09; chuỗi băm theo bước báo ảnh chụp nào đã cũ. QD21/C1–C4 đạt ngày 2026-10-20 (mục 5).
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
- `scripts/replay.py` → `lab/replay.py`; `scripts/pl_cluster.py` → `lab/pl_cluster.py`; `notes/nb/README.vi.md` → `lab/README.vi.md`.
- Tiền đăng ký của E09.04, E09.05 trỏ tới đường dẫn cũ: chạy lại đúng lệnh cũ bằng commit `89e94d4` hoặc tag `nb-v1`.

Cảnh báo trùng số: `12_cluster` cũ (khoang 12 = phân cụm) khác hẳn `12-filter` mới.

## 9. Quy tắc notebook

- Khung mỗi notebook:
  - Cell đầu (Markdown, trước cell 0): tiêu đề trùng tên file, ví dụ `# 09a-overcluster — Phân cụm thừa K' = 20`, rồi bốn dòng: **Câu hỏi** (kèm mã Q/E); **Vị trí** (bước nào; loại: gốc, thay thế, chèn, phân tích; dựa trên notebook nào); **Bối cảnh** (bộ đặc trưng, seed); **Kết luận** (1–2 câu kèm số, điền khi xong).
  - Mỗi cell code kẹp giữa hai cell Markdown có nhãn cố định:
    - Trước, tối đa 3 dòng: **Mục đích** (câu hỏi, mã Q/E); **Vào → Ra** (tên biến, shape); **Đọc kết quả** (nhìn vào đâu, giá trị nào nghĩa là gì). Cell có công thức hay thuật toán mới thêm dòng **Cách làm** (công thức kèm nguồn).
    - Sau, tối đa 3 gạch: **Kết quả** (số chính, chép đúng từ output); **Ý nghĩa** (nói gì cho câu hỏi, khớp hay trái dự đoán; phần diễn giải ghi [SUY LUẬN]); **Tiếp** (hệ quả cho bước sau, bỏ nếu không có).
  - Notebook dữ liệu: tiêu đề mục `## 1. Giao thức`, `## 2. Ảnh`, `## 3. Nhãn`.
  - Cuối notebook: "Summary (EN)".
- Cách viết Markdown: cho người chưa đọc code; dùng số cụ thể, không diễn giải lại code; thuật ngữ lần đầu kèm tiếng Anh (mục 10). Mỗi cell code làm một việc: phần Trước cần quá 3 dòng thì tách cell. Cell thiết lập (cell 0, import) cũng có Sau, một dòng kiểm tra, ví dụ "dirty phải là False". Chạy lại mà số đổi thì sửa Sau.
- Code, chú thích code, tên biến, nhãn trục, tiêu đề và chú giải hình: tiếng Anh. Markdown: tiếng Việt, kèm thuật ngữ tiếng Anh.
- Notebook mô hình ngắn, một câu hỏi chính, Restart & Run All không quá 5 phút; dài thì tách thành chữ cái mới. Notebook dữ liệu được phép dài.
- Cell code đầu tiên (cell 0) in commit, trạng thái dirty, đường dẫn đặc trưng (mẫu ở mục 11). Trước khi commit: Restart & Run All; commit kèm output, hình nhỏ.
- Không huấn luyện nặng trong notebook: đọc đặc trưng có sẵn; tối đa chạy SVI cho 1 seed.
- Bộ đo viết một lần trong `.py`, notebook import lại. Xem code trong model: in đúng hàm bằng `inspect.getsource` cạnh đầu vào/đầu ra của nó, không chép code.
- Đánh giá hai tầng: tầng 1 chấm đầu ra của một bước bằng nhãn thật (chỉ để chấm), để lọc nhanh; tầng 2 chạy cả pipeline qua `exp.py`, ít nhất 5 seed. Thứ hạng ở tầng 1 phải được xác nhận ở tầng 2.
- Bối cảnh mặc định: cấu hình paper (`ft`); `fz` là bối cảnh phụ (QD19).
- Không nhìn trước dữ liệu: cấu hình thuộc một thí nghiệm sẽ tiền đăng ký không chạy trên pool lớp mới của GC10 trước khi tiền đăng ký; muốn thử thì dùng dữ liệu đồ chơi hoặc mô phỏng khám phá trên lớp cơ sở.
- Cuối mỗi notebook có "Summary (EN)": 3–5 phát biểu dạng kết quả, mỗi câu kèm số và nguồn (mã E hoặc run trong LOG).

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

## 11. Cell 0 (mẫu)

Chép nguyên khối dưới vào cell code đầu tiên của mỗi notebook:

    # cell 0: provenance (run first)
    import os, subprocess, sys
    from pathlib import Path
    ROOT = Path.cwd().resolve()
    while not (ROOT / "main.py").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    os.chdir(ROOT); sys.path[:0] = [str(ROOT), str(ROOT / "lab")]
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    print("commit", git("rev-parse", "--short", "HEAD"),
          "| code dirty", bool(git("status", "--porcelain", "--", ".", ":(exclude)runs", ":(exclude)notes",
                                   ":(exclude)lab/*.ipynb", ":(exclude)lab/*.md")),
          "| features", Path("runs/inputs.txt").read_text().strip())
