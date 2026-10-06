# Notebook nghiên cứu

Notebook để giải trình pipeline VB-CGCD theo từng module (đầu vào, đầu ra thật), tìm chỗ nghẽn, cấy phương án thay thế, và học các thuật toán trong và ngoài model. Số liệu chính thức không nằm ở đây.

## Ba lớp

- Bàn mổ: `scripts/replay.py` chạy lại pipeline đúng thứ tự, giữ mọi kết quả trung gian.
- Atlas (00–04): giải phẫu theo luồng, mỏng (khoảng 10 cell); 00 làm cuối.
- Khoang (10–13): một nhóm module thay được; Học → Mổ → Cấy → Summary (EN); chỉ mở khi được làm.

## Atlas (theo luồng dữ liệu)

- `00_map.ipynb` [làm cuối]: bản đồ từng module (shape thật, code, mã D; trỏ về mã M; sinh từ `replay`, 1 seed, cấu hình paper) và bảng mất điểm theo từng khâu (tổng hợp từ run chính thức, nhiều seed).
- `01_data.ipynb` [xong]: GC10 + NEU, dữ liệu thô và giao thức (chia theo cuộn, S0, pool S1, ngân sách nhãn).
- `02_phase1_features.ipynb` [chưa có]: DINO, tiền xử lý, CLS/pooler, đóng băng/FT, độ tách lớp.
- `03_s0_gaussians.ipynb` [chưa có]: PCA, Gaussian theo lớp, SVI/closed form, dừng sớm, dự đoán Mahalanobis; đếm dự đoán test S0 rơi vào slot chưa học (ứng viên D24).
- `04_s1_discovery.ipynb` [chưa có]: atlas mỏng của luồng S1: pool → phân cụm → đặt tên → lượt F → lọc → lượt S; phần can thiệp nằm ở khoang 12, 13.

## Module

Mã M là định danh, không phải vị trí; module mới lấy mã mới (M14, ...).
Ký hiệu: d = `--num_dim`, C = `--num_classes`, K = `--increment`, N = số ảnh.

- M1 chia dữ liệu: ảnh thô → `splits/*.json` (`scripts/make_*_split.py`) [cố định]
- M2 tiền xử lý: ảnh → tensor 3×224×224 (`dino-neudet.py`: `build`, `to_dataset`)
- M3 fine-tune, tuỳ chọn: ảnh có nhãn lớp cơ sở → ViT BOFT (`finetune_dino`)
- M4 trích đặc trưng: tensor → (N, 768), CLS hoặc pooler (`extract`)
- M5 loader: `.npy` + split + seed → dữ liệu từng phiên (`dataloaders/`) [cố định]
- M6 PCA: S0 train → phép chiếu 768 → d (`pre_processing`)
- M7 SVI: (N, d) + nhãn → μ (C, d), Σ (C, d, d) (`run_inference`)
- M8 dừng sớm: Σ lớp mới + Σ lớp cũ → cờ dừng (`calculate_metrics_on_covariances`)
- M9 phân lớp: (N, d) + (μ, Σ) → nhãn, log-likelihood (N, C) (`_predict`)
- M10 phân cụm: pool (N, 768) + K → id cụm (N,) (`clustering/`)
- M11 đặt tên cụm, chỉ khi đánh giá: id cụm + nhãn thật test → π, LUT (`hungarian_map`) [cố định]
- M12 lọc cũ/mới: pool (N, d) + tham số lượt F → mặt nạ (N,) (`train`, `novel_idx`)
- M13 đánh giá: tham số + tập test → All/Old/New/S0test, `metrics.json` (`test`, `extract_vbcgcd.py`) [cố định]

## Thứ tự chạy một seed

M5 → M6 → M7 (S0) → M9 + M13, rồi với mỗi phiên online:
M10 → M11 → M7 (F) + M8 → M9 + M12 → M7 (S) + M8 → M9 + M13.

- Rút RNG toàn cục: M5, M6 (solver randomized), M10.
- Trạng thái chảy theo thứ tự: RNG; tham số Gaussian (S0 → F → lọc → S); PCA (fit ở S0, dùng về sau).
- Thứ tự chỉ bắt buộc trong `replay.py` và script; notebook lấy ảnh chụp đầu vào từ `replay`.
- Cấy ghép: bắt đầu từ cùng trạng thái RNG ở lối vào module bị thay, rồi chạy lại phần phía sau.

## Khoang (đánh số theo vị trí trong pipeline)

- `10_feature.ipynb` [chưa có]: M2–M4.
- `11_gauss.ipynb` [chưa có]: M6–M9.
- `12_cluster.ipynb` [chưa có]: M10, cộng M11 để chấm.
- `13_filter.ipynb` [chưa có]: M12.

## Khung mỗi notebook

- Atlas: Mục đích → bản đồ code (file, hàm, Eq. hoặc dòng Algorithm 1) → đầu vào → chạy từng bước → đầu ra → paper ↔ code (mã D) → nghẽn (trỏ LOG, PL-E) → ý tưởng (trỏ PL-Q) → Summary (EN).
- Khoang: Học (thuật toán trong và ngoài model, trên dữ liệu đồ chơi) → Mổ (bản hiện tại, dữ liệu thật) → Cấy (phương án thay thế, đo hai tầng) → Summary (EN).

## Quy tắc

- Notebook để giải thích. Số liệu chính thức đến từ `exp.py` (LOG, `notes/pl/experiments.vi.md`); notebook chỉ trỏ bằng mã. Câu hỏi mới ghi ở `notes/pl/questions.vi.md`.
- Đặt trong `notes/` nên sửa notebook không làm run bị dirty.
- Cell đầu in commit, trạng thái dirty, đường dẫn đặc trưng (mẫu ở cuối). Trước khi commit: Restart & Run All. Commit kèm output, hình nhỏ.
- Không huấn luyện nặng: đọc đặc trưng có sẵn; tối đa chạy SVI cho 1 seed.
- Một notebook = một câu hỏi chính; tối đa 20 cell (atlas khoảng 10); Restart & Run All không quá 5 phút; vượt thì tách (ví dụ 12a, 12b).
- Notebook import từ `.py`, không có chiều ngược lại; bộ đo viết một lần trong script, notebook import lại.
- Xem code trong model: in đúng hàm bằng `inspect.getsource` cạnh đầu vào/đầu ra của nó, không chép code.
- Khoang đo hai tầng: tầng 1 chấm đầu ra của module bằng nhãn thật (chỉ để chấm), để lọc nhanh; tầng 2 chạy cả pipeline qua `exp.py`, ít nhất 5 seed. Thứ hạng ở tầng 1 phải được xác nhận ở tầng 2.
- Bối cảnh mặc định: cấu hình paper (`ft`, fine-tune trên tập có nhãn); `fz` là bối cảnh phụ (PL-QD4).

## Ngôn ngữ

- Giải thích: tiếng Việt, kèm thuật ngữ tiếng Anh.
- Code, chú thích code, tên biến, nhãn trục, tiêu đề và chú giải hình: tiếng Anh, để dùng lại cho paper.
- Cuối mỗi notebook có "Summary (EN)": 3–5 phát biểu dạng kết quả, mỗi câu kèm số và nguồn (PL-E hoặc run trong LOG), viết sao cho chép được vào paper.
- Thuật ngữ dùng theo bảng dưới; thêm từ mới vào bảng trước khi dùng.

## Thuật ngữ (Vi — En)

- lớp cơ sở — base class; lớp mới — novel class; phiên — session
- nhãn giả — pseudo-label; khám phá lớp mới — novel-class discovery
- đặc trưng đóng băng / fine-tune — frozen / fine-tuned features
- cụm — cluster; sở hữu cụm — cluster ownership (lớp đông nhất trong cụm)
- bảng chéo — contingency table; đặt tên cụm — cluster naming (Hungarian matching)
- bộ lọc cũ/mới — old/new filter; quên — forgetting
- hiệu ứng đồng đều — uniform effect (of k-means)
- ngân sách nhãn — label budget; tiền đăng ký — pre-registration
- bàn mổ — replay harness (`scripts/replay.py`); khoang — slot (nhóm module thay được)
- cấy ghép — graft; ảnh chụp đầu vào — input snapshot
- đánh giá tầng 1 / tầng 2 — tier-1 (module output) / tier-2 (full pipeline) evaluation

## Cell đầu (mẫu)

```python
# cell 0: provenance (run first)
import os, subprocess, sys
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / "main.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT); sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
print("commit", git("rev-parse", "--short", "HEAD"),
      "| code dirty", bool(git("status", "--porcelain", "--", ".", ":(exclude)runs", ":(exclude)notes")),
      "| features", Path("runs/inputs.txt").read_text().strip())
```
