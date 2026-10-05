# Notebook nghiên cứu

Giải thích pipeline VB-CGCD theo từng bước, với đầu vào và đầu ra thật, để giải trình, tìm chỗ nghẽn và nảy ý tưởng. Số liệu chính thức không nằm ở đây.

## Danh sách (theo luồng dữ liệu)

- `00_map.ipynb` [chưa có]: chạy 1 seed từ đầu đến cuối, in shape ở mỗi bước; bảng mất điểm theo từng khâu.
- `01_data.ipynb` [xong]: GC10 + NEU, dữ liệu thô và giao thức (chia theo cuộn, S0, pool S1, ngân sách nhãn).
- `02_phase1_features.ipynb` [chưa có]: DINO, tiền xử lý, CLS/pooler, đóng băng/FT, độ tách lớp.
- `03_s0_gaussians.ipynb` [chưa có]: PCA, Gaussian theo lớp, SVI/closed form, dừng sớm, dự đoán Mahalanobis.
- `04_s1_discovery.ipynb` [chưa có]: pool, k-means, bảng chéo, đặt tên, lượt F, bộ lọc, lượt S; nơi can thiệp của dòng PL.

## Khung mỗi notebook

Mục đích → bản đồ code (file, hàm, Eq. hoặc dòng Algorithm 1) → đầu vào → chạy từng bước → đầu ra → paper ↔ code (mã D) → nghẽn (trỏ LOG, PL-E) → ý tưởng (trỏ PL-Q) → Summary (EN).

## Quy tắc

- Notebook để giải thích. Số liệu chính thức đến từ `exp.py` (LOG, `notes/pl/experiments.vi.md`); notebook chỉ trỏ bằng mã. Câu hỏi mới ghi ở `notes/pl/questions.vi.md`.
- Đặt trong `notes/` nên sửa notebook không làm run bị dirty.
- Cell đầu in commit, trạng thái dirty, đường dẫn đặc trưng (mẫu ở cuối). Trước khi commit: Restart & Run All. Commit kèm output, hình nhỏ.
- Không huấn luyện nặng: đọc đặc trưng có sẵn; tối đa chạy SVI cho 1 seed.

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
