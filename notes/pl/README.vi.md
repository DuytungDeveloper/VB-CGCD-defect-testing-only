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

- Câu hỏi `PL-Q<n>`, quyết định `PL-QD<n>`, thí nghiệm `PL-E<nn>`. Không dùng lại mã; mục bị bỏ ghi "bỏ" kèm lý do, không xoá.
- Tag run: `<dataset>_pl<nn>_<biến thể>_s<N>`, ví dụ `gc10_pl01_k20_s0`, để `exp.py agg gc10_pl01_k20` gom đúng các seed.
- Trong `notes/LOG.md`, dòng `why:` của run thuộc dòng PL bắt đầu bằng mã thí nghiệm, ví dụ `PL-E01: ...`.

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
