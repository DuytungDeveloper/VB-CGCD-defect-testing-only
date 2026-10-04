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
