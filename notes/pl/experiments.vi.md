# Thí nghiệm

Danh mục mọi thí nghiệm của repo (mục 0), rồi mỗi thí nghiệm của dòng PL một mục: tiền đăng ký (không sửa sau khi commit), rồi Kết quả, Đánh đổi, Kết luận. Quy trình ở `README.vi.md` §4. Mã cũ: `notes/old-codes.vi.md`.

## 0. Danh mục

Số theo thứ tự thời gian trong mỗi bước (QD22). Mỗi dòng: mã: nội dung (ngày; trạng thái, kết luận) — nơi ghi chi tiết.

- E01.01: GC10 đóng băng, ảnh 224×448 (2026-10-01; xong, LOẠI: tốt hơn ở proxy nhưng kém hơn ở pipeline) — note LOG `gc10_fr224x448_km_evhung_d64_s4`
- E02.01: lưới (λ, τ) của Eq. 3 trên fold F5, 3 epoch (2026-09-19 → 20; xong: không dự đoán được kết quả full-scale) — D19
- E02.02: Eq. 3 ở full-scale: lưới (λ, τ), nhiễu FT_SEED 42–46, nhánh augonly (2026-09-20 → 27; xong: Eq. 3 không giúp) — D19
- E02.03: sửa D9, fine-tune chỉ trên 400 ảnh có nhãn mỗi lớp (`base400`) (2026-09-29 → 30; xong: tăng nhẹ, chưa kết luận) — D9
- E02.04: GC10, fine-tune trên lớp cơ sở so với đóng băng, FT seed 42, 0, 1 (2026-10-01; xong: giữ lớp cũ tốt hơn; khám phá không ổn định theo FT seed) — note LOG `gc10_ftseed0_km_evhung_d64_s4`
- E02.05: cấu hình paper chuẩn hoá, FT seed r0–r2, GC10 và NEU, 30 run phase 2 (2026-10-05; xong: mốc so sánh của QD19) — LOG, các run `*_cls5first_img80pct_r*`, `*_cls3first_img80pct_r*`
- E02.06: ngân sách nhãn ở phase 1: ít ảnh mỗi lớp (10, 25, 50), hoặc đổi số lớp có nhãn và cách chọn (đông nhất, ngẫu nhiên) (2026-10-05; chờ phase 2: mới trích đặc trưng) — LOG ngày 2026-10-05, dòng "label budget"
- E03.01: GC10, chuẩn hoá đặc trưng theo camera (2026-10-01; xong, LOẠI: không vượt nhiễu k-means; 10 seed, không lưu run) — `notes/plan.vi.md` §5
- E03.02: probe theo lớp trên đặc trưng GC10 (`fz`, `ft` r0–r2) so với recall của pipeline: điểm mất ở biểu diễn hay ở khám phá (2026-11-01; xong: crease mất ở khám phá, rolled_pit chưa xác định; E09.06 vẫn là bước kế tiếp) — mục E03.02 dưới đây
- E04.01: GC10, S0 không nhãn (`--s0_labels none`) (2026-10-01; xong: khám phá lớp mới không cần nhãn S0) — note LOG `gc10_fr_nolab_km_evhung_d64_s4`
- E05.01: NEU, `num_dim` 384 so với 128 (2026-09-30; xong: 384 làm Σ sụp ở S0, dùng 128) — D21
- E06.01: CT1, tiên nghiệm co Inverse-Wishart tối ưu qua SVI (2026-10-01; xong, LOẠI: tiên nghiệm gần như vô hiệu, D23) — `notes/improvement-1.vi.md`
- E06.02: CT1, tăng ngân sách bước: lr ×10, ×100 (2026-10-01; xong: xác nhận D23) — `notes/improvement-1.vi.md` §3.1
- E06.03: CT1, ước lượng dạng đóng (Ledoit–Wolf), cờ `--estimator closed`, biến thể `06a-closed-form` (2026-10-02; xong: trượt tiêu chí 2b, giữ cờ opt-in) — `notes/improvement-1.vi.md`
- E06.04: phiên đầu bằng hậu nghiệm NIW và dự đoán Student-t, biến thể `06b-vb-s0`, GC10 và CUB (2026-11-11 → 12; xong: ĐẠT cả hai, D = +1.91 GC10, +1.59 CUB; A3 là lớp cũ cho E09.08) — mục E06.04 dưới đây
- E09.01: D18, bảng 2×2 (GMM hay k-means, `n_init` 1 hay 10), CIFAR-100 (2026-09-19; xong: số lần khởi tạo quyết định độ chính xác; chọn k-means `n_init` 10) — D18
- E09.02: GC10 đóng băng, GMM `n_init` 10 thay k-means (2026-09-30; xong, LOẠI; kết luận sửa ngày 2026-10-02) — note LOG `gc10_frozen_gmm10_evhung_d64_s4`
- E09.03: CT2, bảng chéo cụm × lớp thật trên pool S1, GC10 đóng băng (2026-10-02; xong: lớp hiếm bị nuốt vì không sở hữu cụm nào) — `notes/improvement-2.vi.md`
- E09.04: phân cụm thừa trên `fz`: lớp hiếm có cụm riêng ở K' = 20 không? (2026-10-05 → 07; xong, LOẠI) — mục E09.04 dưới đây
- E09.05: phân cụm thừa trên cấu hình paper (`ft`) (2026-10-07; xong, SỐNG) — mục E09.05 dưới đây
- E09.06: phân cụm thừa K' = 20 rồi gộp về 5 cụm bằng Bayes factor NIW, biến thể `09a` + `09b`, GC10 (2026-11-12 → 13; xong: CHẤP NHẬN, Δ = 0.064, Δ_Old = −0.53 điểm; rolled_pit vẫn mất cụm) — mục E09.06 dưới đây
- E09.07: nhiều lần khởi tạo k-means, best-of-1000 so với best-of-10, 11 lần rút lồng nhau trong mỗi ảnh chụp (2026-11-04 → 11; xong: không đủ lớn, Δ = 0.0148 < T = 0.0273; giữ `n_init` 10) — mục E09.07 dưới đây
- E09.08: CUB t5, VB-GMM cho phiên không nhãn, biến thể `09e-vbgmm` (thay bước 11–13): trách nhiệm hai phía với thành phần cũ cố định, mọi khe dùng hậu nghiệm NIW, khởi tạo từ k-means của bước 09; nhánh chẩn đoán KW (k-means làm trắng theo P) và V-new (E-M trong nhóm mới); luật xác nhận trên r3, r4 (đề xuất 2026-11-05; tiền đăng ký 2026-11-18; xong, Rito duyệt 2026-11-19: Bác, ΔS5_S0test −8.522; CAVI khoá ở khởi tạo k-means, KW −0.140; điều kiện bỏ cơ chế đạt; VB thật: hỏng nhãn mềm và Dirichlet, NIW còn mất lớp cũ, prior phân cấp chưa thử) — mục E09.08 dưới đây
- E09.09: dừng phép gộp của `09b` khi log Bayes factor lớn nhất không còn dương (k tự do, không dưới 5), biến thể `09f-merge-stop`, tầng 1, GC10 (2026-11-14; xong: KHÔNG RÕ, Δ ≈ 0; rolled_pit giữ được cụm con ở 98% số lần; rất nhạy với off) — Q09.14; mục E09.09 dưới đây
- E10.01: D13, Hungarian khi đánh giá (`--hungarian_on eval`) (2026-09-29 → 30; xong: trung bình gần như không đổi, lệch ở 1/25 phiên) — D13
- E11.01: cột 11 và phản thực tế bước dừng trên nhánh `09a` + `09b`: điểm mất ở lượt F hay ở bộ lọc và lượt S; lớp cũ mất do bước dừng hay do phân hoạch; phép đo mô tả, GC10 (2026-11-14 → 15; xong: LƯỢT F, sát nút, 4/5 FT seed; BƯỚC DỪNG GIẢI THÍCH, D_old = +0.67 ảnh; phần mất thêm của cụm đã gộp nằm nhiều hơn ở bộ lọc và lượt S) — mục E11.01 dưới đây
- E12.01: WSM của BSW (CVPR 2026; OT nới một biên cộng RSQ, chạy trên đặc trưng cố định) ở bước 12 (đề xuất 2026-11-05; chưa tiền đăng ký; thí nghiệm riêng hay ablation chốt khi tiền đăng ký E09.08) — QD27; tài liệu chiến lược 2026-11-05 §3, §7 (ngoài repo)
- E12.02: CUB t5, bộ lọc cũ/mới không nhãn dựa trên mô hình lớp cũ (A3 hoặc A1, chọn bằng mô phỏng bỏ-một-lớp trên S0; ngưỡng p-value conformal α = 0.10) thay `keep` của bước 12, rồi khớp mô hình N của E14.06: lấy lại bao nhiêu phần khoảng lớp cũ giữa N và N sạch (tiền đăng ký 2026-11-17) — mục E12.02 dưới đây
- E14.01: điểm lớp mới theo bốn quy ước ghép tên P0–P3 trên cùng dự đoán, CUB t5 (2026-11-16; xong, Rito duyệt tạm: P1–P3 cao hơn P0 0.75–0.79 điểm M_d; P3 trùng P2) — mục E14.01 dưới đây
- E14.02: CUB t5, quên theo đợt lớp và sổ mất điểm theo phiên, kèm Sửa đổi (bảng 4, 5) (2026-11-16; xong, Rito duyệt: L_10 = 10.10 điểm; quên là ảnh cũ bị khe mới giành, 30/30 chuỗi) — mục E14.02 dưới đây
- E14.03: CUB t5, L_10 mất bao nhiêu vì khe cũ giành ảnh test lớp mới, khe cũ thắng nhờ số hạng nào, pipeline gán theo tâm, đường cong đánh đổi theo độ lệch b (tiền đăng ký 2026-11-17; xong, Rito duyệt tạm: L_comp = 10.18 điểm, gần bằng L_10; khe cũ thắng nhờ số hạng Mahalanobis) — mục E14.03 dưới đây
- E14.04 (= "E14.01" của đề xuất 2026-11-05; mã cũ trùng với E14.01 ở trên): đường cong δ và phép so ở cùng mức ảnh cũ (cánh sườn S–P / PMB), chỉ đọc pickle (đề xuất 2026-11-05; chưa tiền đăng ký) — tài liệu chiến lược 2026-11-05 §3 (ngoài repo)
- E14.05: CUB t5, khe cũ giành ảnh test lớp mới vì mean khe mới đặt lệch (bộ ước lượng, ảnh cũ lẫn vào cụm, cụm sai lớp) hay vì chồng lấn trong biểu diễn: thang Mk, Mc, M1 và Σ chung (tiền đăng ký 2026-11-17; xong, Rito duyệt tạm, có Sửa cùng ngày: R(Mk) 0.98 nhưng Old giảm 52 điểm, là đánh đổi; mean gần 0 ở mọi khe, như D23 và E06.04; E09.08 phải dùng một bộ ước lượng cho mọi khe) — mục E14.05 dưới đây
- E14.06: CUB t5, L_10 và L_comp khi mọi khe dùng hậu nghiệm NIW với dự đoán Student-t (A3 của E06.04 cho lớp gốc, cùng tiên nghiệm cho khe của các phiên); phụ: mô hình đó có hơn pipeline không (tiền đăng ký 2026-11-17; xong, Rito duyệt tạm: L_comp_N = 6.50, cạnh tranh vẫn là đích; N không hơn pipeline vì Old giảm 4.9; N sạch hơn pipeline ở cả New và Old) — mục E14.06 dưới đây
- E14.07: CUB t5, trần dưới bộ ước lượng nhất quán: L_comp và lớp mới khi khe mới khớp đúng lớp (Nlab); S5_S0test của N, Nc, Nlab; đường cong độ lệch b trên N (tiền đăng ký 2026-11-18; xong, Rito duyệt tạm: L_comp_Nlab = 8.81 > T, cạnh tranh là giới hạn của biểu diễn dưới NIW, bỏ ý (c); Nlab hơn Nc 25.2 điểm New: đích của E09.08 là phân hoạch trong nhóm mới; ràng buộc S5_S0test đạt được với bộ lọc hoàn hảo) — mục E14.07 dưới đây
- E14.08: CUB t5, vì sao N mất lớp cũ: bốn phép làm yếu khe mới của N (O1 tempering likelihood, O2 co mean về m0, O3 tempering lúc dự đoán so với p_ref, O4 lệch cộng b ≤ 0) đọc tại ΔS5_S0test = −B; ảnh cũ gây hại qua mean hay hiệp phương sai (Mb, Ψb); phân bổ ảnh test gốc bị chiếm (tiền đăng ký 2026-11-19; xong, Rito duyệt 2026-11-19: R1, G_O4 +3.087 ở 3/3 FT seed nhưng ΔOld −3.488 và b chọn bằng nhãn test; ảnh cũ hại qua khe bản sao, tỉ số H8 15.278, qua cả hiệp phương sai lẫn mean; bước kế E14.09: chọn b không nhãn, thêm ràng buộc Old) — mục E14.08 dưới đây
- E14.09: CUB t5, prior phân cấp cho khe mới của N (HB-off: off chọn bằng LOO trên hàng được giữ; HB-full: m0, P, off của nhóm mới; HB-slot: off theo evidence từng khe; HB-near: mượn lớp gốc gần nhất, kiểm phản biện), đọc dưới ràng buộc hai vế của QD28 (tiền đăng ký 2026-11-19; xong, Rito duyệt 2026-11-19: không đạt, mọi nhánh trượt dọc đường đánh đổi; ΔS5_All của N −3.513, của Nc +2.004; HB-near ủng hộ phản biện; bước kế (1) và (2)) — mục E14.09 dưới đây

## E03.02 — Probe theo lớp: điểm mất ở biểu diễn hay ở khám phá? (Q03.02, Q14.04)

### Tiền đăng ký (2026-11-01)

- Câu hỏi: với mỗi lớp mới của GC10, trên đặc trưng `ft` r0–r2 (cấu hình paper, QD19), một bộ phân lớp tuyến tính có nhãn thật đọc được lớp đó tới đâu, so với recall cuối pipeline? Phần chênh nói điểm mất nằm ở biểu diễn (bước 01–03, 05) hay ở khám phá (bước 09–13).
- Vì sao có thí nghiệm này: kết quả quyết định cải tiến tiếp theo nhắm vào bước nào (E09.06 ở bước 09, hay Q03.02 và Q05.02 ở đặc trưng). Trên `fz` đã có số cũ: probe balanced novel 69.0 so với pipeline 37.6 (LOG, CORRECTION ngày 2026-10-02 ở note `gc10_frozen_gmm10_evhung_d64_s4`); trên `ft` chưa có số theo lớp.
- Cơ sở: phân hoạch của k-means là bộ phân lớp tuyến tính theo vùng Voronoi (argmin ‖x − c_k‖² = argmax 2c_kᵀx − ‖c_k‖²), nên bộ tuyến tính học có nhãn là trần xấp xỉ của bước 09 trên cùng đặc trưng; probe tuyến tính là cách đánh giá đặc trưng chuẩn (Alain & Bengio 2016; Caron et al. 2021). Chỉ là trần xấp xỉ: logistic regression có điều chuẩn, probe học trên ảnh train còn k-means chạy trên pool, và sau bước 09 còn bộ phân lớp Gaussian.
- Dữ liệu: 4 bộ đặc trưng GC10 (`gc10_fz_cls5first_img80pct`, `gc10_ft_cls5first_img80pct_r0`–`r2`). Học trên các ảnh train có vai trò S0, reserve, S1 (`lab/data_io.py`), nhãn thật của cả 10 lớp; đo trên 455 ảnh test.
- Cách làm: `python scripts/exp.py run gc10_e03_02 -- python lab/score.py 03`, hàm `tier1_03` của `lab/score.py` (commit `b89f1d6`). Một run; ngẫu nhiên duy nhất là PCA với `random_state=0`.
- Probe (scikit-learn): `lr` = LogisticRegression(max_iter=5000); `lr_bal` = như `lr`, `class_weight="balanced"`; `knn` = 20 láng giềng gần nhất theo cosine; `pca_lr_bal` = `lr_bal` trên PCA 64 chiều fit ở các ảnh S0 (như bước 05).
- Mốc so sánh: recall từng lớp trên test của pipeline, cộng 5 seed (dòng `confusion stage 1` trong `console.log`) của các run E02.05 `gc10_ft_cls5first_img80pct_r<r>_km_evhung_d64_s0`–`s4`; với `fz`: `gc10_frozen_km_evhung_d64_head_s0`–`s4`.
- Đại lượng: recall từng lớp mới dạng k/n với khoảng tin cậy 95% Wilson (Wilson 1927); balanced novel = trung bình recall của 5 lớp mới. Với pipeline, khoảng Wilson tính trên n ảnh test của lớp (5 seed dùng chung các ảnh đó, nên cộng seed không làm n lớn hơn).
- Phép kiểm toàn vẹn; chỉ đọc kết quả khi cả bốn đạt:
  - C1: mã băm của 4 bộ đặc trưng trùng `config.json` của các run pipeline tương ứng;
  - C2: kiểm đầu cuối `ft`: đặc trưng trích lại từ checkpoint trùng file (`lab/03-features.ipynb` §1.2, đạt ngày 2026-11-01, trùng từng bit);
  - C3: số ảnh test: rolled_pit 6, crease 11, inclusion 43, silk_spot 124, waist_folding 30;
  - C4: `lr` học trên toàn bộ ảnh train của `fz` tái lập độ chính xác 80.2% (LOG, `20260930_232359__gc10_extract_frozen`).
- Quy tắc xếp loại (probe `lr_bal`, 768 chiều thô), cho mỗi lớp mới và mỗi bộ đặc trưng:
  - D (mất ở khám phá): cận dưới khoảng tin cậy của probe lớn hơn cận trên của pipeline;
  - R (biểu diễn yếu): recall của probe < 50%;
  - một lớp có thể vừa D vừa R (DR); không D, không R là U (chưa xác định).
- Quy tắc quyết định, cho rolled_pit và crease; một lớp mang nhãn D (hay R) nếu nhãn đó, kể cả trong DR, xuất hiện ở ít nhất 2/3 bộ `ft`:
  - có lớp mang D → E09.06 giữ là thí nghiệm kế tiếp của dòng PL;
  - có lớp mang R → nâng ưu tiên Q03.02 và Q05.02 (làm sau E09.06 nếu cũng có D);
  - không lớp nào mang D hay R → không đổi kế hoạch; ghi "chưa xác định do số ảnh test nhỏ".
- Chỉ để mô tả, không dùng cho quyết định: `lr`, `knn`, `pca_lr_bal`; inclusion, silk_spot, waist_folding; bộ `fz`.
- Dự đoán [suy luận, dựa trên số cũ của `fz`: probe rolled_pit 67%, crease 45%; pipeline 0% và 11%]: rolled_pit mang D; crease U hoặc R. Không đoán số của `ft`.
- Nếu C1–C4 không đạt: không đọc kết quả, sửa lỗi rồi chạy lại một lần, ghi lý do.
- Lưu ý: probe dùng nhãn thật của lớp mới, chỉ để chấm. Tiền đăng ký E09.06 phải ghi là đã thấy các số này.
- Tag: `gc10_e03_02` (một run).

### Kết quả (2026-11-01; run `20261101_165324__gc10_e03_02`, code `b89f1d6`, tiền đăng ký `afdeb1d`)

- Toàn vẹn: C1 và C3 đạt ở cả 4 bộ; C2 đạt trước khi chạy (`lab/03-features.ipynb` §1.2); C4: `lr` trên toàn bộ ảnh train của `fz` cho 80.2%. `checks_ok` = 1.
- Probe quyết định (`lr_bal`, 768 chiều), số ảnh test đoán đúng / tổng, so với recall của pipeline (cộng 5 seed):

  | bộ | silk_spot | inclusion | rolled_pit | crease | waist_folding | balanced novel |
  |---|---|---|---|---|---|---|
  | `fz` probe | 111/124 | 22/43 | 4/6 | 5/11 | 25/30 | 0.672 |
  | `fz` pipeline | 0.665 | 0.140 | 0.000 | 0.109 | 0.967 | 0.376 |
  | `ft` r0 probe | 84/124 | 24/43 | 3/6 | 5/11 | 26/30 | 0.611 |
  | `ft` r0 pipeline | 0.682 | 0.163 | 0.000 | 0.345 | 0.780 | 0.394 |
  | `ft` r1 probe | 96/124 | 27/43 | 3/6 | 6/11 | 28/30 | 0.676 |
  | `ft` r1 pipeline | 0.421 | 0.074 | 0.000 | 0.000 | 0.967 | 0.292 |
  | `ft` r2 probe | 105/124 | 25/43 | 3/6 | 8/11 | 27/30 | 0.711 |
  | `ft` r2 pipeline | 0.639 | 0.181 | 0.000 | 0.055 | 0.933 | 0.362 |

- Xếp loại (D, R, U, theo quy tắc):
  - rolled_pit: U ở `fz`, r0, r1, r2 (probe 3/6 hoặc 4/6, pipeline 0/6; khoảng Wilson chồng nhau);
  - crease: R ở `fz` và r0, D ở r1 và r2;
  - inclusion: D ở cả 4 bộ; silk_spot: D ở `fz`, r1, r2, U ở r0; waist_folding: U ở cả 4 bộ.
- Quyết định: crease mang D (2/3 bộ `ft`), rolled_pit không mang D hay R → E09.06 giữ là thí nghiệm kế tiếp; không lớp nào mang R ở 2/3 bộ `ft` → không nâng ưu tiên Q03.02, Q05.02.
- Dự đoán: "rolled_pit mang D" sai (U ở cả ba bộ `ft`); "crease U hoặc R" chỉ đúng ở r0.
- Mô tả (không dùng cho quyết định), balanced novel `fz` / r0 / r1 / r2:
  - `lr`: 0.666 / 0.623 / 0.657 / 0.685;
  - `knn`: 0.445 / 0.509 / 0.484 / 0.499; crease 0/11 ở cả 4 bộ, rolled_pit 1–2/6;
  - `pca_lr_bal`: 0.562 / 0.641 / 0.570 / 0.612, so với `lr_bal` thô 0.672 / 0.611 / 0.676 / 0.711.

### Kết luận

- Trên cấu hình paper, phần lớn điểm của lớp mới mất ở khâu khám phá, không ở đặc trưng: probe `lr_bal` đạt balanced novel 0.61–0.71 trên `ft`, pipeline 0.29–0.39; rõ nhất ở inclusion (D ở mọi bộ) và crease (D ở 2/3 bộ `ft`). Vì vậy E09.06 (bước 09) vẫn là thí nghiệm kế tiếp.
- rolled_pit không xếp loại được: chỉ có 6 ảnh test, probe 3/6. Không kết luận gì về lớp này từ tập test hiện tại.
- [SUY LUẬN] Fine-tune trên lớp cũ không làm lớp mới dễ tách hơn theo đường thẳng (`fz` 0.672 nằm giữa ba bộ `ft`). PCA-64 của bước 05 có thể mất tới 11 điểm (r1, r2) nhưng không mất ở r0. k-NN thua hẳn probe tuyến tính ở lớp hiếm, nên các phương pháp phân cụm dựa trên láng giềng gần (Q09.06) cần được kiểm riêng cho lớp hiếm.
- Đã thấy các số này trước khi tiền đăng ký E09.06; E09.06 phải ghi điều này.

## E06.04 — Hậu nghiệm NIW và dự đoán Student-t ở phiên đầu (QD27)

### Tiền đăng ký (2026-11-11; commit: df2e846)

- Câu hỏi: thay bộ phân lớp phiên đầu bằng hậu nghiệm Normal–Inverse-Wishart (NIW) và phân phối dự đoán Student-t có tăng độ chính xác test phiên đầu không, trên GC10 (n = 130–184 mỗi lớp, d = 64) và CUB (n = 25 mỗi lớp, d = 384)?
- Vì sao có thí nghiệm này: nó là nền của E09.08 (QD27). Thành phần cũ của VB-GMM là hậu nghiệm này, và tiên nghiệm của lớp mới lấy từ nó. Code hiện ước lượng MLE với guide rỗng (D11) và gần như đứng ở khởi tạo (D23).
- Nền lý thuyết (Murphy 2012 §4.6.3; Murphy 2007; Gelman et al. 2013, BDA3 §3.6):
  - Tiên nghiệm NIW(m0, κ0, ν0, Ψ0) liên hợp với Gaussian. Với n mẫu, trung bình x̄, ma trận tán xạ S: κn = κ0 + n; νn = ν0 + n; mn = (κ0 m0 + n x̄)/κn; Ψn = Ψ0 + S + (κ0 n/κn)(x̄ − m0)(x̄ − m0)ᵀ.
  - Kỳ vọng hậu nghiệm của Σ là Ψn/(νn − d − 1). Phân phối dự đoán là Student-t bậc νn − d + 1, tâm mn, ma trận tỉ lệ Ψn(κn + 1)/(κn(νn − d + 1)).
  - Phiên đầu là Bayes chính xác: hậu nghiệm này là nghiệm tối ưu của ELBO (Eq. 7 của paper), không phải xấp xỉ. Phần biến phân nằm ở E09.08. Đây là Bayesian QDA đã có (Geisser 1964; Srivastava, Gupta & Frigyik 2007), là thành phần nền, không phải đóng góp mới.
- Dữ liệu:
  - GC10: 15 ảnh chụp `gc10_v2_ft_r<r>_s<N>`, r = 0–2, N = 0–4 (chuỗi mặc định, QD21); đặc trưng PCA sau bước 05.
  - CUB: 15 chuỗi lab t5 trên đặc trưng `ft` r0–r2, seed 0–4, bước 04–08, tạo sau tiền đăng ký này bằng biến thể bước 04 trong `lab/variants.py`; không sửa `dataloaders/` (QD27).
- Bốn nhánh, cùng đặc trưng PCA và cùng tập test trong từng chuỗi:
  - A0: mặc định (SVI của pipeline, bước 06–08).
  - A1: E06.03, ước lượng dạng đóng với Ledoit–Wolf về đẳng hướng; dùng lại `classifier/closed_form.py`.
  - A2: NIW, phân lớp bằng Gaussian plug-in với trung bình mn và hiệp phương sai Ψn/(νn − d − 1).
  - A3: NIW, phân lớp bằng phân phối dự đoán Student-t.
  - Mọi nhánh lấy argmax log mật độ với tiên nghiệm lớp đều, như pipeline.
- Tiên nghiệm của A2 và A3:
  - m0 = trung bình toàn cục của ảnh S0 có nhãn; κ0 = 0.01.
  - Ψ0 = (ν0 − d − 1)·Σ_gộp, với Σ_gộp là hiệp phương sai gộp trong lớp của mọi lớp cơ sở (tổng tán xạ chia cho N − C). Khi đó kỳ vọng tiên nghiệm của Σ bằng Σ_gộp, và ν0 lớn cho đúng LDA.
  - ν0 chung cho mọi lớp, chọn trong từng chuỗi trên lưới ν0 − d − 1 ∈ {2, 5, 10, 25, 50, 100, 250, 1000, 5000, 100000}.
  - Luật chọn: cực đại tổng, trên mọi ảnh S0 có nhãn, của log mật độ dự đoán Student-t của ảnh đó khi bỏ chính nó ra (leave-one-out). Ảnh bị bỏ được rút khỏi cả lớp của nó lẫn Σ_gộp. Hoà thì lấy ν0 lớn hơn. Không đọc test, không đọc pool.
  - Vì sao không chọn bằng hợp lý biên như bản nháp ngày 2026-11-11: với κ0 nhỏ, số hạng dự đoán tiên nghiệm của một mẫu xấp xỉ κ0^(d/2)·(2π)^(−d/2)·E[|Σ|^(−1/2)], không đo độ khớp nhưng tăng mạnh khi ν0 nhỏ: log E[|Σ|^(−1/2)] − log |Σ_gộp|^(−1/2) = (d/2)·ln(2/(ν0 − d − 1)) + lnΓ((ν0 + 1)/2) − lnΓ((ν0 − d + 1)/2). Hợp lý biên nhạy với tiên nghiệm mơ hồ là điều đã biết (O'Hagan 1995; Berger & Pericchi 1996); chọn theo dự đoán: Geisser & Eddy 1979. BDA7 và Quadratic Bayes cũng chọn siêu tham số bằng kiểm định chéo (Srivastava, Gupta & Frigyik 2007).
  - Luật này vẫn là "empirical Bayes trên S0, không nhìn test" của QD27.
- Đại lượng chính: D = độ chính xác test phiên đầu của A3 trừ của A0 (điểm %), hiệu trong từng chuỗi; D_r = trung bình trên 5 seed của FT seed r; D = trung bình D_r trên 3 FT seed. Tính riêng cho GC10 và cho CUB. Đơn vị: FT seed (n = 3).
- Quy tắc quyết định, cho từng dataset:
  - Đạt: D ≥ 1 điểm và D_r > 0 ở ít nhất 2/3 FT seed.
  - Bác: D ≤ 0.
  - Không đủ lớn: các trường hợp còn lại.
  - Không dùng ngưỡng theo SD của baseline: SD gộp của S0_All ở A0 nhỏ hơn 0.5 điểm trên cả hai dataset, nên ngưỡng 1 điểm đã chặt hơn 2 SD. Trên GC10, 1 điểm bằng 2.41 ảnh trong 241 ảnh test lớp cũ.
- Hệ quả:
  - CUB đạt và GC10 không bị bác: A3 là biểu diễn lớp cũ cho E09.08.
  - CUB bị bác: không dùng hậu nghiệm NIW làm lớp cũ trên CUB; rà lại chiến lược trước khi tiền đăng ký E09.08.
  - A3 không đạt nhưng A2 − A0 đạt theo cùng tiêu chí: ghi "tiên nghiệm đạt, Student-t không". E09.08 vẫn đi tiếp, vì bước E của VB-GMM dùng dạng bậc hai (Bishop 2006, Eq. 10.64), không dùng Student-t.
  - Các trường hợp khác: báo cáo; quyết định về E09.08 lấy ở lần rà chiến lược sau E06.04 và E09.06.
- Báo cáo phụ (không dùng để quyết định):
  - A1 − A0, A2 − A0, A3 − A2, tính như D;
  - balanced accuracy, recall từng lớp, recall thấp nhất;
  - SD (ddof = 1) giữa 5 seed của từng nhánh trong từng FT seed (thước ổn định);
  - log mật độ dự đoán trung bình của lớp đúng trên test, cho A2 và A3;
  - tách cũ/mới ở tầng 1: AUROC trên pool phiên 1, điểm là log mật độ lớn nhất trong các lớp cũ, nhãn thật chỉ để chấm, cho cả bốn nhánh. Trên CUB con số này lạc quan, vì ảnh cũ trong pool đã nằm trong tập fine-tune (rò kiểu D9);
  - bản tiên nghiệm bỏ-một-lớp (Ψ0 của lớp k lấy Σ gộp của các lớp khác): ν0 nó chọn và độ chính xác của A2, A3 theo nó. Dùng để hiệu chuẩn tiên nghiệm cho lớp mới ở E09.08;
  - ν0 được chọn ở từng chuỗi, và đường cong leave-one-out trên cả lưới.
- Cơ chế, cách đọc chốt trước:
  - ν0 ở biên trên của lưới: dữ liệu không đòi hiệp phương sai riêng từng lớp; A2 và A3 khi đó gần trùng LDA.
  - ν0 ở biên dưới: tiên nghiệm gần như không được dùng.
  - Log mật độ test của A3 phải cao hơn của A2 (Aitchison 1975); nếu không, kiểm lại cài đặt trước khi đọc kết quả.
  - Student-t có bậc tự do nhỏ so với d gần như bất biến khi nhân tỉ lệ ma trận, nên A3 có thể kém A2 về độ chính xác dù hơn về mật độ.
- Phép kiểm toàn vẹn; chỉ đọc kết quả khi cả ba đạt:
  - C1: A0 trùng `acc0` của ảnh chụp ở 15/15 chuỗi GC10; trên CUB, A0 của chuỗi lab trùng S0_All của run `cub_base_ft_r<r>_s<N>` cùng FT seed và seed, ở 15/15.
  - C2: trên dữ liệu thật của một chuỗi mỗi dataset, log hợp lý biên bằng tổng log dự đoán tuần tự (lệch tương đối ≤ 1e-8); leave-one-out tính nhanh trùng tính trực tiếp ở một lớp (≤ 1e-8).
  - C3: A1 trùng kết quả của `--estimator closed` trên cùng đặc trưng (nhãn dự đoán trùng hẳn).
- Nếu C1–C3 không đạt: không đọc kết quả, sửa lỗi rồi chạy lại một lần, ghi lý do.
- Đã thấy trước tiền đăng ký này:
  - S0_All của A0 trên 15 chuỗi mặc định GC10 và 15 run CUB upstream (`50d72f3`);
  - E06.03 trên GC10 `fz` và CIFAR-100 (`notes/improvement-1.vi.md`);
  - giả lập Gaussian ngoài repo (dữ liệu tự sinh cỡ CUB và cỡ GC10), dùng để chọn luật leave-one-out và tiên nghiệm gộp mọi lớp; không phải dữ liệu dự án.
  - Chưa thấy: mọi số của A1, A2, A3 trên `ft` của GC10 và trên CUB; tham số phiên đầu của A0 trên CUB.
- Dự đoán [suy luận, xác suất chủ quan]:
  - GC10: D > 0 (khoảng 60%); D ≥ 1 (khoảng 40%);
  - CUB: D > 0 (khoảng 55%); D ≥ 1 (khoảng 40%);
  - |A3 − A2| < 0.5 điểm: GC10 (khoảng 85%), CUB (khoảng 60%); trên CUB, A3 có thể thấp hơn A2;
  - CUB: A1 thấp hơn A2 (khoảng 75%).
- Điểm yếu đã biết, không sửa: D_r > 0 ở ít nhất 2/3 FT seed đạt với xác suất 0.5 khi không có hiệu ứng; chỉ có 3 FT seed; đặc trưng thật không Gaussian; trên GC10, trần do nhãn của S0 khoảng 92.5% (Q08.03).
- Ngoài phạm vi: chạy cả chuỗi với lớp cũ từ A3 (tầng 2); chọn κ0; VB-GMM (E09.08); gộp cụm (E09.06).
- Cách chạy: biến thể `06b-vb-s0` (trải 06–08) trong `lab/variants.py`; runner trong `lab/score.py`; mỗi chuỗi một run qua `exp.py`, ghi mọi số vào `metrics.json`; code commit trước khi chạy.
- Tag: `gc10_e06_04_r<r>_s<N>` và `cub_e06_04_r<r>_s<N>`, r = 0–2, N = 0–4.

### Kết quả (2026-11-12; run `gc10_e06_04_r0_s0`–`cub_e06_04_r2_s4`, code `a5b4ff4`, commit run `7b97b22`; số in từ `python lab/score.py e0604-summary`)

- Sửa ngày 2026-11-13: tiêu đề mục này từng ghi code `60bbef6`; repo không có commit đó, còn `config.json` của cả 30 run ghi `a5b4ff4`, cây sạch, mã thoát 0 (lệnh chỉ đọc 2026-11-13).
- Toàn vẹn: `checks_ok` = 1 ở 30/30 run; C2 đạt ở r0 s0 của cả hai dataset (GC10: 1.5e-15 và 8.4e-16; CUB: 4.9e-14 và 4.2e-16); C1 trùng mốc (GC10 206/241, CUB 2513/2929 ở r0 s0); C3 không có nhãn khác.
- SD gộp của độ chính xác A0 giữa các seed: 0.448 điểm (GC10), 0.179 điểm (CUB); đúng câu "nhỏ hơn 0.5 điểm" của tiền đăng ký.
- Quyết định:
  - GC10: D = +1.909 điểm; D_r = 2.573 / 2.905 / 0.249, dương ở 3/3 FT seed: ĐẠT. A2 − A0 = +1.770: ĐẠT.
  - CUB: D = +1.586 điểm; D_r = 1.325 / 1.598 / 1.837, dương ở 3/3 FT seed: ĐẠT. A2 − A0 = +1.600: ĐẠT.
  - Hệ quả (theo tiền đăng ký): A3 là biểu diễn lớp cũ cho E09.08.
- ν0 − d − 1 được chọn: GC10 10 ở 13/15 chuỗi, 25 ở 2/15; CUB 250 ở 15/15. Không chuỗi nào dính biên lưới. Bản bỏ-một-lớp: GC10 10; CUB 100 (r0, r1) và 250 (r2).
- Báo cáo phụ (trung bình theo FT seed r0 / r1 / r2):
  - A3 − A2: GC10 +0.25 / −0.17 / +0.33; CUB −0.01 / 0.00 / −0.03 điểm.
  - A1 − A0: GC10 +1.91 / +1.66 / −1.49; CUB −1.66 / −2.08 / −1.71 điểm.
  - Balanced accuracy A0 → A3: GC10 85.93 → 88.28, 85.35 → 88.59, 86.63 → 87.09; CUB 86.05 → 87.32, 86.17 → 87.76, 85.59 → 87.43.
  - Recall thấp nhất A0 → A3: GC10 69.6 → 69.3, 72.2 → 69.6, 66.3 → 66.7; CUB 31.3 → 40.7, 32.0 → 30.7, 25.3 → 41.3.
  - SD giữa 5 seed, A0 / A3: GC10 0.371 / 0.371, 0.455 / 0.541, 0.508 / 0.473; CUB 0.227 / 0.129, 0.180 / 0.147, 0.112 / 0.178.
  - Log mật độ test của lớp đúng: A3 > A2 ở 15/15 chuỗi mỗi dataset.
  - AUROC tách cũ/mới trên pool phiên 1 (A0 / A1 / A2 / A3): GC10 0.34 / 0.87 / 0.87 / 0.87, 0.25 / 0.90 / 0.89 / 0.89, 0.24 / 0.88 / 0.87 / 0.88; CUB 0.76 / 0.85 / 0.79 / 0.79, 0.75 / 0.84 / 0.79 / 0.79, 0.76 / 0.86 / 0.79 / 0.79.
  - Tiên nghiệm bỏ-một-lớp, độ chính xác A3: GC10 87.72 / 87.88 / 86.81; CUB 86.79 / 87.35 / 87.31.
  - A0 không đẩy ảnh test nào vào khe chưa thấy (`A0_unseen` = 0 ở 30/30), nên A0 và `A0base` trùng nhau.
  - D23 trên CUB, đo trực tiếp: trung vị ‖μ_k‖/‖x̄_k‖ = 0.0107; trị riêng của Σ_k trong 0.85–1.82. Trên GC10: 0.009, trị riêng 0.82–1.34.
- Dự đoán: GC10 D > 0 và D ≥ 1: đúng cả hai; CUB D > 0 và D ≥ 1: đúng cả hai; |A3 − A2| < 0.5 ở GC10 và CUB: đúng; CUB A1 thấp hơn A2: đúng.
- Cơ chế: log mật độ A3 > A2 ở 15/15, đúng như chốt trước; ν0 không ở biên.

### Đánh đổi

- GC10 phụ thuộc FT seed: ở r2, D_r chỉ +0.25 điểm (khoảng 0.6 ảnh test, tính tay) và 3/5 chuỗi có D ≤ 0; mức +1.91 đến chủ yếu từ r0, r1. CUB đều: mọi chuỗi trong khoảng +1.09 đến +2.01.
- Leave-one-out chọn hai chế độ ngược nhau. Tỉ trọng tiên nghiệm off/(off + n), tính tay: GC10 khoảng 10/160, gần hiệp phương sai riêng từng lớp; CUB khoảng 250/275, gần một hiệp phương sai chung (LDA).
- Phân lớp và tách cũ/mới kéo về hai phía trên CUB: A3 hơn A1 khoảng 3.3 điểm độ chính xác (tính tay) nhưng kém 0.05–0.07 AUROC. [SUY LUẬN] Co mạnh về Σ gộp làm lớp cũ rộng hơn, nên giải thích được cả ảnh mới. Mức co tốt nhất cho phân lớp chưa chắc tốt nhất cho khám phá ở E09.08.
- Độ ổn định theo SD giữa các seed không tốt hơn A0. Mức tăng trên GC10 không nằm ở lớp yếu nhất; trên CUB recall thấp nhất tăng ở r0, r2 nhưng giảm ở r1.
- Student-t không thêm gì vào độ chính xác (A3 − A2 trong ±0.35 điểm), dù luôn hơn về log mật độ. Phần có ích ở phiên đầu là tiên nghiệm.

### Kết luận

- ĐẠT trên cả GC10 và CUB: A3 (hậu nghiệm NIW, dự đoán Student-t, ν0 chọn bằng leave-one-out) là biểu diễn lớp cũ cho E09.08.
- A2 về bản chất là co hiệp phương sai về Σ gộp (gần RDA, Friedman 1989); đóng góp mới không nằm ở phiên đầu mà phải nằm ở phần liên tục của E09.08.
- Mang sang E09.08: (1) mức co của thành phần cũ là một núm đánh đổi giữa phân lớp và tách cũ/mới, phải đo chứ không giả định; (2) tiên nghiệm lấy từ các lớp khác dự đoán gần bằng tiên nghiệm gộp (CUB thấp hơn 0.3–0.6 điểm, tính tay), ủng hộ tiên nghiệm phân cấp cho lớp mới.
- Giới hạn: số laptop, cần server chạy lại (QD27); chỉ phiên đầu; n = 3 FT seed; AUROC trên CUB lạc quan vì rò kiểu D9.
- Theo chiến lược 2026-11-05 §4: rà lại toàn bộ chiến lược sau E06.04 và E09.06; bước tiếp theo là E09.06.

## E09.04 — Phân cụm thừa: lớp hiếm có cụm riêng không? (Q09.11)

Mã đổi theo QD22 (2026-10-18); nội dung không đổi. Bản gốc: commit `e02f0af` (tiền đăng ký), `7937b1c` (sửa đổi), `37db095` (kết quả).

### Tiền đăng ký (2026-10-05; commit: e02f0af)

- Câu hỏi (Q09.11): trên đặc trưng DINO đóng băng của GC10, nếu k-means có nhiều cụm hơn số lớp mới (K' > 5), crease hoặc rolled_pit có sở hữu một cụm chứa ít nhất một nửa số ảnh của nó không?
- Làm rõ Q09.11: "chiếm đa số" hiểu là lớp đông nhất trong cụm (điều kiện sở hữu, như E09.03 ở `notes/improvement-2.vi.md`).
- Giả thuyết: lỗi ở E09.03 (crease, rolled_pit nằm trong cụm của waist_folding; silk_spot chia 3 cụm) là do thiếu sức chứa (uniform effect của k-means; Xiong, Wu & Chen 2009, IEEE TSMC-B), không phải do đặc trưng không tách được các lớp này.
- Dữ liệu: pool S1 đúng như pipeline, 748 ảnh (silk 420, inclusion 135, waist 94, crease 32, rolled 17, ảnh cũ 50); đặc trưng `gc10_fz_cls5first_img80pct`, 768 chiều thô như bước phân cụm của pipeline.
- Cách làm: script offline gọi lại loader → PCA của S0 → k-means đúng thứ tự RNG như `main.py` (PCA ở S0 dùng solver randomized nên cũng rút RNG; PCA chỉ gọi để giữ thứ tự, phân cụm vẫn trên 768 chiều). Với mỗi seed 0–9, chạy k-means (`n_init` 10) với K' ∈ {5, 10, 15, 20, 30}, mỗi K' bắt đầu từ cùng một trạng thái RNG. Nhãn thật chỉ dùng để chấm.
- Đại lượng (n_ck = số ảnh lớp c trong cụm k, từ bảng chéo `contingency_matrix`):
  - c sở hữu k nếu c là lớp đông nhất trong k (tính cả 5 lớp cũ).
  - best(c) = max n_ck / n_c trên các cụm c sở hữu; bằng 0 nếu c không sở hữu cụm nào.
  - Kèm theo: số cụm c sở hữu, tổng recall trên các cụm c sở hữu, độ thuần n_ck / n_k của cụm tốt nhất (purity; Manning, Raghavan & Schütze 2008, §16.3), inertia, CV cỡ cụm.
  - Độ phân giải: một ảnh rolled = 5.9 điểm recall (100/17), một ảnh crease = 3.1 điểm (100/32). Ngưỡng 50% nghĩa là rolled ≥ 9 ảnh, crease ≥ 16 ảnh.
- Phép kiểm toàn vẹn; chỉ đọc kết quả K' > 5 khi cả ba đạt:
  - C1: ở K' = 5, seed 0–4, inertia trùng khít inertia mà các run `gc10_frozen_km_evhung_d64_head_s*` in ra (792075.8125, 789687.4375, 790486.1875, 788922.1875, 790444.375). Nếu trượt: dừng, chẩn đoán thứ tự RNG.
  - C2: pool đúng 748 ảnh với số ảnh từng lớp như trên.
  - C3 (đối chứng dương): ở K' = 5, waist_folding có best ≥ 50% ở mọi seed (E09.03: 92/94 ảnh trong một cụm).
- Dự đoán [suy luận, xác suất chủ quan khoảng 60%]: ở K' = 20, ít nhất một trong hai lớp đạt; rolled_pit dễ đạt hơn crease (probe tuyến tính: rolled 67, crease 45).
- Quy tắc quyết định (chốt ở K' = 20; các K' khác chỉ để mô tả):
  - Một lớp đạt nếu best ≥ 50% ở ít nhất 6/10 seed.
  - Sống: crease hoặc rolled_pit đạt → E09.06, gộp cụm con theo khoảng cách Bhattacharyya, ngưỡng hiệu chỉnh trên lớp cơ sở có nhãn.
  - Loại: cả hai không đạt → bỏ hướng phân cụm thừa trên đặc trưng CLS đóng băng; chuyển sang câu hỏi đặc trưng (phase 1) hoặc Active GCD (Q09.05).
- Ngoài phạm vi: đặc trưng fine-tune (làm sau nếu cần).
- Tag: `gc10_pl01_s<N>`, N = 0–9; mỗi run ghi mọi K' vào `metrics.json`.

### Sửa đổi (2026-10-06, quyết định 2026-10-05; trước khi chạy; phần tiền đăng ký ở trên giữ nguyên)

- C1 đổi thành: ở K' = 5, seed 0–4, bảng chéo lớp × cụm (contingency table) của bản phát lại (`scripts/replay.py`) trùng khít bảng mà các run `gc10_frozen_km_evhung_d64_head_s*` in ra; inertia chỉ cần lệch tương đối < 1e-6 so với các giá trị ở C1 cũ.
- Lý do: `scripts/replay.py` cho bảng chéo trùng khít ở cả 5 seed, nhưng inertia lệch đúng 1 ULP float32 (0.0625 ở độ lớn khoảng 790 nghìn); seed bị lệch đổi theo thứ tự import thư viện (lần đầu seed 1–3, sau khi đổi thứ tự import là seed 0). [suy luận] Bit cuối của inertia phụ thuộc thứ tự cộng số dấu phẩy động (Goldberg 1991), không phụ thuộc luồng RNG.
- Tiêu chí mới đủ cho C1: mọi đại lượng quyết định của E09.04 (sở hữu, best, độ thuần) là hàm của bảng chéo. Bảng trùng chưa chắc phân hoạch trùng từng ảnh (đổi chỗ hai ảnh cùng lớp giữa hai cụm cho cùng bảng), nhưng điều đó không đổi kết quả E09.04.
- Nguồn tham chiếu: bảng chéo chỉ có trong `console.log` của các run trên (không vào git); phần có trong git là inertia ở C1 cũ và cỡ cụm từng seed (`notes/improvement-2.vi.md`, E09.03). Vì vậy `metrics.json` của E09.04 phải ghi bảng chéo K' = 5 của bản phát lại.
- Sửa đổi làm trước khi có bất kỳ dữ liệu nào của E09.04 (K' > 5 chưa chạy).

### Kết quả (2026-10-07; run `gc10_pl01_s0`–`s9`, commit 89e94d4)

- Cài đặt: `scripts/pl_cluster.py` gọi `scripts/replay.py` (`stop="cluster"`, nên SVI ở S0 vẫn chạy) và chụp trạng thái RNG ở lối vào `KMeansCluster.fit`; mọi K' bắt đầu từ trạng thái đó. Thứ tự rút RNG như tiền đăng ký; C1 xác nhận.
- Toàn vẹn: C1 (seed 0–4: bảng chéo trùng các run `gc10_frozen_km_evhung_d64_head`, inertia lệch tương đối < 1e-6; seed 5–9: phân hoạch K' = 5 trùng pipeline), C2, C3 đạt ở cả 10 seed.
- Quyết định ở K' = 20, số ảnh trong cụm tốt nhất theo seed 0–9:
  - rolled_pit: 13, 0, 0, 0, 0, 0, 0, 11, 0, 0 → đạt 2/10 seed.
  - crease: 8, 15, 13, 13, 0, 14, 0, 8, 16, 13 → đạt 1/10 seed.
  - Cả hai không đạt: LOẠI.
- Kèm theo ở K' = 20 (trung bình 10 seed): rolled sở hữu 0.2 cụm, độ thuần trung vị của cụm tốt nhất 0.25 (2 seed có cụm); crease best 10.0/32, tổng trên các cụm sở hữu 11.6/32, độ thuần trung vị 0.41.
- Mô tả theo K' = 5, 10, 15, 20, 30 (số seed đạt): crease 0, 7, 4, 1, 1; rolled 0, 0, 2, 2, 0. Balanced accuracy ghép một–một trên pool: 0.366, 0.407, 0.391, 0.342, 0.282.
- Dự đoán (ít nhất một lớp đạt; rolled dễ hơn crease): sai.

### Đánh đổi

- Chốt ở K' = 20 tránh vườn rẽ nhánh, nhưng bỏ qua việc crease đạt nhiều nhất ở K' = 10; số đó chỉ để mô tả.
- best(c) chỉ tính cụm lớn nhất, nên lớp bị chẻ thành nhiều cụm nhỏ bị đánh giá thấp; bước gộp có thể nhặt lại.

### Kết luận

- LOẠI theo quy tắc: trên đặc trưng CLS đóng băng, phân cụm thừa không cho rolled_pit hay crease cụm riêng; khi có, cụm đó lẫn nhiều lớp khác. Đây là bối cảnh phụ (QD19); kết luận về hướng lấy theo E09.05.

## E09.05 — Phân cụm thừa trên cấu hình paper (Q09.11, QD19)

Mã đổi theo QD22 (2026-10-18); nội dung không đổi. Bản gốc: commit `885e2dc` (tiền đăng ký), `37db095` (kết quả).

### Tiền đăng ký (2026-10-07; commit: 885e2dc)

- Câu hỏi (Q09.11, trên cấu hình paper): với đặc trưng fine-tune trên tập có nhãn, nếu k-means có K' > 5 cụm, crease hoặc rolled_pit có sở hữu một cụm chứa ít nhất một nửa số ảnh của nó không?
- Vì sao có thí nghiệm này: QD19 đặt mốc so sánh là cấu hình paper; E09.04 (`fz`) là bối cảnh phụ. Kết luận về hướng phân cụm thừa lấy theo E09.05.
- Giả thuyết: như E09.04 (thiếu sức chứa do hiệu ứng đồng đều; Xiong, Wu & Chen 2009). Ở `ft` r0 seed 4 với K = 5, 27/32 ảnh crease và 16/17 ảnh rolled nằm chung cụm do waist sở hữu (`notes/nb/04_s1_discovery.ipynb`).
- Dữ liệu: pool S1 đúng như pipeline (748 ảnh, thành phần như E09.04); đặc trưng `gc10_ft_cls5first_img80pct_r0`, `_r1`, `_r2` (`dino-vitb16-sl`, 768 chiều, đúng đầu vào của bước phân cụm).
- Cách làm: với mỗi FT seed r ∈ {0, 1, 2} và mỗi seed 0–9, `scripts/replay.py` (`stop="cluster"`) chạy lại pipeline tới hết bước 09; trạng thái RNG ở lối vào `KMeansCluster.fit` được chụp từ bên ngoài (`notes/nb/12_cluster.ipynb` cell 8). Với mỗi K' ∈ {5, 10, 15, 20, 30}, khôi phục trạng thái đó rồi chạy `KMeansCluster` (`n_init` 10). Nhãn thật chỉ dùng để chấm.
- Đại lượng: như E09.04 (sở hữu, best(c), các đại lượng kèm theo), tính bằng `scripts/pl_cluster.py`. Ngưỡng 50% nghĩa là rolled ≥ 9/17 ảnh, crease ≥ 16/32 ảnh.
- Phép kiểm toàn vẹn; chỉ đọc kết quả K' > 5 khi cả ba đạt:
  - C1: ở K' = 5, phân hoạch chạy lại từ trạng thái đã chụp trùng phân hoạch của pipeline trong cùng lần phát lại; với seed 0–4, cỡ cụm (sắp xếp) còn phải trùng cỡ cụm in trong `console.log` của run `gc10_ft_cls5first_img80pct_r<r>_km_evhung_d64_s<N>` (run của E02.05, 2026-10-05).
  - C2: pool đúng 748 ảnh với thành phần như E09.04.
  - C3 (đối chứng dương): ở K' = 5, waist_folding có best ≥ 50% ở mọi cặp (FT seed, seed).
- Dự đoán [suy luận, xác suất chủ quan khoảng 50%]: ở K' = 20, ít nhất một trong hai lớp đạt. Không đoán lớp nào dễ hơn: trên `ft` cả hai cùng nằm trong cụm của waist.
- Quy tắc quyết định (chốt ở K' = 20; các K' khác chỉ để mô tả):
  - Trong một FT seed, một lớp đạt nếu best ≥ 50% ở ít nhất 6/10 seed.
  - Một lớp đạt E09.05 nếu đạt ở ít nhất 2/3 FT seed.
  - Sống: crease hoặc rolled_pit đạt → E09.06 (gộp cụm con), làm trên cấu hình paper.
  - Loại: cả hai không đạt → bỏ hướng phân cụm thừa cho cấu hình paper, dù E09.04 cho kết quả nào; chuyển sang Q09.06 (phương pháp khác ở bước 09), Q03.03 (đặc trưng) hoặc Q09.05.
- Ngoài phạm vi: phương pháp phân cụm khác (Q09.06); bước gộp (E09.06).
- Tag: `gc10_pl01b_r<r>_s<N>`, r = 0–2, N = 0–9 (30 run); mỗi run ghi mọi K' vào `metrics.json`.

### Kết quả (2026-10-07; run `gc10_pl01b_r0_s0`–`gc10_pl01b_r2_s9`, commit 89e94d4)

- Toàn vẹn: C1 (phân hoạch K' = 5 trùng pipeline; seed 0–4: cỡ cụm trùng run của E02.05), C2, C3 đạt ở cả 30 run.
- Quyết định ở K' = 20, số ảnh trong cụm tốt nhất theo seed 0–9:
  - rolled_pit r0: 13, 0, 11, 12, 12, 0, 0, 0, 12, 10 (6/10); r1: 13, 13, 14, 0, 0, 13, 11, 12, 12, 14 (8/10); r2: 11, 0, 8, 13, 0, 12, 12, 8, 0, 14 (5/10) → đạt 2/3 FT seed.
  - crease r0: 19, 19, 19, 19, 20, 15, 20, 14, 17, 19 (8/10); r1: 14, 13, 15, 14, 26, 15, 15, 15, 14, 11 (1/10); r2: 14, 16, 15, 15, 15, 16, 23, 19, 23, 20 (6/10) → đạt 2/3 FT seed.
  - Cả hai đạt: SỐNG.
- Kèm theo ở K' = 20 (trung bình 10 seed, r0/r1/r2): độ thuần trung vị của cụm tốt nhất: rolled 0.48/0.40/0.46, crease 0.75/0.71/0.62. Riêng crease r1: best 15.2/32 nhưng tổng trên các cụm sở hữu 20.2/32 (trung bình 1.5 cụm).
- Mô tả theo K' = 5, 10, 15, 20, 30 (số lần đạt trên 30 run): rolled 0, 2, 3, 19, 21; crease 15, 12, 19, 15, 11.
- Balanced accuracy ghép một–một trên pool ở K' = 5 (trung bình 10 seed): r0/r1/r2 0.497/0.491/0.392, so với `fz` 0.366. Cùng cấu hình, balanced novel cuối pipeline trên test (run của E02.05, 5 seed) là 39.4/29.2/36.2, so với `fz` 37.6: tính trung bình, tầng 1 xếp `ft` trên `fz`, tầng 2 xếp ngược lại (Q14.04).
- Dự đoán (ít nhất một lớp đạt): đúng.

### Đánh đổi

- Biên mỏng: rolled r0 và crease r2 đạt đúng 6/10. Kết luận SỐNG chịu được một seed đổi kết quả, không chịu được hai seed đổi đúng ở hai ô này. crease r1 trượt sát ngưỡng (9/10 seed có 13–15 ảnh).
- Cụm của rolled lẫn nhiều (độ thuần trung vị 0.40–0.48), nên bước gộp có nguy cơ nuốt lại.
- Khi chưa gộp, phân cụm thừa làm giảm balanced accuracy một–một trên pool (r0: 0.497 ở K' = 5, 0.413 ở K' = 20), nên bước gộp là bắt buộc.
- Kết luận ràng buộc theo `ft` đúng với cấu hình paper, nhưng thừa hưởng nhiễu giữa các FT seed (crease r1).

### Kết luận

- SỐNG theo quy tắc, ràng buộc cho cấu hình paper (QD19): với K' = 20, sức chứa đủ để rolled_pit và crease có cụm riêng ở 2/3 FT seed. Đây chỉ là điều kiện cần.
- Bước tiếp: E09.06 (gộp cụm con), tiền đăng ký riêng. K' và ngưỡng gộp phải chốt trước hoặc hiệu chỉnh trên lớp cơ sở; không chọn theo các số mô tả ở trên.

### Ghi chú (2026-10-20)

- So trên cùng tập seed: dòng balanced accuracy ở mục Kết quả lấy tầng 1 trên seed 0–9 nhưng tầng 2 trên seed 0–4. Tầng 1 tính lại trên seed 0–4 (khoá `k5_bal_acc_hungarian` trong `metrics.json` của `gc10_pl01b_r<r>_s<N>` và `gc10_pl01_s<N>`): r0/r1/r2 0.469/0.469/0.399, `fz` 0.374. Số trên seed 0–9 ở mục Kết quả giữ nguyên.
- Trên cùng 5 seed, r0 và r1 bằng nhau ở tầng 1 (0.469) nhưng cách nhau 10.2 điểm ở tầng 2 (39.4 so với 29.2). [suy luận] Phần r1 mất nằm sau phân hoạch pool: chuyển từ pool sang test, đặt tên cụm, hoặc các bước 11–14 (Q14.04). Điều này củng cố quy tắc của QD18: thứ hạng ở tầng 1 phải được xác nhận ở tầng 2.
- Bước tiếp, cập nhật mục Kết luận: E09.06 gộp cụm con tới đúng k = 5 cụm vì k đã biết, không dùng ngưỡng gộp; quy tắc quyết định đặt ở tầng 2. K' vẫn phải chốt trước khi chạy.

## E09.06 — Phân cụm thừa rồi gộp bằng Bayes factor NIW (Q09.11, QD25, QD27)

### Tiền đăng ký (2026-11-12; commit: 61b9eb9)

- Câu hỏi: ở bước 09, phân cụm pool thành K' = 20 cụm con rồi gộp về đúng 5 cụm bằng Bayes factor dưới tiên nghiệm Normal–Inverse-Wishart (NIW) có tăng balanced novel ở tầng 2 không, khi S1_Old không giảm quá B?
- Vì sao có thí nghiệm này:
  - Phần lớn điểm của lớp mới mất ở khâu khám phá, không ở đặc trưng (E03.02: probe `lr_bal` 0.61–0.71 trên `ft`, pipeline 0.29–0.39).
  - E09.05 (SỐNG): ở K' = 20, rolled_pit và crease có cụm riêng ở 2/3 FT seed; chưa gộp thì tầng 1 giảm (r0: 0.497 ở K' = 5, 0.413 ở K' = 20). Gộp về đúng k = 5 vì k đã biết, không dùng ngưỡng gộp (E09.05, Ghi chú 2026-10-20).
  - E09.07: tối ưu k-means sâu hơn ở K = 5 không đủ lớn, nên E09.06 chạy trên k-means mặc định (`n_init` 10).
  - QD27: `09b` cũng là khởi tạo dự kiến cho VB-GMM (`09e`, E09.08).
- Giả thuyết [suy luận]: ở K = 5, lớp hiếm bị nuốt vì thiếu sức chứa (E09.03–E09.05). Ở K' = 20 chúng có cụm riêng; một phép gộp ráp các mảnh của lớp lớn (silk_spot, inclusion) trước sẽ giữ được cụm đó khi về lại 5 cụm, nên nhãn giả của lớp hiếm tốt hơn và balanced novel ở tầng 2 tăng. Giả thuyết ngược: xem Rủi ro chính.
- Tiêu chí gộp đổi so với kế hoạch cũ (dòng danh mục và quy tắc Sống của E09.04, 2026-10-05, trước E06.04: khoảng cách Bhattacharyya, ngưỡng hiệu chỉnh trên lớp cơ sở; ngưỡng đã bỏ ở Ghi chú 2026-10-20 của E09.05): tiêu chí chính là Bayes factor NIW; Bhattacharyya xuống nhánh chỉ báo cáo.
  - Log Bayes factor của phép gộp hai cụm a, b là ln p(X_a ∪ X_b) − ln p(X_a) − ln p(X_b), với p(X) là hợp lý biên dưới NIW (Murphy 2007, eq. 266; `niw.log_evidence`). Gộp tham lam theo Bayes factor là mỗi bước chọn phép gộp làm tổng log hợp lý biên của phân hoạch giảm ít nhất.
  - Vì k đã biết, chỉ cần thứ tự gộp. Mỗi bước giảm đúng một cụm, nên phần hằng trong log hợp lý biên của mỗi cụm, (d/2)·ln κ0 + (ν0/2)·ln|Ψ0| − ln Γ_d(ν0/2), có mặt như nhau ở mọi cặp ứng viên và triệt tiêu khi xếp hạng. Không siêu tham số nào được chọn bằng hợp lý biên (điều E06.04 đã tránh); ν0 lấy bằng leave-one-out như E06.04.
  - Cùng phép so "một cụm hay hai cụm" bằng hợp lý biên có trong Bayesian Hierarchical Clustering (Heller & Ghahramani 2005, kèm trọng số tiên nghiệm của Dirichlet process), trong phép thử merge của split–merge cho DPMM (Jain & Neal 2004; Chang & Fisher 2013) và trong DeepDPM (Ronen et al. 2022).
  - Cơ chế [suy luận]: log Bayes factor trừ phần mất độ khớp khi hai cụm dùng chung một Gaussian (với Σ cố định, đúng bằng chi phí Ward theo Mahalanobis, ½·h·Δμᵀ Σ⁻¹ Δμ, với h = n_a·n_b/(n_a + n_b)), và cộng phần thưởng vì bớt một bộ tham số (riêng phần trung bình là (d/2)·ln h, từ công thức đóng khi κ0 ≪ n). Phần mất độ khớp nhỏ khi một trong hai cụm nhỏ, nên vẫn nghiêng về nuốt cụm nhỏ (hiệu ứng đồng đều; Xiong, Wu & Chen 2009); phần thưởng tăng theo h, nên kéo về phía ráp các mảnh lớn. Ward thuần (Ward 1963) chỉ có phần mất độ khớp, đo bằng Euclid. Bhattacharyya không phụ thuộc cỡ cụm, nhưng so một Σ co mạnh về tiên nghiệm (cụm nhỏ) với một Σ lấy từ dữ liệu (cụm lớn).
  - Giả lập ngoài repo (pool tự sinh cỡ GC10, không phải dữ liệu dự án): Bayes factor giữ cụm của lớp hiếm nhiều nhất trong bốn tiêu chí; Bhattacharyya chỉ số hạng trung bình có balanced trên pool ngang hoặc hơn Bayes factor chút ít; Ward cho lại kết quả như K = 5.
- Đã thấy trước tiền đăng ký này:
  - E03.02 (Kết luận của E03.02 yêu cầu ghi điều này), E09.03, E09.04, E09.05;
  - tầng 1 của các phân hoạch K' = 5–30 trên 15 ảnh chụp (khoá `k*_` trong `metrics.json` của `gc10_e09_05_v2_*`; QD24);
  - E09.07 (mọi số), `09-cluster`, `09d-ownership` (gồm sổ Ward và kết luận Q09.13), `10-naming` đến `14-eval`;
  - E06.04 (mọi số, gồm `off_lc` = 10 ở 15/15 chuỗi GC10);
  - giả lập ngoài repo nói trên, dùng để chọn Bayes factor.
  - Chưa thấy: mọi phân hoạch đã gộp trên dữ liệu thật; mọi số tầng 2 của phân hoạch dựng từ K' = 20.
- Dữ liệu: 15 ảnh chụp `gc10_v2_ft_r<r>_s<N>`, r = 0–2, N = 0–4 (chuỗi mặc định, QD21). Bước 04–08 giữ nguyên; chỉ đổi bước 09; bước 10–14 chạy code mặc định.
- Biến thể (`09a` rồi `09b`):
  - `09a`: `v09a_overcluster` (đã có trong `lab/variants.py`; dùng cho tầng 1 của E09.05 trên bàn mổ v2): `KMeansCluster` với K' = 20, `n_init` 10, trên 768 chiều thô của pool, rút RNG toàn cục. Giữ không gian thô để giữ đúng điều kiện của E09.05.
  - `09b`: `v09b_merge` (viết sau tiền đăng ký này): gộp tích tụ 20 cụm con về đúng 5 cụm trong không gian PCA-64 của bước 05 (`z_pool`). Mỗi bước gộp cặp có log Bayes factor lớn nhất, rồi tính lại hợp lý biên của cụm vừa gộp.
  - Tiên nghiệm, chung cho mọi cụm: m0 = trung bình của `z_pool`; κ0 = 0.01; ν0 = d + 1 + off; Ψ0 = off·Σ_gộp, với Σ_gộp là hiệp phương sai gộp trong lớp của 5 lớp cơ sở trên ảnh S0 có nhãn (tổng tán xạ chia cho N − C, như E06.04).
  - off chọn trong từng ảnh chụp bằng leave-one-out trên S0 với tiên nghiệm bỏ-một-lớp (`niw.loo_curve(..., loo_class=True)` rồi `niw.choose`), đúng luật cho ra `off_lc` ở E06.04. Lý do: cụm mới là lớp không có trong Σ_gộp, đúng tình huống mà tiên nghiệm bỏ-một-lớp giả lập.
  - Vì sao gộp trong PCA-64: trong 768 chiều thô, Σ_gộp ước từ 764 ảnh S0 nên suy biến (hạng tối đa 764 − 5 = 759 < 768; tính tay từ `n_use` của `splits/gc10_split.json`), và Ψ0 không xác định dương. Bước 11–13 cũng mô hình hoá cụm bằng Gaussian trong chính PCA-64. Giá phải trả: xem Rủi ro thứ hai.
  - Ảnh test (cho bước 10): tâm gần nhất trong 20 tâm của `09a` (768 chiều, như `predict_raw` mặc định), rồi theo ánh xạ cụm con → cụm gộp. Cụm gộp đánh số 0–4 theo thứ tự xuất hiện đầu tiên trong pool.
  - `09b` không đọc nhãn pool, không đọc test, không rút RNG; trả `ids` và `cluster` (có `predict_raw`) như bước 09 (`_check_09`).
- Thiết kế (lặp phép so trong cùng ảnh chụp, như E09.07; QD26):
  - Mỗi ảnh chụp có 11 lần rút. Lần k = 0: RNG toàn cục đặt về `rng_in` của bước 09 (phép so một lần chạy của QD25). Lần k = 1–10: RNG toàn cục đặt về trạng thái của `np.random.RandomState(906000 + 1000·r + 100·N + k)`.
  - Trong mỗi lần rút, nhánh mặc định (bước 09 mặc định: `KMeansCluster`, K = 5, `n_init` 10) và nhánh biến thể (`09a` rồi `09b`) bắt đầu từ cùng trạng thái RNG đó (số ngẫu nhiên chung, CRN).
  - Mỗi phân hoạch khác nhau (so ảnh pool và ảnh test sau khi đánh lại số cụm) chạy tiếp bước 10–14 một lần, trong cùng tiến trình, từ trạng thái của ảnh chụp sau bước 08; phân hoạch trùng dùng lại kết quả. Hợp lệ như ở E09.07: sau bước 09 không bước nào rút RNG toàn cục, và SVI dùng `PRNGKey(0)`.
- Đại lượng chính (QD25; cách so như E09.07): balanced novel ở tầng 2 (trung bình recall của 5 lớp mới trên 214 ảnh test lớp mới, tên cụm QD15).
  - D_i = trung bình theo các lần rút của (balanced novel của biến thể − của mặc định) ở ảnh chụp i.
  - Δ_r = trung bình D_i trên 5 seed của FT seed r; Δ = trung bình Δ_r trên 3 FT seed. Đơn vị: FT seed (n = 3).
  - Ràng buộc lớp cũ: Δ_Old, tính cùng cách với S1_Old (241 ảnh test lớp cũ, cuối phiên S1).
  - Sai số Monte Carlo như E09.07: SE_MC = √(Σ_i s_i²/K)/15, với s_i là SD (ddof = 1) của hiệu giữa các lần rút ở ảnh chụp i và K là số lần rút.
- Ngưỡng: T = 0.0273 (đơn vị tỉ lệ, như Δ) và B = 0.615 điểm % (như Δ_Old), đúng hai số của E09.07 phần 2 (SD gộp giữa các seed của chuỗi mặc định; `14-eval` §3.3). Runner tính lại hai số này từ các khoá `ref_*`; số tính lại phải làm tròn ra đúng như trên.
- Quy tắc quyết định:
  - Chấp nhận: Δ ≥ T, Δ_r > 0 ở ít nhất 2/3 FT seed, và Δ_Old ≥ −B → `09a` + `09b` (K' = 20, Bayes factor NIW) thành bước 09 mặc định của dòng PL; E09.08 khởi tạo VB-GMM từ phân hoạch này.
  - Bác: Δ ≤ 0, hoặc Δ_Old < −B → giữ bước 09 mặc định.
  - Không đủ lớn: các trường hợp còn lại (0 < Δ < T, hoặc Δ ≥ T nhưng Δ_r > 0 ở dưới 2/3 FT seed) → giữ bước 09 mặc định; báo cáo.
  - Nếu Δ hoặc Δ_Old cách một ranh giới quyết định (T, 0, −B) dưới 2·SE_MC: thêm 10 lần rút (k = 11–20, cùng công thức seed) cho cả 15 ảnh chụp, một lần duy nhất, rồi quyết theo 21 lần rút.
  - Sau E09.06: rà lại chiến lược trước khi tiền đăng ký E09.08 (chiến lược 2026-11-05 §4).
- Điều kiện bỏ ý tưởng (điều kiện bỏ mũi A, tức hướng cải tiến khám phá ở bước 09, của chiến lược 2026-11-05 §4, viết thành phép đo):
  - Lớp c có cụm riêng ở một cặp (seed, lần rút) nếu best(c) ≥ 50% trên phân hoạch pool đã gộp (định nghĩa của E09.04: rolled_pit ≥ 9/17 ảnh, crease ≥ 16/32 ảnh). Trong FT seed r, lớp c giữ cụm nếu điều này đúng ở ít nhất một nửa số cặp (seed, lần rút) của r (28/55 với 11 lần rút; 53/105 nếu thêm lần rút).
  - Bỏ: E09.06 không được chấp nhận, và cả crease lẫn rolled_pit đều giữ cụm ở dưới 2/3 FT seed → bỏ mũi A; chuyển sang đặc trưng mức vùng lỗi (Q03.03) hoặc nhãn thật tối thiểu (trục lượng nhãn; QD16, QD27).
  - Không bỏ: E09.06 không được chấp nhận nhưng ít nhất một lớp hiếm giữ cụm ở ít nhất 2/3 FT seed → ghi rằng nút thắt nằm sau bước 09 (bước 10–14), là phần việc của E09.08 (trách nhiệm mềm thay nhãn cứng).
- Báo cáo phụ (không dùng để quyết định):
  - kết quả riêng của k = 0;
  - M1: Δ balanced ở cột 10 của sổ (mức cụm, tên QD15); M3: Δ bước dừng của lượt S (`stop_S`); cả hai tính như Δ;
  - S1_New, S1_All, M_f, balanced accuracy 10 lớp; recall từng lớp mới kèm khoảng Wilson (Wilson 1927);
  - sau gộp (nhãn pool chỉ để chấm): tỉ lệ cặp (seed, lần rút) mà crease, rolled_pit có cụm riêng, và mà chúng sở hữu ít nhất một cụm; số cụm silk_spot sở hữu; độ thuần của phép gộp (tỉ lệ bước gộp nối hai cụm có cùng lớp đông nhất);
  - tầng 1 (`score09`: `bal_acc_hungarian`, số cụm sở hữu, best) của sáu phân hoạch trong mỗi lần rút: K = 5 mặc định, K' = 20 chưa gộp, gộp Bayes factor, và ba nhánh dưới đây;
  - ba nhánh chỉ báo cáo, trên cùng 20 cụm con, chỉ chấm ở tầng 1, không chạy bước 10–14; mỗi bước gộp cặp có chi phí nhỏ nhất rồi tính lại cụm vừa gộp:
    - Ward trong 768 chiều thô: chi phí n_a·n_b/(n_a + n_b)·‖μ_a − μ_b‖², tức mức tăng inertia (Ward 1963);
    - Bhattacharyya đầy đủ (Eq. 9 của paper) giữa hai Gaussian kỳ vọng hậu nghiệm NIW (trung bình mn, hiệp phương sai Ψn/(νn − d − 1)) trong PCA-64, cùng tiên nghiệm với `09b`;
    - như trên, chỉ số hạng trung bình ⅛·Δμᵀ Σ̄⁻¹ Δμ, với Σ̄ = (Σ_a + Σ_b)/2;
  - off của từng ảnh chụp; inertia ở K = 5 và K' = 20; số phân hoạch khác nhau đã chạy 10–14.
- Phép kiểm toàn vẹn; chỉ đọc kết quả khi cả năm đạt:
  - C1: ở k = 0, nhánh mặc định cho đúng `ids` và id test của ảnh chụp, inertia lệch tương đối ≤ 1e-6, và các số ở bước 14 của chuỗi mặc định (kết quả rời rạc trùng hẳn; số thực lệch tương đối ≤ 1e-6), ở 15/15 ảnh chụp (như C1 của E09.07).
  - C2: ở k = 0, phân hoạch K' = 20 của `09a` trùng số tầng 1 đã lưu của `gc10_e09_05_v2_r<r>_s<N>` (cùng hàm, cùng `rng_in`): `k20_inertia` lệch tương đối ≤ 1e-6, `k20_owned_*` và `k20_best_*` trùng hẳn, ở 15/15 ảnh chụp.
  - C3: mọi phân hoạch chạy 10–14 qua `_check_09`.
  - C4: ở mọi phân hoạch đã chạy 10–14, gained 08→14 = 0, và số ảnh đúng của S1_Old = số đúng ở 08 − lost 08→14 (như C4 của E09.07).
  - C5: mỗi phân hoạch gộp có đúng 5 cụm không rỗng, và mỗi cụm con nằm trong đúng một cụm gộp (cả pool lẫn test); ở mọi bước gộp của `09b`, cặp được chọn có log Bayes factor lớn nhất khi tính lại trực tiếp bằng `niw.log_evidence` trên mọi cặp hiện có, và giá trị dùng để chọn trùng giá trị tính lại (lệch ≤ 1e-6 nat); off trùng `off_lc` của run `gc10_e06_04_r<r>_s<N>`.
- Nếu C1–C5 không đạt: không đọc kết quả, sửa lỗi rồi chạy lại một lần, ghi lý do.
- Dự đoán [suy luận, xác suất chủ quan]:
  - với mỗi lớp crease, rolled_pit: trong các cặp (seed, lần rút) mà lớp có cụm riêng ở K' = 20 chưa gộp, sau gộp vẫn còn ở ít nhất một nửa (khoảng 60%);
  - silk_spot sở hữu đúng 1 cụm sau gộp ở hơn một nửa số cặp (seed, lần rút) (khoảng 55%);
  - M1 > 0 (khoảng 65%);
  - Δ > 0 (khoảng 55%); Δ ≥ T (khoảng 30%). Tầng 1 và tầng 2 từng xếp hạng ngược nhau (E09.05, Ghi chú 2026-10-20; E09.07, Đánh đổi).
- Rủi ro chính [suy luận]: các cụm con của silk_spot là nhóm camera/cuộn thật (Q09.13: trong silk, camera và cuộn giải thích 0.87–0.92 thành viên cụm; `09d-ownership` §6), không phải mảnh cắt của một Gaussian; crease ở r1, r2 cũng chia theo cuộn. Khi đó Bayes factor có thể giữ các nhóm silk riêng và gộp cụm lớp hiếm vào waist_folding trước. Dấu hiệu: sau gộp, silk sở hữu từ 2 cụm trở lên và lớp hiếm mất cụm riêng. Ba nhánh chỉ báo cáo cho biết hiện tượng đó riêng của Bayes factor hay chung cho mọi tiêu chí gộp.
- Rủi ro thứ hai [suy luận]: PCA-64 fit trên S0 làm mất một phần tách tuyến tính của lớp mới (E03.02, Kết luận: tới 11 điểm ở r1, r2, không mất ở r0), nên phép gộp có thể nối những cụm con mà không gian thô còn tách được. Thiết kế không tách riêng yếu tố không gian (Ward chạy trong không gian thô nhưng khác cả tiêu chí). Nếu lớp hiếm mất cụm riêng sau gộp ở r1, r2 mà không mất ở r0, kết luận ghi là chưa tách được hai nguyên nhân.
- Điểm yếu đã biết, không sửa:
  - Δ_r > 0 ở ít nhất 2/3 FT seed đạt với xác suất 0.5 khi không có hiệu ứng; chỉ có 3 FT seed;
  - T là ngưỡng thực tiễn (d ≥ 1), không phải phép kiểm thống kê; SE_MC chỉ đo nhiễu của các lần rút;
  - tiên nghiệm bỏ-một-lớp mới được kiểm trên lớp cơ sở (E06.04), chưa kiểm trên lớp mới;
  - gộp tham lam không bảo đảm cực đại toàn cục của tổng log hợp lý biên;
  - K' = 20 lấy từ quy tắc của E09.05, không chỉnh.
- Ngoài phạm vi: CUB (cần bàn mổ nhiều phiên; thiết kế riêng, chưa có quyết định); ước lượng K (Q09.02); VB-GMM (E09.08); K' khác 20; chọn κ0; tầng 2 cho các nhánh chỉ báo cáo.
- Chi phí ước lượng [suy luận, từ thời gian của E09.07: 30–51 phút mỗi ảnh chụp theo `config.json` của 15 run]: khoảng 8–10 giờ cho 15 ảnh chụp.
- Cách chạy: biến thể trong `lab/variants.py` (`v09a_overcluster` đã có, `v09b_merge` mới); hàm tầng 2 `tier2_e0906` và bảng quyết định `e0906-summary` trong `lab/score.py`, dùng lại `_tail_0907`, `_measures_0907`, `score09`; 15 run qua `exp.py`, mỗi run một ảnh chụp, ghi mọi lần rút vào `metrics.json`; code viết sau tiền đăng ký và commit trước khi chạy.
- Tag: `gc10_e09_06_r<r>_s<N>`, r = 0–2, N = 0–4.

### Kết quả (2026-11-13; run `gc10_e09_06_r0_s0`–`gc10_e09_06_r2_s4` (commit `994656a`) và `gc10_e09_06x_r0_s0`–`gc10_e09_06x_r2_s4` (commit kế tiếp); code `6996b57`; số in từ `python lab/score.py e0906-summary`)

- Toàn vẹn: `checks_ok` = 1 ở 30/30 run (C1, C2 ở 15 run có lần rút 0; C3, C4, C5 ở cả 30). off = 10 ở 15/15 ảnh chụp, trùng `off_lc` của E06.04. Ngưỡng tính lại: T 0.02728, B 0.61545, làm tròn đúng hai số đã tiền đăng ký.
- Sau 11 lần rút: Δ = 0.0655, Δ_Old = −0.5533 điểm, cách −B dưới 2·SE_MC (SE_MC 0.0569). Theo quy tắc: thêm 10 lần rút (k = 11–20) cho cả 15 ảnh chụp, một lần.
- Sau 21 lần rút:
  - Δ = 0.0642 ≥ T; Δ_r = 0.0080 / 0.1122 / 0.0724, dương ở 3/3 FT seed; Δ_Old = −0.5295 điểm ≥ −B; SE_MC = 0.0036 (Δ), 0.0425 (Δ_Old); không ranh giới nào trong 2·SE_MC. Quyết định: CHẤP NHẬN.
  - Ghi chú: Δ_Old cách −B 0.0855 điểm, còn 2·SE_MC = 0.085 điểm (tính tay); nằm ngay ngoài vùng phải thêm lần rút. Quy tắc chỉ cho thêm một lần.
  - Hệ quả (theo tiền đăng ký): `09a` + `09b` thành bước 09 mặc định của dòng PL; E09.08 khởi tạo VB-GMM từ phân hoạch này.
- Điều kiện bỏ ý tưởng: không áp dụng (chấp nhận). Số cặp (seed, lần rút) có cụm riêng sau gộp, trên 105 mỗi FT seed (cần 53): crease 88 / 65 / 44; rolled_pit 1 / 2 / 0.
- Báo cáo phụ (theo FT seed r0 / r1 / r2):
  - M1 (Δ balanced cột 10): 0.0519 / 0.1263 / 0.1199, trung bình 0.0994. M3 (Δ bước dừng của lượt S): 16.2 / 44.7 / 52.6, trung bình 37.85 bước.
  - Δ_Old: −0.138 / −0.553 / −0.897 điểm. Δ S1_New: −2.38 / +9.15 / +0.89; Δ S1_All: −1.19 / +4.01 / −0.06 điểm; Δ balanced 10 lớp: 0.0033 / 0.0535 / 0.0315. Δ M_f = −Δ S1_Old (acc0 chung hai nhánh).
  - Riêng k = 0: Δ 0.0625, Δ_Old −0.4426 điểm.
  - Tầng 1 (`bal_acc_hungarian` trên pool): mặc định 0.501 / 0.470 / 0.391; K' = 20 chưa gộp 0.410 / 0.407 / 0.381; Bayes factor 0.556 / 0.525 / 0.515; Ward 0.558 / 0.432 / 0.412; Bhattacharyya 0.302 / 0.299 / 0.313; Bhattacharyya chỉ số hạng trung bình 0.258 / 0.248 / 0.205.
  - Tỉ lệ cặp (seed, lần rút) có cụm riêng / sở hữu ít nhất một cụm, gộp 315 cặp: crease: mặc định 0.473 / 0.476; K' = 20 0.594 / 0.994; Bayes factor 0.625 / 0.641; Ward 0.317 / 0.330; Bhattacharyya 0.095 / 0.187; chỉ số hạng trung bình 0.000 / 0.057. rolled_pit: mặc định 0 / 0; K' = 20 0.644 / 0.679; Bayes factor 0.010 / 0.010; Ward và Bhattacharyya 0.003; chỉ số hạng trung bình 0.
  - Độ thuần của phép gộp Bayes factor: 0.523 / 0.566 / 0.622. Số cụm silk_spot sở hữu sau gộp: 2.06 / 2.02 / 2.10. Mỗi ảnh chụp chạy 38–42 phân hoạch khác nhau qua bước 10–14.
- Dự đoán:
  - crease còn cụm riêng sau gộp ở 0.615 số cặp có cụm ở K' = 20 (ít nhất một nửa): đúng; rolled_pit 0.015: sai.
  - silk_spot sở hữu đúng 1 cụm ở hơn một nửa số cặp: sai (0 cặp).
  - M1 > 0: đúng. Δ > 0: đúng. Δ ≥ T: đúng (đã đặt khoảng 30%).
- Rủi ro chính: xảy ra với rolled_pit, không với crease. silk_spot giữ khoảng 2 cụm ở mọi FT seed; rolled_pit mất cụm ở mọi tiêu chí gộp (≤ 0.010), nên hiện tượng chung cho mọi phép gộp về k = 5, không riêng Bayes factor.
- Rủi ro thứ hai (PCA-64): rolled_pit mất cụm ở cả ba FT seed, không riêng r1, r2; số cặp crease giữ cụm giảm từ r0 tới r2 (88, 65, 44). Chưa tách được tiêu chí gộp với không gian gộp.

### Đánh đổi

- Mức tăng đến từ r1, r2. Ở r0, Δ_r chỉ 0.008 dù tầng 1 tăng 0.055 (0.501 → 0.556, tính tay): tầng 1 và tầng 2 lại xếp hạng khác nhau.
- Lớp cũ trả giá: Δ_Old trung bình −0.53 điểm; riêng r2 là −0.90, vượt B nếu xét một FT seed (quy tắc xét trung bình). [SUY LUẬN] Lượt S dừng muộn hơn trung bình 38 bước (M3), Gaussian lớp mới nở rộng hơn và lấn sang lớp cũ, đúng cơ chế Hình 2 của paper; chưa kiểm bằng sổ mất điểm.
- Balanced tăng nhưng độ chính xác thường không tăng đều: ở r0, S1_New giảm 2.4 và S1_All giảm 1.2 điểm. [SUY LUẬN] Phần lợi ở crease (lớp hiếm, trọng số lớn trong balanced) được trả bằng lớp đông.
- rolled_pit: ở K' = 20 có cụm riêng 64% số cặp, sau mọi phép gộp còn tối đa 1%. [SUY LUẬN] Gộp về đúng k = 5 trong khi silk_spot giữ 2 mode (camera/cuộn, Q09.13) làm mất chỗ của lớp nhỏ nhất.
- Đổi tiêu chí sang Bayes factor là đúng: Ward ngang Bayes factor ở r0 nhưng thấp hơn ở r1, r2; hai bản Bhattacharyya kém cả mặc định ở tầng 1.

### Kết luận

- CHẤP NHẬN (số laptop, QD27): `09a` + `09b` (K' = 20, Bayes factor NIW) là bước 09 mặc định của dòng PL; E09.08 khởi tạo VB-GMM từ phân hoạch này.
- Giới hạn: Δ_Old sát −B và r2 trả giá lớn; rolled_pit không được cứu; mức tăng dựa vào r1, r2; n = 3 FT seed.
- Mang sang E09.08 [SUY LUẬN]: (1) với k cố định, lớp nhỏ nhất bị gộp mất khi một lớp có nhiều mode; ứng viên: cho phép nhiều thành phần mỗi lớp rồi gộp nhãn, hoặc tiên nghiệm Dirichlet không ép cụm đều; (2) kiểm cơ chế lớp cũ mất điểm (bước dừng của lượt S) bằng sổ mất điểm trước khi thiết kế; (3) rà lại chiến lược trước khi tiền đăng ký E09.08 (chiến lược 2026-11-05 §4).

### Ghi chú (2026-11-14, B13; notebook `lab/09b-overcluster-merge.ipynb`, `cc4212c`; chỉ đọc 30 run của E09.06 và chạy lại phần phân cụm, không đổi phần Kết quả)

- Mức FT seed (§1.3): Δ = 0.0642, SD 0.0526, t = 2.11 với 2 bậc tự do, khoảng tin cậy 95% [−0.067, 0.195], p = 0.169. Quyết định CHẤP NHẬN theo luật đã tiền đăng ký giữ nguyên; với FT seed làm đơn vị thì lợi chưa vững, chờ T2. Độ mịn của thước: một ảnh test rolled_pit = 0.0333 balanced novel (lớn hơn T), crease 0.0182; một ảnh test cũ = 0.415 điểm, B = 1.48 ảnh.
- Lợi đi qua các bước thế nào (§3.1): hiệu hai nhánh ở tầng 1 / cột 10 / cột 14 là 0.056 / 0.052 / 0.008 (r0), 0.055 / 0.126 / 0.112 (r1), 0.123 / 0.120 / 0.072 (r2). Lợi qua được khâu đặt tên; phần mất nằm ở bước 11–14, nơi nhánh `09a` + `09b` mất 0.114 / 0.113 / 0.092 so với 0.070 / 0.099 / 0.045 của nhánh mặc định. [SUY LUẬN] Cụm sau gộp là hợp của nhiều cụm con, một Gaussian mỗi cụm ở lượt F và S khớp kém. Cột 11 của nhánh `09a` + `09b` chưa đo.
- Theo lớp (§3.2): recall ở cột 14 tăng ở inclusion (+0.103), crease (+0.155), rolled_pit (+0.084); silk −0.006, waist −0.015. rolled_pit và crease góp 0.073 trên 0.112 của Δ_r ở r1 và 0.064 trên 0.072 ở r2 (tính tay), tức 65% và 89%, trên 6 và 11 ảnh test. rolled_pit gần như không sở hữu cụm trên pool (`best` 0.000–0.016) nhưng có recall 0.163 (r1), 0.119 (r2). [SUY LUẬN] Phần lợi của rolled_pit đến từ đặt tên trên test (QD15), không từ khám phá.
- Tầng 1 báo trước tầng 2 yếu (§3.3): hệ số góc trong từng ảnh chụp từ hiệu tầng 1 sang hiệu cột 14 là 0.29 / 0.27 / 0.35; trong các cặp tầng 1 tăng, cột 14 giảm ở 19/71, 4/78, 11/95.
- Lớp cũ (§4): nhánh `09a` + `09b` giữ ít hơn 0.33 / 1.33 / 2.16 ảnh test cũ và dừng lượt S muộn hơn 16 / 45 / 53 bước. Trong từng ảnh chụp, trên 613 phân hoạch: −2.5 / −3.6 / −3.3 ảnh cũ trên 100 bước dừng muộn (Spearman −0.70 / −0.87 / −0.68), và balanced novel tăng theo bước dừng (Spearman +0.75 / +0.54 / +0.64). Ở cùng bước dừng, hai nhánh giữ số ảnh cũ như nhau ở r0, r1 (hình §4.3). Ở r2 bước dừng chỉ giải thích −1.06 trên −2.16 ảnh và hai nhánh không có vùng bước dừng chung. Giả thuyết ở mục Đánh đổi (dừng muộn 38 bước) đứng được ở mức tương quan cho r0, r1; chưa có phản thực tế. [SUY LUẬN] Bước dừng là một trục đánh đổi cũ/mới.
- Vết gộp (§5; 315/315 lần rút dựng lại trùng `metrics.json`): cụm con riêng của rolled_pit (203/315 lần rút; 25 ảnh, 12 ảnh rolled_pit) luôn bị gộp, ở bước 10–12 trên 15, vào nhóm do crease làm chủ ở 135/203 lần (tính tay từ bảng đối tác), với log Bayes factor trung vị −85.3, trong khi silk còn làm chủ 3 nhóm. Chủ cuối của nhóm chứa nó: crease 132, waist 48, inclusion 19, rolled_pit 3 (tính tay). log Bayes factor của cặp được chọn đổi dấu giữa bước 8 (+34) và bước 9 (−10); các bước 1–7 gần như chỉ gộp cùng chủ, từ bước 8 chủ yếu gộp khác chủ. inclusion gộp vào silk sớm với log Bayes factor dương (863 bước, trung vị bước 8, +57).
- Bổ sung cho dòng rolled_pit ở mục Đánh đổi: vết gộp khớp suy luận ở đó. rolled_pit mất ở phần gộp bị ép bởi k = 5, sau khi log Bayes factor đã âm, không phải vì luật gộp ưu tiên cụm nhỏ. [SUY LUẬN] log Bayes factor = 0 là điểm dừng tự nhiên, ở khoảng 12 nhóm (tính tay: 20 − 8).
- Lưu ý về các luật báo cáo: `bf` tính trên PCA-64 (`z_pool`), `ward` tính trên đặc trưng thô 768 chiều (`merge_costs`). So sánh Ward với Bayes factor ở mục Đánh đổi vì vậy trộn luật gộp với không gian.
- Bước tiếp (duyệt 2026-11-14): (1) tiền đăng ký thí nghiệm dừng gộp ở log Bayes factor = 0, chấm Hungarian một–một như QD15; (2) phản thực tế bước dừng trên mẫu con các lần rút; (3) viết lại thiết kế E09.08 theo hai kết quả đó.

### Tiền đăng ký lặp lại trên FT seed mới (2026-11-14; viết trước khi có bất kỳ đặc trưng hay run nào của r3, r4)

- Câu hỏi: lợi của `09a` + `09b` có lặp lại trên FT seed chưa từng dùng không? Lý do: với n = 3 FT seed, khoảng tin cậy 95% của Δ chứa 0 (Ghi chú 2026-11-14, §1.3), và Δ_r lệch mạnh theo FT seed.
- Dữ liệu: hai FT seed mới r3, r4 (`FT_SEED` 3, 4), phase 1 bằng đúng lệnh của r0–r2 (`run_phase1_neu.sh --split splits/gc10_split.json --resize full`, 10 epoch); mỗi FT seed 5 chuỗi mặc định (seed 0–4).
- Cách chạy: không đổi code. Mỗi chuỗi: `replay.py run`, `score.py 09` (mốc của cổng C2), `score.py e0604` (mốc `off_lc` của cổng C5), rồi `score.py e0906` với lần rút 0–10 và 11–20. Luật RNG của lần rút, các cổng C1–C5, thước đo và nhánh so sánh giữ nguyên như phần Tiền đăng ký ở trên. T = 0.0273 và B = 0.615 cố định (QD25, sửa 2026-11-13).
- Đại lượng: Δ_r của r3 và r4 (trung bình 5 ảnh chụp × 21 lần rút của hiệu balanced novel ở cột 14); Δ_Old_r tương ứng.
- Quy tắc, chốt trước:
  - LẶP LẠI ĐƯỢC: Δ_r > 0 ở cả r3 và r4, và trung bình hai Δ_r ≥ T;
  - KHÔNG LẶP LẠI: trung bình hai Δ_r ≤ 0;
  - KHÔNG RÕ: các trường hợp còn lại.
- Báo cáo kèm, không dùng để quyết định: Δ gộp trên 5 FT seed với khoảng tin cậy 95% (Student-t, 4 bậc tự do); Δ_Old gộp so với −B; số cặp mà rolled_pit, crease có cụm riêng sau gộp; hiệu ở tầng 1 và cột 10.
- Hệ quả: nếu KHÔNG LẶP LẠI, xét lại việc lấy `09a` + `09b` làm mốc quyết định của dòng PL trước khi tiền đăng ký E09.08. Các trường hợp khác: giữ mốc, ghi khoảng tin cậy mới.
- Dự đoán [SUY LUẬN, xác suất chủ quan khoảng 60%]: LẶP LẠI ĐƯỢC; rolled_pit có cụm riêng sau gộp ở không quá 2/105 cặp mỗi FT seed.
- Giới hạn: `e0906_summary` hiện chỉ đọc r0–r2; bảng của r3, r4 tính từ `metrics.json` sau khi mọi run xong (không sửa code khi loop đang chạy). r3, r4 sau thí nghiệm này không còn là FT seed chưa dùng: T2 phải lấy FT seed mới.

### Kết quả lặp lại (2026-11-14; run `gc10_ft_cls5first_img80pct_r3`, `_r4` và các chuỗi, mốc, run E09.06 của r3, r4, commit run `e4e1113`; số in bằng một lệnh chỉ đọc `metrics.json` của các run đã commit; mọi run checks_ok = 1)

- Theo FT seed (Δ_r | Δ_Old_r điểm | hiệu tầng 1 | hiệu cột 10 | cặp còn cụm riêng sau gộp, rolled_pit / crease, trên 105): r3 0.0328 | −0.782 | 0.1897 | 0.0610 | 6 / 70; r4 0.0535 | +0.937 | 0.1411 | 0.0966 | 0 / 47. Để so: r0 0.0080, r1 0.1122, r2 0.0724 (không đổi so với phần Kết quả).
- Quy tắc đã chốt: Δ_r3 > 0, Δ_r4 > 0, trung bình 0.0432 ≥ T = 0.0273 → LẶP LẠI ĐƯỢC.
- Báo cáo kèm, trên 5 FT seed: Δ = 0.0558, khoảng tin cậy 95% [0.0066, 0.1050], t = 3.15 với 4 bậc tự do; Δ_r > 0 ở 5/5; Δ_Old = −0.287 điểm (B = 0.615).
- Dự đoán: "LẶP LẠI ĐƯỢC" đúng; "rolled_pit có cụm riêng sau gộp ở không quá 2/105 cặp mỗi FT seed" sai ở r3 (6/105).

### Kết luận lặp lại

- GIỮ `09a` + `09b` làm mốc quyết định của dòng PL (số laptop). Không nói "+2T": r3, r4 nay đã dùng, T2 phải lấy FT seed mới (r5 trở đi).
- Giới hạn: khoảng tin cậy gộp có cận dưới 0.007, và r0–r2 là các FT seed đã dùng để chọn phương pháp (ước lượng gộp có thiên lệch chọn lọc); bằng chứng sạch là r3, r4, trong đó r3 chỉ hơn T 0.006. Lớp cũ đổi dấu theo FT seed (r4 +0.94, r3 −0.78). [SUY LUẬN] Khớp cơ chế bước dừng của Ghi chú 2026-11-14 (Q07.04). Ở r3, hiệu tầng 1 (0.19) còn 0.061 ở cột 10 và 0.033 ở cột 14: khâu pool → test lại là chỗ mất lớn (Ghi chú 2026-11-14, §3.3).
- rolled_pit vẫn không được cứu (6/105 và 0/105); Q09.14 giữ nguyên ưu tiên.

## E09.07 — Nhiều lần khởi tạo k-means, giữ inertia thấp nhất (Q09.12, QD24, QD26)

### Tiền đăng ký, phần 1 (2026-11-04; commit: 1ce3233)

- Câu hỏi (Q09.12, phần có nhãn): ở bước 09, thay `n_init` 10 bằng 1000 lần khởi tạo k-means++ và giữ nghiệm inertia thấp nhất có làm tăng balanced novel ở tầng 2 không? Kèm theo: trong cùng một ảnh chụp, phân hoạch có inertia thấp hơn có cho balanced novel cao hơn không?
- Vì sao có thí nghiệm này: E09.06 cũng chạy k-means, nên phải đứng trên k-means đã ổn định, để phần thắng do gộp cụm không lẫn với phần thắng do tối ưu (QD24). Trên CIFAR-100, số lần khởi tạo quyết định độ chính xác nhiều hơn thuật toán phân cụm (D18, E09.01).
- Giả thuyết [suy luận]: nghiệm best-of-10 đổi theo khởi tạo (`04-loader` §3.3); nghiệm best-of-1000 nằm ở cực tiểu sâu hơn, ổn định hơn, và cho nhãn giả tốt hơn. Giả thuyết ngược cũng có cơ sở: do hiệu ứng đồng đều (Xiong, Wu & Chen 2009), mục tiêu SSE ưa chia silk_spot hơn là tách lớp hiếm, nên inertia thấp hơn có thể không giúp, thậm chí làm giảm, balanced novel.
- Đã thấy trước tiền đăng ký này:
  - tầng 1 của 15 nghiệm mặc định (khoá `k5_*` trong `metrics.json` của `gc10_e09_05_v2_*`; QD24) và tầng 2 của 15 chuỗi mặc định (`runs/INDEX.txt`);
  - `04-loader` §3.3: ở r0 s4, cùng trạng thái RNG, chỉ đổi thứ tự hàng: inertia 82926 thành 82757, balanced accuracy lớp mới trên pool 0.32 thành 0.56, crease có cụm riêng (một lần rút);
  - thử trên dữ liệu đồ chơi (QD26): tách `n_init` 10 thành 10 lần khởi tạo đơn; hoán vị hàng cùng phân phối với rút khởi tạo mới.
  - Chưa thấy: landscape của `09-cluster`, mọi số của `09d-ownership`, mọi phân hoạch best-of-1000.
- Dữ liệu: 15 ảnh chụp `gc10_v2_ft_r<r>_s<N>`, r = 0–2, N = 0–4 (chuỗi mặc định, QD21). Bước 04–08 giữ nguyên; chỉ đổi bước 09; bước 10–14 chạy code mặc định.
- Biến thể `v09c_multi_init` (`lab/variants.py`): k-means của bước 09 (`KMeans` của scikit-learn 1.5.2, K = increment, k-means++, Lloyd, các tham số khác mặc định) chạy M lần khởi tạo đơn liên tiếp từ một luồng RNG, giữ nghiệm theo đúng luật của scikit-learn (thay khi inertia nhỏ hơn hẳn và phân hoạch khác); không đọc nhãn; trả `ids`, `cluster` như bước 09 (`_check_09`).
- Thiết kế (lặp phép so trong cùng ảnh chụp, QD26):
  - Mỗi ảnh chụp có 11 lần rút. Lần k = 0: luồng là RNG toàn cục đặt về `rng_in` của bước 09 (đúng phép so một lần chạy của QD25). Lần k = 1–10: `np.random.RandomState(907000 + 1000·r + 100·N + k)`.
  - Mỗi lần rút: 1000 lần khởi tạo đơn liên tiếp. Nhánh mặc định: nghiệm được giữ sau 10 lần đầu (đúng thủ tục `n_init` 10). Nhánh biến thể: nghiệm được giữ sau cả 1000 lần (M = 1000). Hai nhánh dùng chung 10 lần đầu (RNG chung), nên inertia của biến thể không lớn hơn của mặc định.
  - Mỗi phân hoạch khác nhau (so sau khi đánh lại số cụm) chạy tiếp bước 10–14 một lần, trong cùng tiến trình, từ trạng thái của ảnh chụp sau bước 08; phân hoạch trùng dùng lại kết quả. Hợp lệ vì sau bước 09 không bước nào rút RNG toàn cục (`RNG_STEPS` = 04, 05, 09) và SVI dùng `PRNGKey(0)`.
- Đại lượng chính (QD25; cách so theo QD26): balanced novel ở tầng 2, tức trung bình recall của 5 lớp mới trên 214 ảnh test lớp mới, tên cụm như chuỗi mặc định (QD15; Brodersen et al. 2010).
  - D_i = trung bình theo các lần rút của (balanced novel của biến thể − của mặc định) ở ảnh chụp i.
  - Δ_r = trung bình D_i trên 5 seed của FT seed r; Δ = trung bình Δ_r trên 3 FT seed. Đơn vị: FT seed (n = 3).
  - Ràng buộc lớp cũ: Δ_Old, tính cùng cách với S1_Old (241 ảnh test lớp cũ, cuối phiên S1).
  - Sai số Monte Carlo: SE_MC = √(Σ_i s_i²/K) / 15, tổng trên 15 ảnh chụp; s_i là độ lệch chuẩn (ddof = 1) của hiệu giữa các lần rút ở ảnh chụp i, K là số lần rút.
- Ngưỡng (công thức chốt ở đây; số điền ở phần 2, đo ở 14 trên 15 chuỗi mặc định, trước mọi run của E09.07):
  - T = trung bình theo 3 FT seed của độ lệch chuẩn (ddof = 1) của balanced novel trên 5 seed của chuỗi mặc định. Lý do: cải thiện nhỏ hơn dao động của một lần chạy thì người chạy pipeline một lần không nhận ra.
  - B = như T, với S1_Old.
- Quy tắc quyết định:
  - Chấp nhận: Δ ≥ T, Δ_r > 0 ở ít nhất 2/3 FT seed, và Δ_Old ≥ −B → `v09c_multi_init` (M = 1000) thành bước 09 mặc định của dòng PL; E09.06 chạy trên nó.
  - Bác: Δ ≤ 0, hoặc Δ_Old < −B → giữ `n_init` 10; E09.06 chạy trên k-means mặc định.
  - Không đủ lớn: các trường hợp còn lại (0 < Δ < T, hoặc Δ ≥ T nhưng Δ_r > 0 ở dưới 2/3 FT seed) → giữ `n_init` 10; báo cáo.
  - Nếu Δ hoặc Δ_Old cách một ranh giới quyết định (T, 0, −B) dưới 2·SE_MC: thêm 10 lần rút (k = 11–20, cùng công thức seed) cho cả 15 ảnh chụp, một lần duy nhất, rồi quyết theo 21 lần rút.
- Điều kiện bỏ ý tưởng: bác hoặc không đủ lớn → bỏ hướng "thêm khởi tạo" ở bước 09; câu "inertia thấp hơn có tốt hơn không" trả lời bằng tương quan ở báo cáo phụ.
- Báo cáo phụ (không dùng để quyết định): kết quả riêng của k = 0; S1_New, S1_All, M_f, balanced accuracy 10 lớp; recall từng lớp mới kèm khoảng Wilson (Wilson 1927); tầng 1 (`score09`: `bal_acc_hungarian`, số cụm sở hữu) của nghiệm được giữ sau 10, 100 và 1000 lần khởi tạo trên cùng các lần rút; inertia; số ảnh chụp và số lần rút mà biến thể đổi phân hoạch; tương quan hạng Spearman giữa inertia và balanced novel trên các phân hoạch đã chạy 10–14 trong từng ảnh chụp, trung bình theo FT seed.
- Phép kiểm toàn vẹn; chỉ đọc kết quả khi cả ba đạt:
  - C1: ở k = 0, phân hoạch của nhánh mặc định trùng `ids` của ảnh chụp, và các số ở bước 14 trùng chuỗi mặc định (kết quả rời rạc trùng hẳn; số thực lệch tương đối ≤ 1e-6), ở 15/15 ảnh chụp.
  - C2: ở mọi lần rút, inertia của biến thể ≤ inertia của mặc định.
  - C3: mọi phân hoạch qua `_check_09` (mỗi ảnh pool và ảnh test một id trong [0, increment)).
- Nếu C1–C3 không đạt: không đọc kết quả, sửa lỗi rồi chạy lại một lần, ghi lý do.
- Dự đoán [suy luận, xác suất chủ quan]: ở lần k = 0, biến thể đổi phân hoạch ở ít nhất 8/15 ảnh chụp (khoảng 70%); Δ > 0 (khoảng 55%); Δ ≥ T (khoảng 30%); tương quan Spearman âm (inertia thấp đi với balanced novel cao) ở ít nhất 2/3 FT seed (khoảng 50%).
- Phần 2 (sau 14, trước mọi run của E09.07): điền T và B; chốt đại lượng cơ chế từ sổ mất điểm (định nghĩa ở 10); không đổi gì khác của phần 1.
- Ngoài phạm vi: phương pháp phân cụm khác (Q09.06); gộp cụm (E09.06); đổi K (Q09.02); M khác 1000 (chỉ ở báo cáo phụ tầng 1).
- Chi phí ước lượng [suy luận]: 165.000 lần khởi tạo k-means; tối đa 330 lần chạy bước 10–14 (cả chuỗi 04–14 mất 2.3–2.5 phút theo `runs/INDEX.txt`); dưới một ngày máy.
- Cách chạy: 15 run qua `exp.py`, mỗi run một ảnh chụp, ghi mọi lần rút vào `metrics.json`; hàm tầng 2 trong `lab/score.py`, biến thể trong `lab/variants.py`; code viết sau 14 và commit trước khi chạy.
- Tag: `gc10_e09_07_r<r>_s<N>`, r = 0–2, N = 0–4.

### Tiền đăng ký, phần 2 (2026-11-05; commit: 8c92521)

- Đã thấy thêm, sau phần 1 và trước mọi run của E09.07: `14-eval` (`58a698e`), mọi số của 15 chuỗi mặc định ở bước 10–14 (balanced theo bước, S1_Old, lỗi theo đích, cách đặt tên khi chấm). Chưa thấy: mọi phân hoạch best-of-1000.
- Sửa đổi phần 1 (một chỗ; phần 1 ở trên giữ nguyên): T và B tính bằng SD gộp thay vì trung bình các SD theo FT seed.
  - Công thức: s_p = √(Σ_r (n_r − 1) s_r² / Σ_r (n_r − 1)), với n_r = 5 và s_r là SD (ddof = 1) trong FT seed r. Đây là căn của trung bình bình phương trong nhóm của ANOVA một chiều, và là mẫu số của Cohen's d (Cohen 1988).
  - Lý do: phần 1 lấy T là "dao động của một lần chạy". Với FT seed chọn ngẫu nhiên, độ lệch của một lần chạy so với trung bình FT seed của nó có phương sai bằng trung bình các σ_r², nên SD của nó là s_p. Trung bình các SD nhỏ hơn hoặc bằng s_p và không phải ước lượng chuẩn.
  - Ý nghĩa: Δ ≥ T tương đương Cohen's d ≥ 1. Với hai lần chạy độc lập, biến thể thắng mặc định với xác suất Φ(1/√2) ≈ 0.76 (giả định chuẩn; McGraw & Wong 1992).
  - Thời điểm và hướng: sửa sau khi thấy SD của chuỗi mặc định (dữ liệu phần 1 hẹn dùng), trước mọi run. T chặt hơn (0.0244 → 0.0273); B lỏng hơn 0.04 ảnh (0.599 → 0.615).
  - Báo cáo kèm quyết định theo công thức phần 1 (Nosek et al. 2018).
- Số (`14-eval` §3.1; balanced novel tính bằng `score.ledger_state`, tên QD15):
  - T = 0.0273, cùng đơn vị tỉ lệ với Δ. SD theo FT seed: 0.0341 / 0.0071 / 0.0320. Tương đương trung bình 0.82 ảnh rolled_pit hoặc 1.5 ảnh crease mỗi ảnh chụp.
  - B = 0.615 điểm %, cùng đơn vị với Δ_Old, tức 1.48 ảnh test cũ. SD theo FT seed: 0.415 / 0.629 / 0.754 (1.00 / 1.52 / 1.82 ảnh).
  - Độ bất định của T (chỉ báo cáo): khoảng tin cậy 95% của σ là 0.0196–0.0451 (χ², 12 bậc tự do; gần đúng, vì r0 có hai đỉnh). Nếu Δ nằm trong khoảng này, kết luận ghi là nhạy với ngưỡng; quyết định vẫn theo T = 0.0273.
  - Không đặt ngưỡng riêng cho từng FT seed, dù r1 dao động ít hơn hẳn (Bartlett χ² = 6.96, p = 0.03). Mỗi SD chỉ có 4 bậc tự do; khoảng tin cậy của σ ở r1 là 0.004–0.020.
- Đại lượng cơ chế (chỉ báo cáo; hiệu theo cặp, trung bình theo lần rút, rồi theo seed, rồi theo FT seed, như Δ):
  - M1: Δ balanced ở cột 10 của sổ (mức cụm, tên QD15).
  - M2: Δ balanced ở cột 14 với tên QD14.
    - Cách tính: Hungarian cân theo số ảnh trên chính `slot_14`, chấm trên cùng tập, như ACC của GCD (Vaze et al. 2022).
    - Hạn chế: ở chuỗi mặc định, chấm trên cùng tập thổi phồng 0.7 / 4.7 / 1.0 điểm (`14-eval` §4.2); một phần triệt tiêu trong hiệu theo cặp.
    - Quy tắc đọc: nếu Δ ≥ T mà M2 ≤ 0, hoặc Δ ≤ 0 mà M2 ≥ T, kết luận ghi rằng phần chênh nằm ở cách đặt tên khi chấm. Quyết định không đổi.
  - M3: Δ bước dừng của lượt S, lấy từ dòng log `early_stop_flag` của lượt S (như `S1_estop_S`; không lấy từ sổ). M3 đo kênh ngân sách bước (`13` §4.1).
- Phép kiểm toàn vẹn thêm, C4: ở mọi phân hoạch đã chạy 10–14, gained 08→14 = 0, và số ảnh đúng của S1_Old = số đúng ở 08 − lost 08→14 (`14-eval` §2.1, §3.1).
- Dự đoán thêm [suy luận, xác suất chủ quan]:
  - M1 cùng dấu với Δ ở ít nhất 2/3 FT seed (khoảng 65%);
  - |M2 − Δ| ≥ 0.01 ở r1 (khoảng 50%);
  - tương quan Spearman giữa Δ bước dừng và Δ_Old trên 15 ảnh chụp là âm (khoảng 65%).
  - Dự đoán của phần 1 giữ nguyên.
- Điểm yếu đã biết, không sửa:
  - Δ_r > 0 ở ít nhất 2/3 FT seed đạt với xác suất 0.5 khi không có hiệu ứng;
  - chỉ có 3 FT seed, nên khả năng khái quát hạn chế;
  - T là ngưỡng thực tiễn (d ≥ 1), không phải phép kiểm thống kê;
  - SE_MC chỉ đo nhiễu của các lần rút.
- Không đổi: mọi thứ khác của phần 1.

### Ghi chú (2026-11-04, sau tiền đăng ký; không đổi phần tiền đăng ký)

- Dòng "thử trên dữ liệu đồ chơi (QD26): ... hoán vị hàng cùng phân phối với rút khởi tạo mới" ghi đúng kết quả thử lúc đó, nhưng trên dữ liệu thật phép so này trượt khi trạng thái RNG cố định (`09-cluster` §3.4b, `8982942`; sửa ở QD26). Thiết kế E09.07 không dùng hoán vị hàng (k = 0 dùng `rng_in`, k = 1–10 dùng luồng RNG mới), nên không đổi.
- Đã thấy sau tiền đăng ký, trước mọi run (không dùng nhãn): landscape của `09-cluster` (`8982942`); ví dụ nghiệm pipeline ở nhóm sâu nhất ở 7/15 ảnh chụp, best-of-10 rơi vào nhóm đó với xác suất 0.11–0.70. Tiền đăng ký chốt trước khi có các số này.
- Đã thấy sau tiền đăng ký, trước mọi run (có nhãn, `09d-ownership`, `abbc3af`): oracle loại (c) ở 15/15 ảnh chụp; trên cùng pool r0, cấu hình crease có inertia cao hơn 0.23–0.35% nhưng balanced accuracy tầng 1 cao hơn (0.56–0.57 so với 0.32–0.33). Kế hoạch của `09d` chốt ở `d282b37`, sau tiền đăng ký E09.07 (`1ce3233`).
- 2026-11-05: T và B của phần 2 được tính chính thức bằng code ở `14-eval` §3.3 (commit 3efbca2), từ sổ mất điểm của 15 chuỗi mặc định: T = 0.02728 (SD theo FT seed 0.03406 / 0.00711 / 0.03198), B = 0.61545 điểm (0.41494 / 0.62928 / 0.75377). Làm tròn ra đúng số đã ghi ở phần 2 (0.0273, 0.615), nên quyết định dùng hai số đó. Công thức phần 1 cho 0.02438 và 0.59933. Khoảng tin cậy 95% của σ: T 0.0196–0.0450, B 0.441–1.016. Bartlett: T χ² = 6.94, p = 0.031; B χ² = 1.22, p = 0.544. Phần 2 ghi khoảng của T là 0.0196–0.0451 và Bartlett χ² = 6.96, p = 0.03, tính tay từ các SD đã làm tròn; số đúng là số ở đây.

### Kết quả (2026-11-11; run `gc10_e09_07_r0_s0`–`gc10_e09_07_r2_s4`, code `aab1178`; số in từ `python lab/score.py e0907-summary`)

- Toàn vẹn: C1–C4 đạt ở 15/15 run (`checks_ok` = 1; `e0907_summary` không đọc kết quả khi thiếu). T và B tính lại từ khoá `ref_*`: 0.02728 và 0.61545, làm tròn đúng số đã tiền đăng ký.
- Quyết định (11 lần rút mỗi ảnh chụp): Δ = 0.01480 (T = 0.0273); Δ_r = 0.0243 / 0.0015 / 0.0186, dương ở 3/3 FT seed; Δ_Old = −0.0201 điểm (B = 0.615); SE_MC = 0.00200 (Δ) và 0.0295 (Δ_Old). Không ranh giới nào nằm trong 2·SE_MC, nên không thêm lần rút.
  - Theo ngưỡng phần 2: KHÔNG ĐỦ LỚN (0 < Δ < T). Theo công thức phần 1 (0.0244, 0.599): cũng không đủ lớn.
  - Δ nằm dưới cận dưới khoảng tin cậy của σ (0.0196), nên kết luận không nhạy với ngưỡng.
- Đại lượng cơ chế (trung bình, rồi theo FT seed r0 / r1 / r2): M1 = 0.01371 (0.0340 / −0.0042 / 0.0114); M2 = 0.01234 (0.0203 / 0.0031 / 0.0136); M3 = +7.32 bước (+26.4 / −2.3 / −2.1).
- Báo cáo phụ:
  - riêng k = 0: Δ = 0.01637, Δ_Old = −0.0553;
  - biến thể đổi phân hoạch ở 161/165 lần rút;
  - theo FT seed: S1_New +0.80 / +0.11 / +3.25 điểm; S1_All +0.26 / +0.09 / +1.58 điểm; balanced accuracy 10 lớp +0.0110 / +0.0010 / +0.0098; Δ_Old −0.226 / +0.068 / +0.098 điểm;
  - tầng 1 (`bal_acc_hungarian` trên pool) của nghiệm giữ sau 10 / 100 / 1000 lần khởi tạo: r0 0.5006 / 0.5674 / 0.5717; r1 0.4688 / 0.4590 / 0.4591; r2 0.3895 / 0.3816 / 0.3810;
  - Spearman giữa inertia và balanced novel trên các phân hoạch đã chạy: trung bình theo FT seed −0.534 / −0.283 / −0.386; âm ở 14/15 ảnh chụp (trừ r1 s2, +0.48);
  - chưa tổng hợp (có trong `metrics.json` của 15 run): recall từng lớp mới kèm khoảng Wilson, số cụm sở hữu, M_f.
- Dự đoán:
  - ở k = 0 biến thể đổi phân hoạch ở ít nhất 8/15 ảnh chụp: đúng (hiệu của k = 0 khác 0 ở 14/15 ảnh chụp; đếm tay từ bảng);
  - Δ > 0: đúng. Δ ≥ T: sai;
  - Spearman âm ở ít nhất 2/3 FT seed: đúng (3/3);
  - M1 cùng dấu với Δ ở ít nhất 2/3 FT seed: đúng (r0, r2; r1 trái dấu);
  - |M2 − Δ| ≥ 0.01 ở r1: sai (0.0016);
  - Spearman giữa Δ bước dừng và Δ_Old trên 15 ảnh chụp là âm: đúng (−0.69; tính tay từ bảng in 4 chữ số).

### Đánh đổi

- r0: tầng 1 tăng 7.1 điểm (0.5006 lên 0.5717) nhưng balanced novel ở tầng 2 chỉ tăng 2.4 điểm. Cùng lúc, lượt S dừng muộn hơn 26 bước và S1_Old giảm 0.23 điểm: phần thắng ở lớp mới đi kèm phần mất ở lớp cũ, vẫn trong B.
- r1, r2: tầng 1 giảm nhẹ (−0.010, −0.009) trong khi tầng 2 không giảm (r2 tăng 0.019). Tầng 1 và tầng 2 xếp hạng khác nhau, như ở E09.05.
- Trong từng ảnh chụp, inertia thấp hơn đi với balanced novel cao hơn, nhưng hiệu trung bình chỉ khoảng 0.54 T (tính tay) và gần 0 ở r1.

### Kết luận

- Không đủ lớn theo quy tắc: giữ `n_init` 10; E09.06 chạy trên k-means mặc định; bỏ hướng "thêm khởi tạo" ở bước 09 (điều kiện bỏ ý tưởng).
- Câu hỏi kèm theo của Q09.12: trong cùng ảnh chụp, phân hoạch có inertia thấp hơn thường cho balanced novel cao hơn (Spearman âm ở 14/15 ảnh chụp), nhưng tối ưu sâu hơn không vượt được dao động của một lần chạy.
- Hạ tầng nhiều lần rút (`multi_init`, `tier2_e0907`) dùng lại được cho E09.06.

## E09.08 — CUB t5: VB-GMM cho phiên không nhãn, sửa phân hoạch trong nhóm mới và trách nhiệm hai phía với thành phần cũ (QD27, QD28)

### Tiền đăng ký (2026-11-18; viết trước khi có code và trước mọi số của thí nghiệm)

- Câu hỏi: trên CUB t5, thay bước 11–13 bằng VB-GMM (Bishop 2006 §10.2; Corduneanu & Bishop 2001) với thành phần cũ cố định, mọi khe dùng hậu nghiệm NIW, trách nhiệm mềm hai phía giữa thành phần cũ và thành phần mới, có tăng M_d so với pipeline mà giữ S5_S0test trong biên B không? Phụ: phần lợi đến từ sửa phân hoạch trong nhóm mới, từ thước đo làm trắng theo hiệp phương sai trong lớp, hay từ trách nhiệm mềm?
- Vì sao (Kết luận E14.07, bộ không rò):
  - Vế M_d: Nlab hơn Nc 25.216 điểm New, còn Nc ngang N (60.932 so với 61.033): phần còn lấy lại được nằm ở phân hoạch trong nhóm mới.
  - Vế S5_S0test: N mất 9.296 điểm so với pipeline, Nc chỉ mất 0.198: ảnh cũ lẫn vào tập khớp khe mới là nguồn mất của lớp cũ.
  - Ý (c) của Kết luận E12.02 (cạnh tranh lúc dự đoán) bỏ khỏi mục tiêu: L_comp_Nlab = 8.807 > T; L_comp chỉ báo cáo.
  - Bộ lọc cứng dựa trên điểm một phía bỏ đúng các ảnh mới sát ranh giới cũ/mới: NF-count thua Nc 6.339 điểm New và có L_comp 12.339 (E12.02). Trách nhiệm ở đây là tỉ số hai phía π_k p(x | k) chuẩn hoá trên cả thành phần cũ lẫn mới, không là một hàm của điểm giống lớp cũ.
- Đã tra trước khi viết: D11, D12, D14, D16, D23, D24, D25 (`notes/paper-code-discrepancies.md`); Kết quả và Kết luận của E06.04, E09.06, E11.01, E12.02, E14.01, E14.02 (cả Sửa đổi), E14.03, E14.05 (cả Sửa), E14.06, E14.07; QD27, QD28 (cả Sửa 2026-11-18); `lab/niw.py`; `_post_1406`, `_score_1406`, `_logd_1202`, `_best_map_1401`, `_rules_1401`, `tier2_e1407` của `lab/score.py`.
- Đã thấy trước tiền đăng ký này: mọi số của E14.01–E14.07 và E12.02 trên `cub_chain_ft_l25` và `cub_chain_ft` (r0–r2). Chưa thấy: mọi số của V, V-new, KW, V-hard, V-πfix trên dữ liệu thật.
- Loại: thí nghiệm cải tiến có luật quyết định (dò hướng trên r0–r2) và luật xác nhận (trên FT seed mới r3, r4), chốt cùng lúc.
- Dữ liệu: 15 chuỗi `cub_chain_ft_l25_r<r>_s<N>` (bộ quyết định, QD28 Sửa 2026-11-16) và 15 chuỗi `cub_chain_ft_r<r>_s<N>` (báo kèm); xác nhận trên 10 chuỗi `cub_chain_ft_l25_r<r>_s<N>`, r = 3, 4, N = 0–4, tạo sau tiền đăng ký này bằng đúng lệnh phase 1 và lệnh chuỗi của r0–r2 (`FT_SEED` 3, 4). Không gian PCA của S0; `ids` (bước 09), `keep` (bước 12), `pi_test`, `lut` lấy từ ảnh chụp. Bước 09 giữ nguyên (k-means mặc định, QD28 Sửa 2026-11-18); các nhánh thay bước 11–13 và cách dự đoán ở bước 14. Phân hoạch ở bước 09 của mọi phiên không phụ thuộc biến thể ở bước 11–14 (QD28, RNG chung; C2 của 3b-1), nên runner chỉ đọc ảnh chụp, như E14.06 và E14.07. Nhãn pool chỉ dùng cho nhánh oracle và để chấm; nhãn test chỉ để chấm.
- Mô hình V:
  - Tiên nghiệm, một cho mọi khe, như N của E14.06: m0 = mean của `z_s0`, κ0 = `niw.K0` (0.01), ν0 = d + 1 + off, Ψ0 = off × P, P = `niw.prior_cov` (hiệp phương sai trong lớp gộp của S0), off = `niw.choose(niw.loo_curve(z_s0, s0_y))`, chọn một lần mỗi chuỗi.
  - Thành phần cũ, cố định trong phiên (không cập nhật từ pool, vì D16): khe 0…99 là hậu nghiệm NIW từ S0 có nhãn (A3 của E06.04); khe của các đợt đã khám phá là hậu nghiệm của chính nhánh ở phiên của đợt đó, đóng băng.
  - Thành phần mới k = 1…20 của phiên: q(μ_k, Σ_k) dạng NIW, cùng tiên nghiệm.
  - Hệ số trộn π ~ Dirichlet(α0 · 1) trên mọi thành phần của phiên (cũ và mới), α0 = 1, chốt trước.
  - q(Z) q(π) ∏ q(μ_k, Σ_k), cập nhật CAVI (Bishop 2006 Eq. 10.46–10.67, viết theo tham số NIW): bước E, ln ρ_ik = E[ln π_k] + ½ E[ln |Σ_k⁻¹|] − d / (2 κ_k) − (ν_k / 2) (x_i − m_k)ᵀ Ψ_k⁻¹ (x_i − m_k) + hằng, với E[ln |Σ_k⁻¹|] = Σ_{j=1..d} ψ((ν_k + 1 − j)/2) + d ln 2 − ln |Ψ_k| và E[ln π_k] = ψ(α_k) − ψ(Σ_j α_j); r_ik = ρ_ik / Σ_j ρ_ij. Bước M, chỉ cho thành phần mới: N_k = Σ_i r_ik, x̄_k và tán xạ S_k có trọng số r_ik, κ_k = κ0 + N_k, ν_k = ν0 + N_k, m_k = (κ0 m0 + N_k x̄_k)/κ_k, Ψ_k = Ψ0 + S_k + (κ0 N_k / κ_k)(x̄_k − m0)(x̄_k − m0)ᵀ; α_k = α0 + N_k cho mọi thành phần.
  - Khởi tạo: r_ik = 1 ở thành phần mới mang id k-means của ảnh i, 0 ở mọi thành phần khác; bước M đầu tiên, rồi lặp E, M.
  - Dừng: thay đổi tương đối của ELBO dưới 1e-8, hoặc 200 vòng; ghi số vòng.
  - Dự đoán (bước 14): argmax log-mật độ dự đoán Student-t của hậu nghiệm NIW (như N) trên các khe đã gặp, prior lớp đều; π không dùng lúc dự đoán.
  - Sang phiên sau: hậu nghiệm của 20 thành phần mới đóng băng, thành thành phần cũ.
- Nhánh, cùng chuỗi, cùng ảnh chụp:
  - PIPE: pipeline (`acc1`).
  - N, Nc, Nlab: như E14.07, tính lại.
  - KW: k-means cứng (`KMeans`, K = 20, `n_init` 10, k-means++) trên mọi ảnh pool trong không gian PCA làm trắng theo P, w = L⁻¹ (z − m0) với P = L Lᵀ; RNG `np.random.RandomState(908000 + 1000 r + 100 N + i)` ở phiên i; rồi khớp N trên các ảnh có `keep` của pipeline theo id KW. Tách phần lợi của thước đo.
  - V-new: CAVI chỉ trên 20 thành phần mới, trên các ảnh có `keep` của pipeline (cố định), khởi tạo từ id k-means; π trên 20 thành phần. Tách phần lợi của sửa phân hoạch trong nhóm mới.
  - V: như mục Mô hình V, trên mọi ảnh pool, không dùng `keep`. Nhánh chính.
  - V-hard: như V, bước E lấy argmax (r_ik ∈ {0, 1}). Tách mềm so với cứng.
  - V-πfix: như V, E[ln π_k] cố định bằng nhau. Tách vai trò của Dirichlet.
  - Khe của các đợt trước ở mỗi nhánh là khe của chính nhánh đó.
- Đặt tên khi chấm: P1 (E14.01) cho mọi nhánh, kể cả PIPE: ở phiên i, khối 20 khe của phiên ghép với 20 lớp mới bằng Hungarian trên dự đoán của bộ phân lớp của nhánh (ô lớp mới thật × khe mới), bảng tên của các phiên trước đóng băng; khe 0…99 giữ nghĩa. P0 báo kèm cho PIPE, N, Nc.
- Đại lượng chính: ΔM_d = M_d(V) − M_d(PIPE), cả hai chấm bằng P1; hiệu trong từng chuỗi, trung bình 5 chuỗi mỗi FT seed, rồi trung bình 3 FT seed.
- Ràng buộc (QD28): ΔS5_S0test = S5_S0test(V) − S5_S0test(PIPE) ≥ −B (0.376). S5_S0test không phụ thuộc cách đặt tên (khe 0…99 giữ nghĩa).
- Quy tắc quyết định (dò hướng, bộ không rò):
  - Chấp nhận: ΔM_d ≥ T (0.935), ΔM_d > 0 ở 3/3 FT seed, và ràng buộc đạt → chạy luật xác nhận.
  - Bác: ΔM_d ≤ 0, hoặc ràng buộc trượt.
  - Không đủ lớn: các trường hợp còn lại.
- Luật xác nhận, chốt cùng lúc, trên r3, r4 (bộ không rò, 5 chuỗi mỗi FT seed): XÁC NHẬN nếu ΔM_d > 0 ở cả r3 và r4, trung bình hai FT seed ≥ T, và trung bình ΔS5_S0test ≥ −B; KHÔNG XÁC NHẬN nếu trung bình ΔM_d ≤ 0 hoặc trung bình ΔS5_S0test < −B; KHÔNG RÕ trong các trường hợp còn lại. Chỉ chạy khi dò hướng ra Chấp nhận; chỉ khi XÁC NHẬN mới báo V như cải tiến (số laptop, QD27: server chạy lại).
- Luật đọc phụ, chẩn đoán, chốt trước (bộ không rò, New dưới P1):
  - New(V-new) − New(N) ≥ T: E-M sửa được phân hoạch trong nhóm mới.
  - New(KW) − New(N) ≥ T: thước đo làm trắng một mình đã đủ; phần lợi của trách nhiệm mềm đọc bằng V − KW.
  - |M_d(V) − M_d(V-hard)| < T/2 và |S5_S0test(V) − S5_S0test(V-hard)| < B: trách nhiệm mềm không thêm gì so với gán cứng.
- Điều kiện bỏ cơ chế: dò hướng ra Bác, và New(V-new) − New(N) < T, và New(KW) − New(N) < T → bỏ hướng sửa phân hoạch bằng mô hình Gaussian trên biểu diễn hiện tại; ghi rằng khoảng Nc → Nlab cần biểu diễn khác hoặc nhãn.
- Báo kèm (không dùng để quyết định): New, Old, All, S5_S0test, g20, L_comp của mọi nhánh; ρ = (Old − Old_N)/(Old_Nc − Old_N); tỉ lệ phần trần lấy được (New − New_N)/(New_Nlab − New_N); c09 của phân hoạch của KW, V-new, V trên pool (nhãn pool chỉ để chấm; với V và V-new là argmax trách nhiệm trên thành phần mới); khối trách nhiệm của ảnh cũ nằm trong thành phần mới; N_k nhỏ nhất và số thành phần có N_k < 2; số vòng CAVI; P0 của PIPE, N, Nc.
- Cổng kiểm:
  - G1: cột PIPE trùng `acc1` của `S<i>_14` (1e-6); N, Nc, Nlab (New, Old, g20, S0test, dưới P0) trùng run E14.07 cùng bộ (1e-4).
  - G2: đường tính của V-new với trách nhiệm khoá ở id k-means (không bước E) cho lại N; đường tính của V với trách nhiệm khoá bằng tập của Nc cho lại Nc (1e-6).
  - G3: ELBO không giảm qua các vòng của V, V-new, V-πfix (sai số 1e-6 tương đối).
  - G4: cập nhật NIW với trọng số 0/1 trùng `_post_1406` (1e-8 tương đối); log-mật độ của thành phần cũ lấy từ cache trùng tính trực tiếp.
  - G5: M_d dưới P1 của PIPE trùng E14.01 (`Md_P1` của run `cub_e14_01_*` cùng chuỗi, 1e-4).
  - Runner chỉ in trạng thái cổng và thời gian, không in số của từng chuỗi. Trước run thật: thử trên chuỗi giả ngoài repo, có đối chứng âm cho từng cổng.
- Code (viết sau tiền đăng ký): runner `tier2_e0908` trong `lab/score.py`, lệnh `e0908` qua `exp.py` (`--prefix`, `--checks-only`), một run mỗi bộ (tag `cub_e09_08_ft_l25`, `cub_e09_08_ft`; xác nhận: `cub_e09_08x_ft_l25`); lệnh `e0908-summary` chỉ đọc. Dùng lại `_post_1406`, `_score_1406`, `_logd_1202`, `_best_map_1401`; không sửa `lab/niw.py`, `lab/sessions.py`, `lab/replay.py`. Runner xác nhận đọc chuỗi r3, r4 bằng cùng hàm với tham số danh sách FT seed.
- Chi phí: đo bằng một lần checks-only có profile trên mỗi bộ đặc trưng trước run thật, không ước lượng tay.
- Dự đoán của Claude [SUY LUẬN], độ tin thấp: New(V-new) − New(N) từ +2 đến +8; New(KW) − New(N) từ −2 đến +6; ΔM_d(V − PIPE) từ +3 đến +9; ΔS5_S0test(V − PIPE) từ −4 đến 0; dò hướng ra Chấp nhận với xác suất khoảng 30%, rủi ro chính là ràng buộc lớp cũ.
- Giới hạn:
  - r0–r2 đã dùng ở E14.01–E14.07 và E12.02: phần dò hướng không phải kiểm định xác nhận; xác nhận dựa vào r3, r4 (n = 2 FT seed).
  - Thành phần cũ cố định: không xử lý việc cùng ảnh cũ lặp lại giữa các phiên (D16).
  - K = 20 biết trước (giao thức); không ước lượng số lớp.
  - Một tiên nghiệm cho mọi khe; off chọn trên S0 (250 ở 30/30 chuỗi, E14.07); với off lớn, Σ của khe mới gần P, nên V gần một k-means mềm theo Mahalanobis với Σ chung.
  - CAVI chỉ đạt cực trị địa phương quanh khởi tạo k-means.
  - Kết luận "giới hạn của biểu diễn" về L_comp (E14.07) chỉ đúng dưới NIW với off = 250.
  - GC10 không thuộc thí nghiệm này (QD28 Sửa 2026-11-18).
- Tag: `cub_e09_08_ft_l25`, `cub_e09_08_ft`; xác nhận `cub_e09_08x_ft_l25`.

### Sửa đổi (2026-11-19; viết sau khi có run `cub_e09_08_ft_l25`, `cub_e09_08_ft` và hai mốc C1 của r3, r4, trước khi chạy `e0908-summary`; phần tiền đăng ký ở trên, `c17e53c`, giữ nguyên)

- Trạng thái lúc viết: hai run E09.08 qua G1–G5 ở 15/15 chuỗi mỗi bộ; chuỗi r3, r4 qua C1 48/48 ở seed 0 so với mốc `main.py` tất định (`cub_km_evhung_det_ft_l25_r3_s0`, `_r4_s0`). Claude chưa thấy số nào của PIPE, N, Nc, Nlab, KW, V-new, V, V-hard, V-πfix từ hai run này: các lệnh đã chạy chỉ in trạng thái cổng và thời gian; dòng `result:` mà `exp.py` tự ghi vào LOG không được in trong lệnh nào.
- Lý do: luật xác nhận chạy trên r3, r4, nhưng cổng G1 (phần N, Nc, Nlab) và G5 so với run E14.07 và E14.01, mà hai run này chỉ có trên r0–r2. Cần chốt cổng của run xác nhận trước khi biết dò hướng ra gì, để không chọn cổng theo kết quả.
- Không đổi: câu hỏi, mô hình V, các nhánh, cách đặt tên P1, đại lượng chính, ràng buộc, T = 0.935, B = 0.376, quy tắc quyết định, luật xác nhận, luật đọc phụ, điều kiện bỏ cơ chế. Run xác nhận chỉ chạy khi `e0908-summary` ra Chấp nhận trên `cub_e09_08_ft_l25`; ra Bác hoặc Không đủ lớn thì Sửa đổi này không dùng và r3, r4 không được chấm bằng các nhánh của E09.08.
- Run xác nhận: tag `cub_e09_08x_ft_l25`; 10 chuỗi `cub_chain_ft_l25_r<r>_s<N>`, r = 3, 4, N = 0–4; cùng hàm `tier2_e0908` với tham số danh sách FT seed (mặc định 0–2, giữ nguyên run dò hướng), lệnh `e0908 --prefix cub_chain_ft_l25 --fts 3,4`; RNG của KW theo công thức đã chốt (`908000 + 1000 r + 100 N + i`); off chọn lại cho từng chuỗi như ở dò hướng.
- Cổng của run xác nhận:
  - G1 phần PIPE trùng `acc1` (1e-6): giữ.
  - G1 phần N, Nc, Nlab trùng E14.07, và G5 (M_d của PIPE dưới P1 trùng E14.01): bỏ, vì không có run tham chiếu. Hai cổng này kiểm code của nhánh N, Nc, Nlab và của P1 so với code đã kiểm; code đó đã qua ở 30/30 chuỗi r0–r2 (hai bộ) và không đổi ở run xác nhận (cùng commit runner). Phần mất đi là kiểm đầu vào của r3, r4; đầu vào đã qua kiểm lệnh phase 1 và lệnh chuỗi (commit `a1b4193`) và C1 ở seed 0.
  - G2, G3, G4: giữ nguyên.
  - Run xác nhận chỉ được đọc khi mọi cổng giữ lại đều đạt ở 10/10 chuỗi.
- `e0908-summary` đọc thêm `cub_e09_08x_ft_l25` khi có: cột r3, r4 (trung bình 5 chuỗi mỗi FT seed) và trung bình hai FT seed của ΔM_d(V − PIPE) và ΔS5_S0test(V − PIPE), số FT seed có ΔM_d > 0, kết luận XÁC NHẬN, KHÔNG XÁC NHẬN hoặc KHÔNG RÕ theo luật đã chốt; báo kèm như dò hướng. Không gộp r3, r4 với r0–r2.
- Code (viết sau Sửa đổi này, chỉ khi dò hướng ra Chấp nhận): thêm tham số FT seed vào `tier2_e0908` và nhánh đọc run xác nhận vào `e0908_summary`; phép thử trên chuỗi giả như runner gốc, có đối chứng âm cho từng cổng giữ lại; checks-only trên `cub_chain_ft_l25_r3_s0` trước run thật.
- Chi phí [tính tay]: khoảng 10 × 15.5 phút ≈ 2.6 giờ, từ thời gian mỗi chuỗi của run dò hướng (13995 s cho 15 chuỗi).

### Kết quả (2026-11-19; run `cub_e09_08_ft_l25`, `cub_e09_08_ft`, code `c162858`, commit run `260a4a0`, tiền đăng ký `c17e53c`, Sửa đổi `c8e5f09`; số in bằng `python lab/score.py e0908-summary`; 30/30 chuỗi qua G1–G5)

- off = 250 ở 30/30 chuỗi. Đặt tên P1 cho mọi nhánh. C1 của chuỗi r3, r4 (seed 0) OK 48/48 so với mốc `main.py` tất định.
- Bộ không rò (bộ quyết định), trung bình 5 phiên, 5 chuỗi, 3 FT seed; S5_S0test là của phiên 5; Δ so với PIPE; c09 là độ đúng của phân hoạch trên ảnh mới của pool (Hungarian).

| Nhánh | New | Old | S5_S0test | ΔM_d | ΔS5_S0test | c09 |
|---|---|---|---|---|---|---|
| PIPE | 57.707 | 74.168 | 78.398 | – | – | – |
| N | 61.511 | 69.257 | 69.102 | +3.803 | −9.296 | 66.547 |
| Nc | 61.516 | 76.060 | 78.200 | +3.808 | −0.198 | – |
| Nlab (oracle) | 86.149 | 82.946 | 79.340 | +28.441 | +0.942 | – |
| KW | 61.370 | 70.061 | 70.117 | +3.663 | −8.280 | 67.245 |
| V-new | 61.511 | 69.257 | 69.102 | +3.803 | −9.296 | 66.560 |
| V | 61.245 | 69.402 | 69.876 | +3.538 | −8.522 | 66.547 |
| V-hard | 61.477 | 68.546 | 68.144 | +3.770 | −10.254 | 66.547 |
| V-πfix | 61.196 | 69.625 | 70.436 | +3.489 | −7.962 | 66.547 |

- V theo FT seed r0, r1, r2: ΔM_d +3.519, +3.482, +3.613; ΔS5_S0test −8.727, −8.453, −8.385.
- CAVI, trung bình theo phiên (bộ không rò): số vòng V-new 2.000, V 14.747, V-hard 6.453, V-πfix 11.067; khối trách nhiệm của ảnh cũ nằm trong thành phần mới: V 0.406, V-hard 0.537, V-πfix 0.377; N_k nhỏ nhất của V 8.626; số thành phần có N_k < 2: V 13, V-hard 10, V-πfix 15 (trên 1500).
- L_comp: PIPE 9.133, N 5.847, V 6.072, Nlab 8.807.
- P0 (đối chiếu với E14.07): PIPE 56.944, N 61.033, Nc 60.932.
- Quy tắc quyết định, bộ không rò: ΔM_d(V − PIPE) = +3.538 ≥ T (0.935) và dương ở 3/3 FT seed, nhưng ΔS5_S0test = −8.522 < −B (−0.376): ràng buộc trượt → **Bác**. Luật xác nhận trên r3, r4 không chạy (Sửa đổi `c8e5f09`).
- Luật đọc phụ, bộ không rò: New(V-new) − New(N) = +0.000 < T; New(KW) − New(N) = −0.140 < T; |M_d(V) − M_d(V-hard)| = 0.232 < T/2, nhưng |S5_S0test(V) − S5_S0test(V-hard)| = 1.732 ≥ B → trách nhiệm mềm có tác dụng (lên phía lớp cũ).
- Điều kiện bỏ cơ chế: Bác, V-new dưới T, KW dưới T → **đạt**.
- Bộ có rò cho cùng kết luận: ΔM_d(V − PIPE) +3.447 dương ở 3/3 FT seed, ΔS5_S0test −8.617 → Bác; New(V-new) − New(N) +0.000; New(KW) − New(N) +0.266; |V − V-hard| M_d 0.213, S5_S0test 1.957; điều kiện bỏ cơ chế đạt.
- Dự đoán đã đăng ký: New(V-new) − New(N) trong +2 tới +8: sai, +0.000. New(KW) − New(N) trong −2 tới +6: đúng, −0.140. ΔM_d(V − PIPE) trong +3 tới +9: đúng, +3.538. ΔS5_S0test(V − PIPE) trong −4 tới 0: sai, −8.522. Chấp nhận với xác suất khoảng 30%: kết quả là Bác.
- Số tính tay (kiểm bằng code trong lệnh ghi mục này), bộ không rò: New(V) − New(N) = −0.27 (tính tay); S5_S0test(V) − S5_S0test(N) = +0.77 (tính tay); c09(KW) − c09(N) = +0.70 (tính tay); c09 của V, V-hard, V-πfix trùng c09 của N đến 3 chữ số ở cả 3 FT seed và cả hai bộ (tính tay, so dòng).
- Chi phí thật: 13995 s (`cub_e09_08_ft_l25`) và 14010 s (`cub_e09_08_ft`), theo dòng tiến độ.

### Kết luận (Rito duyệt 2026-11-19)

- **Bác.** Cách gom nhóm mới (VB-GMM) nhận ra lớp mới tốt hơn một chút (M_d +3.538), nhưng làm mô hình quên lớp cũ nhiều đến mức không chấp nhận được (S5_S0test −8.522, biên cho phép −0.376). Luật đặt ra từ đầu là không được đổi lớp cũ lấy lớp mới, nên biến thể này bị loại. Hệ quả: không chạy bước xác nhận trên r3, r4 (r3, r4 chưa bị chấm bằng nhánh nào của E09.08, còn dùng được làm FT seed xác nhận), và VB-GMM không được dùng làm cải tiến.
- Phần tăng lớp mới không đến từ VB-GMM: N một mình đã tăng +3.803, cao hơn V.
- Cách chia nhóm trong số lớp mới không đổi: CAVI bị khoá ở điểm khởi tạo k-means (V-new dừng sau 2 vòng, New trùng N; c09 của V, V-hard, V-πfix trùng N). KW cho một cách chia khác (c09 +0.70, tính tay) mà lớp mới không tăng (New −0.140). [suy luận] Biểu diễn hiện tại không cho gom nhóm đúng hơn bằng các cách kiểu k-means; trách nhiệm gần one-hot ở 384 chiều là giải thích cho việc CAVI bị khoá, chưa đo trực tiếp.
- Lớp cũ mất vì thành phần cũ cố định không giành lại được ảnh cũ: 0.406 khối trách nhiệm của ảnh cũ nằm trong thành phần mới (V).
- Điều kiện bỏ cơ chế đạt. Phạm vi: k-means mặc định, KW, CAVI khởi tạo từ k-means với một tiên nghiệm NIW (off = 250), trên PCA-384 của CUB t5. Khoảng Nc → Nlab (+25.216 New) cần biểu diễn khác hoặc nhãn.
- **Về VB, ghi lại cho rõ.** Paper mô tả mô hình theo Variational Bayes (ELBO, prior liên hợp), nhưng code chính thức chỉ ước lượng MLE (D23). "VB thật" là đề xuất của nhóm (báo thầy 2026-11-10), gồm 4 thành phần. Trạng thái sau E09.08:
  - (1) Hậu nghiệm NIW của lớp cũ làm prior, dự đoán Student-t (Bayesian QDA): có kết quả. E06.04 đạt (D = +1.91 GC10, +1.59 CUB). Trên CUB t5, N (NIW cho mọi khe, cùng tập ảnh đã lọc `keep` của pipeline) tăng New +3.803 nhưng mất S5_S0test −9.296 so với pipeline.
  - (2) Prior phân cấp cho hiệp phương sai lớp mới: chưa thử (E09.08 dùng một prior chung cho mọi khe).
  - (3) Nhãn mềm thay khâu lọc cũ/mới: thua (E09.08, nhánh V).
  - (4) Trọng số Dirichlet cân bằng cũ/mới: không giúp (bỏ Dirichlet, V-πfix, giữ lớp cũ hơn V: S5_S0test 70.436 so với 69.876).
  - VB thật chưa thua hẳn: hỏng (3), (4); (1) có kết quả nhưng còn mất lớp cũ; (2) chưa thử.
- **Câu hỏi then chốt, việc đầu tiên của phiên sau:** N khớp trên đúng tập ảnh đã lọc của pipeline mà vẫn mất 9.296 điểm S5_S0test, còn pipeline (MLE qua SVI, có dừng sớm) thì không. Vì sao? Sửa được thì VB thật còn cơ hội thắng bản gốc; không sửa được thì dừng hướng VB thật.
- Giới hạn: r0–r2 đã dùng ở E12.02 và E14.01–E14.07, nên đây không phải kiểm định xác nhận; GC10 không thuộc thí nghiệm này.

## E09.09 — Dừng gộp theo dấu của log Bayes factor (Q09.14, Q09.02)

### Tiền đăng ký (2026-11-14; viết trước khi chạy và trước khi đọc các khoá `t1_k20_*` của r3, r4)

- Câu hỏi: dừng phép gộp của `09b` khi log Bayes factor lớn nhất không còn dương, thay vì ép về đúng 5 cụm, có cho phân hoạch pool tốt hơn, và giữ được cụm riêng của rolled_pit không? Nguồn: E09.06, Ghi chú 2026-11-14 (vết gộp: log Bayes factor đổi dấu ở khoảng bước 8–9; rolled_pit bị gộp sau đó).
- Nhánh, trên cùng lần rút (cùng luật RNG như E09.06: lần rút 0 dùng `rng_in`, lần rút k ≥ 1 dùng `RandomState(906000 + 1000 r + 100 N + k)`), 21 lần rút mỗi ảnh chụp:
  - `bf5`: `09a` + `09b` như E09.06 (gộp về đúng 5 cụm);
  - `bf0`: cùng cụm con K' = 20 và cùng tiên nghiệm (`merge_prior`), cùng luật gộp, nhưng dừng khi cặp tốt nhất có log Bayes factor ≤ 0, và không gộp xuống dưới 5 cụm. Số cụm K̂ ≥ 5 do dữ liệu quyết định.
- Thước chính (tầng 1): `bal_acc_hungarian` của `score09` trên ảnh lớp mới của pool, ghép một–một 5 lớp với K̂ cụm (Kuhn 1955); ảnh trong cụm không được ghép tính là sai. Nhãn pool chỉ dùng để chấm.
- Đại lượng: Δ_r = trung bình trên 5 ảnh chụp × 21 lần rút của (`bf0` − `bf5`), theo FT seed. Cụm riêng: `best` ≥ 0.5 (như mục Kết quả của E09.06). Tỷ lệ giữ của rolled_pit: trong các lần rút mà rolled_pit có cụm riêng ở K' = 20, tỷ lệ lần mà nó còn cụm riêng ở `bf0`.
- Dữ liệu quyết định: r3, r4 (vết gộp của hai FT seed này chưa ai xem). r0–r2 chỉ báo cáo, ghi rõ là không sạch (luật được nghĩ ra từ vết gộp của chúng).
- Quy tắc, chốt trước:
  - SỐNG: Δ_r3 > 0 và Δ_r4 > 0, và tỷ lệ giữ của rolled_pit gộp trên r3, r4 ≥ 0.5 (mốc: `bf5` giữ 0.015 trên r0–r2);
  - LOẠI: trung bình Δ_r3, Δ_r4 ≤ 0;
  - KHÔNG RÕ: còn lại.
  Nếu không có lần rút nào của r3, r4 mà rolled_pit có cụm riêng ở K' = 20, điều kiện thứ hai không xác định và kết luận là KHÔNG RÕ.
- Hệ quả: SỐNG → thiết kế tầng 2 cho nhánh k tự do (bộ phân lớp K̂ khe mới, đặt tên nhiều–một), tiền đăng ký riêng. LOẠI → giữ k = 5; câu (c) về rolled_pit chuyển sang hướng khác. KHÔNG RÕ → không xây tầng 2, ghi lại và để sau E09.08.
- Báo cáo kèm, không dùng để quyết định: phân phối K̂; tỷ lệ cụm riêng của rolled_pit và crease ở cả hai nhánh; `best` theo lớp (silk mất bao nhiêu khi bị tách); Δ với off nhân 0.5 và 2 (độ nhạy theo tiên nghiệm).
- Code (viết sau tiền đăng ký, mặc định tắt): tham số dừng cho `agglomerate` / `v09b_merge` trong `lab/variants.py`, biến thể `09f-merge-stop`; runner `score.py e0909 --from <chuỗi mặc định>` qua `exp.py`, không chạy SVI, 25 run (r0–r4 × s0–s4). Cổng kiểm: nhánh `bf5` trùng `d<k>_t1_bf_bal`, `J20`, `purity` của E09.06 ở mọi lần rút; tắt tham số dừng thì phân hoạch trùng `bf5` từng ảnh.
- Dự đoán của Claude [SUY LUẬN, khoảng 55%]: SỐNG, với Δ_r nhỏ (0.02–0.06); silk mất `best` rõ rệt vì bị tách. Rito ghi dự đoán riêng trong sổ giấy.
- Giới hạn: tầng 1 báo trước tầng 2 yếu (E09.06, Ghi chú 2026-11-14, §3.3), nên SỐNG chỉ là giấy phép xây tầng 2. Ngưỡng 0.5 của tỷ lệ giữ là ngưỡng tự đặt. Điểm 0 của log Bayes factor phụ thuộc tiên nghiệm (off).

### Sửa đổi (2026-11-14; trước khi viết code và trước mọi run; phần tiền đăng ký ở trên, `cff7c46`, giữ nguyên)

- Cách cài: phép gộp tham lam và tất định, nên đường gộp của `bf0` là đoạn đầu của đường gộp `bf5`. Runner cắt `merge_steps` của `v09b_merge` (luật `bf`) ở bước đầu tiên có log Bayes factor ≤ 0 (không dưới 5 nhóm) và dựng phân hoạch bằng `merged_ids`. Không sửa `lab/variants.py`; biến thể `v09f_merge_stop` chỉ viết khi E09.09 SỐNG và cần tầng 2, khi đó phải trùng phân hoạch tầng 1 của runner. Kết quả toán học như phần tiền đăng ký.
- Cổng kiểm, thay dòng cổng kiểm ở trên: G1 `bf5` trùng `d<k>_t1_bf_bal`, `J20`, `purity` của E09.06 ở cả 21 lần rút; G2 `off` trùng `off_lc` của E06.04; G3 cắt đủ 15 bước thì phân hoạch trùng `bf5` từng ảnh, và điểm cắt của `bf0` là bước đầu tiên có log Bayes factor ≤ 0 (hoặc 5 nhóm).
- Nhánh độ nhạy (off × 0.5, × 2): `v09b_merge` với tiên nghiệm đã đổi off (cùng m0, k0; ν0 và Ψ0 tính lại theo off mới), cắt cùng cách; mỗi lần rút chỉ ghi balanced, K̂, `best` của rolled_pit và crease.
- Bảng quyết định do lệnh `score.py e0909-summary` in ra (không dùng lệnh rời), như `e0906-summary`.
- Dự đoán của Claude, sửa (bản ở trên giữ nguyên): trên r0–r2, 20 cụm chưa gộp cho tầng 1 một–một 0.41 / 0.41 / 0.38 so với 0.56 / 0.53 / 0.51 của `bf5` (E09.06, bảng quyết định). `bf0` dừng ở khoảng 12 cụm, nằm giữa. [SUY LUẬN] rolled_pit được thêm nhưng silk và inclusion mất khi bị tách; Δ nhiều khả năng quanh 0. Xác suất SỐNG khoảng 35%; kết quả dễ xảy ra nhất là KHÔNG RÕ.

### Kết quả (2026-11-14; run `gc10_e09_09_r0_s0`–`gc10_e09_09_r4_s4`, code `2917741`, commit run `c76ffcd`; số in bằng `python lab/score.py e0909-summary`; 25/25 run qua G1–G3)

- Quyết định trên r3, r4: Δ_r3 = −0.0003, Δ_r4 = +0.0028, trung bình +0.0013; tỷ lệ giữ của rolled_pit 0.982 (108/110; mốc `bf5` 0.015) → KHÔNG RÕ (điều kiện Δ_r3 > 0 không đạt).
- Theo FT seed (Δ_r | K̂ trung vị [min, max] | rolled_pit: có cụm con ở K' = 20, giữ được ở `bf0` | crease có cụm riêng `bf5` → `bf0`): r3 −0.0003 | 12 [10, 13] | 47, 46 | 70 → 95; r4 +0.0028 | 12 [10, 13] | 63, 62 | 47 → 87. Chỉ báo cáo, không sạch: r0 +0.0105 | 11 [9, 13] | 71, 66 | 88 → 92; r1 +0.0134 | 12 [10, 13] | 86, 86 | 65 → 31; r2 −0.0377 | 13 [11, 15] | 46, 42 | 44 → 80.
- `best` trung bình `bf5` → `bf0`, r3 / r4: rolled_pit 0.042 → 0.308 / 0.000 → 0.452; crease 0.552 → 0.566 / 0.379 → 0.564; inclusion 0.416 → 0.319 / 0.502 → 0.298; silk 0.537 → 0.448 / 0.539 → 0.536; waist 0.801 → 0.764 / 0.666 → 0.646. Ở cả 5 FT seed, inclusion mất 0.10–0.24.
- Độ nhạy theo tiên nghiệm (báo cáo kèm): off × 0.5 cho K̂ trung vị 14–17 và Δ −0.08 tới −0.11 ở cả 5 FT seed; off × 2 cho K̂ 8–9 và Δ +0.03 tới +0.06 ở cả 5 FT seed.
- Dự đoán: bản đầu của Claude (SỐNG, khoảng 55%) sai; bản sửa trước khi code (KHÔNG RÕ là dễ nhất, SỐNG khoảng 35%) đúng.

### Kết luận

- KHÔNG RÕ theo quy tắc: không xây tầng 2 cho nhánh k tự do; k = 5 và `09a` + `09b` giữ nguyên làm mốc; để sau E09.08.
- Cơ chế của B13 được xác nhận trên dữ liệu sạch: khi rolled_pit có cụm con riêng ở K' = 20, phép gộp không bị ép giữ được nó ở 108/110 lần. Nút thắt của rolled_pit nay là bước K' = 20 (chỉ 47/105 và 63/105 lần rút có cụm con riêng) và ràng buộc k = 5, không phải luật gộp.
- Lợi ở lớp hiếm bị trả bằng việc tách lớp đông, nhất là inclusion. Với Hungarian một–một, mỗi lớp chỉ được tính một cụm, nên lớp bị tách mất điểm. [SUY LUẬN] Nhánh k tự do chỉ có ích nếu có cách gộp nhãn các cụm thừa về 5 lớp mà không dùng nhãn; luật gộp theo Bayes factor không làm được việc đó, vì nó coi các mode của inclusion (và silk) khác nhau hơn crease so với rolled_pit (Ghi chú 2026-11-14 của E09.06, §5.3).
- Điểm 0 của log Bayes factor không phải điểm dừng tự nhiên: nó dịch theo off. off × 2 cho Δ dương ở cả 5 FT seed, nhưng đó là phát hiện sau khi xem dữ liệu, trên các FT seed đã dùng; muốn kiểm phải tiền đăng ký mới và lấy FT seed chưa dùng (r5 trở đi). [SUY LUẬN] Hướng hợp lý hơn một ngưỡng chỉnh tay là để mô hình học số thành phần (tiên nghiệm Dirichlet trong E09.08).

## E11.01 — Cột 11 và phản thực tế bước dừng trên nhánh `09a` + `09b` (Q11.01, Q07.04)

### Tiền đăng ký (2026-11-14; viết trước khi có code và trước mọi run)

- Câu hỏi:
  - Q11.01: phần điểm mất từ cột 10 tới cột 14 nằm ở lượt F, hay ở bộ lọc và lượt S, trong từng nhánh?
  - Q07.04: lớp cũ mất thêm ở nhánh `09a` + `09b` là do lượt S dừng muộn hơn, hay do chính phân hoạch?
  - Nguồn: E09.06, Ghi chú 2026-11-14 (`09b-overcluster-merge` §3.1, §4).
- Vì sao: câu 3 của thiết kế lại E09.08 (trách nhiệm mềm thay khâu nào của bước 11–13), chiến lược 2026-11-14 §6 (ngoài repo).
- Loại: phép đo mô tả. r0–r4 đều đã dùng để chọn hoặc lặp lại `09a` + `09b`, nên đây không phải kiểm định xác nhận và không đổi mặc định nào của dòng PL.
- Dữ liệu: 25 chuỗi `gc10_v2_ft_r<r>_s<N>` (r0–r4, s0–s4), lần rút 0–3 của mỗi chuỗi, cùng luật RNG như E09.06 (lần rút 0 dùng `rng_in`, lần rút k ≥ 1 dùng `RandomState(906000 + 1000 r + 100 N + k)`). Lấy theo chỉ số lần rút, không theo kết quả.
- Nhánh, trên cùng lần rút:
  - `def`: phân hoạch của `s09_cluster`, bước 10–14 mặc định;
  - `var`: phân hoạch `09a` + `09b` (K' = 20, luật `bf`), bước 10–14 mặc định;
  - `var@def`: phân hoạch và bước 10–12 của `var`; lượt S bị ép dừng ở bước dừng lượt S của `def` trong cùng lần rút; rồi bước 14.
- Cách ép bước dừng (đọc từ `classifier/mngmm.py`, `run_inference`):
  - ở vòng lặp s, khi cờ dừng bật và s > n_epochs / 3, mngmm trả về trạng thái trước lần cập nhật gần nhất, tức sau s − 1 lần cập nhật;
  - thành phần dừng mới thay `calculate_metrics_on_covariances`: trả `dets` như bản gốc (vì `dets` đi vào `svi.update`), cờ bật đúng ở vòng lặp s và tắt ở mọi vòng khác; s = n_epochs nghĩa là không bật cờ;
  - lượt S dùng `PRNGKey(0)` và lịch học không phụ thuộc `n_epochs`, nên quỹ đạo trùng quỹ đạo tự nhiên cho tới bước s;
  - nếu bước dừng của `def` không lớn hơn n_epochs / 3 thì lần rút đó không có nhánh `var@def`; ghi số lần như vậy.
- Đại lượng, mỗi nhánh và mỗi lần rút (balanced novel như QD25, tên cụm như QD15; các cột là cột của sổ mất điểm, `score.ledger_state`):
  - c10, c11, c14: balanced novel ở cột 10, 11, 14; L_F = c10 − c11; L_S = c11 − c14;
  - old11, old14: số ảnh test cũ đúng ở cột 11, 14;
  - bộ lọc (cột 12 của pool): số ảnh lớp mới được giữ và mang đúng tên; số ảnh cũ không bị loại;
  - stop_F, stop_S.
  - Theo FT seed: trung bình trên 5 chuỗi × 4 lần rút.
- Quy tắc, chốt trước:
  - Q11.01, trên nhánh `var`: LƯỢT F nếu L_F ≥ L_S ở ít nhất 4/5 FT seed; BỘ LỌC VÀ LƯỢT S nếu L_F < L_S ở ít nhất 4/5 FT seed; CẢ HAI trong các trường hợp còn lại. Báo kèm cùng phép đếm cho `def`, và hiệu `var` − `def` của L_F, L_S.
  - Q07.04: D_old = trung bình trên 5 FT seed của (old14 của `var@def` − old14 của `def`), đơn vị ảnh. BƯỚC DỪNG GIẢI THÍCH nếu |D_old| < 1.48 ảnh; PHÂN HOẠCH CÓ PHẦN nếu |D_old| ≥ 1.48 ảnh. 1.48 ảnh = B (0.615 điểm) trên 241 ảnh test cũ (tính tay). Báo kèm: old14 của `var` − `def` khi dừng tự nhiên; giá của việc dời bước dừng với lớp mới, c14 của `var` − c14 của `var@def`.
- Hệ quả:
  - LƯỢT F → trong E09.08, trách nhiệm mềm thay trước hết việc khớp một Gaussian cho mỗi cụm trên cả pool.
  - BỘ LỌC VÀ LƯỢT S → chạy mở rộng (2) rồi mới chốt câu 3 của E09.08.
  - CẢ HAI → chạy mở rộng (1).
  - BƯỚC DỪNG GIẢI THÍCH → E09.08 coi luật dừng và khối xác suất cũ/mới là núm đánh đổi cũ/mới; các nhánh được so ở cùng mức ảnh cũ.
  - PHÂN HOẠCH CÓ PHẦN → ghi vào thiết kế E09.08 rằng phân hoạch đổi ranh giới cũ/mới ngoài đường bước dừng.
- Mở rộng, chốt trước, mỗi cái chạy đúng một lần:
  - (1) nếu Q11.01 ra CẢ HAI: thêm lần rút 4–7, áp lại quy tắc trên 8 lần rút; kết luận đó là cuối.
  - (2) nếu Q11.01 ra BỘ LỌC VÀ LƯỢT S: thêm nhánh bộ lọc dùng nhãn thật (giữ đúng các ảnh lớp mới của pool; chỉ để phân tích, không thuộc phương pháp), lượt S và bước 14 như thường, cho cả `def` và `var` trên cùng lần rút; tách L_S thành phần do bộ lọc và phần của lượt S.
- Cổng kiểm:
  - G1: `def` và `var` trùng các khoá `d<k>_def_*`, `d<k>_var_*` của run E09.06 cùng chuỗi, cùng lần rút, ở bal10, bal14, stop_F, stop_S, S1_Old, old14.
  - G2: ép nhánh `var` dừng ở chính bước dừng tự nhiên của nó cho lại `params1` trùng từng bit (lần rút 0 của mỗi run).
  - G3: stop_S của `var@def` bằng stop_S của `def`.
  - G4: C3 và C4 của E09.06.
- Code (viết sau tiền đăng ký, mặc định tắt): một thành phần dừng mới trong `lab/variants.py`, chỉ dùng cho lượt S của nhánh `var@def`; không sửa `classifier/mngmm.py` và `lab/replay.py`. Runner `score.py e1101 --from <chuỗi>` qua `exp.py`, 25 run. Bảng quyết định do `score.py e1101-summary` in ra.
- Chi phí: khoảng 6 giờ máy (tính tay: run `gc10_e09_06_r1_s0` chạy 21 phân hoạch trong 30.1 phút; ở đây 100 lần rút, mỗi lần hai chuỗi 10–14 và một lượt S).
- Dự đoán của Claude [SUY LUẬN]:
  - Q11.01: LƯỢT F khoảng 45%, CẢ HAI khoảng 40%, BỘ LỌC VÀ LƯỢT S khoảng 15%. Căn cứ: ở nhánh mặc định, lượt F mất 0.037 / 0.075 / 0.019 và bộ lọc cộng lượt S mất 0.027 / 0.019 / 0.024 trên r0 / r1 / r2 (tính tay từ bảng `14-eval` §4.1 dẫn trong chiến lược 2026-11-05 §1).
  - Q07.04: BƯỚC DỪNG GIẢI THÍCH khoảng 60%. Căn cứ: ở r0, r1 hai nhánh giữ số ảnh cũ như nhau khi cùng bước dừng; ở r2 còn khoảng −1.1 ảnh không giải thích được (E09.06, Ghi chú 2026-11-14, §4).
- Giới hạn:
  - Cột 11 là phản thực tế (bộ phân lớp với Gaussian của lượt F), không phải đầu ra của pipeline.
  - L_S gộp bộ lọc với lượt S, trừ khi chạy mở rộng (2). Lượt S khởi động lại từ Gaussian của S0, không tiếp từ lượt F (`s13_gauss_s`), nên L_S có thể âm.
  - Mỗi chuỗi chỉ 4 lần rút. Ngưỡng 4/5 FT seed và 1.48 ảnh là ngưỡng tự đặt.
  - Chỉ ép một chiều (`var` theo bước dừng của `def`).
  - Hiệu `var` − `def` của phần mất ở bước 11–14 chỉ là 0.044 / 0.014 / 0.047 trên r0 / r1 / r2 (tính tay từ E09.06, Ghi chú 2026-11-14), cùng cỡ với T = 0.0273; vì vậy quy tắc dùng mức của từng nhánh, không dùng hiệu.

### Kết quả (2026-11-15; run `gc10_e11_01_r0_s0`–`gc10_e11_01_r4_s4`, code `a4f9578` + `d21273a`, commit run `e787ebe`; số in bằng `python lab/score.py e1101-summary`; 25/25 run qua G1–G4, không lần rút nào thiếu nhánh `var@def`)

- Q11.01: trên nhánh `var`, L_F ≥ L_S ở 4/5 FT seed → LƯỢT F. Nhánh `def` cũng 4/5.
- Theo FT seed (r0 / r1 / r2 / r3 / r4):
  - `var`: L_F 0.0597 / 0.0677 / 0.0490 / 0.0636 / 0.0621; L_S 0.0485 / 0.0528 / 0.0550 / 0.0436 / 0.0595.
  - `def`: L_F 0.0339 / 0.0762 / 0.0228 / 0.0558 / 0.0444; L_S 0.0325 / 0.0191 / 0.0221 / 0.0203 / 0.0452.
  - L_F − L_S của `var` (tính tay): +0.011 / +0.015 / −0.006 / +0.020 / +0.003.
- Hiệu `var` − `def`, trung bình 5 FT seed: L_F +0.0138, L_S +0.0240.
- Bộ lọc, trung bình mỗi lần rút, `var` / `def`: ảnh lớp mới được giữ và mang đúng tên 316.25 / 299.10 (r0), 325.05 / 283.90 (r1), 324.00 / 285.45 (r2), 295.60 / 207.70 (r3), 279.70 / 235.00 (r4); ảnh cũ không bị loại 7.85 / 6.30, 7.95 / 7.90, 10.50 / 6.85, 9.10 / 6.00, 9.45 / 7.30.
- Q07.04: D_old = +0.670 ảnh (B = 1.48 ảnh) → BƯỚC DỪNG GIẢI THÍCH. Theo FT seed: +0.25 / +0.50 / +0.05 / −0.70 / +3.25.
- Báo kèm: old14 `var` − `def` khi dừng tự nhiên −0.740 ảnh; giá của việc dời bước dừng với lớp mới (bal14 `var` − `var@def`) +0.0127.
- Bước dừng lượt S trung bình, `def` / `var`: 582.00 / 603.2, 520.95 / 565.8, 496.15 / 558.5, 532.25 / 616.8, 537.10 / 570.9.
- Dự đoán của Claude (LƯỢT F khoảng 45%, BƯỚC DỪNG GIẢI THÍCH khoảng 60%): cả hai trúng.

### Kết luận

- LƯỢT F theo quy tắc, nhưng sát nút: r4 chỉ hơn 0.003, r2 ngược dấu; đổi r4 là ra CẢ HAI. Đọc đúng: hai khâu mất ngang cỡ nhau, lượt F nhỉnh hơn.
- Câu gốc của Q11.01 (vì sao bước 11–14 làm cụm đã gộp mất nhiều hơn) nghiêng về bộ lọc và lượt S: phần mất thêm so với `def` là +0.024 ở L_S, so với +0.014 ở L_F. Quy tắc dùng mức của từng nhánh nên kết luận chính không phản ánh điều này.
- Bộ lọc của `var` giữ nhiều hơn ở cả hai chiều: thêm khoảng 17–88 ảnh mới mang đúng tên, và thêm khoảng 0–4 ảnh cũ lọt qua mỗi lần rút (tính tay từ các dòng trên). [SUY LUẬN] Cụm đã gộp rộng hơn nên ranh giới chấp nhận của bộ lọc rộng hơn; là một ứng viên giải thích L_S lớn hơn, chưa kiểm.
- Lớp cũ mất ở `09a` + `09b` đi qua bước dừng: dừng muộn hơn khoảng 49 bước (tính tay), giữ ít hơn 0.74 ảnh cũ; ép dừng ở bước của `def` lấy lại 1.41 ảnh cũ (tính tay) và mất 0.013 balanced novel. r4 lệch xu hướng (+3.25 ảnh) và đẩy trung bình lên.
- Hệ quả cho E09.08:
  - câu 3: trách nhiệm mềm thay trước hết việc khớp một Gaussian cho mỗi cụm trên cả pool (lượt F), theo hệ quả đã chốt; vì hai điểm trên, thiết kế cũng phải xử lý bộ lọc, không coi là phụ;
  - câu 4: luật dừng là núm đánh đổi cũ/mới thật; E09.08 so các nhánh ở cùng mức ảnh cũ.
- Không chạy mở rộng (1) và (2): điều kiện của chúng (CẢ HAI; BỘ LỌC VÀ LƯỢT S) không xảy ra.
- Giới hạn: phép đo mô tả trên r0–r4 (đã dùng để chọn phương pháp), 4 lần rút mỗi chuỗi; cột 11 là phản thực tế, không phải đầu ra của pipeline.

## E12.02 — CUB t5: bộ lọc cũ/mới không nhãn dựa trên mô hình lớp cũ lấy lại bao nhiêu lớp cũ cho mô hình N? (QD27, QD28)

### Tiền đăng ký (2026-11-17; viết trước khi có code và trước mọi số của thí nghiệm)

- Mã: theo bước 12 (bộ lọc cũ/mới), QD22. E12.01 (WSM) là một đề xuất khác, chưa tiền đăng ký.
- Câu hỏi: trên CUB t5, nếu thay `keep` của bước 12 bằng một bộ lọc chỉ dựa trên mô hình lớp cũ, không dùng nhãn pool, rồi khớp mô hình N của E14.06 trên tập được giữ, thì lấy lại được bao nhiêu phần ρ của khoảng lớp cũ giữa N và N sạch, và lớp mới mất bao nhiêu?
- Vì sao: E14.06 cho thấy N (hậu nghiệm NIW cho mọi khe) đổi khoảng 4 điểm lớp mới lấy khoảng 5 điểm lớp cũ; N sạch (bỏ ảnh cũ khỏi cụm bằng nhãn) cho lớp cũ +6.774 so với N, lớp mới gần như không đổi. Bộ lọc của pipeline giữ trung bình 50% ảnh pool cũ, 87% ở phiên 1 (E14.02, Sửa đổi). Đây là bước nhỏ đi trước trách nhiệm mềm của E09.08 (Kết luận E14.06). Nó chỉ làm sạch tập khớp khe mới, không nhắm vào cạnh tranh lúc dự đoán: L_comp của N sạch là 6.993, không nhỏ hơn của N (6.502).
- Đã tra trước khi viết: D9, D12, D16, D23, D24, D25 (`notes/paper-code-discrepancies.md`); Kết quả và Kết luận của E06.04, E14.02 (cả Sửa đổi), E14.03, E14.05 (cả Sửa), E14.06; Q12.02; `lab/niw.py`; `_post_1406`, `_logd_1406`, `_score_1406` của `lab/score.py`.
- Loại: phép đo mô tả có can thiệp; một luật đọc chốt trước trên ρ.
- Dữ liệu: như E14.06: 15 chuỗi `cub_chain_ft_l25_r<r>_s<N>` (bộ đọc chính) và 15 chuỗi `cub_chain_ft_r<r>_s<N>` (báo kèm); không gian PCA của S0; `ids`, `pi_test`, `lut` (P0, QD15) lấy từ ảnh chụp. Không SVI, không rút RNG. Nhãn pool chỉ dùng để chấm bộ lọc và cho các nhánh oracle; nhãn test chỉ để chấm.
- Mô hình N: đúng như E14.06 (tiên nghiệm m0 = mean của `z_s0`, κ0 = `niw.K0`, Ψ0 = off × P, off = `niw.choose(niw.loo_curve(z_s0, s0_y))`; khe lớp gốc từ S0; khe off_j + c từ các ảnh pool của phiên j có `ids` = c và được giữ; cụm rỗng là tiên nghiệm; dự đoán Student-t, argmax trên các khe đã gặp). Chỉ đổi tập giữ.
- Mô hình điểm ở phiên i (một ứng viên, chọn theo dòng "Chọn ứng viên"): khe lớp gốc 0…99 khớp trên S0; khe đã khám phá off_j + c, j < i, khớp trên các ảnh mà chính bộ lọc này giữ ở phiên j, theo `ids` (cụm có dưới 2 ảnh được giữ thì không có khe trong mô hình điểm). Ứng viên:
  - A1: trung bình mẫu cộng Ledoit–Wolf về đẳng hướng (như `_fit_closed`, A1 của E06.04), log mật độ Gaussian;
  - A3 với off ∈ {25, 100, 250, 1000}: Ψ0 = off × P, cùng m0, κ0 như N; log mật độ dự đoán Student-t.
- Điểm: s(x) = max trên các khe của mô hình điểm của log mật độ của x. Điểm thấp nghĩa là lạ với mọi lớp cũ.
- Ngưỡng, p-value conformal (Bates, Candès, Lei, Romano & Sesia 2023, Ann. Statist.): điểm hiệu chuẩn s_j của mọi ảnh S0, tính như s nhưng khe của chính ảnh j khớp lại khi bỏ ảnh j (leave-one-out); p(x) = (1 + #{j : s_j ≤ s(x)}) / (n + 1). Bộ lọc giữ x (coi là mới) khi p(x) ≤ α. Nếu ảnh cũ của pool có cùng phân phối với ảnh S0 khi bỏ ra thì P(giữ ảnh cũ) ≤ α, nên α là tỉ lệ ảnh cũ bị giữ theo danh định, so được với mức 0.50 của pipeline. α chính = 0.10.
- Chọn ứng viên, trong từng chuỗi, chỉ bằng S0 (mô phỏng bỏ-một-lớp): điểm "cũ giả" là s_j như trên; điểm "mới giả" của một ảnh lớp k là max trên các khe j ≠ k của log mật độ, với các khe khớp trên đủ S0 (khe j ≠ k chưa thấy ảnh đó). Chọn ứng viên có AUROC lớn nhất (cũ giả là lớp dương, s cao là cũ); hoà thì lấy A3 có off gần off của N nhất. Xấp xỉ đã biết, không sửa: P, do đó Ψ0, vẫn chứa tán xạ của lớp k, trọng số khoảng 1/100.
- Nhánh, cùng chuỗi, cùng ảnh chụp:
  - N, Nc: như E14.06, tính lại; Nc dùng nhãn pool (oracle);
  - NF (nhánh chính): N trên tập giữ của bộ lọc, α = 0.10; khe đã khám phá dùng ở các phiên sau cũng khớp trên tập giữ của bộ lọc;
  - NF∧ (báo cáo): giữ khi cả bộ lọc lẫn `keep` của pipeline cùng giữ;
  - NF-α (báo cáo): α ∈ {0.02, 0.05, 0.20, 0.30, 0.50};
  - NF-count (oracle, báo cáo): giữ đúng số ảnh lớp mới thật của pool, chọn các ảnh có s thấp nhất; tách phần thua do ngưỡng khỏi phần thua do điểm.
- Đại lượng chính: ρ = (Old_NF − Old_N) / (Old_Nc − Old_N), tính trên trung bình 5 phiên và 5 chuỗi của mỗi FT seed, rồi lấy trung bình 3 FT seed (như R của E14.05).
- Báo kèm: New, Old, All, g20, L_comp của mọi nhánh; ΔNew, ΔOld, ΔAll so với pipeline và so với N; ứng viên được chọn và AUROC bỏ-một-lớp của mọi ứng viên; trên pool, theo phiên (nhãn chỉ để chấm): tỉ lệ ảnh cũ bị giữ, tách ảnh cũ lớp gốc và ảnh cũ của đợt đã khám phá, đặt cạnh α; tỉ lệ ảnh mới bị loại; AUROC của s; số khe rỗng của N ở mỗi nhánh.
- Cách đọc, chốt trước (bộ không rò; bộ có rò báo kèm):
  - ρ ≥ 0.67 và ΔNew của NF so với pipeline > T (0.935): NF hơn pipeline theo luật đọc phụ của E14.06 và là nấc khởi đầu của E09.08; kiểm lại trên r3, r4 hoặc server trước khi báo như cải tiến. Căn cứ của 0.67 (tính tay từ E14.06): NF cần ΔOld > −B = −0.376, tức lấy lại ít nhất 4.509 trong 6.774 điểm.
  - 1/3 ≤ ρ < 0.67: bộ lọc cứng chỉ lấy lại một phần; E09.08 giữ trách nhiệm mềm làm cơ chế chính, NF là nhánh đối chứng.
  - ρ < 1/3: mô hình lớp cũ đứng một mình không tách đủ; NF-count và AUROC của s cho biết phần thua do điểm hay do ngưỡng.
  - Mọi kết luận đọc kèm L_comp của NF: bộ lọc không nhắm vào cạnh tranh lúc dự đoán.
- Cổng kiểm:
  - G1: cột pipeline trùng `acc1` của `S<i>_14` (1e-6); N và Nc tính lại trùng `S<i>_N_*`, `S<i>_Nc_*` của run E14.06 cùng bộ (1e-4, vì E14.06 lưu số làm tròn 4 chữ số).
  - G2: đưa `keep` của pipeline vào đúng đường tính của NF cho lại N; đưa tập giữ oracle vào cho lại Nc (1e-6).
  - G3: tập giữ lồng nhau theo α (α nhỏ hơn cho tập con); tập của NF∧ là tập con của NF.
  - G4: khe lớp gốc của A3 trùng `niw.fit` (1e-8 tương đối); s_j của một lớp tính nhanh trùng khớp lại trực tiếp (1e-8); khe lớp gốc của A1 cho cùng nhãn với `closed_form_params` trên tập test S0 (như C3 của E06.04).
  - G5: trên S0, tỉ lệ ảnh có p ≤ α lệch α không quá 1/(n + 1) cộng phần hoà (kiểm cài đặt).
  - Runner chỉ in trạng thái cổng và thời gian, không in số của từng chuỗi.
- Code (viết sau tiền đăng ký): runner `tier2_e1202` trong `lab/score.py`, lệnh `e1202` qua `exp.py`, một run mỗi bộ (tag `cub_e12_02_ft_l25`, `cub_e12_02_ft`); lệnh `e1202-summary` chỉ đọc. Dùng lại `_post_1406`, `_logd_1406`, `_score_1406`; không sửa `lab/niw.py`, `lab/sessions.py`, `lab/replay.py`.
- Chi phí [suy luận]: 20–40 phút mỗi run (E14.06: 10.8 phút; thêm điểm leave-one-out trên S0 cho 5 ứng viên).
- Dự đoán của Claude [SUY LUẬN], độ tin thấp: AUROC của s trên pool (bộ không rò) trong 0.70–0.82; tỉ lệ ảnh cũ bị giữ ở α = 0.10 trong 0.15–0.30, cao hơn α vì ảnh S0 đã được fine-tune thấy; ρ trong 0.25–0.55; New của NF thấp hơn N 0–2 điểm; L_comp của NF trong 6–7.5.
- Giới hạn:
  - Ảnh S0 có nhãn đã được fine-tune thấy (đúng giao thức), nên điểm hiệu chuẩn lạc quan so với ảnh cũ không nhãn của pool; mức lọt thực tế được đo ở phần báo kèm, không sửa.
  - Điểm hiệu chuẩn chỉ có cho lớp gốc; dùng chung cho khe đã khám phá là một giả định (với off lớn, κn và νn của hai loại khe cùng cỡ); kiểm bằng tỉ lệ giữ tách theo đợt.
  - Bộ lọc đứng sau bước 09, nên không sửa phân cụm; "bỏ ảnh cũ trước bước 09" là một ablation riêng.
  - Ảnh cũ lặp lại giữa các phiên (D16); mỗi phiên quyết định độc lập.
  - α, hai ngưỡng 1/3 và 0.67 của luật đọc và lưới ứng viên do Claude đặt; r0–r2 đã dùng ở E14.01–E14.06, nên đây không phải kiểm định xác nhận.

### Sửa đổi (2026-11-17; trước khi viết code và trước mọi run; phần tiền đăng ký ở trên, `d566669`, giữ nguyên)

- G3 thay bằng: (a) ở phiên 1, tập giữ lồng nhau theo α và tập của NF∧ là tập con của NF; (b) ở mọi phiên, áp các α của nhánh NF và NF-α lên cùng một vectơ điểm của nhánh NF cho các tập lồng nhau (kiểm cài đặt của luật giữ).
- Lý do: từ phiên 2, mỗi nhánh có khe đã khám phá riêng, khớp trên tập giữ của chính nhánh đó, nên mô hình điểm khác nhau giữa các nhánh và tập giữ của hai nhánh không bắt buộc lồng nhau. G3 như bản gốc có thể báo lỗi khi code đúng. Chỉ ở phiên 1 mọi nhánh mới có cùng mô hình điểm (khe lớp gốc).
- Chi phí [suy luận], thay dòng Chi phí: 45–90 phút mỗi run, vì runner chạy 10 nhánh tuần tự qua 5 phiên (N, Nc, NF, NF∧, 5 nhánh NF-α, NF-count), không chỉ một nhánh.
- Không đổi gì khác: câu hỏi, đại lượng chính, luật đọc, các nhánh, dự đoán.

### Sửa đổi thứ hai (2026-11-17; sau khi viết code, trước mọi run trên dữ liệu thật; tiền đăng ký `d566669` và Sửa đổi `571660b` giữ nguyên)

- Dòng "Mô hình điểm", ý "cụm có dưới 2 ảnh được giữ thì không có khe trong mô hình điểm": với ứng viên A1, ngưỡng là dưới 3 ảnh; với A3 giữ dưới 2 ảnh.
- Lý do: Ledoit–Wolf của scikit-learn ở n = 2 cho hệ số co bằng 0 và hiệp phương sai hạng 1, nên Cholesky báo lỗi. Phát hiện khi thử runner trên chuỗi giả ngoài repo (code `e680c16`), trước mọi số trên dữ liệu thật. Với n = 1, nhánh của `closed_form_params` dùng Σ đẳng hướng gộp, nhưng tiền đăng ký đã bỏ khe khi n < 2, nên không dùng nhánh đó.
- Không đổi gì khác.

### Kết quả (2026-11-18; run `cub_e12_02_ft_l25`, `cub_e12_02_ft`, code `2e66ccb`, commit run `a110103`; số in bằng `python lab/score.py e1202-summary`; 30/30 chuỗi qua G1–G5)

- Luật đọc, bộ không rò: ρ(NF) 0.465, ΔNew(NF) so với pipeline −25.751 (T 0.935): vùng giữa, "bộ lọc cứng chỉ lấy lại một phần; E09.08 giữ trách nhiệm mềm làm cơ chế chính". Bộ có rò: ρ 0.486, ΔNew −24.291. L_comp(NF) 23.200 (không rò) / 23.183 (có rò).
- Mô hình điểm được chọn: A1 ở 15/15 chuỗi mỗi bộ. AUROC bỏ-một-lớp trên S0, không rò / có rò: A1 0.822 / 0.830; A3 off 25: 0.686 / 0.693; off 100: 0.684 / 0.690; off 250: 0.681 / 0.687; off 1000: 0.674 / 0.680.
- Bộ không rò, trung bình 5 phiên, 5 chuỗi, 3 FT seed. Pipeline: New 56.944, Old 74.046. Cột "giữ cũ gốc" và "bỏ mới" là tỉ lệ trên pool; "khe rỗng" là số khe N rỗng trên 1500.

| Nhánh | New | Old | ρ | L_comp | ΔNew so với pipeline | giữ cũ gốc | giữ cũ đã khám phá (S2–S5) | bỏ mới | AUROC pool của s | khe rỗng |
|---|---|---|---|---|---|---|---|---|---|---|
| N | 61.033 | 69.161 | 0 | 6.502 | +4.089 | 0.417 | 0.651 | 0.001 | – | 0 |
| Nc | 60.932 | 75.935 | 1 | 6.993 | +3.988 | 0.000 | 0.000 | 0.001 | – | 70 |
| NF (α 0.10) | 31.193 | 72.327 | 0.465 | 23.200 | −25.751 | 0.018 | 0.063 | 0.704 | 0.798 | 55 |
| NF∧ | 31.163 | 72.367 | 0.471 | 23.225 | −25.781 | 0.018 | 0.060 | 0.703 | 0.796 | 57 |
| NF-α 0.02 | 7.797 | 67.071 | −0.312 | 13.640 | −49.147 | 0.003 | 0.017 | 0.937 | 0.722 | 599 |
| NF-α 0.05 | 19.681 | 69.892 | 0.107 | 20.556 | −37.264 | 0.009 | 0.043 | 0.837 | 0.754 | 169 |
| NF-α 0.20 | 42.577 | 73.913 | 0.700 | 20.715 | −14.367 | 0.037 | 0.092 | 0.493 | 0.844 | 28 |
| NF-α 0.30 | 47.827 | 74.239 | 0.749 | 18.006 | −9.117 | 0.056 | 0.147 | 0.359 | 0.871 | 25 |
| NF-α 0.50 | 53.720 | 73.859 | 0.693 | 13.139 | −3.224 | 0.092 | 0.251 | 0.191 | 0.902 | 14 |
| NF-count (oracle) | 54.593 | 73.905 | 0.700 | 12.339 | −2.351 | 0.104 | 0.280 | 0.195 | 0.896 | 13 |

- Bộ có rò, cùng cột (New / Old / ρ / L_comp / ΔNew so với pipeline / giữ cũ gốc / bỏ mới / AUROC pool / khe rỗng): pipeline New 57.142, Old 74.427; N 61.032 / 69.329 / 0 / 6.402 / +3.890; Nc 60.959 / 76.277 / 1 / 6.786 / +3.817; NF 32.851 / 72.710 / 0.486 / 23.183 / −24.291 / 0.014 / 0.687 / 0.813 / 76; NF-α 0.20: ρ 0.715, New 43.707; NF-α 0.30: ρ 0.736, New 48.475; NF-α 0.50: ρ 0.697, New 54.042, ΔNew −3.100; NF-count 54.732 / 74.202 / 0.701 / 11.990 / −2.410 / 0.097 / 0.186 / 0.903 / 9.
- Ở cả hai bộ, mọi nhánh lọc có ΔNew so với pipeline âm, nên không nhánh nào đồng thời có ρ ≥ 0.67 và ΔNew > T, kể cả NF-count.
- NF theo phiên, bộ không rò (giữ cũ gốc / giữ cũ đã khám phá / bỏ mới / AUROC pool): S1 0.086 / – / 0.569 / 0.824; S2 0.003 / 0.099 / 0.713 / 0.838; S3 0.002 / 0.062 / 0.767 / 0.745; S4 0.001 / 0.043 / 0.765 / 0.799; S5 0.000 / 0.050 / 0.704 / 0.782. Bộ có rò: S1 0.065 / – / 0.543 / 0.853; S5 0.000 / 0.049 / 0.691 / 0.794.
- Kiểm giả định "điểm hiệu chuẩn của lớp gốc dùng chung cho khe đã khám phá" (Giới hạn của tiền đăng ký), bằng tỉ lệ giữ tách theo đợt: ở phiên 1, khi mô hình điểm chỉ có khe lớp gốc, ảnh cũ gốc bị giữ 0.086 (α 0.10); từ phiên 2 tỉ lệ này còn 0.003 trở xuống. Cài đặt: điểm hiệu chuẩn là max trên 100 khe lớp gốc (`_select_1202`); điểm pool từ phiên 2 là max trên khe lớp gốc và khe đã khám phá của nhánh (`_branch_1202`).
- Cách đọc các cột báo kèm: từ phiên 2, mỗi nhánh có mô hình điểm riêng (khe đã khám phá khớp trên tập giữ của chính nhánh), nên AUROC pool và tỉ lệ giữ khác nhau giữa các nhánh; chúng không phải thuộc tính riêng của s. Quét α và NF-count là báo kèm; luật đọc chỉ dùng NF.
- Chi phí thật: 181.6 phút (không rò) và 398.0 phút (có rò), cột `min` của `runs/INDEX.txt`, không phải 45–90 phút như Sửa đổi.
- Dự đoán đã đăng ký: AUROC pool trong 0.70–0.82: đúng (0.798). Tỉ lệ ảnh cũ bị giữ ở α 0.10 trong 0.15–0.30: sai, và sai chiều: 0.018 (cũ gốc), 0.063 (cũ đã khám phá), dưới α. ρ trong 0.25–0.55: đúng (0.465). New của NF thấp hơn N 0–2 điểm: sai, thấp hơn 29.840. L_comp của NF trong 6–7.5: sai, 23.200.

### Kết luận (Rito duyệt 2026-11-18)

- Theo luật đọc chốt trước: vùng giữa, ở cả hai bộ đặc trưng. Trên CUB t5, bộ lọc cứng dựa trên mô hình lớp cũ không phải cải tiến độc lập; NF là nhánh đối chứng của E09.08, trách nhiệm mềm giữ vai trò cơ chế chính.
- Biết đúng số ảnh mới cũng không đủ: NF-count thua pipeline 2.351 điểm New (> T), còn Nc (nhãn oracle) không mất New so với N (−0.101) mà vẫn tăng Old 6.774. Khoảng New giữa NF-count và Nc là 6.339 điểm (tính tay, 60.932 − 54.593): phần thua này đến từ thứ hạng của s (chọn sai ảnh nào bị bỏ), không chỉ từ ngưỡng.
- [suy luận] Bộ lọc chặt hơn α danh định vì điểm hiệu chuẩn là max trên 100 khe lớp gốc, còn điểm pool từ phiên 2 là max trên cả khe lớp gốc và khe đã khám phá; max trên nhiều khe hơn chỉ làm điểm tăng, nên p tăng và ít ảnh được giữ. Khớp với tỉ lệ giữ ảnh cũ gốc theo phiên (0.086 ở phiên 1, từ 0.003 trở xuống từ phiên 2). E12.02 không kiểm trực tiếp cơ chế này; muốn kiểm thì hiệu chuẩn trên đúng tập khe mà điểm pool dùng.
- [suy luận] Không chạy bản E12.02 sửa hiệu chuẩn: kể cả khi tỉ lệ giữ về đúng α, điểm hoạt động hợp lý nhất dự kiến cỡ NF-α 0.50 hoặc NF-count (ΔNew so với pipeline −3.224, −2.351), vẫn thua pipeline quá T. Đây là ngoại suy từ các nhánh chưa sửa hiệu chuẩn, độ tin vừa.
- L_comp: làm sạch tập khớp không giảm cạnh tranh lúc dự đoán (Nc 6.993, N 6.502). Khi lọc quá chặt, L_comp không đơn điệu theo α vì chính g20 sụp (NF-α 0.02: g20 21.438, L_comp 13.640; NF: g20 54.393, L_comp 23.200), nên L_comp của các nhánh lọc chặt không đọc như số đo thuần của cạnh tranh.
- Hệ quả cho thiết kế E09.08 [suy luận]: (a) trọng số mềm tránh khe rỗng của quyết định 0/1 (NF: 55 khe N rỗng trên 1500); (b) nếu dùng s làm trọng số thì hiệu chuẩn trên cùng tập khe với điểm pool; (c) cạnh tranh lúc dự đoán cần cơ chế riêng, làm sạch tập khớp không thay được.
- Bước kế (Rito duyệt): sang thiết kế E09.08, mang theo ý (b); không chạy E12.02b.
- Giới hạn: như phần tiền đăng ký; r0–r2 đã dùng ở E14.01–E14.06, nên đây không phải kiểm định xác nhận; quét α và NF-count là báo kèm, không dùng để chọn α. Chi phí thật gấp 2–4 lần ước lượng; từ nay ước lượng chi phí từ một lần chạy checks-only.

## E14.01 — Điểm lớp mới đổi bao nhiêu theo quy ước ghép tên? (QD15, QD28, D13)

### Tiền đăng ký (2026-11-16; viết trước khi có code và trước mọi số)

- Câu hỏi: trên cùng dự đoán của bộ phân lớp, M_d và các bảng S1–S5 của CUB t5 đổi bao nhiêu khi đổi cách ghép khe với lớp lúc chấm?
- Vì sao: nhánh `09e` của E09.08 không có k-means để đặt tên như QD15, nên luật đặt tên phải chốt trước tiền đăng ký E09.08; và số của các bài C-GCD khác chấm bằng Hungarian toàn cục (chiến lược 2026-11-16 §2.5, ngoài repo).
- Loại: phép đo mô tả, không có luật đạt/trượt. Không đổi đại lượng chính của QD28 (M_d, tên cụm như QD15) cho các nhánh có k-means.
- Dữ liệu: 30 chuỗi mặc định `cub_chain_ft_r<r>_s<N>` và `cub_chain_ft_l25_r<r>_s<N>` (r0–r2, s0–s4). Ở mỗi phiên i = 1…5: `params1` của ảnh chụp `S<i>_13`, dự đoán trên tập test của phiên bằng thành phần dự đoán mặc định. Không SVI, không rút RNG, không đọc nhãn pool.
- Bốn cách chấm trên cùng dự đoán (off_i = 100 + 20 (i − 1); khối của phiên i là khe off_i … off_i + 19):
  - P0 (QD15, hiện tại): khe 0–99 giữ nghĩa; khối của phiên i ghép với 20 lớp mới của phiên bằng Hungarian trên cụm k-means của ảnh test lớp mới (`pi_test` của ảnh chụp `S<i>_10`); bảng tên của các phiên trước đóng băng.
  - P1: như P0, nhưng khối của phiên i ghép bằng Hungarian trên dự đoán của bộ phân lớp; bảng tên của khối j < i là bảng P1 của phiên j, đóng băng.
  - P2: khe 0–99 giữ nghĩa; ở mỗi phiên ghép lại mọi khe ≥ 100 đã có với mọi lớp mới đã gặp, một phép Hungarian, trên dự đoán của bộ phân lớp. Đây là SPE (Wang et al. 2026, Eq. 5–7) áp cho từng phiên với Y_old = 100 lớp có nhãn.
  - P3: Hungarian toàn cục trên mọi khe và mọi lớp đã gặp, tính trên toàn tập test của phiên (Eq. 1 của cùng bài).
  - Ở P1, P2 ma trận ghép chỉ gồm ô (lớp mới thật, khe mới), như Alg. 1 của bài: ảnh cũ rơi vào khe mới và ảnh mới rơi vào khe cũ là lỗi với mọi bảng tên, nên tính trên toàn tập test hay chỉ ảnh test lớp mới cho cùng kết quả.
  - P1 và việc lặp P2 theo phiên là phần mở rộng của ta: bài SPE chỉ xét GCD một giai đoạn và chấm trên tập không nhãn D_u, không trên tập test.
- Đại lượng, mỗi cách chấm, mỗi chuỗi, mỗi phiên: All, Old, New (cùng ba tập test như `metrics.json`); Old tách thành Base (lớp 0–99) và Disc (lớp đã khám phá ở phiên trước). M_d = trung bình S<i>_New của 5 phiên. Theo FT seed: trung bình 5 chuỗi. Báo hiệu Pk − P0 của M_d, S5_All, S5_Old cạnh T = 0.935 và B = 0.376 (QD28), cho từng bộ đặc trưng.
- Chốt trước:
  - Luật đặt tên của `09e` là P1 (cách duy nhất trong bốn cách không cần k-means mà vẫn đóng băng tên của lớp đã khám phá). Mọi so sánh `09e` với nhánh có k-means chấm cả hai nhánh bằng P1; số P0 của nhánh có k-means báo kèm.
  - P3 chỉ là cột phụ để đặt cạnh số của bài khác; P2 chỉ báo cáo.
  - Kết quả của E14.01 không dùng để đổi đại lượng chính sau khi thấy số.
- Cổng kiểm:
  - G1: P0 tính lại trùng `acc1` của ảnh chụp `S<i>_14` và S<i>_All, S<i>_Old, S<i>_New của `metrics.json` ở cả 30 chuỗi.
  - G2, thứ tự suy ra từ định nghĩa: S<i>_New(P1) ≥ S<i>_New(P0) ở mọi phiên; số ảnh đúng trên toàn tập test của phiên: P3 ≥ P2 ≥ max(P1, P0). Vi phạm là lỗi code. Không có thứ tự bảo đảm giữa All(P1) và All(P0) từ phiên 2 (bảng tên đóng băng của hai cách khác nhau).
- Code (viết sau tiền đăng ký): runner trong `lab/score.py`, lệnh `e1401` qua `exp.py`, một run cho mỗi bộ đặc trưng (tag `cub_e14_01_ft`, `cub_e14_01_ft_l25`), ghi `metrics.json` và `checks_ok`; lệnh `e1401-summary` chỉ đọc. Không sửa `lab/sessions.py`, `lab/replay.py`.
- Chi phí [suy luận]: dưới 15 phút máy mỗi run (150 lần dự đoán trên 3–6 nghìn ảnh test); chưa đo.
- Dự đoán của Claude [SUY LUẬN], độ tin thấp: M_d(P1) − M_d(P0) từ 0 tới 1.5 điểm; S5_New(P3) − S5_New(P0) từ 2 tới 8 điểm; S5_Old(P3) thấp hơn S5_Old(P0). Căn cứ: 20 lớp mới mỗi phiên với khoảng 57% đúng nên bảng tên từ k-means và từ bộ phân lớp ít khi khác nhau nhiều; bài SPE báo hiệu GH − SPE của New trên CUB dưới 1 điểm (Bảng 7, 8: SimGCD với số lớp ước lượng, và SelEx), nhưng ở đây lớp mới yếu hơn nhiều.
- Giới hạn:
  - Cả bốn cách dùng nhãn test để đặt tên (D13); phép đo là độ nhạy theo quy ước ghép, không phải điểm thật.
  - P1 chọn tên sau khi thấy lỗi của bộ phân lớp, P0 chọn tên độc lập với bộ phân lớp; hiệu P1 − P0 dương không có nghĩa P0 sai.
  - r0–r2 của hai bộ đặc trưng; bộ `cub_chain_ft` có rò D9.

### Kết quả (2026-11-16; run `cub_e14_01_ft`, `cub_e14_01_ft_l25`, code `c3cb144`, commit run `f1ee1e0`; số in bằng `python lab/score.py e1401-summary`; 30/30 chuỗi qua G1–G2)

- Hiệu so với P0, trung bình ba FT seed, bộ có rò / bộ không rò (điểm):
  - M_d: P1 +0.788 / +0.763; P2 +0.751 / +0.758; P3 bằng P2.
  - S5_All: P1 +0.379 / +0.354; P2 +0.611 / +0.624; P3 bằng P2.
  - S5_New: P1 +1.006 / +1.041; P2 +1.158 / +1.123; P3 bằng P2.
  - S5_Old: P1 +0.310 / +0.279; P2 +0.551 / +0.569; P3 bằng P2.
  - S5_Disc: P1 +0.706 / +0.636; P2 +1.255 / +1.296; P3 +1.296 / +1.296.
  - S5_Base: P1, P2 0.000 / 0.000; P3 −0.032 / 0.000.
- M_d(P1) − M_d(P0) theo FT seed r0 / r1 / r2: +0.764 / +0.782 / +0.820 (có rò); +0.538 / +0.797 / +0.953 (không rò). T = 0.935: một trong sáu FT seed vượt T.
- Mức tuyệt đối của M_d, bộ không rò: P0 56.944, P1 57.707, P2 57.702, P3 57.702; bộ có rò: 57.142, 57.930, 57.893, 57.893.
- P3 khác P2 chỉ ở FT seed r1 của bộ có rò (S5_Base 78.628 so với 78.723; S5_Disc 55.904 so với 55.782).
- Không ảnh test nào được dự đoán vào khe chưa học: 0 ở cả 30 chuỗi × 5 phiên.
- Chi phí thật: 109 s và 111 s mỗi run (15 chuỗi), không phải "dưới 15 phút" như ước lượng.
- Dự đoán đã đăng ký: M_d(P1) − M_d(P0) trong 0–1.5: đúng. S5_New(P3) − S5_New(P0) trong 2–8: sai, kết quả +1.158 / +1.123. S5_Old(P3) thấp hơn S5_Old(P0): sai chiều, kết quả +0.551 / +0.569 (P3 không lấy khe lớp gốc; Old tăng vì phần Disc được ghép lại).

### Kết luận (Rito duyệt tạm 2026-11-16)

- Trên CUB t5, quy ước ghép tên đổi M_d dưới T: P1, P2, P3 cao hơn P0 từ 0.75 tới 0.79 điểm ở cả hai bộ đặc trưng. Hungarian toàn cục trùng SPE theo phiên (P3 = P2), trừ −0.032 điểm S5_Base trên bộ có rò.
- Đại lượng chính của QD28 giữ nguyên (M_d dưới P0), như đã chốt trước. Ràng buộc S5_S0test (= S5_Base) không đổi dưới P0, P1, P2.
- `09e` đặt tên bằng P1; mọi so sánh `09e` với nhánh có k-means chấm cả hai nhánh bằng P1. Lý do thêm sau khi có số: hiệu P1 − P0 của M_d (0.763 trên bộ không rò) bằng khoảng 82% của T (tính tay), nên chấm hai nhánh bằng hai thước sẽ cho `09e` gần trọn một ngưỡng T.
- Đặt cạnh bài khác: báo P3 làm cột phụ, kèm hiệu P3 − P0 (+0.75 điểm M_d, +0.62 điểm S5_All).
- D24 trên CUB: không mở câu hỏi mới về mất điểm test (0 ảnh vào khe chưa học).
- [suy luận] Lý do P3 = P2: khe lớp gốc là Gaussian khớp trên ảnh có nhãn và không được cập nhật ở các phiên sau (C5 của QD28), nên ảnh lớp mới hiếm khi dồn đủ nhiều vào một khe lớp gốc để Hungarian đổi tên khe đó. E14.01 không kiểm giả thuyết này.
- Giới hạn: như phần tiền đăng ký. "P3 = P2 trên pipeline này" không suy ra số của các bài khác so được với số của ta: các bài đó dùng đầu phân lớp huấn luyện chung, fine-tune mỗi phiên và chia dữ liệu khác.

## E14.02 — CUB t5: quên theo đợt lớp và sổ mất điểm theo phiên (QD28)

### Tiền đăng ký (2026-11-16; viết trước khi có code và trước mọi số)

- Câu hỏi:
  - (a) một đợt lớp mất bao nhiêu điểm từ phiên nó được khám phá tới phiên 5?
  - (b) điểm lớp mới của một phiên mất ở phân cụm (bước 09–10) hay ở phần sau phân cụm (bước 11–14)?
- Vì sao: thay phép tính tay "điểm mất chủ yếu ở lúc khám phá, không ở quên" (chiến lược 2026-11-16 §2.1, ngoài repo) bằng số do code in; chọn khâu mà E09.08 nhắm tới và mức ưu tiên của hướng Plasticity–Stability. Đây là bản theo phiên của sổ mất điểm mà QD28 để lại làm sau.
- Loại: phép đo mô tả trên pipeline mặc định, không có luật đạt/trượt, không thêm biến thể. Việc thay bộ phân lớp trên cả chuỗi không thuộc E14.02 (để ở thiết kế E09.08, làm nhánh đối chứng đơn giản); với lớp gốc, E06.04 đã đo.
- Dữ liệu: 30 chuỗi mặc định như E14.01. Bộ không rò `cub_chain_ft_l25` là bộ đọc chính; `cub_chain_ft` báo kèm. Tên cụm theo P0 (QD15), bảng tên của các phiên trước đóng băng. Không SVI, không rút RNG. Đọc nhãn thật của pool chỉ để chấm cột 09.
- Bảng 1, đợt lớp × phiên: đợt 0 = 100 lớp gốc, đợt c = 20 lớp của phiên c (c = 1…5). Ô (c, i), i ≥ c: độ chính xác của bộ phân lớp sau phiên i (`params1` của `S<i>_13`) trên ảnh test của đợt c; ô (0, 0) = S0_All của `metrics.json`. Quên của đợt c = ô (c, c) − ô (c, 5), c = 0…4.
- Bảng 2, sổ của lớp mới ở phiên i, cùng 20 lớp:
  - c09: trên ảnh pool lớp mới của phiên, độ chính xác của phân hoạch k-means (`ids`) sau khi ghép cụm với lớp bằng Hungarian trên chính các ảnh đó;
  - c10: trên ảnh test lớp mới, độ chính xác của k-means (`cluster.predict_raw`) với bảng tên `pi_test`;
  - c11: như c14 nhưng bộ phân lớp là `params_f` của lượt F (`S<i>_11`); phản thực tế, như cột 11 của E11.01;
  - c14: S<i>_New.
  - L_10 = c10 − c14; L_F = c10 − c11; L_S = c11 − c14.
- Bảng 3, dòng lỗi ở phiên i, bộ phân lớp cuối: ảnh test lớp mới vào đúng khe / khe khác của phiên / khe của đợt đã khám phá / khe lớp gốc; ảnh test cũ vào đúng khe / khe của phiên / khe cũ khác. Tỉ lệ trên số ảnh của tập.
- Tổng hợp: trung bình 5 chuỗi cho mỗi FT seed, rồi trung bình và SD của ba FT seed; quên và L_10, L_F, L_S báo cạnh T = 0.935 và B = 0.376 (QD28).
- Cách đọc, chốt trước:
  - quên của các đợt 1–4 (trung bình) nhỏ hơn B và c10 thấp: nút thắt ở phân cụm; E09.08 nhắm bước 09, hướng Plasticity–Stability xếp sau;
  - L_10 lớn hơn T: phần sau phân cụm làm mất điểm; bảng 3 cho biết ảnh lớp mới rơi vào khe cũ hay khe mới khác, L_F và L_S cho biết lượt nào;
  - quên của các đợt 1–4 lớn hơn B: giữ hướng Plasticity–Stability;
  - các trường hợp không loại trừ nhau; ghi mọi trường hợp xảy ra.
- Cổng kiểm:
  - G1: các ô suy ra được từ `metrics.json` trùng `metrics.json` (S<i>_All, S<i>_Old, S<i>_New, S<i>_S0test = ô (0, i)) và `acc1` của ảnh chụp, ở cả 30 chuỗi.
  - G2: số ảnh đúng của c10 bằng nghiệm tối ưu của Hungarian tính lại trên ảnh test lớp mới.
  - G3: mỗi dòng của bảng 3 cộng lại bằng 1; số ảnh của các đợt cộng lại bằng số ảnh test của phiên.
- Code (viết sau tiền đăng ký): runner trong `lab/score.py`, lệnh `e1402` qua `exp.py`, một run mỗi bộ đặc trưng (tag `cub_e14_02_ft`, `cub_e14_02_ft_l25`); lệnh `e1402-summary` chỉ đọc. Dùng lại cách nạp và dự đoán đã qua G1 của E14.01. Không sửa `lab/sessions.py`, `lab/replay.py`.
- Chi phí [suy luận]: khoảng 2–4 phút mỗi run (E14.01: 109 s và 111 s; thêm một lần dự đoán với `params_f` và một lần gán cụm trên ảnh test mỗi phiên).
- Dự đoán của Claude [SUY LUẬN], độ tin thấp: quên của các đợt 1–4 trung bình 2–3 điểm, lớn hơn B; L_10 dưới 2 điểm; ảnh test lớp mới sai chủ yếu vào khe khác của cùng phiên; c09 cao hơn c10 không quá 3 điểm. Căn cứ: E14.01 cho S5_Disc 54.03 so với M_d 56.94 trên bộ không rò (hai tập lớp khác nhau, nên chỉ là gợi ý).
- Giới hạn:
  - c10 biết trước ảnh nào là lớp mới (k-means chỉ gán ảnh test lớp mới): cận trên của khâu đặt tên, không phải một bộ phân lớp thay thế.
  - c09 đọc nhãn pool của r0–r2; các FT seed này không dùng cho kiểm định xác nhận về sau (T2 dùng FT seed mới).
  - B được đo trên 100 lớp gốc; một đợt chỉ có 20 lớp nên nhiễu lớn hơn. SD giữa FT seed của quên theo đợt được in kèm.
  - Bảng tên P0 đóng băng; E14.01 cho biết ghép lại tên (P2) thêm khoảng 1.3 điểm S5_Disc.

### Kết quả (2026-11-16; run `cub_e14_02_ft_l25`, `cub_e14_02_ft`, code `a1c1d00`, commit run `0706ec8`; số in bằng `python lab/score.py e1402-summary`; 30/30 chuỗi qua G1–G3)

Số ghi là bộ không rò / bộ có rò; trung bình ba FT seed, SD giữa FT seed trong ngoặc khi có.

- Bảng 2, trung bình 5 phiên: c09 66.547 / 66.552; c10 67.041 / 66.901; c11 61.124 / 60.996; c14 = M_d 56.944 / 57.142.
  - L_10 = +10.096 (0.306) / +9.758 (0.681); L_F = +5.917 (0.462) / +5.904 (0.807); L_S = +4.179 (0.179) / +3.854 (0.333); c09 − c10 = −0.494 (0.142) / −0.349 (0.097).
  - L_10 theo phiên 1…5, bộ không rò: 10.094, 6.474, 8.963, 11.303, 13.649. c10 theo phiên: 68.592, 62.567, 74.977, 61.056, 68.012.
- Bảng 3, trung bình 5 phiên, tỉ lệ trên ảnh test lớp mới: đúng khe 0.569 / 0.571; khe khác của phiên 0.241 / 0.242; khe của đợt đã khám phá 0.062 / 0.058; khe lớp gốc 0.127 / 0.128; khe chưa học 0.000.
  - Trên ảnh test cũ: đúng 0.740 / 0.744; khe của phiên 0.039 / 0.037; khe cũ khác 0.221 / 0.219; khe chưa học 0.000.
- Bảng 1, bộ không rò, ô (c, i) với i = c … 5:
  - đợt 0: 85.397, 82.365, 81.115, 79.989, 79.217, 78.398;
  - đợt 1: 58.498, 54.613, 54.131, 53.627, 53.016;
  - đợt 2: 56.093, 53.765, 52.572, 51.902;
  - đợt 3: 66.014, 62.960, 62.599;
  - đợt 4: 49.754, 48.615;
  - đợt 5: 54.363.
- Quên theo đợt 0…4: 6.999 (0.101), 5.481 (0.285), 4.191 (0.426), 3.415 (0.713), 1.138 (0.377) / 6.817 (0.421), 5.575 (0.108), 3.907 (0.718), 3.368 (0.496), 1.056 (0.323). Trung bình đợt 1–4: 3.556 (0.222) / 3.477 (0.217).
- Cách đọc đã chốt: quên của đợt 1–4 lớn hơn B (0.376); L_10 lớn hơn T (0.935). Trường hợp "quên nhỏ hơn B và c10 thấp" không xảy ra.
- Chi phí thật: 129 s và 139 s mỗi run.
- Dự đoán đã đăng ký: quên đợt 1–4 trong 2–3 điểm: sai cỡ (3.556), đúng ở chỗ lớn hơn B. L_10 dưới 2 điểm: sai, kết quả +10.096. Ảnh lớp mới sai chủ yếu vào khe khác của cùng phiên: đúng (0.241 so với 0.189 vào khe cũ, tính tay 0.062 + 0.127). c09 − c10 không quá 3 điểm: đúng (c09 thấp hơn c10 0.494).

### Kết luận (Rito duyệt tạm 2026-11-16)

- Trên CUB t5 không rò, điểm lớp mới mất ở ba chỗ cùng cỡ: phân cụm (c10 = 67.0), phần sau phân cụm (L_10 = 10.1 điểm, lớn hơn T; lượt F 5.9, bộ lọc cộng lượt S 4.2), và quên sau khi học (trung bình 3.6 điểm mỗi đợt, lớn hơn B). Hai bộ đặc trưng cho cùng hình dạng.
- Nhận định "trên CUB điểm mất chủ yếu ở lúc khám phá, không ở quên" (chiến lược 2026-11-16 §2.1, tính tay, ngoài repo) được thay bằng dòng trên.
- E09.08 có đích trên CUB: bước 11–13. Đại lượng đích là L_10; 10.1 điểm M_d là cận trên của phần lấy lại được, không phải mức kỳ vọng, vì c10 biết trước ảnh nào là lớp mới.
- Trong ảnh test lớp mới, 18.9% (tính tay) rơi vào khe cũ và 24.1% vào khe khác của cùng phiên: cả tách cũ/mới lẫn ghép trong phiên đều có chỗ để tăng. Bước D (thang oracle) giữ nguyên ba nhánh.
- Hướng Plasticity–Stability giữ lại, với câu hỏi đặt lại. [suy luận] Khe của lớp cũ giữ nguyên từng bit qua các phiên (C5 của QD28), nên một ảnh cũ đang đúng chỉ thành sai khi một khe mới giành lấy nó; dòng "ảnh cũ vào khe của phiên" (0.039) khớp với cách hiểu này. E14.02 chưa kiểm trực tiếp: cần tách mức giảm của từng đợt theo khe đích.
- Giới hạn thêm sau khi có số: c11 là cột duy nhất không có cổng kiểm trực tiếp, nên cách chia L_F / L_S kém chắc hơn tổng L_10; độ khó khác nhau giữa các đợt (ô đầu 49.8 tới 66.0), nên so quên giữa các đợt lẫn cả độ khó.

### Sửa đổi (2026-11-16): bảng 4 và 5 của E14.02 (viết sau Kết quả ở trên, trước khi code của hai bảng chạy trên ảnh chụp thật)

- Lý do: Kết luận ở trên nêu một suy luận chưa kiểm: quên là do khe mới giành ảnh cũ. Hai bảng này kiểm suy luận đó và cho biết khe nào giành. Chúng được thêm sau khi thấy bảng 1–3, nên là phần khám phá của E14.02; cách đọc dưới đây chốt trước khi thấy số của chính chúng.
- Dữ liệu: 30 chuỗi như trên; run mới `cub_e14_02b_ft_l25`, `cub_e14_02b_ft` (lệnh `e1402b` qua `exp.py`; `e1402b-summary` chỉ đọc). Không SVI, không rút RNG; đọc nhãn pool cho bảng 5.
- Bảng 4: ở phiên i, bộ phân lớp sau phiên i − 1 (phiên 1: `params0`) và bộ phân lớp sau phiên i dự đoán cùng các ảnh test của phiên i. Với ảnh test của đợt c < i, tính theo % số ảnh của đợt: lost_new (trước đúng, sau nằm ở một khe của phiên i), lost_other (trước đúng, sau sai ở khe khác), gained (trước sai, sau đúng). Mức giảm của ô (c, i − 1) → (c, i) ở bảng 1 bằng lost_new + lost_other − gained.
- Dự đoán có thể bị bác (suy từ C5 của QD28, từ việc dự đoán là argmax trên mọi khe, và từ kết quả 0 ảnh vào khe chưa học của E14.01): lost_other = 0 và gained = 0 ở mọi đợt, mọi phiên, mọi chuỗi. Một ô khác 0 là đủ bác câu "quên = bị khe mới giành"; khi đó phải tìm phần nào của phép dự đoán đổi theo phiên.
- Bảng 5: ở phiên i, xếp 20 khe của phiên theo tỉ lệ ảnh cũ trong cụm k-means của khe trên pool (nhãn pool chỉ để chấm), và theo tỉ lệ ảnh cũ trong số ảnh của cụm được bộ lọc giữ lại. Đại lượng: phần ảnh test cũ bị các khe của phiên lấy rơi vào k khe đứng đầu, k = 1, 2, 5, 10; mức đều là k / 20. Kèm: tỉ lệ ảnh cũ trong pool và trong số ảnh được giữ; phần ảnh pool cũ và phần ảnh pool mới được bộ lọc giữ.
- Cách đọc, chốt trước: phần rơi vào 5 khe đứng đầu từ 0.50 trở lên (gấp đôi mức đều 0.25): quên tập trung ở các khe có cụm nhiều ảnh cũ, phù hợp với nhánh "bỏ ảnh cũ trước bước 09" của thang oracle; dưới 0.35: quên rải đều, cụm lẫn ảnh cũ không phải nguồn chính. Bảng 5 là tương quan; phép can thiệp là thang oracle.
- Cổng kiểm: G1 (All của bộ phân lớp sau phiên i trùng `acc1`); G4 (bộ phân lớp trước, chấm trên ảnh của phiên i, cho đúng điểm của nó trên tập test của chính phiên trước, từng đợt; phiên 1 so với `acc0`). Dự đoán lost_other = gained = 0 là kết quả, không phải cổng.
- Dự đoán của Claude [SUY LUẬN]: lost_other = gained = 0 (độ tin cao); phần rơi vào 5 khe đứng đầu trong 0.50–0.70 (độ tin thấp); bộ lọc giữ 10–25% ảnh pool cũ (độ tin thấp).
- Giới hạn: ngưỡng 0.50 và 0.35 do Claude chọn; bảng in cả k = 1, 2, 5, 10 để kết quả không phụ thuộc một ngưỡng.

### Kết quả của Sửa đổi (2026-11-16; run `cub_e14_02b_ft_l25`, `cub_e14_02b_ft`, code `dd876fc`, commit run `ea43b5e`; số in bằng `python lab/score.py e1402b-summary`; 30/30 chuỗi qua G1, G4)

Số ghi là bộ không rò / bộ có rò; trung bình ba FT seed, SD giữa FT seed trong ngoặc khi có.

- Bảng 4: `pure` = 1 ở 30/30 chuỗi; tổng mọi ô lost_other và gained bằng 0.0000 ở cả hai bộ.
  - lost_new, bộ không rò, đợt c ở phiên i (% số ảnh của đợt): đợt 0, phiên 1…5: 3.032, 1.250, 1.127, 0.772, 0.819; đợt 1, phiên 2…5: 3.885, 0.481, 0.505, 0.610; đợt 2, phiên 3…5: 2.328, 1.193, 0.670; đợt 3, phiên 4…5: 3.054, 0.361; đợt 4, phiên 5: 1.139.
  - Đợt 1–4, phiên ngay sau khi học: 2.601 (0.116) / 2.499 (0.048); các phiên sau đó: 0.637 (0.080) / 0.652 (0.141). Đợt 0, trung bình mỗi phiên: 1.400 (0.020) / 1.363 (0.084).
- Bảng 5, trung bình 5 phiên: phần ảnh test cũ bị các khe của phiên lấy rơi vào k khe có cụm nhiều ảnh cũ nhất: k = 1: 0.010 / 0.009; k = 2: 0.019 / 0.018; k = 5: 0.103 (0.014) / 0.098 (0.005); k = 10: 0.383 / 0.371. Mức đều: 0.05, 0.10, 0.25, 0.50. Xếp theo tỉ lệ ảnh cũ trong số ảnh được giữ, k = 5: 0.103 / 0.099.
  - Theo phiên 1…5, k = 5, bộ không rò: 0.120, 0.078, 0.234, 0.043, 0.043.
  - Ảnh test cũ nằm trong khe của phiên (% số ảnh test cũ), phiên 1…5: 6.120, 4.453, 3.252, 3.209, 2.233.
- Ngữ cảnh, bộ không rò, phiên 1…5:
  - tỉ lệ ảnh cũ trong pool: 0.498, 0.543, 0.582, 0.614, 0.641 (theo giao thức t5: 500 ảnh mới và 5 ảnh mỗi lớp cũ, tính tay 500/1000 … 900/1400);
  - phần ảnh pool cũ được bộ lọc giữ: 0.873, 0.611, 0.413, 0.338, 0.273; trung bình 0.502 (0.015) / 0.498 (0.001);
  - phần ảnh pool mới được giữ: 1.000, 1.000, 1.000, 0.998, 0.999;
  - tỉ lệ ảnh cũ trong số ảnh được giữ: 0.464, 0.420, 0.363, 0.349, 0.328.
- Cách đọc đã chốt: k = 5 cho 0.103, dưới 0.35: cụm lẫn ảnh cũ không phải nguồn chính của quên. Kết quả còn thấp hơn mức đều 0.25, một trường hợp mà cách đọc không lường.
- Chi phí thật: 132 s và 143 s mỗi run.
- Dự đoán đã ghi: lost_other = gained = 0: đúng. Phần rơi vào 5 khe đứng đầu trong 0.50–0.70: sai, ngược chiều (0.103). Bộ lọc giữ 10–25% ảnh pool cũ: sai (trung bình 50%, 87% ở phiên 1).

### Kết luận của Sửa đổi (Rito duyệt 2026-11-16)

- Quên trên CUB t5 là ảnh cũ bị khe của phiên mới giành, ở 30/30 chuỗi; không có trôi tham số (đúng với C5 của QD28). Câu [suy luận] trong Kết luận ở trên nay đã được kiểm.
- Mức giành tập trung ở phiên ngay sau khi một đợt được học (2.60% số ảnh, so với 0.64% ở mỗi phiên sau).
- Các khe có cụm nhiều ảnh cũ lấy ít ảnh test cũ hơn mức đều (5 khe đứng đầu: 0.103 so với 0.25). Giả thuyết "cụm lẫn ảnh cũ gây quên" bị bác.
- Hướng Plasticity–Stability, theo nghĩa bảo vệ tham số cũ, không có đối tượng trong pipeline này; nó nhập vào bài toán tách cũ/mới của E09.08 (cân mật độ giữa thành phần cũ và mới).
- Số cho thang oracle (bước D): pool có 50–64% ảnh cũ; bộ lọc giữ gần như mọi ảnh mới (0.999) và trung bình 50% ảnh cũ (87% ở phiên 1), nên lượt S khớp khe mới trên một tập có 33–46% ảnh cũ.
- [suy luận, sau khi thấy số] Khe khớp trên cụm gọn của một lớp mới là Gaussian sắc và lấy các ảnh cũ giống lớp đó; cụm gồm ảnh cũ của nhiều lớp thì loãng. Nếu đúng, bỏ ảnh cũ trước bước 09 không giảm quên. Chưa kiểm; nhánh oracle của bước D là phép can thiệp.
- Chưa giải thích: phần ảnh pool cũ lọt qua bộ lọc giảm đều từ 0.873 (phiên 1) xuống 0.273 (phiên 5).
- Giới hạn: hai bảng là phần khám phá, thêm sau khi thấy bảng 1–3; bảng 5 là tương quan trên 20 khe mỗi phiên; r0–r2.

## E14.03 — CUB t5: L_10 mất bao nhiêu vì khe cũ giành ảnh test lớp mới, và khe cũ thắng nhờ số hạng nào? (QD28)

### Tiền đăng ký (2026-11-17; viết trước khi có code và trước mọi số)

- Câu hỏi:
  - (a) trong L_10 = c10 − c14 của E14.02, bao nhiêu điểm mất vì ảnh test lớp mới bị khe cũ giành lúc dự đoán (khe lớp gốc, khe của các đợt trước)?
  - (b) với các ảnh bị giành, khe cũ thắng nhờ số hạng log det hay số hạng Mahalanobis của log-mật độ?
  - (c) phụ: pipeline gán theo tâm trên cả chuỗi cho bao nhiêu điểm; điểm hiện tại cách đường biên đánh đổi lớp mới / lớp cũ theo độ lệch b bao xa?
- Vì sao: E14.02 chốt đích của E09.08 là bước 11–13, nhưng đi từ c10 sang c14 đổi hai thứ cùng lúc: bộ phân lớp, và tập khe được xét (từ 20 khe của phiên sang mọi khe). Ba nhánh của thang oracle (bước D) chỉ làm sạch dữ liệu huấn luyện, không chặn khe cũ giành ảnh test; nếu phần (a) lớn thì mức nâng nhỏ ở D không có nghĩa là cơ chế cũ/mới không có đích. Phần (b) nối với luận điểm của paper (lệch hiệp phương sai, Eq. 9–11) và với luật dừng theo định thức (`classifier/mngmm.py` `calculate_metrics_on_covariances`).
- Loại: phép đo mô tả; phần A có luật đọc chốt trước, phần B, C, D chỉ báo cáo.
- Dữ liệu: 15 chuỗi `cub_chain_ft_l25_r<r>_s<N>` (bộ đọc chính) và 15 chuỗi `cub_chain_ft_r<r>_s<N>` (báo kèm). Ở mỗi phiên i = 1…5: `params1` của `S<i>_13`, log-mật độ của `_predict` (thành phần dự đoán mặc định) trên tập test của phiên; tên theo P0 (QD15), bảng tên đóng băng. Không SVI, không rút RNG. Nhãn thật chỉ để chấm, và để lấy tâm lớp gốc từ `s0_x`, `s0_y` (S0 có nhãn theo giao thức).
- Phần A, can thiệp sạch: cùng `params1`, argmax trên các tập khe khác nhau, chấm trên ảnh test lớp mới (off = 100 + 20 (i − 1)):
  - c14: mọi khe (= S<i>_New, như pipeline);
  - g_nobase: bỏ khe 0…99;
  - g_nodisc: bỏ khe 100 … off − 1 (ở phiên 1 trùng c14);
  - g20: chỉ khe off … off + 19.
  - Đại lượng chính: L_comp = g20 − c14, trung bình 5 phiên. Báo kèm L_base = g_nobase − c14, L_disc = g_nodisc − c14 (không cộng lại thành L_comp) và L_10 = c10 − c14 tính lại trong cùng run.
  - Biết trước, không phải kết quả: L_comp ≥ 0 (mặt nạ chỉ đổi dự đoán của ảnh đang nằm ngoài khe của phiên, mà với ảnh lớp mới các ảnh đó đều sai) và L_comp ≤ 100 × (new_disc + new_base) của bảng 3 E14.02 (trung bình 18.9 điểm, tính tay).
- Phần B, cơ chế (chỉ báo cáo): với mỗi ảnh test lớp mới mà khe dự đoán o là khe cũ (o < off), n = khe đúng: Δ_det = −½ (log det Σ_o − log det Σ_n), Δ_maha = −½ (d²_o − d²_n), tính bằng numpy float64 (Cholesky) trong không gian PCA; Δ_det + Δ_maha = log p_o − log p_n. In theo nhóm khe o (lớp gốc, đợt trước): số ảnh, trung vị Δ_det, trung vị Δ_maha, tỉ lệ ảnh có Δ_det > 0, tỉ lệ ảnh có |Δ_det| > |Δ_maha|; kèm trung bình log det Σ của khe lớp gốc, khe các đợt trước và khe của phiên.
- Phần C, pipeline gán theo tâm (chỉ báo cáo): tâm lớp gốc = trung bình `s0_x` theo lớp (đặc trưng thô); tâm của khe off_j + c = `cluster_centers_` của mô hình k-means ở `S<j>_09`. Ở phiên i, mỗi ảnh test vào khe có tâm gần nhất (Euclid, đặc trưng thô) trong các khe đã gặp. In S<i>_All, S<i>_Old, S<i>_New, S<i>_S0test, M_d, M_f proxy, quên theo đợt như bảng 1 của E14.02. Ghi rõ: so với c14, phần C đổi cùng lúc không gian (thô 768 so với PCA-384), cách ước lượng (tâm k-means so với SVI) và dữ liệu huấn luyện (cả pool so với ảnh được giữ); không dùng trong luật đọc.
- Phần D, đường cong đánh đổi (chỉ báo cáo; không phải phương pháp): cộng độ lệch b vào log-mật độ của 20 khe của phiên rồi lấy argmax trên mọi khe. Lưới b của phiên i: b = 0 và b = −Q_q(m) với q ∈ {0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98}, m = (log-mật độ lớn nhất trên 20 khe của phiên) − (lớn nhất trên các khe cũ), tính trên mọi ảnh test của phiên, không dùng nhãn. In (S<i>_New, S<i>_Old) theo b; mức tăng S<i>_New nội suy tại mức giảm S<i>_Old bằng B (0.376) và bằng 1 điểm; trung bình 5 phiên. b không được chọn: không có b nào được báo như kết quả của một phương pháp.
- Tổng hợp: trung bình 5 chuỗi cho mỗi FT seed, rồi trung bình và SD của ba FT seed (như E14.02); in kèm 15 giá trị theo chuỗi của L_comp.
- Cách đọc phần A, chốt trước:
  - L_comp > T (0.935) và L_comp ≥ ½ L_10: đích chính của E09.08 là cạnh tranh cũ/mới lúc dự đoán (cân thành phần mới với thành phần cũ cố định), không phải chất lượng ước lượng Gaussian trong phiên; kết luận E14.02 "đích là bước 11–13" ghi lại theo đó.
  - L_comp ≤ T: đích là ước lượng ở bước 11–13, như E14.02.
  - Còn lại (L_comp > T nhưng dưới ½ L_10): cả hai.
  - Ngưỡng ½: phần lớn hơn là ưu tiên; không có căn cứ cho ngưỡng chặt hơn.
  - g20 biết trước ảnh test nào là lớp mới: L_comp là lợi ích của một bộ định tuyến cũ/mới hoàn hảo lúc test, không phải mức đạt được. Phần D cho biết lấy lại một phần của nó tốn bao nhiêu điểm lớp cũ.
- Hệ quả cho bước D: g20 là nhánh thứ tư của thang oracle (bộ định tuyến lúc test), có từ E14.03 mà không cần chạy thêm; luật ba vùng của D đọc kèm L_comp.
- Cổng kiểm:
  - G1: c14 và các cột lớp cũ tính lại trùng `acc1` của `S<i>_14` (1e-6) và `metrics.json` (0.006); c10 tính lại trùng `S<i>_c10` của run E14.02 cùng bộ.
  - G2: ở mỗi phiên, g20 − c14 ≤ 100 × (new_disc + new_base) của run E14.02 (sai số 1e-6).
  - G3: mặt nạ mở ra mọi khe cho lại đúng dự đoán của c14 ở mọi ảnh.
  - G4: ở mỗi ảnh của phần B, |Δ_det + Δ_maha − (log p_o − log p_n)| ≤ 1e-3 × max(1, |log p_o|), với log p của `_predict`.
  - G5: tâm của phần C chỉ trong 20 khe của phiên cho cùng cụm với `cluster.predict_raw` ở mọi ảnh test lớp mới; `s0_y` có đúng 25 ảnh mỗi lớp 0…99.
- Code (viết sau tiền đăng ký): runner `tier2_e1403` trong `lab/score.py`, lệnh `e1403` qua `exp.py`, một run mỗi bộ đặc trưng (tag `cub_e14_03_ft_l25`, `cub_e14_03_ft`); lệnh `e1403-summary` chỉ đọc. Dùng lại cách nạp và dự đoán đã qua G1 của E14.01, E14.02. Không sửa `lab/sessions.py`, `lab/replay.py`.
- Chi phí [suy luận]: khoảng 3–4 phút mỗi run (E14.02: 129 s và 139 s; thêm log det và Cholesky của các khe, và phần C).
- Dự đoán của Claude [SUY LUẬN], độ tin thấp:
  - L_comp trong 7–11 điểm, tức hơn nửa L_10; L_base khoảng gấp đôi L_disc (12.7% so với 6.2% ảnh test lớp mới, bảng 3 E14.02).
  - Phần B: Δ_det > 0 và |Δ_det| > |Δ_maha| ở hơn nửa số ảnh bị giành: khe cũ thắng nhờ định thức nhỏ hơn. Căn cứ: Gaussian gần khởi tạo (D23), nên số hạng Mahalanobis giữa các khe khác nhau ít; luật dừng cho khe mới dừng khi định thức trung bình vượt hệ số dừng nhân với định thức trung bình của khe cũ, hệ số của lượt F tăng từ 1.2 lên 2.0 qua 5 phiên (tính tay).
  - Phần C: M_d của pipeline theo tâm thấp hơn c14.
  - Phần D: tại mức giảm lớp cũ bằng B, S<i>_New tăng 2–5 điểm.
- Giới hạn:
  - Phần B chỉ xét các ảnh đã bị giành (chọn theo kết quả), nên mô tả cơ chế của lỗi, không phải của mọi quyết định.
  - Phần C kế thừa tâm cụm khớp trên pool có 50–64% ảnh cũ; tâm lớp gốc từ 25 ảnh trong 768 chiều; là một mốc đơn giản, không phải bộ phân lớp tốt nhất dựa trên tâm.
  - Phần D lấy lưới b từ phân phối của m trên ảnh test (không nhãn); đường cong là mô tả trên tập test.
  - r0–r2 đã dùng ở E14.01, E14.02; không phải kiểm định xác nhận.

### Kết quả (2026-11-17; run `cub_e14_03_ft_l25`, `cub_e14_03_ft`, code `c7359f8`, commit run `6e7920f`; số in bằng `python lab/score.py e1403-summary`; 30/30 chuỗi qua G1–G5)

Số ghi là bộ không rò / bộ có rò; trung bình ba FT seed, SD giữa FT seed trong ngoặc khi có.

- Phần A, trung bình 5 phiên, ảnh test lớp mới: c10 67.041 / 66.901; g20 67.121 / 67.002; g_nobase 62.970 / 63.195; g_nodisc 58.733 / 58.704; c14 56.944 / 57.142.
  - L_comp = +10.177 (0.372) / +9.860 (0.500); L_10 = +10.096 (0.306) / +9.758 (0.681), trùng E14.02; L_base = +6.026 (0.315) / +6.053 (0.498); L_disc = +1.788 (0.157) / +1.562 (0.085).
  - L_comp theo chuỗi, bộ không rò (r0 s0…s4 | r1 | r2): 9.79 10.38 9.41 9.84 9.31 | 11.01 9.99 9.91 9.90 11.13 | 10.09 10.92 10.60 10.06 10.32; bộ có rò: 8.81 9.59 9.55 9.37 10.04 | 9.61 9.70 9.37 10.10 9.65 | 11.02 10.55 9.88 10.38 10.29.
  - c10 − g20 = −0.080 / −0.101 (tính tay): khi chỉ xét 20 khe của phiên, bộ Gaussian cho điểm ngang phép gán k-means.
  - L_comp bằng khoảng 54% (tính tay, 10.177 / 18.9) phần ảnh test lớp mới rơi vào khe cũ ở bảng 3 E14.02; phần còn lại, khi bị chặn khỏi khe cũ, rơi vào khe sai của phiên.
- Luật đọc: L_comp 10.177 > T (0.935) và ≥ ½ L_10 (5.048); bộ có rò 9.860 ≥ 4.879. Vùng 1: đích chính của E09.08 là cạnh tranh cũ/mới lúc dự đoán.
- Phần B, gộp 5 phiên mỗi chuỗi, trung bình ba FT seed; khe lớp gốc | khe đợt trước:
  - số ảnh mỗi chuỗi 364.0 | 177.5 (bộ có rò 366.8 | 165.9);
  - trung vị Δ_det 0.602 | 0.460 (0.622 | 0.445); trung vị Δ_maha 4.347 | 5.609 (4.375 | 6.364);
  - tỉ lệ ảnh có Δ_det > 0: 0.962 | 0.868 (0.968 | 0.867); tỉ lệ ảnh có |Δ_det| > |Δ_maha|: 0.139 | 0.088 (0.135 | 0.073).
  - log det trung bình của khe, bộ không rò, S1…S5: khe lớp gốc −10.036 ở mọi phiên; khe của phiên −9.283, −9.005, −8.696, −8.498, −8.146; khe đợt trước từ −9.283 (S2) tới −8.870 (S5).
- Phần C: M_d theo tâm 53.887 (0.474) / 53.600 (0.659), dưới c14; S5_All 61.170 / 61.620; S5_Old 62.531 / 62.908; S5_S0test 73.292 / 74.109; M_f proxy 10.513 / 10.333; quên đợt 1–4 6.372 / 5.912 (pipeline, E14.02: 3.556 / 3.477).
- Phần D: mức tăng S<i>_New tại mức giảm S<i>_Old bằng B: +0.874 (0.050) / +0.843 (0.128); tại mức giảm 1 điểm: +2.098 (0.116) / +2.016 (0.219). Lưới b chạm cả hai mức giảm ở 75/75 phiên.
- Dự đoán đã đăng ký:
  - L_comp trong 7–11 điểm, hơn nửa L_10: đúng (10.177), nhưng đã thấy số trước khi đọc bảng (xem dưới).
  - L_base khoảng gấp đôi L_disc: sai cỡ, gấp 3.37 lần (tính tay, 6.026 / 1.788).
  - Phần B, |Δ_det| > |Δ_maha| ở hơn nửa số ảnh bị giành: sai; chỉ 13.9% và 8.8%. Đúng dấu: Δ_det > 0 ở 96.2% và 86.8%.
  - Phần C, M_d theo tâm thấp hơn c14: đúng (53.887 so với 56.944).
  - Phần D, tăng 2–5 điểm tại mức giảm B: sai, +0.874.
- Lệch so với tiền đăng ký và về quy trình:
  - G2 dùng sai số 1e-6 + 100 × 1e-4 điểm thay cho 1e-6, vì `new_disc`, `new_base` của run E14.02 lưu làm tròn 4 chữ số. Chặn trên của G2 chỉ đúng khi không ảnh nào vào khe chưa học; điều này đã đo (new_unseen = 0 ở 30/30 chuỗi, E14.02).
  - Kết quả lộ trước khi đọc bảng: dòng tiến độ của runner in L_comp, L_10, M_d theo tâm và mức tăng tại B của từng chuỗi. Rito thấy chuỗi r0 s0 ở lần kiểm cổng và cả 30 chuỗi trước khi chạy `e1403-summary`. Phần B là phần duy nhất còn mù. Luật đọc phần A không đổi.
  - Chi phí: run `cub_e14_03_ft` 5.6 phút. Run `cub_e14_03_ft_l25` ghi 117.2 phút, nhưng khoảng giữa hai chuỗi liền nhau thất thường, từ 8 s tới 1248 s (tính tay từ dòng tiến độ) [suy luận: máy ngủ trong đêm]. Runner không dùng RNG hay SVI, nên số không phụ thuộc thời gian chạy.

### Kết luận (Rito duyệt tạm 2026-11-17)

- Trên CUB t5 không rò, gần như toàn bộ L_10 (khoảng 10 điểm M_d) nằm ở ranh giới cũ/mới lúc dự đoán: khi chỉ xét 20 khe của phiên, bộ Gaussian của bước 11–13 cho điểm ngang phép gán k-means (c10 − g20 = −0.08, tính tay). Hai bộ đặc trưng cho cùng hình dạng.
- Kết luận E14.02 "E09.08 có đích trên CUB: bước 11–13" sửa thành: đích là cạnh tranh cũ/mới lúc dự đoán; bước 11–13 chỉ liên quan qua chỗ đặt khe mới so với khe cũ. 10.2 điểm là lợi ích của bộ định tuyến cũ/mới hoàn hảo lúc test, không phải mức kỳ vọng.
- Khe cũ thắng nhờ số hạng Mahalanobis, không nhờ định thức: định thức của các khe gần như cùng cỡ (khe của phiên S5 hơn khe lớp gốc 1.89 nat trên 384 chiều, tính tay). Phép tách chưa phân biệt vị trí mean với hình dạng Σ, vì d² chứa cả Σ⁻¹.
- Phần chú thích của vùng 1 trong luật đọc ("cân thành phần mới với thành phần cũ cố định") không được ủng hộ: một độ lệch đều cho khe của phiên chỉ lấy lại +0.874 điểm lớp mới tại mức giảm lớp cũ bằng B, +2.098 tại mức giảm 1 điểm. [suy luận] Tính theo số ảnh, độ lệch đều đổi nhiều ảnh cũ lấy ít ảnh mới, vì lớp cũ đông hơn lớp mới 5 lần trở lên.
- Câu hỏi mở cho bước kế: ảnh lớp mới gần khe cũ hơn khe đúng vì mean khe mới đặt lệch (ước lượng, ví dụ ảnh cũ lẫn vào pool) hay vì lớp mới và lớp cũ chồng lấn thật trong không gian đặc trưng (biểu diễn)? Phép đo kế, làm trước thang oracle: L_comp khi thay mean khe mới bằng mean thật của lớp từ ảnh pool có nhãn, và khi tính d² với một Σ chung. Thiết kế E09.08 chờ phép đo này.
- Giới hạn thêm sau khi có số: phần A, C, D lộ trước khi đọc bảng (chỉ phần B còn mù); phần B chỉ xét ảnh đã bị giành; r0–r2 không phải kiểm định xác nhận.

## E14.05 — CUB t5: khe cũ giành ảnh test lớp mới vì mean khe mới đặt lệch hay vì lớp mới và lớp cũ chồng lấn? (QD28)

### Tiền đăng ký (2026-11-17; viết trước khi có code và trước mọi số của thí nghiệm)

- Câu hỏi: trong L_comp của E14.03 (10.18 điểm, bộ không rò), phần nào lấy lại được khi chỉ đổi mean của 20 khe của phiên, và nguồn lệch nằm ở bộ ước lượng (SVI của bước 13), ở ảnh cũ lẫn vào cụm, hay ở cụm sai lớp? Nếu không nấc nào lấy lại được phần lớn: lớp mới và lớp cũ chồng lấn trong không gian đặc trưng (giới hạn của biểu diễn).
- Vì sao: Kết luận E14.03 để mở câu hỏi này và đặt nó trước thang oracle; phần B của E14.03 cho thấy khe cũ thắng nhờ số hạng Mahalanobis, nhưng chưa tách mean với hình dạng Σ. Bước 13 (`lab/replay.py` `s13_gauss_s`) chạy SVI trên ảnh được giữ với nhãn là id cụm (`ids + off`), khởi động lại từ Gaussian của S0; D23 ghi tham số gần như không rời khởi tạo (đo trên GC10).
- Loại: phép đo mô tả có can thiệp; một luật đọc chốt trước trên R(M).
- Dữ liệu đầu vào đã xem trước khi viết (đếm, không phải kết quả; lệnh chỉ đọc 2026-11-17 trên `cub_chain_ft_l25_r0_s0…s4`): mỗi lớp mới có 25 ảnh pool ở mọi phiên; bộ lọc giữ 24–25 trong 25 ảnh của mỗi lớp mới; ảnh cũ được giữ 231–440 mỗi phiên, giảm từ S1 tới S5; ảnh test mỗi lớp mới từ 11 tới 30 (trung vị 30).
- Dữ liệu: 15 chuỗi `cub_chain_ft_l25_r<r>_s<N>` (bộ đọc chính) và 15 chuỗi `cub_chain_ft_r<r>_s<N>` (báo kèm). Ở mỗi phiên i = 1…5: `params1` của `S<i>_13`; log-mật độ tính bằng `_predict` (MVN, không trọng số) với tham số đã thay; không gian PCA của S0; tập test của phiên; tên theo P0 (QD15), `lut` cộng dồn. Không SVI, không rút RNG. Nhãn pool chỉ dùng ở nấc Mc, M1, M3; nhãn test chỉ để chấm.
- Can thiệp: chỉ thay mean của khe off + c, c = 0…19 (off = 100 + 20 (i − 1)); khe cũ giữ nguyên.
  - M0: mean của `params1` (= pipeline; c14 như E14.03).
  - Mk: trung bình `z_pool` của các ảnh được giữ (`keep`) có `ids` = c (đúng tập ảnh và nhãn mà bước 13 dùng); không dùng nhãn.
  - Mc: như Mk nhưng bỏ các ảnh có `true_pool_y` < off.
  - M1: trung bình `z_pool` của mọi ảnh pool thuộc lớp y với `lut[y]` = off + c (25 ảnh, kể cả ảnh bộ lọc bỏ).
  - Cụm không còn ảnh ở Mk hoặc Mc: giữ mean M0 cho khe đó và đếm số khe như vậy.
- Nhánh Σ chung (chỉ báo cáo): Σ chung = trung bình `class_covs` của khe 0…99 trong `params1` (không dùng thêm nhãn); M2 = mean M0 với Σ chung cho mọi khe, M3 = mean M1 với Σ chung cho mọi khe.
- Đại lượng ở mỗi nấc M: New(M) = điểm ảnh test lớp mới khi argmax trên mọi khe (c14 của nấc), Old(M), g20(M) (argmax chỉ trên 20 khe của phiên), L_comp(M) = g20(M) − New(M). Trung bình 5 phiên mỗi chuỗi, rồi trung bình 5 chuỗi mỗi FT seed, rồi trung bình và SD của ba FT seed (như E14.03).
- Đại lượng chính: R(M) = [New(M) − New(M0)] / L_comp(M0), tính trên trung bình 5 phiên của mỗi FT seed rồi lấy trung bình ba FT seed, với M ∈ {Mk, Mc, M1}. R có thể vượt 1 (đổi mean cũng có thể sửa lỗi ghép trong phiên) hoặc âm.
- Báo kèm: ΔOld(M) = Old(M) − Old(M0); khoảng cách Mahalanobis d(μ_M0, μ_Mk) của mỗi khe mới theo Σ của khe đó, trung vị theo phiên, kèm trung vị khoảng cách từ μ_Mk tới mean khe cũ gần nhất (thang so); số ảnh nhỏ nhất của một cụm trong tập giữ; số khe rỗng ở Mk, Mc.
- Cách đọc, chốt trước (bộ không rò; bộ có rò báo kèm):
  - R(Mk) ≥ ½: nguồn chính là bộ ước lượng của bước 13; E09.08 nhắm vào cách ước lượng mean khe mới (mean mẫu, hoặc hậu nghiệm VB thật theo Eq. 7). Vì Mk không dùng nhãn, đây là ứng viên phương pháp, cần kiểm lại trên FT seed khác (r3, r4) hoặc trên server trước khi báo như kết quả.
  - R(Mk) < ½ và R(M1) ≥ ½: nguồn chính là dữ liệu đưa vào bước 13 (ảnh cũ lẫn vào cụm hoặc cụm sai lớp); thang oracle của bước D có đích; R(Mc) − R(Mk) là phần do ảnh cũ lẫn vào cụm, R(M1) − R(Mc) là phần do cụm sai lớp.
  - R(M1) < ½: lớp mới và lớp cũ chồng lấn trong không gian đặc trưng; trần của mọi biến thể bước 09–13 thấp, kể cả E09.08 với VB-GMM; hướng tiếp theo phải đụng tới biểu diễn hoặc tới cách định tuyến cũ/mới.
  - Ngưỡng ½ theo cùng lý do như E14.03: phần lớn hơn là ưu tiên.
  - Mọi kết luận đọc kèm ΔOld: một nấc lấy lại lớp mới bằng cách làm mất nhiều hơn B (0.376) điểm lớp cũ được ghi rõ là đánh đổi.
- Cổng kiểm:
  - G1: nấc M0 (tham số thay bằng chính nó, qua cùng đường tính) cho New, Old trùng `acc1` của `S<i>_14` (1e-6) và trùng `S<i>_c14`, `S<i>_g20` của run E14.03 cùng bộ (sai số 1e-4, vì E14.03 lưu số làm tròn 4 chữ số).
  - G2: tổng số ảnh dùng ở Mk qua 20 cụm bằng `keep.sum()`; mỗi lớp mới có đúng 25 ảnh pool; `lut` là song ánh từ 20 lớp mới lên 20 khe của phiên.
  - G3: với cụm không có ảnh cũ, mean Mc trùng mean Mk từng bit; tập ảnh của Mc là tập con của Mk.
  - G4: Σ chung xác định dương (Cholesky không lỗi).
  - Runner chỉ in trạng thái cổng và thời gian, không in số của từng chuỗi (bài học của E14.03).
- Code (viết sau tiền đăng ký): runner `tier2_e1405` trong `lab/score.py`, lệnh `e1405` qua `exp.py`, một run mỗi bộ đặc trưng (tag `cub_e14_05_ft_l25`, `cub_e14_05_ft`); lệnh `e1405-summary` chỉ đọc. Dùng lại cách nạp và dự đoán đã qua G1 của E14.03. Không sửa `lab/sessions.py`, `lab/replay.py`.
- Chi phí [suy luận]: 5–8 phút mỗi run (E14.03: 5.6 phút; sáu lần dự đoán mỗi phiên).
- Dự đoán của Claude [SUY LUẬN], độ tin thấp:
  - R(Mk) trong 0.1–0.4: mean SVI lệch khỏi mean mẫu chưa đủ để giải thích phần lớn L_comp, vì g20 ≈ c10 ở E14.03 cho thấy các khe mới đặt đúng tương đối với nhau.
  - R(Mc) − R(Mk) trong 0.1–0.2: 33–47% ảnh được giữ là ảnh cũ (tính tay từ số đếm trên).
  - R(M1) trong 0.4–0.7, tức gần ngưỡng; ΔOld(M1) trong khoảng −1 tới 0 điểm.
  - d(μ_M0, μ_Mk) nhỏ hơn ⅓ khoảng cách từ μ_Mk tới mean khe cũ gần nhất.
- Giới hạn:
  - Mean thật từ 25 ảnh trong 384 chiều; Σ của khe giữ nguyên khi đổi mean, nên Σ không còn khớp với mean mới (trừ nhánh Σ chung).
  - Mc, M1, M3 dùng nhãn pool: là trần, không phải phương pháp; không đụng tập test.
  - Chỉ đổi khe của phiên đang xét; khe đợt trước giữ mean của pipeline, nên ảnh bị khe đợt trước giành không được sửa ở phiên sau.
  - r0–r2 đã dùng ở E14.01–E14.03; không phải kiểm định xác nhận.

### Kết quả (2026-11-17; run `cub_e14_05_ft_l25`, `cub_e14_05_ft`, code `0777048`, commit run `7f90f57`; số in bằng `python lab/score.py e1405-summary`; 30/30 chuỗi qua G1–G4)

Số ghi là bộ không rò / bộ có rò; trung bình ba FT seed, SD giữa FT seed trong ngoặc khi có.

- Theo nấc, trung bình 5 phiên và 5 chuỗi (New, Old, L_comp):
  - M0: 56.944 / 57.142; 74.046 / 74.427; 10.177 / 9.860 (trùng E14.03);
  - Mk: 66.960 / 66.998; 22.098 / 22.078; 0.536 / 0.508;
  - Mc: 67.354 / 67.271; 39.518 / 40.121; 0.579 / 0.571;
  - M1: 91.216 / 91.662; 43.588 / 44.209; 1.322 / 1.256;
  - M2: 47.032 / 47.194; 72.992 / 73.340; 19.802 / 19.517;
  - M3: 92.199 / 92.406; 29.681 / 30.143; 1.074 / 1.111.
- R(M) và ΔOld(M):
  - Mk: R 0.984 (0.007) / 0.999 (0.036); ΔOld −51.948 (0.480) / −52.348 (0.619);
  - Mc: R 1.023 (0.010) / 1.027 (0.033); ΔOld −34.528 / −34.305;
  - M1: R 3.372 (0.158) / 3.506 (0.170); ΔOld −30.458 / −30.217;
  - M2: R −0.975 / −1.009; ΔOld −1.054 / −1.086;
  - M3: R 3.468 / 3.582; ΔOld −44.365 / −44.284.
- Khoảng cách, trung vị trên 20 khe rồi trung bình: d(μ_M0, μ_Mk) 9.142 / 9.203; d(μ_Mk, mean khe cũ gần nhất) 9.156 / 9.215; tỉ lệ 0.998 / 0.999.
- Cụm giữ nhỏ nhất 5 / 4 ảnh; khe rỗng ở Mk 0 / 0, ở Mc 70 / 87 trên 1500 (cụm chỉ có ảnh cũ).
- Luật đọc: R(Mk) 0.984 ≥ ½, nhánh "bộ ước lượng của bước 13 là nguồn chính". Điều khoản ΔOld của tiền đăng ký áp vào: ΔOld(Mk) = −51.95 điểm, lớn hơn B (0.376) hơn 100 lần, nên mức tăng lớp mới ở mọi nấc Mk, Mc, M1, M3 là đánh đổi, không phải phần L_comp lấy lại được; câu "Mk là ứng viên phương pháp" của luật đọc không đứng.
- Dự đoán đã đăng ký, cả năm sai: R(Mk) 0.1–0.4 (kết quả 0.984); R(Mc) − R(Mk) 0.1–0.2 (0.039, tính tay); R(M1) 0.4–0.7 (3.372); ΔOld(M1) từ −1 tới 0 (−30.458); tỉ lệ khoảng cách dưới ⅓ (0.998).
- Lệch so với tiền đăng ký: runner cắt tham số về các khe đã gặp (off + 20) và lấy argmax trên các khe này, để khe chưa học (mean 0, Σ = I) không vào nhánh Σ chung; ở M0 điều này không đổi dự đoán (G1 qua ở 30/30 chuỗi, so với `acc1` và với E14.03).
- Phân tích sau khi thấy số (không thuộc tiền đăng ký; một lệnh chỉ đọc ngày 2026-11-17, chưa đưa vào repo, trên `cub_chain_ft_l25_r0_s0…s4`, 5 phiên mỗi chuỗi). Giả thuyết: mọi mean của pipeline bị kéo về gần gốc tọa độ với cùng một hệ số. Luật đọc viết trước khi chạy: đứng nếu ở cả khe lớp gốc và khe mới α trung vị ≤ 0.5 và cos trung vị ≥ 0.8; đổ nếu α của khe lớp gốc ≥ 0.8. α = ⟨μ_pipeline, μ_mẫu⟩ / ‖μ_mẫu‖², μ_mẫu là mean của ảnh S0 có nhãn (khe lớp gốc) hoặc của ảnh giữ theo cụm (khe mới).
  - Khe lớp gốc: α trung vị 0.009 (từ 0.006 tới 0.013), cos 0.812–0.814, ‖μ_pipeline‖ 0.132–0.133 so với ‖μ_mẫu‖ 12.143–12.145.
  - Khe mới: α trung vị 0.008–0.010 (từ 0.005 tới 0.014), cos 0.709–0.748, ‖μ_pipeline‖ 0.122–0.131 so với ‖μ_mẫu‖ 9.631–11.116.
  - Chuẩn trung vị của ảnh test trong không gian PCA: 18.075–18.101.
  - Đọc: phần α đứng ở cả hai nhóm (mean của pipeline bằng khoảng 1% mean mẫu, cùng hệ số ở hai nhóm); phần cos đứng ở khe lớp gốc, không đạt ở khe mới (0.71–0.75 dưới 0.8). Mean mẫu theo cụm có lẫn ảnh cũ nên là mốc nhiễu hơn mean S0.
- Phân tích sau khi thấy số, thứ hai (lệnh chỉ đọc ngày 2026-11-17, chưa đưa vào repo): chuẩn trung vị của mean các khe đã học trong `saved_models` của 83 run `main.py` trên laptop. Luật đọc viết trước khi chạy, cho các run CUB: dưới 1 là chế độ của chính code upstream; từ 5 trở lên là lỗi tái lập của dự án; ở giữa không kết luận.
  - CUB, 48 run (`cub_base_ft`, `cub_km_evhung_ft`, `cub_km_evhung_det_ft` r0–r2 × s0–s4 và `cub_km_evhung_det_ft_l25` r0–r2 s0): stage 0 từ 0.1308 tới 0.1328, stage 5 từ 0.1291 tới 0.1308; khe chưa học 0.0000; log det trung vị của Σ ở stage 0 từ −10.077 tới −9.922 (384 chiều).
  - Báo kèm, không thuộc luật đọc (không gian và thang khác): GC10 d64 0.0866–0.1159; NEU d128 0.0548–0.0623 ở stage 0.
  - Đọc: dưới 1 ở 48/48 run CUB, gồm lưới `cub_base_ft` đã tái lập S0 và S5 của paper (Q05.01, `50d72f3`): chế độ mean gần 0 là của code upstream. Phép đo chỉ dùng chuẩn của mean, không có mean mẫu đi kèm, nên yếu hơn phép đo α ở trên; nó dựa trên việc chuỗi lab trùng `main.py` ở C1.

### Kết luận (Rito duyệt tạm 2026-11-17)

- Trên CUB t5, mean của mọi khe (lớp gốc và lớp mới) bằng khoảng 1% mean mẫu (α trung vị 0.009) và Σ gần I (log det khoảng −10 trên 384 chiều, tức phương sai mỗi chiều khoảng 0.974, tính tay). Chế độ này có ở cả 48 run `main.py` trên CUB, kể cả lưới tái lập số của paper. D23 được xác nhận trên CUB, ở cỡ lớn hơn dự kiến.
- [suy luận] Với μ_k ≈ α m_k và Σ ≈ I, log p_k(x) ≈ −½‖x‖² + α xᵀm_k − ½α²‖m_k‖²; số hạng đầu chung cho mọi khe, số hạng cuối khoảng 0.006. Bộ phân lớp Gaussian của pipeline vì vậy vận hành như một bộ phân lớp theo hướng prototype (argmax xᵀm_k). Luận điểm lệch hiệp phương sai của paper (Eq. 9–11) không hoạt động trong các run này, vì Σ gần như chưa rời khởi tạo.
- Thang của E14.05 bị nhiễu bởi chỗ lệch tầm mean giữa các nhóm khe: đặt một nhóm khe về đúng tầm trong khi nhóm kia còn bị co làm nhóm đúng tầm hút ảnh của nhóm kia (ví dụ tính tay: ảnh cũ ‖x‖ = 18, khe đúng bị co cho −162, khe mới đúng tầm ‖m‖ = 10 với cos 0.3 cho −158). Không nấc nào là phần lấy lại được; mọi mức tăng lớp mới đổi bằng 30–52 điểm lớp cũ.
- Kết luận E14.03 đọc lại: "khe cũ thắng nhờ số hạng Mahalanobis" nghĩa là thắng nhờ tích vô hướng với hướng prototype; L_comp là chuyện hướng của prototype, không phải chuyện hình dạng Gaussian.
- M1 cho thấy khi mean đặt đúng, 20 lớp mới tách nhau tới 92.5% (g20 của M1), so với 67.0% của c10: giả thuyết "chồng lấn trong biểu diễn" không đứng cho việc tách các lớp mới với nhau.
- Đích của E09.08 chuyển từ "bước 13" sang bộ ước lượng của cả mô hình: một cách ước lượng nhất quán cho mọi khe (mean mẫu, hoặc hậu nghiệm VB thật theo Eq. 7). Bước kế: tiền đăng ký E14.06, ước lượng lại mọi khe nhất quán (mean mẫu với Σ mẫu có co hoặc Σ chung) và so New, Old với pipeline.
- Giới hạn: hai phép phân tích sau làm sau khi thấy số, bằng lệnh chưa có trong repo; phép đo α chỉ trên r0; mô tả "bộ phân lớp theo hướng" là suy luận từ phép khai triển, chưa có thí nghiệm riêng.
- Sửa ngày 2026-11-17 (sau khi đọc lại D23 trong `notes/paper-code-discrepancies.md` và Kết quả E06.04; các dòng trên giữ nguyên làm bản ghi):
  - Mean gần 0 trên CUB không phải phát hiện mới: E06.04 đã đo trực tiếp ở phiên đầu, trung vị ‖μ_k‖/‖x̄_k‖ = 0.0107, trị riêng của Σ_k trong 0.85–1.82; D23 ghi tổng bước của lịch học là 0.0334 mỗi tham số và đã đo CIFAR-100 d384 với max ‖μ_k‖ 0.147. Câu "D23 được xác nhận trên CUB, ở cỡ lớn hơn dự kiến" sai: cỡ khớp với D23 và E06.04. Tiền đăng ký E14.05 ghi D23 "đo trên GC10", bỏ sót phép đo CUB của E06.04; nếu đọc lại hai nguồn này trước, các nấc Mk, Mc, M1 với Σ của pipeline đã đoán trước được là phép đổi tầm, không phải phép đo phần lấy lại.
  - Câu "bộ phân lớp theo hướng prototype (argmax xᵀm_k)" sai về cơ chế: phép khai triển bỏ số hạng bậc hai −½ xᵀ(Σ⁻¹ − I) x. D23 đo trên GC10 rằng số hạng này quyết định ở phiên đầu (trải 10.2–11.0 mỗi ảnh, so với 0.73–0.75 của số hạng tuyến tính; riêng nó cho đúng nhãn của pipeline ở 240/241 ảnh), tức một QDA lượng tử hoá theo dấu. Trên CUB số hạng này chưa đo. Cách giải thích chỗ sập Old ở Mk, M1, M3 vẫn đứng ở mức suy luận, vì khi một nhóm khe có mean đúng tầm thì chênh lệch của số hạng mean cỡ hàng trăm (ví dụ tính tay −162 so với −158 ở trên chỉ tính số hạng mean), lớn hơn trải của số hạng bậc hai đo trên GC10.
  - Câu "đích của E09.08 chuyển từ bước 13 sang bộ ước lượng của cả mô hình" không phải hướng mới: QD27 và E06.04 đã đặt E09.08 trên hậu nghiệm NIW (A3) cho lớp cũ, và E06.04 đo A3 hơn pipeline +1.586 điểm ở phiên đầu trên CUB. E14.05 bổ sung hai điều: (a) khe mới trong các phiên cũng ở chế độ này (α 0.008–0.010); (b) thay bộ ước lượng cho một nhóm khe mà giữ nhóm kia làm Old giảm 30–52 điểm, nên E09.08 phải dùng cùng một bộ ước lượng cho mọi khe (A3 cho lớp cũ và hậu nghiệm cùng họ cho lớp mới). Kết luận E14.02 ("đích là bước 11–13") và E14.03 ("đích là cạnh tranh cũ/mới") đều đo trên pipeline có mean gần 0 và cần đo lại trên mô hình dùng bộ ước lượng nhất quán trước khi dùng để thiết kế.
  - Không cần E14.06 như đã ghi ở trên; bước kế là thiết kế E09.08 theo QD27, với yêu cầu (b).

## E14.06 — CUB t5: L_10 và L_comp còn bao nhiêu khi mọi khe dùng hậu nghiệm NIW? (QD27, QD28)

### Tiền đăng ký (2026-11-17; viết trước khi có code và trước mọi số của thí nghiệm)

- Mã: E14.06 ở đây là thiết kế NIW này; thiết kế "mean mẫu với Σ mẫu có co" nhắc ở Kết luận E14.05 không làm (dòng Sửa của E14.05).
- Câu hỏi: khi mọi khe (lớp gốc và khe của mọi phiên) dùng cùng một bộ ước lượng là hậu nghiệm Normal–Inverse-Wishart với dự đoán Student-t (A3 của E06.04), L_10 = c10 − c14 và L_comp = g20 − c14 (định nghĩa của E14.03) còn bao nhiêu? Phụ: mô hình đó có hơn pipeline không?
- Vì sao: Sửa E14.05 ghi rằng kết luận E14.02 và E14.03 đo trên pipeline có mean gần 0 (D23, E06.04) và cần đo lại trên mô hình dùng một bộ ước lượng cho mọi khe trước khi thiết kế E09.08; thay bộ ước lượng cho một nhóm khe mà giữ nhóm kia làm lớp cũ giảm 30–52 điểm (E14.05). Mô hình ở đây là nấc đầu của E09.08 theo QD27.
- Đã tra trước khi viết: D23 (`notes/paper-code-discrepancies.md`), Kết quả E06.04, `lab/niw.py`, Kết quả và Sửa của E14.03, E14.05.
- Loại: phép đo mô tả có can thiệp; một luật đọc chính và một luật đọc phụ chốt trước.
- Dữ liệu: 15 chuỗi `cub_chain_ft_l25_r<r>_s<N>` (bộ đọc chính) và 15 chuỗi `cub_chain_ft_r<r>_s<N>` (báo kèm); không gian PCA của S0 (`z_s0`, `z_pool`, ảnh test qua `pca.transform`); tên theo P0 (QD15), `lut` cộng dồn. Bước 09–12 giữ nguyên của pipeline (`ids`, `keep`, `pi_test` từ ảnh chụp); không SVI, không rút RNG.
- Mô hình N (công thức của `lab/niw.py`; Murphy 2012 §4.6.3): với n hàng, mean x̄, tán xạ S: κn = κ0 + n, νn = ν0 + n, mn = (κ0 m0 + n x̄)/κn, Ψn = Ψ0 + S + (κ0 n/κn)(x̄ − m0)(x̄ − m0)ᵀ; dự đoán Student-t, νn − d + 1 bậc, tâm mn, ma trận tỉ lệ Ψn(κn + 1)/(κn(νn − d + 1)).
  - Tiên nghiệm, một cho mọi khe: m0 = mean của `z_s0`, κ0 = `niw.K0` (0.01), Ψ0 = off × P, P = hiệp phương sai trong lớp gộp của S0 (`niw.prior_cov`), off = `niw.choose(niw.loo_curve(z_s0, s0_y))` trên lưới `niw.OFFS`, chọn một lần mỗi chuỗi.
  - Khe lớp gốc 0…99: hậu nghiệm từ ảnh S0 có nhãn (= A3 của E06.04).
  - Khe off_j + c của phiên j ≤ i: hậu nghiệm từ các ảnh pool của phiên j có `keep` và `ids` = c (đúng tập ảnh và nhãn của bước 13); cụm không có ảnh: hậu nghiệm bằng tiên nghiệm (đếm số khe như vậy).
  - Dự đoán ở phiên i: argmax log-mật độ Student-t trên các khe 0 … off_i + 19 (khe chưa gặp không tham gia, như E14.05; với mô hình này khe chưa gặp không có tham số).
- Đại lượng ở mỗi phiên, như phần A của E14.03 nhưng trên N: c14_N (ảnh test lớp mới, mọi khe đã gặp), g_nobase_N, g_nodisc_N, g20_N, Old_N, All_N; c10 lấy lại như E14.03 (không phụ thuộc bộ phân lớp). L_comp_N = g20_N − c14_N; L_10_N = c10 − c14_N. So với pipeline: ΔNew = c14_N − c14, ΔOld = Old_N − Old, ΔAll = All_N − All (c14, Old, All của pipeline tính lại trong cùng run). Trung bình 5 phiên, 5 chuỗi mỗi FT seed, rồi trung bình và SD của ba FT seed.
- Báo kèm, không thuộc luật đọc:
  - A2: cùng hậu nghiệm, dự đoán bằng Gaussian với mean mn và hiệp phương sai Ψn/(νn − d − 1) (`kind = "gauss"`).
  - N sạch (oracle, dùng nhãn pool): như N nhưng khe của phiên chỉ dùng ảnh giữ có nhãn thật là lớp mới của phiên đó; tách phần L_comp_N do ảnh cũ lẫn vào cụm. Không phải phương pháp.
  - off được chọn; số khe rỗng ở N và N sạch.
- Cách đọc chính, chốt trước (bộ không rò; bộ có rò báo kèm):
  - L_comp_N > T (0.935) và L_comp_N ≥ ½ L_10_N: cạnh tranh cũ/mới lúc dự đoán vẫn là đích chính của E09.08 khi bộ ước lượng đã nhất quán.
  - L_comp_N ≤ T: cạnh tranh đo ở E14.03 là sản phẩm của chế độ mean gần 0; E09.08 nhắm vào phần còn lại của L_10_N.
  - Còn lại: cả hai.
- Cách đọc phụ, chốt trước: N hơn pipeline khi ΔNew > T và ΔOld > −B (B = 0.376); khi đó N là nấc khởi đầu của E09.08. ΔAll báo kèm, không dùng trong luật.
- Cổng kiểm:
  - G1: c10 tính lại trùng `S<i>_c10` của run E14.03 cùng bộ (1e-4); c14, Old của pipeline tính lại trùng `acc1` của `S<i>_14` (1e-6).
  - G2: trên bộ có rò, off chọn được và độ chính xác của phần lớp gốc trên tập test S0 trùng `off` và `acc_A3` của run `cub_e06_04_r<r>_s<N>`. Nếu cổng này báo lỗi vì E06.04 dùng dữ liệu S0 khác chuỗi, dừng để điều tra, không đọc số. Trên bộ không rò chỉ báo off.
  - G3: hậu nghiệm của khe lớp gốc tính trong runner trùng `niw.fit` trên cùng hàng (1e-8 tương đối).
  - G4: tổng số ảnh của 20 cụm bằng `keep.sum()` ở mọi phiên; tập ảnh của N sạch là tập con của N.
  - Runner chỉ in trạng thái cổng và thời gian.
- Code (viết sau tiền đăng ký): runner `tier2_e1406` trong `lab/score.py`, lệnh `e1406` qua `exp.py`, một run mỗi bộ (tag `cub_e14_06_ft_l25`, `cub_e14_06_ft`); `e1406-summary` chỉ đọc. Dùng `lab/niw.py` không sửa; không sửa `lab/sessions.py`, `lab/replay.py`.
- Chi phí [suy luận]: 10–20 phút mỗi run (một Cholesky 384 × 384 mỗi khe mỗi phiên, Student-t trên ảnh test; E06.04 làm việc tương tự ở phiên đầu).
- Dự đoán của Claude [SUY LUẬN], độ tin thấp:
  - c14_N trong 55–65; L_comp_N trong 3–8 điểm, tức vẫn trên T: cụm của pipeline lẫn 33–47% ảnh cũ nên mean khe mới bị kéo về phía lớp cũ.
  - ΔOld trong khoảng −5 tới +2: khe mới đúng tầm và học trên cụm lẫn ảnh cũ có thể giành ảnh cũ.
  - N sạch có c14 cao hơn N 3–8 điểm.
  - |A3 − A2| dưới 1 điểm ở c14.
- Giới hạn:
  - Bộ lọc (bước 12) dùng bộ phân lớp lượt F của pipeline, vẫn ở chế độ mean gần 0; tập ảnh giữ là sản phẩm của pipeline cũ. E14.06 chỉ thay bộ ước lượng cuối; thay cả chuỗi là việc của E09.08.
  - off chọn trên S0 và dùng cho mọi khe; với off lớn, Σ của cụm nhỏ (4–5 ảnh) gần như bằng P.
  - Một tiên nghiệm cho mọi khe; độ trải của lớp mới có thể khác lớp gốc.
  - r0–r2 đã dùng ở E14.01–E14.05; không phải kiểm định xác nhận.

### Kết quả (2026-11-17; run `cub_e14_06_ft_l25`, `cub_e14_06_ft`, code `49792bc`, commit run `c7c4815`; số in bằng `python lab/score.py e1406-summary`; 30/30 chuỗi qua G1–G4)

Số ghi là bộ không rò / bộ có rò; trung bình ba FT seed, SD giữa FT seed trong ngoặc khi có.

- Pipeline (tính lại trong run, trùng `acc1`): New 56.944 / 57.142; Old 74.046 / 74.427; All 71.794 / 72.138; c10 67.041 / 66.901 (trùng E14.03).
- Mô hình N (hậu nghiệm NIW cho mọi khe, Student-t): New 61.033 (0.560) / 61.032 (0.455); Old 69.161 / 69.329; All 68.045 / 68.181; g20 67.535 / 67.433; g_nobase 63.377 / 63.296; g_nodisc 63.965 / 63.884.
  - L_comp_N = +6.502 (0.286) / +6.402 (0.437); L_10_N = +6.008 (0.113) / +5.869 (0.501).
  - ΔNew +4.089 (0.302) / +3.890 (0.495); ΔOld −4.885 (0.365) / −5.098 (0.593); ΔAll −3.749 (0.340) / −3.956 (0.530).
- A2 (Gaussian plug-in của cùng hậu nghiệm): New 61.026 / 60.974; Old 69.133 / 69.317; L_comp 6.516 / 6.464; |A3 − A2| ở New 0.007 / 0.058 (tính tay).
- N sạch (oracle, nhãn pool): New 60.932 / 60.959; Old 75.935 / 76.277; All 73.927 / 74.218; L_comp 6.993 / 6.786; so với pipeline ΔNew +3.988 / +3.817, ΔOld +1.889 / +1.850, ΔAll +2.133 / +2.080; so với N: New −0.101, Old +6.774 (tính tay, bộ không rò).
- Phần lớp gốc ở phiên đầu (A3 trên tập test S0): 86.981 / 87.343; off = 250 ở 30/30 chuỗi (bộ có rò trùng E06.04 ở 15/15, G2). Khe rỗng: N 0, N sạch 70 / 87 trên 1500 (cụm chỉ có ảnh cũ, cùng số với Mc của E14.05).
- Luật đọc chính: L_comp_N 6.502 > T (0.935) và ≥ ½ L_10_N (3.004); bộ có rò 6.402 ≥ 2.934. Cạnh tranh cũ/mới lúc dự đoán vẫn là đích chính của E09.08 khi bộ ước lượng đã nhất quán.
- Luật đọc phụ: ΔNew +4.089 > T nhưng ΔOld −4.885 < −B: N không hơn pipeline.
- Dự đoán đã đăng ký: c14_N trong 55–65: đúng (61.033); L_comp_N trong 3–8: đúng (6.502); ΔOld trong −5 tới +2: đúng, sát biên dưới (−4.885); N sạch cao hơn N 3–8 điểm ở c14: sai (−0.101); |A3 − A2| dưới 1 điểm: đúng.
- Chi phí: run `cub_e14_06_ft_l25` 10.8 phút. Run `cub_e14_06_ft` ghi 177.1 phút, nhưng dòng tiến độ có hai bước nhảy khoảng 5000 s (sau r0_s4 và sau r2_s3) giữa các chuỗi khoảng 43–46 s [suy luận: máy ngủ hai lần]; runner không dùng RNG hay SVI, nên số không phụ thuộc thời gian chạy.

### Kết luận (Rito duyệt tạm 2026-11-17)

- Dưới bộ ước lượng nhất quán (hậu nghiệm NIW cho mọi khe), cạnh tranh cũ/mới lúc dự đoán vẫn là đích chính của E09.08: L_comp_N = 6.50 điểm. Khoảng 3.7 điểm của L_comp ở E14.03 (10.18) là sản phẩm của chế độ mean gần 0 (tính tay: 10.18 − 6.50).
- Mô hình N đổi khoảng 4 điểm lớp mới lấy khoảng 5 điểm lớp cũ, nên không hơn pipeline. [suy luận] Lớp cũ cao của pipeline một phần là nhờ khe mới yếu (mean gần 0, Σ gần I), nên khe mới ít giành ảnh cũ.
- Phần lớp cũ mà N mất đến từ ảnh cũ lẫn trong cụm: N sạch cho lớp cũ +6.77 điểm so với N, lớp mới gần như không đổi (−0.10). Trong chế độ NIW, câu "cụm lẫn ảnh cũ không gây quên" của Kết luận Sửa đổi E14.02 không còn đúng; câu đó vẫn đúng cho pipeline, nơi nó được đo. Câu "Plasticity–Stability không có đối tượng" của E14.02 đọc lại: khi khe mới học đúng tầm, đối tượng cần bảo vệ là tập ảnh dùng để khớp khe mới. N sạch bỏ ảnh cũ sau bước 09, không phải trước bước 09 như suy luận của E14.02.
- N sạch hơn pipeline ở lớp mới (+3.99) và lớp cũ (+1.89): trần của "bộ ước lượng nhất quán cộng bộ lọc cũ/mới hoàn hảo". Khoảng 6.8 điểm lớp cũ giữa N và N sạch là giá trị của một bộ lọc cũ/mới hoàn hảo.
- Hệ quả cho E09.08: cần cả bộ ước lượng nhất quán lẫn cách giữ ảnh cũ khỏi tập khớp khe mới. Trách nhiệm mềm với thành phần cũ A3 (thiết kế VB-GMM của QD27) là một cách; một bộ lọc không nhãn dựa trên mô hình lớp cũ (A3, hoặc A1 có AUROC tách cũ/mới 0.85 trên pool phiên 1 theo E06.04) là bước nhỏ hơn đi trước.
- A2 và A3 cho kết quả như nhau (chênh dưới 0.06 điểm ở lớp mới); E09.08 có thể dùng A2.
- Giới hạn: bộ lọc (bước 12) vẫn là của pipeline; N sạch dùng nhãn pool; r0–r2 không phải kiểm định xác nhận.

## E14.07 — CUB t5: dưới hậu nghiệm NIW, cạnh tranh lúc dự đoán còn bao nhiêu khi khe mới đúng lớp và sạch? (QD28)

### Tiền đăng ký (2026-11-18; viết trước khi có code và trước mọi số)

- Câu hỏi: (a) khi khe của mỗi lớp mới khớp trên đúng ảnh pool của lớp đó (Nlab), L_comp còn bao nhiêu? (b) lớp mới lên tới đâu khi phân hoạch đúng? (c) S5_S0test của N, Nc, Nlab so với pipeline; (d) một độ lệch đều trên khe mới của N lấy lại được bao nhiêu lớp mới tại mức giảm lớp cũ bằng B?
- Vì sao: thiết kế E09.08 (bàn giao 2026-11-18 mục 4, ngoài repo) cần ba trần trước khi tiền đăng ký.
  - L_comp của N là 6.502, của Nc là 6.993 (E14.06): làm sạch tập khớp không giảm cạnh tranh, nên ý (c) của Kết luận E12.02 có thể là giới hạn của biểu diễn.
  - Chưa có trần cho phần sửa phân hoạch: chỉ có c10 = 67.041; M1 của E14.05 dùng Σ của pipeline, không nhất quán.
  - QD28 đặt ràng buộc lớp cũ là S5_S0test, nhưng N và Nc chưa có số này.
- Đã tra trước khi viết: D16, D23, D24, D25 (`notes/paper-code-discrepancies.md`); Kết quả và Kết luận của E14.02 (cả Sửa đổi), E14.03, E14.05 (cả Sửa), E14.06, E12.02; `_post_1406`, `_score_1406`, `_parts_1403` của `lab/score.py`; `lab/niw.py`; QD28 Sửa 2026-11-18.
- Loại: phép đo mô tả có can thiệp oracle; hai luật đọc chốt trước.
- Dữ liệu: 15 chuỗi `cub_chain_ft_l25_r<r>_s<N>` (bộ đọc chính) và 15 chuỗi `cub_chain_ft_r<r>_s<N>` (báo kèm); không gian PCA của S0; `ids`, `keep`, `lut` (P0, QD15) lấy từ ảnh chụp. Không SVI, không rút RNG. Nhãn pool chỉ dùng cho nhánh oracle; nhãn test chỉ để chấm.
- Nhánh: mô hình N đúng như E14.06 (tiên nghiệm m0 = mean của `z_s0`, κ0 = `niw.K0`, Ψ0 = off × P, off = `niw.choose(niw.loo_curve(z_s0, s0_y))`; dự đoán Student-t, argmax trên các khe đã gặp); chỉ đổi tập hàng của khe mới.
  - N, Nc: như E14.06, tính lại.
  - Nlab (oracle): khe `lut[y]` của mỗi lớp mới y khớp trên mọi ảnh pool có nhãn y (25 ảnh, kể cả ảnh bộ lọc bỏ); khe của các đợt trước là khe Nlab của chính nhánh. Khác Nc ở cả phân hoạch lẫn tập giữ; bộ lọc giữ 24–25 trong 25 ảnh của mỗi lớp mới (E14.05), nên phần do tập giữ nhỏ.
- Đại lượng, mỗi nhánh, trung bình 5 phiên, 5 chuỗi mỗi FT seed, rồi trung bình và SD của ba FT seed: New, Old, All, g20, nobase, nodisc, L_comp = g20 − New; S5_S0test của mỗi nhánh và của pipeline.
- Phần D, trên N: cộng b vào log-mật độ của 20 khe của phiên rồi lấy argmax trên mọi khe đã gặp; lưới b như E14.03 (b = 0 và b = −Q_q(m), m = log-mật độ lớn nhất trên 20 khe của phiên trừ lớn nhất trên các khe cũ, không dùng nhãn). In mức tăng New tại mức giảm Old bằng B (0.376) và bằng 1 điểm; báo kèm cùng đường cong trên Nlab.
- Cách đọc, chốt trước (bộ không rò; bộ có rò báo kèm):
  - (a) L_comp_Nlab ≤ T (0.935): cạnh tranh của N là do khe mới bẩn hoặc sai; E09.08 giữ ý (c) trong mục tiêu. L_comp_Nlab > T: phần đó là giới hạn của biểu diễn dưới bộ ước lượng NIW này; E09.08 không nhắm ý (c), chỉ báo cáo. In kèm L_comp_Nlab / L_comp_N.
  - (b) New_Nlab − New_Nc ≥ T: E-M trong nhóm mới có chỗ để tăng; nhánh V-new của E09.08 giữ vai trò chẩn đoán chính. Dưới T: phân hoạch k-means gần trần dưới N; V-new chỉ báo cáo.
  - (c) S5_S0test của Nc trừ của pipeline < −B: ràng buộc của QD28 không đạt được kể cả với bộ lọc hoàn hảo; rà lại ràng buộc hoặc mục tiêu trước khi tiền đăng ký E09.08.
  - (d) chỉ báo cáo.
- Cổng kiểm:
  - G1: cột pipeline trùng `acc1` của `S<i>_14` (1e-6); N, Nc, c10 trùng run E14.06 cùng bộ (1e-4).
  - G2: Nlab có đúng 25 ảnh mỗi lớp mới; `lut` là song ánh từ 20 lớp mới lên 20 khe của phiên; tập ảnh của Nc là tập con của các ảnh được giữ.
  - G3: phần D tại b = 0 cho lại New, Old của N (1e-6); New không giảm và Old không tăng khi b tăng.
  - G4: S5_S0test của pipeline trùng dòng S0 của `acc1` ở `S5_14` (1e-6) và `metrics.json` (0.006).
  - Runner chỉ in trạng thái cổng và thời gian, không in số của từng chuỗi.
- Code (viết sau tiền đăng ký): runner `tier2_e1407` trong `lab/score.py`, lệnh `e1407` qua `exp.py` (`--prefix`, `--checks-only`), một run mỗi bộ (tag `cub_e14_07_ft_l25`, `cub_e14_07_ft`); lệnh `e1407-summary` chỉ đọc. Dùng lại `_post_1406`, `_score_1406` và phần giải tam giác của `_logd_1202`; không sửa `lab/niw.py`, `lab/sessions.py`, `lab/replay.py`.
- Chi phí: đo bằng một lần checks-only trên mỗi bộ đặc trưng trước run thật, không ước lượng tay (E14.06: 10.8 phút trên bộ không rò, chỉ để tham khảo).
- Dự đoán của Claude [SUY LUẬN], độ tin thấp: L_comp_Nlab trong 3–6 (vẫn trên T); New_Nlab trong 70–80; S5_S0test của Nc trừ của pipeline trong −0.5 tới +1.5; phần D trên N tăng New 1–2 điểm tại mức giảm Old bằng B.
- Giới hạn:
  - Nlab dùng nhãn pool: là trần, không phải phương pháp.
  - Kết luận "giới hạn của biểu diễn" chỉ đúng dưới hậu nghiệm NIW với off chọn như E14.06 (250 ở 30/30 chuỗi); bộ ước lượng khác có thể cho số khác.
  - r0–r2 đã dùng ở E14.01–E14.06 và E12.02; không phải kiểm định xác nhận.

### Kết quả (2026-11-18; run `cub_e14_07_ft_l25`, `cub_e14_07_ft`, code `5fbce01`, commit run `7271ffc`, tiền đăng ký `d1ac606`; số in bằng `python lab/score.py e1407-summary`; 30/30 chuỗi qua G1–G4)

- off = 250 ở 30/30 chuỗi.
- Bộ không rò, trung bình 5 phiên, 5 chuỗi, 3 FT seed; S5_S0test là của phiên 5; c10 = 67.041.

| Nhánh | New | Old | g20 | L_comp | S5_S0test | ΔS5_S0test so với pipeline |
|---|---|---|---|---|---|---|
| Pipeline | 56.944 | 74.046 | – | – | 78.398 | – |
| N | 61.033 | 69.161 | 67.535 | 6.502 | 69.102 | −9.296 |
| Nc | 60.932 | 75.935 | 67.925 | 6.993 | 78.200 | −0.198 |
| Nlab (oracle) | 86.149 | 82.946 | 94.956 | 8.807 | 79.340 | +0.942 |

- Bộ có rò, cùng cột (New / Old / L_comp / S5_S0test / ΔS5_S0test): pipeline 57.142 / 74.427 / – / 78.939 / –; N 61.032 / 69.329 / 6.402 / 69.459 / −9.480; Nc 60.959 / 76.277 / 6.786 / 78.732 / −0.207; Nlab 86.249 / 83.361 / 8.671 / 79.713 / +0.774.
- Phần D, mức tăng New tại mức giảm Old bằng B / bằng 1 điểm, bộ không rò: N 0.466 / 1.237; Nlab 0.665 / 1.717. Bộ có rò: N 0.469 / 1.242; Nlab 0.630 / 1.644. Lưới b chạm cả hai mức giảm ở 75/75 phiên, cả hai nhánh, cả hai bộ.
- Luật đọc chốt trước, bộ không rò:
  - (a) L_comp_Nlab 8.807 > T (0.935); L_comp_Nlab / L_comp_N = 1.355: giới hạn của biểu diễn dưới bộ ước lượng NIW này; E09.08 không nhắm ý (c), chỉ báo cáo.
  - (b) New_Nlab − New_Nc = +25.216 ≥ T: E-M trong nhóm mới có chỗ để tăng; nhánh V-new của E09.08 giữ vai trò chẩn đoán chính.
  - (c) S5_S0test của Nc trừ của pipeline = −0.198 ≥ −B (−0.376): ràng buộc của QD28 đạt được với bộ lọc hoàn hảo.
  - Bộ có rò cho cùng ba kết luận (8.671; +25.289; −0.207).
- Dự đoán đã đăng ký: L_comp_Nlab trong 3–6: sai, 8.807, lớn hơn cả L_comp_N. New_Nlab trong 70–80: sai, 86.149. S5_S0test của Nc trừ của pipeline trong −0.5 tới +1.5: đúng (−0.198). Phần D trên N tăng New 1–2 điểm tại mức giảm Old bằng B: sai, 0.466.
- Chi phí thật: 2491 s (`cub_e14_07_ft_l25`) và 541 s (`cub_e14_07_ft`), theo dòng tiến độ; checks-only trước đó 43 s và 41 s một chuỗi. Run `cub_e14_07_ft_l25` có một khoảng 1929 s giữa hai dòng tiến độ r2_s2 và r2_s3 (tính tay; các khoảng khác 31–51 s; note trong LOG). Runner không dùng RNG hay SVI, nên số không phụ thuộc thời gian chạy.

### Kết luận (Rito duyệt tạm 2026-11-18)

- Luật (a): dưới hậu nghiệm NIW, kể cả khi khe mới đúng lớp và sạch (Nlab), khe cũ vẫn giành 8.807 điểm New lúc dự đoán. Cạnh tranh không do khe mới bẩn; E09.08 bỏ ý (c) khỏi mục tiêu, chỉ báo cáo L_comp.
  - [suy luận] Tính theo tỉ lệ trên g20, khe cũ giành 9.3% số ảnh mà bộ định tuyến hoàn hảo cho đúng ở Nlab (8.807 / 94.956) và 9.6% ở N (6.502 / 67.535) (tính tay): cạnh tranh gần như tỉ lệ thuận với phần đúng trong phiên.
  - Kết luận chỉ đúng dưới hậu nghiệm NIW với off = 250; A1 tách cũ/mới tốt hơn A3 (E12.02), nên một bộ ước lượng khác có thể cho số khác.
- Một độ lệch đều không phải lối ra: trên N chỉ +0.466 New tại mức giảm Old bằng B, thấp hơn trên pipeline (0.874, E14.03).
- Luật (b): đòn bẩy lớn nhất là phân hoạch trong nhóm mới. Nlab hơn Nc 25.216 điểm New và 7.011 điểm Old (tính tay, 82.946 − 75.935): lỗi phân cụm làm mất cả lớp mới lẫn lớp cũ, vì cụm sai lớp tạo khe giành ảnh cũ. Nlab dùng nhãn pool, nên chỉ là trần: số này cho biết khoảng cách tới trần, không cho biết phần lấy được khi không có nhãn.
- Luật (c): N mất 9.296 điểm S5_S0test so với pipeline, gần gấp đôi mức giảm Old trung bình của N ở E14.06 (4.885); bỏ ảnh cũ khỏi tập khớp lấy lại 9.098 điểm (tính tay, 78.200 − 69.102). Bộ lọc cũ/mới là điều kiện cần cho ràng buộc S5_S0test của QD28; phân hoạch là đòn bẩy cho M_d.
- Hệ quả cho thiết kế E09.08: mục tiêu có hai vế, M_d tăng nhờ sửa phân hoạch trong nhóm mới (V-new là nhánh chính, so với Nc), và S5_S0test giữ được nhờ trách nhiệm hai phía với thành phần cũ; ý (c) bỏ khỏi mục tiêu; dự đoán của E09.08 viết lại theo các số này.
- Khoảng 1929 s trong run `cub_e14_07_ft_l25`: Rito xác nhận máy không ngủ; nguyên nhân chưa rõ. Runner không dùng RNG hay SVI, nên số không bị ảnh hưởng.
- Giới hạn: Nlab dùng nhãn pool; r0–r2 đã dùng ở E14.01–E14.06 và E12.02, nên đây không phải kiểm định xác nhận; kết luận về biểu diễn chỉ đúng dưới NIW với off = 250.

## E14.08 — CUB t5: vì sao N mất lớp cũ? Khe mới yếu đi theo bốn phép, và ảnh cũ trong tập khớp gây hại qua mean hay qua hiệp phương sai (QD27, QD28)

### Tiền đăng ký (2026-11-19; viết trước khi có code và trước mọi số của thí nghiệm)

- Câu hỏi:
  - (B) Có phép làm yếu khe mới nào (không đụng khe gốc) đưa N về ΔS5_S0test ≥ −B so với pipeline mà New vẫn hơn pipeline ít nhất T không?
  - (A) Phần S5_S0test mà N mất so với Nc đi qua mean hay qua hiệp phương sai của khe mới? Khe chiếm ảnh test gốc có phải khe mà tập khớp chứa ảnh pool cùng lớp với ảnh bị chiếm (D16) không? S0test của PIPE, N, Nc theo từng phiên ra sao?
- Vì sao:
  - Kết luận E14.07 (bộ không rò): N mất 9.296 điểm S5_S0test so với pipeline, Nc chỉ mất 0.198; New của Nc ngang N (60.932 so với 61.033). Ảnh cũ trong `keep` là nguồn mất và không giúp lớp mới.
  - Pipeline khớp trên cùng `keep` mà S5_S0test ngang Nc (78.398 so với 78.200). Câu hỏi là cơ chế nào làm ảnh cũ trong `keep` gần như vô hại ở pipeline. Bàn giao 2026-11-19 (ngoài repo): sửa được thì VB thật còn cơ hội thắng bản gốc; không sửa được thì dừng hướng VB thật.
  - Ứng viên chính [suy luận] (H2): khe mới của pipeline yếu hơn khe gốc một cách có hệ thống.
    - Lượt S chỉ dùng 48–64% tổng learning rate và dừng ngay trên ngưỡng S (`lab/13-gauss-s.ipynb` §2.1, GC10).
    - Ở bộ phân lớp cuối, số hạng log det nghiêng về khe cũ (`lab/14-eval.ipynb` §2.2).
    - Dừng ở bước 525 thay vì 1000 giữ thêm 9 ảnh test cũ (`lab/13-gauss-s.ipynb` §3.2).
    - Bảng 5 của paper (CUB, w/o early-stopping): M_f 29.22 so với 6.72, M_d 60.77 so với 56.46. Cùng chiều với N so với pipeline (paper chạy một seed; luật dừng của code khác paper, D14).
  - Các ứng viên khác [suy luận]:
    - H11: N gần LDA (off = 250) nên quyết định theo mean, mà mean của khe gần như không được chính quy hoá. Với n = 40, trọng số tiên nghiệm của mean là κ0/(κ0 + n) = 0.01/40.01 ≈ 0.00025 (tính tay), còn của hiệp phương sai là 250/(250 + 40) ≈ 0.86 (tính tay). Ở pipeline, μ ≈ 0 (D23, E06.04).
    - H8: ảnh cũ của pool lặp lại ở mọi phiên (D16), nên sinh ra khe "bản sao" của lớp cũ.
    - H4: khe mới rộng hơn khe gốc.
- Đã tra trước khi viết:
  - D11, D14, D16, D23, D25.
  - Kết quả và Kết luận của E06.04, E12.02, E14.02 (cả Sửa đổi), E14.03, E14.05 (cả Sửa), E14.06, E14.07, E09.08.
  - Summary (EN) của `lab/06`, `07`, `08`, `11`, `12`, `13`, `14`.
  - `_post_1406`, `_logd_1406`, `_score_1406`, `_logd_1202`, `_parts_1403`, `_gain_1403`, `tier2_e1407` của `lab/score.py`; `lab/niw.py`.
  - Bảng 5 và Eq. 9–11 của paper.
  - Miller & Dunson 2019 (posterior coarsened xấp xỉ bằng tempering likelihood); Grünwald & van Ommen 2017 (Safe Bayes, learning rate của likelihood); Ali, Kolter & Tibshirani 2019 (dừng sớm xấp xỉ ridge).
- Đã thấy trước tiền đăng ký này: mọi số của E14.01–E14.07, E12.02, E09.08 trên `cub_chain_ft_l25` và `cub_chain_ft` (r0–r2). Claude chưa thấy: S0test theo phiên của PIPE, N, Nc (chỉ thấy phiên 5), và mọi số của O1–O4, Mb, Ψb và phần phân bổ.
- Loại: phép đo mô tả có can thiệp, luật đọc chốt trước.
  - Tầng B: lưới tham số, điểm đọc chọn bằng nhãn test, nên là trần của mỗi phép, không phải phương pháp.
  - Tầng A: oracle nhãn pool.
- Dữ liệu: 15 chuỗi `cub_chain_ft_l25_r<r>_s<N>` (bộ quyết định) và 15 chuỗi `cub_chain_ft_r<r>_s<N>` (báo kèm).
  - Không gian PCA của S0; `ids`, `keep`, `lut` (P0, QD15) lấy từ ảnh chụp.
  - Không SVI, không rút RNG.
  - Nhãn pool chỉ dùng cho Nc, Mb, Ψb và phần phân bổ; nhãn test chỉ để chấm và để chọn điểm đọc.
  - T = 0.935, B = 0.376 (CUB, như E14.06–E09.08).
- Mô hình gốc: N đúng như E14.06/E14.07.
  - Tiên nghiệm: m0 = mean của `z_s0`, κ0 = `niw.K0`, ν0 = d + 1 + off, Ψ0 = off × P, P = `niw.prior_cov(·, 0)`, off = `niw.choose(niw.loo_curve(z_s0, s0_y))`.
  - Dự đoán Student-t; argmax trên mọi khe đã gặp; prior lớp đều.
  - Mọi phép dưới đây chỉ đụng khe mới của mọi phiên đã qua (khe base … n − 1). Khe gốc 0 … base − 1 giữ nguyên.
  - Ở mọi phép, khe không có hàng nào là tiên nghiệm.
- Tầng B, bốn phép. Điểm đầu của mỗi lưới là điểm trung tính, cho lại N.
  - O1, tempering likelihood (posterior lũy thừa, Miller & Dunson 2019):
    - Từ n hàng có mean x̄ và tán xạ S: κ_n = κ0 + ηn, ν_n = ν0 + ηn, m_n = (κ0 m0 + ηn x̄)/κ_n, Ψ_n = Ψ0 + ηS + (κ0 ηn/κ_n)(x̄ − m0)(x̄ − m0)ᵀ.
    - η ∈ {1, 0.5, 0.25, 0.1, 0.05, 0.02}.
    - Ghi chú [suy luận]: vì κ0 nhỏ, mean gần như không dịch, nhưng thang dự đoán nhân thêm (κ_n + 1)/κ_n. Với n = 40, η = 0.02 cho κ_n = 0.81 và hệ số 2.23, so với 1.025 ở η = 1 (tính tay). O1 làm khe rộng ra và có thể còn làm tăng việc chiếm ảnh ở rìa.
  - O2, co mean về m0:
    - κ riêng cho khe mới, κ = n(1 − w)/w, nên m_n = (1 − w) m0 + w x̄, κ_n = n/w, ν_n = ν0 + n, Ψ_n = Ψ0 + S + n(1 − w)(x̄ − m0)(x̄ − m0)ᵀ.
    - Lưới: κ0 (trung tính), rồi w ∈ {0.75, 0.5, 0.25, 0.1}.
  - O3, tempering lúc dự đoán so với một mật độ tham chiếu:
    - Với khe mới k, điểm s_k(x) = τ (L_k(x) − L_ref(x)) + L_ref(x). L_k là log-mật độ Student-t của N; L_ref là log-mật độ Gaussian N(m0, P), tính bằng float64 qua Cholesky. Khe gốc giữ L_c.
    - τ ∈ {1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}.
    - [suy luận] Đây là đối ứng bậc nhất của "ít bước Adam hơn tính từ điểm khởi tạo": ở pipeline, log N(x | μ, I + E) − log N(x | 0, I) gần tuyến tính theo (μ, E), mà (μ, E) tỉ lệ với số bước (E06.04; `lab/08-classifier.ipynb` §3).
    - p_ref chốt một lựa chọn; thí nghiệm này không thử lựa chọn khác.
  - O4, lệch cộng đều b ≤ 0 trên mọi khe mới:
    - Ở mỗi phiên, m′ = max log-mật độ trên khe base … n − 1 trừ max trên khe 0 … base − 1 (không dùng nhãn).
    - Lưới: b = 0 và b = −Q_q(m′) với q theo `Q_1403`; chỉ giữ các điểm có b ≤ 0.
    - Khác phần D của E14.07 ở hai chỗ: chiều b âm, và cộng vào mọi khe mới chứ không chỉ 20 khe của phiên, vì S5_S0test bị khe mới của mọi phiên chiếm.
- Tầng A:
  - A1: S0test (các lớp gốc của tập test) của PIPE, N, Nc ở mỗi phiên 1–5.
  - A2, ghép chéo (oracle nhãn pool):
    - "Mb": m_n từ hàng của N (bẩn); κ_n, ν_n, Ψ_n từ hàng của Nc (sạch).
    - "Ψb": m_n từ hàng của Nc; κ_n, ν_n, Ψ_n từ hàng của N.
    - Phần nào lấy từ một tập hàng rỗng thì nhận giá trị tiên nghiệm (m0; κ0, ν0, Ψ0).
    - R = S5_S0test(Nc) − S5_S0test(N); R_mean = S5_S0test(Ψb) − S5_S0test(N); R_cov = S5_S0test(Mb) − S5_S0test(N).
  - A3, phân bổ ở phiên 5 trên N:
    - Ảnh bị chiếm: ảnh test thuộc lớp gốc y mà argmax trên các khe gốc là y, nhưng argmax trên mọi khe là một khe mới k.
    - Với mỗi ảnh, ghi:
      - phiên của k, số hàng n_k, tỉ lệ ảnh cũ trong hàng của k;
      - chỉ báo H8: hàng của k có ít nhất một ảnh pool nhãn y;
      - phép tách L_k − L_y = ΔA + ΔB, trong đó A là phần không phụ thuộc x của log-mật độ Student-t (các hàm gamma, log det của thang) và B là phần phụ thuộc x.
    - Tỉ lệ nền của H8 cho một ảnh lớp y: phần khe mới ở phiên 5 có ít nhất một hàng nhãn y.
    - Tỉ số H8 = trung bình chỉ báo chia cho trung bình tỉ lệ nền, trên mọi ảnh bị chiếm của một FT seed.
- Đại lượng, cho mỗi phép X ∈ {O1, O2, O3, O4} và mỗi FT seed:
  - Ở mỗi điểm lưới j:
    - New: trung bình 5 phiên và 5 chuỗi;
    - ΔS5_S0test = S5_S0test(X, j) − S5_S0test(PIPE), trung bình 5 chuỗi;
    - ΔOld: như ΔS5_S0test nhưng cho Old, trung bình 5 phiên;
    - với O4, các chuỗi ghép theo chỉ số q.
  - Đi từ điểm trung tính về phía mạnh, lấy điểm đầu tiên có ΔS5_S0test ≥ −B. Nội suy tuyến tính New theo ΔS5_S0test giữa điểm đó và điểm liền trước, tại ΔS5_S0test = −B.
  - G_X = New nội suy − New của pipeline (P0). Nếu không điểm nào đạt: "không chạm".
  - Báo trung bình và SD (ddof 1) của G_X qua 3 FT seed, kèm số FT seed chạm.
- Cách đọc, chốt trước (bộ không rò; bộ có rò báo kèm):
  - R1: có X với trung bình G_X ≥ T và chạm ở 3/3 FT seed → VB thật còn đường.
    - Bước kế: chọn tham số của X không dùng nhãn (Safe Bayes hoặc prequential trên pool) và xác nhận trên r3, r4 bằng một tiền đăng ký riêng.
    - Nhiều X cùng đạt thì ghi cả; X có G_X lớn nhất đi trước.
    - Nếu bộ có rò cho G_X ≤ 0 ở X đó, ghi "không vững"; phải xác nhận trước khi dùng.
  - R2: không X nào đạt R1, nhưng ít nhất một X chạm ở 3/3 FT seed → làm yếu khe mới chỉ trả lại lớp cũ bằng phần lớp mới. H2 giải thích được cơ chế, nhưng hậu nghiệm NIW không hơn pipeline → dừng hướng VB thật (tiêu chí của bàn giao 2026-11-19).
  - R3: không X nào chạm ở 3/3 FT seed → làm yếu khe mới không đủ; cơ chế nằm ở chỗ khác (H11, H8). Đọc A2, A3 trước khi quyết.
  - Đọc phụ A2, chỉ để định hướng, không đè R1–R3:
    - R_mean/R ≥ 0.75 → nhiễm bẩn đi qua mean;
    - R_cov/R ≥ 0.75 → đi qua hiệp phương sai;
    - còn lại → cả hai.
  - Báo kèm, không dùng để quyết: ΔOld tại điểm đọc; mọi đường của các phép; A1 theo phiên; tỉ số H8; phân bố ΔA, ΔB và phiên của khe chiếm.
- Cổng kiểm:
  - G1: PIPE (New, Old, S0test mọi phiên) và N, Nc (New, Old, S0test mọi phiên) trùng run E14.07 cùng bộ (1e-4).
  - G2: mỗi phép tại điểm trung tính cho lại New, Old, S0test của N (1e-6); O3 với τ = 1 và O4 với b = 0 cho lại đúng argmax của N.
  - G3: Mb và Ψb cho lại N khi cả hai nguồn là hàng của N, và cho lại Nc khi cả hai nguồn là hàng của Nc (1e-6).
  - G4: số ảnh bị chiếm ở A3 bằng một phép đếm độc lập (argmax trên khe gốc đúng, argmax trên mọi khe ≥ base); ΔA + ΔB = L_k − L_y (tương đối 1e-6).
  - Runner chỉ in trạng thái cổng và thời gian, không in số của từng chuỗi.
- Code (viết sau tiền đăng ký):
  - Runner `tier2_e1408` trong `lab/score.py`; lệnh `e1408` qua `exp.py` (`--prefix`, `--checks-only`); một run mỗi bộ (tag `cub_e14_08_ft_l25`, `cub_e14_08_ft`).
  - Lệnh `e1408-summary` chỉ đọc và áp luật.
  - Dùng lại `_post_1406`, `_logd_1202("A3")`, `_score_1406`. Không sửa `lab/niw.py`, `lab/sessions.py`, `lab/replay.py`.
- Chi phí: đo bằng một lần checks-only có profile trên mỗi bộ trước run thật, không ước lượng tay (E14.07: 2491 s và 541 s, chỉ để tham khảo).
- Dự đoán của Claude [SUY LUẬN], độ tin thấp:
  - O4 chạm, với G_O4 trong khoảng −4 tới 0 (phần D của E14.07: gần b = 0, mỗi điểm Old đổi được khoảng 1.2 điểm New; tính tay 0.466/0.376).
  - O3 gần O4; O2 có G lớn nhất trong bốn phép; O1 không chạm hoặc chạm muộn nhất.
  - Kết cục R2 nhiều khả năng hơn R1.
  - R_mean/R ≥ 0.75; tỉ số H8 > 1.5; phần S0test N mất so với PIPE tăng theo phiên.
- Giới hạn:
  - Điểm đọc chọn bằng nhãn test, nên G_X là trần của phép, không phải số của một phương pháp.
  - Bốn phép với nhiều điểm lưới: R1 chỉ mở một bước kế, cần xác nhận trên r3, r4.
  - p_ref của O3 chỉ có một lựa chọn. O1–O3 chỉ xét dưới NIW với off = 250, trên PCA-384 của CUB t5.
  - r0–r2 đã dùng ở E14.01–E14.07, E12.02, E09.08, nên đây không phải kiểm định xác nhận.
  - Tầng C (lượt S của pipeline không dừng sớm trên CUB, Q07.03) không nằm trong thí nghiệm này.

### Sửa (2026-11-19; viết sau code, trước checks-only và trước mọi số của thí nghiệm)

- Ba chỗ cài đặt trong `tier2_e1408` không ghi chữ-theo-chữ ở tiền đăng ký; không đổi đại lượng hay luật đọc:
  - O4: điểm có b > 0 được cắt về b = 0 (trùng N), để chỉ số q thẳng hàng giữa các chuỗi khi lấy trung bình. Phép đi trên đường cho kết quả như khi chỉ giữ các điểm b ≤ 0, vì điểm trùng N đã được xét ở điểm trung tính.
  - G3: kiểm từng tham số (mn, κ_n, ν_n, Cholesky của Ψ_n, log det) của mọi khe ở mọi phiên, sai lệch tương đối ≤ 1e-6, thay vì kiểm New, Old, S0test; chặt hơn.
  - G2 của O1 và O2: điểm trung tính tính lại qua `_postx_1408` (η = 1, κ0), không dùng thẳng ma trận của N, để cổng kiểm code của hai phép.
- Code đã thử trên 15 chuỗi giả (d = 6, 30 lớp, 5 phiên, mock `sessions`, `replay`, `jax`; mốc là `tier2_e1407` chạy trên cùng dữ liệu giả): G1–G4 OK ở 15/15; đối chứng âm (một số của E14.07 lệch 0.01) cho G1 DIFF. Không có số nào của dữ liệu thật.

### Kết quả (2026-11-19; run `20261119_104040__cub_e14_08_ft_l25`, `20261119_110129__cub_e14_08_ft`, code `ebe51a3`, Sửa `81f36c4`, commit run `958bef5`; số in bằng `python lab/score.py e1408-summary`; 30/30 chuỗi qua G1–G4)

- off = 250 ở 30/30 chuỗi.
- Sự cố trước run: lúc tra lệnh run, Claude in nhầm dòng `result:` của E14.07 trong `notes/LOG.md`, trong đó có S0test theo phiên của PIPE, N, Nc. Câu "Claude chưa thấy S0test theo phiên" của tiền đăng ký không còn đúng từ lúc đó. A1 chỉ để báo cáo, nên luật R1–R3 không bị ảnh hưởng; A1 dưới đây ghi là đã thấy một phần trước run.
- Bộ không rò, mỗi số là trung bình 5 chuỗi (New, Old: và 5 phiên) của một FT seed, rồi trung bình 3 FT seed:
  - Pipeline: New 56.944, Old 74.046, S5_S0test 78.398.
  - N: New 61.033, Old 69.161, S5_S0test 69.102. Nc: New 60.932, Old 75.935, S5_S0test 78.200.
- Tầng B, bộ không rò. G_X = New tại ΔS5_S0test = −B trừ New của pipeline:

| Phép | G_X (mean / SD) | Chạm −B | ΔOld tại điểm đọc |
|---|---|---|---|
| O1 | −2.356 / 1.382 | 3/3 | −1.354 |
| O2 | – | 0/3 | – |
| O3 | +0.173 / 0.660 | 3/3 | −0.613 |
| O4 | +3.087 / 0.354 | 3/3 | −3.488 |

  - O2 không chạm ở điểm nào; ở w = 0.75, ΔS5_S0test còn xấu hơn N (−10.318 so với −9.296).
  - Điểm đọc của O4 nằm giữa O4[6] (ΔS5_S0test −7.643, New 60.946) và O4[7] (ΔS5_S0test +1.564, New 59.787). Riêng điểm đo O4[7] đã có New hơn pipeline 2.843 (tính tay) với ΔS5_S0test dương, nên kết luận không phụ thuộc phép nội suy. b ở phiên 5 tại O4[7]: trung bình −12.552.
  - Bộ có rò: G_O1 −3.818, O2 không chạm, G_O3 −0.259, G_O4 +2.824 (3/3).
- Tầng A, bộ không rò:
  - A1 (đã thấy một phần trước run): ΔS0test của N so với pipeline ở phiên 1–5: −4.737, −6.760, −7.787, −8.583, −9.296; của Nc: +2.053, +1.659, +1.049, +0.417, −0.198.
  - A2: R = 9.098, R_mean = 3.332, R_cov = 5.317; R_mean/R = 0.366, R_cov/R = 0.584 → "cả hai". R_mean + R_cov = 8.649 (tính tay).
  - A3 (phiên 5 trên N): 523.667 ảnh test gốc bị khe mới chiếm mỗi chuỗi. Chỉ báo H8 = 0.820, tỉ lệ nền 0.054, tỉ số H8 = 15.278. Khe chiếm có tỉ lệ ảnh cũ 0.451 và trung bình 48.358 hàng; ΔA = 1.481, ΔB = 11.177. Phần chiếm theo phiên của khe: 0.299, 0.210, 0.167, 0.156, 0.169.
- Luật đọc chốt trước, bộ không rò:
  - R1: O4 chạm −B ở 3/3 FT seed với G_O4 = +3.087 ≥ T → VB thật còn đường. Bộ có rò cho G_O4 +2.824 > 0, nên không ghi "không vững".
  - Đọc phụ A2: R_mean/R 0.366, R_cov/R 0.584 → cả hai.
- Dự đoán đã đăng ký:
  - G_O4 trong −4 tới 0: sai (+3.087). O3 gần O4: sai. O2 có G lớn nhất: sai (không chạm). O1 không chạm hoặc chạm muộn nhất: sai một nửa (chạm, G thấp nhất trong các phép chạm).
  - R2 nhiều khả năng hơn R1: sai.
  - R_mean/R ≥ 0.75: sai (0.366). Tỉ số H8 > 1.5: đúng (15.278). Mất tăng theo phiên: đúng (đã thấy một phần trước run).

### Kết luận (Rito duyệt 2026-11-19)

- Theo luật chốt trước, E14.08 ra R1. Trừ một lệch đều vào mọi khe mới của N lấy lại được lớp gốc mà vẫn hơn pipeline khoảng 3 điểm lớp mới, ở 3/3 FT seed và cả trên bộ có rò. Hướng VB thật chưa phải dừng. Nhưng lệch được chọn bằng nhãn test, lớn bất thường so với một prior, và làm mất 3.5 điểm ở các lớp cũ của các đợt giữa. Vì vậy đây là trần của phép, chưa phải bằng chứng thắng.
- Cái giá ở Old:
  - Tại điểm đọc của O4, ΔOld = −3.488, gấp 9.277 lần B (tính tay); Nc thì Old hơn pipeline 1.889 (tính tay).
  - [suy luận] b trừ đều vào mọi khe mới, kể cả khe của các đợt trước, nên ảnh test của các đợt đó bị giành lại. Ràng buộc của QD28 chỉ nhìn S5_S0test nên luật R1 không thấy cái giá này. Old chiếm phần lớn tập test, nên All ở S5 của O4 có thể thấp hơn pipeline; runner không lưu All, nên đây chưa có số.
- b không phải một prior hợp lý: b trung bình −12.552 ở phiên 5 ứng với tỉ số prior khoảng e^−12.552 ≈ 3.5·10⁻⁶ (tính tay). [suy luận] b bù cho độ chênh log-mật độ quá lớn của dự đoán Student-t ở 384 chiều, không phải niềm tin về tỉ lệ lớp.
- Cơ chế:
  - Ảnh cũ trong `keep` hại chủ yếu qua khe "bản sao" (H8): 0.820 số ảnh bị chiếm rơi vào khe có ảnh pool cùng lớp, so với tỉ lệ nền 0.054. Đây là hệ quả của D16 (ảnh cũ của pool lặp lại mọi phiên).
  - Phần hại đi qua cả hiệp phương sai (R_cov/R 0.584) lẫn mean (0.366); H11 dạng "chỉ qua mean" bị bác. Co mean về m0 (O2) không cứu được, còn làm xấu hơn.
  - Làm yếu kiểu nhân (O1, O3) cứu được lớp gốc nhưng trả lại phần lớp mới; chỉ phép cộng (O4) giữ được New. [suy luận] Phép cộng chỉ lật các ca sát nút; phép nhân co mọi tỉ số log-likelihood, kể cả của ảnh mới đúng.
- Bước kế: tiền đăng ký E14.09.
  - Chọn b không dùng nhãn (ứng viên: luật dựa trên tỉ lệ ảnh pool được gán vào khe mới; Safe Bayes hoặc prequential).
  - Thêm ràng buộc Old ≥ −B và báo All, để không đổi lớp gốc lấy các đợt giữa.
  - Xác nhận trên r3, r4.
  - Biến thể chỉ trừ b khi khe mới tranh với khe gốc là ý viết sau khi thấy kết quả, nên phải đăng ký riêng.
- Giới hạn: b chọn bằng nhãn test; r0–r2 đã dùng ở E14.01–E14.07, E12.02, E09.08; A1 đã thấy một phần trước run; lưới O4 thô giữa O4[6] và O4[7], nên vị trí tốt nhất của b chưa rõ.

## E14.09 — CUB t5: prior phân cấp cho khe mới của N có qua được ràng buộc hai vế không? (QD27, QD28)

### Tiền đăng ký (2026-11-19; viết trước khi có code và trước mọi số của thí nghiệm)

- Câu hỏi: một prior phân cấp cho khe mới có đưa N qua được ràng buộc hai vế của QD28 (Sửa 2026-11-19: ΔS5_S0test ≥ −B và ΔOld ≥ −B) mà vẫn giữ New hơn pipeline ít nhất T không?
- Vì sao:
  - Bàn giao 2026-11-19 (ngoài repo): prior phân cấp là thành phần VB thật duy nhất chưa thử. Kết luận E14.08: ảnh cũ trong `keep` hại qua khe bản sao, qua cả hiệp phương sai (R_cov/R 0.584) lẫn mean (R_mean/R 0.366).
  - Rito chọn làm E14.09 trước, sau đó (1) tách cũ/mới không nhãn và (2) phân hoạch trong nhóm mới, song song.
- Phản biện trước khi đo [suy luận]:
  - Trần: prior trên hiệp phương sai chỉ chạm phần R_cov của khoảng N mất; phần qua mean không chạm.
  - Đoán trước một phần: tăng off cho khe mới gần giống O1 ở η nhỏ (cùng kéo Ψ_n về prior); O1 trả lại lớp gốc nhưng mất New (G_O1 −2.356). Hai phép không trùng nhau: O1 còn giảm κ_n, ν_n; tăng off giữ n.
  - Mượn từ lớp gần nhất (distribution calibration, Yang et al. 2021) làm khe mới giống lớp cũ hơn, tức tăng khe bản sao; đưa vào làm nhánh bác bỏ được.
- Đã tra trước khi viết: Kết quả và Kết luận của E14.06, E14.07, E14.08, E12.02, E09.08; QD28 (cả Sửa 2026-11-19); `lab/niw.py` (`loo_curve`, `choose`, `prior_cov`, `log_evidence`, `log_sequential`); `tier2_e1408` của `lab/score.py`.
- Đã thấy trước tiền đăng ký này: mọi số của E14.01–E14.08, E12.02, E09.08 trên `cub_chain_ft_l25` và `cub_chain_ft` (r0–r2). Chưa thấy: mọi số của HB-off, HB-full, HB-slot, HB-near.
- Loại: thí nghiệm cải tiến có luật quyết định, chốt trước; nhánh HB-near là kiểm phản biện.
- Dữ liệu: 15 chuỗi `cub_chain_ft_l25_r<r>_s<N>` (bộ quyết định) và 15 chuỗi `cub_chain_ft_r<r>_s<N>` (báo kèm); không gian PCA của S0; `ids`, `keep`, `lut` (P0) từ ảnh chụp. Không SVI, không rút RNG. Nhãn pool chỉ dùng cho Nc; nhãn test chỉ để chấm. T = 0.935, B = 0.376.
- Nhánh: khe gốc như N (m0 = mean của `z_s0`, κ0 = `niw.K0`, Ψ0 = off0 × P, P = `niw.prior_cov(·, 0)` trên S0, off0 = `niw.choose(niw.loo_curve(z_s0, s0_y))`); hàng của khe mới như N (ảnh được giữ theo id cụm); chỉ tiên nghiệm của khe mới đổi. Dự đoán Student-t, argmax trên mọi khe đã gặp.
  - N: tiên nghiệm của khe mới như khe gốc (mốc).
  - HB-off: m0, P từ S0; off_new của các khe của phiên i = `niw.choose(niw.loo_curve(z_keep, ids_keep))` trên các hàng được giữ của phiên, nhãn là id cụm; cụm có dưới 2 hàng được giữ bị bỏ khỏi phép LOO (không bỏ khỏi mô hình).
  - HB-full: m0_new, P_new, off_new của phiên i đều ước lượng từ các hàng được giữ của phiên bằng cùng luật LOO (`niw.class_stats`, `niw.prior_cov(·, 0)`, `niw.loo_curve` trên `z_keep`, `ids_keep`, cụm dưới 2 hàng bị bỏ).
  - HB-slot: m0, P từ S0; off của từng khe chọn trên lưới `niw.OFFS` bằng `niw.log_evidence` của các hàng của khe đó (ν0 = d + 1 + off, Ψ0 = off × P); hoà thì lấy off lớn hơn; khe dưới 2 hàng dùng off0.
  - HB-near (kiểm phản biện): với khe mới k, lớp gốc gần nhất c* = argmin_c Mahalanobis theo P giữa tâm hàng của khe k và tâm lớp c ở S0; tiên nghiệm m0_k = tâm lớp c*, Ψ0_k = off0 × Σ̂_c*, Σ̂_c* = Ψ_n của khe gốc c* chia (ν_n − d − 1); κ0, ν0 như N. Khe không có hàng dùng tiên nghiệm của N.
  - Nc (oracle, trần): như E14.07.
- Đại lượng, mỗi nhánh: New (M_d), Old, S5_S0test, S5_All. Mỗi số là trung bình 5 phiên và 5 chuỗi của một FT seed (S5_* chỉ của phiên 5), rồi trung bình và SD (ddof 1) qua 3 FT seed. Báo kèm: off chọn được ở mỗi phiên (HB-off, HB-full), phân bố off của HB-slot, SD giữa các seed của Old trên pipeline (QD28 Sửa 2026-11-19).
- Cách đọc, chốt trước (bộ không rò; bộ có rò báo kèm):
  - Đạt: có X ∈ {HB-off, HB-full, HB-slot} với ΔM_d(X − PIPE) ≥ T, dương ở 3/3 FT seed, ΔS5_S0test(X − PIPE) ≥ −B và ΔOld(X − PIPE) ≥ −B → viết tiền đăng ký riêng để xác nhận trên r3, r4.
  - Không đạt nhưng có ích: không X nào đạt, nhưng có X với ΔS5_S0test(X − N) ≥ T và ΔM_d(X − N) > −T → prior phân cấp giảm được một phần mất lớp cũ; chuyển sang (1) và (2).
  - Không đạt: còn lại → prior phân cấp trên hiệp phương sai không gỡ được khe bản sao; chuyển sang (1) và (2).
  - HB-near, chỉ kiểm phản biện: ΔS5_S0test(HB-near − N) < 0 → phản biện "mượn từ lớp gần nhất làm khe giống lớp cũ hơn" được ủng hộ.
- Cổng kiểm:
  - G1: PIPE (New, Old, S0test mọi phiên), N, Nc (New, Old, S0test mọi phiên) trùng run E14.08 cùng bộ (1e-4).
  - G2: HB-off và HB-slot với off ép bằng off0 cho lại New, Old, S0test của N (1e-6).
  - G3: `niw.log_evidence` trùng `niw.log_sequential` trên khe đầu tiên có ít nhất 2 hàng của mỗi phiên (tương đối 1e-8).
  - G4: mọi off chọn được nằm trong `niw.OFFS`; P_new của HB-full xác định dương (Cholesky không lỗi).
  - Runner chỉ in trạng thái cổng và thời gian.
- Code (viết sau tiền đăng ký): runner `tier2_e1409` và `e1409-summary` trong `lab/score.py`, lệnh `e1409` qua `exp.py`, một run mỗi bộ (tag `cub_e14_09_ft_l25`, `cub_e14_09_ft`); S5_All tính trong runner. Không sửa `lab/niw.py`, `lab/sessions.py`, `lab/replay.py`.
- Chi phí: đo bằng một lần checks-only có profile trước run thật.
- Dự đoán của Claude [SUY LUẬN], độ tin thấp: không nhánh nào đạt; HB-off chọn off ≥ 250 và nằm gần đường O1 của E14.08; HB-full xấu hơn N ở S5_S0test vì P_new bị ảnh cũ làm rộng; HB-near xấu hơn N ở S5_S0test.
- Giới hạn: chỉ dưới NIW trên PCA-384 của CUB t5; r0–r2 đã dùng ở E14.01–E14.08, E12.02, E09.08; không thử prior phân cấp trên mean, vì O2 của E14.08 cho thấy co mean về m0 làm xấu đi.

### Kết quả (2026-11-19; run `20261119_141626__cub_e14_09_ft_l25`, `20261119_143918__cub_e14_09_ft`, code `70dcf98`, commit run `b17c285`; số in bằng `python lab/score.py e1409-summary`; 30/30 chuỗi qua G1–G4)

- off0 = 250 ở 30/30 chuỗi. Hai run chạy với một luồng BLAS (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`); lý do ghi ở note của run trong `notes/LOG.md`.
- Bộ không rò, hiệu so với pipeline (trung bình 3 FT seed; pipeline: M_d 56.944, Old 74.046, S5_S0test 78.398, S5_All 66.381):

| Nhánh | ΔM_d | ΔOld | ΔS5_S0test | ΔS5_All |
|---|---|---|---|---|
| N | +4.089 | −4.885 | −9.296 | −3.513 |
| HB-off | +3.748 | −4.879 | −9.184 | −3.601 |
| HB-full | −39.327 | −3.702 | +8.196 | −14.015 |
| HB-slot | +2.456 | −4.198 | −7.452 | −3.395 |
| HB-near | +4.163 | −5.465 | −10.677 | −4.195 |
| Nc (oracle) | +3.988 | +1.889 | −0.198 | +2.004 |

  - So với N: HB-off ΔM_d −0.341, ΔS5_S0test +0.112; HB-full −43.416, +17.492 (New còn 17.617); HB-slot −1.632, +1.844; HB-near +0.074, −1.382.
  - off của HB-off và HB-full: 250 ở 15/15 chuỗi tại phiên 1 và 2; 100 ở 4, 1, 2 chuỗi tại phiên 3, 4, 5, còn lại 250. off của HB-slot trên mọi khe: 2 ×133, 5 ×31, 10 ×54, 25 ×79, 50 ×246, 100 ×480, 250 ×477.
  - SD gộp giữa các seed (trong FT seed) của Old trên pipeline: 0.477, lớn hơn B = 0.376 (QD28, Sửa 2026-11-19).
  - Bộ có rò cùng chiều ở mọi nhánh: ΔS5_All của N −3.704, HB-off −3.746, HB-full −14.191, HB-slot −3.349, HB-near −4.413, Nc +1.985; HB-near so với N ở S5_S0test −1.375.
- Luật đọc chốt trước, bộ không rò:
  - Đạt: không nhánh nào.
  - Không đạt nhưng có ích: không nhánh nào. HB-slot có ΔS5_S0test so với N +1.844 ≥ T, nhưng ΔM_d so với N −1.632 < −T.
  - → Không đạt: prior phân cấp trên hiệp phương sai không gỡ được khe bản sao; chuyển sang (1) và (2).
  - HB-near: ΔS5_S0test(HB-near − N) = −1.382 < 0 → phản biện "mượn từ lớp gần nhất làm khe giống lớp cũ hơn" được ủng hộ.
- Dự đoán đã đăng ký: không nhánh nào đạt: đúng. HB-off chọn off ≥ 250 và nằm gần đường O1: sai một nửa (off chủ yếu là 250, đôi khi 100; nhánh gần như trùng N, không gần O1). HB-full xấu hơn N ở S5_S0test: sai (tốt hơn +17.492, nhưng New sụp). HB-near xấu hơn N ở S5_S0test: đúng.
- Giới hạn cài đặt: off của HB-off chọn bằng `niw.loo_curve`, vốn dùng hiệp phương sai gộp của chính các hàng (P_new), rồi đem dùng với P của S0. Vì off rơi vào 250 ở phần lớn chuỗi, giới hạn này gần như không ảnh hưởng.

### Kết luận (Rito duyệt 2026-11-19)

- Prior phân cấp cho khe mới không đưa N qua được ràng buộc hai vế, và không thuộc loại "có ích" theo luật đã đăng ký. Ba cách học prior cho nhóm mới chỉ trượt dọc cùng một đường đánh đổi giữa lớp cũ và lớp mới, như các phép làm yếu của E14.08. Chỉ khi tách cũ/mới bằng nhãn (Nc) mới vượt pipeline, kể cả ở S5_All (+2.004). Thành phần VB còn lại không cứu được VB thật; đường còn lại là tách cũ/mới không nhãn (1) và phân hoạch trong nhóm mới (2).
- S5_All lần đầu có số: N thua pipeline −3.513, Nc hơn +2.004. Theo chỉ số chính của paper, chỉ tách cũ/mới hoàn hảo thắng pipeline; mọi nhánh không nhãn ở đây thua khoảng 3.4–4.2 điểm All. [suy luận] O4 của E14.08 (ΔOld −3.488) nhiều khả năng cũng thua ở All; chưa đo.
- Cơ chế:
  - HB-near xác nhận mượn thống kê của lớp gốc gần nhất làm khe mới giống lớp cũ hơn.
  - HB-full: P_new ước lượng từ các cụm lẫn ảnh cũ nên rộng ra, khe mới gần như không thắng; S5_S0test lên 86.594, New còn 17.617. [suy luận] Cùng với đường O1–O4 của E14.08, đây là bằng chứng mọi cách làm yếu khe mới không nhãn nằm trên cùng một đường; chỉ Nc vượt ra ngoài, nên nút thắt là chất lượng tách cũ/mới, không phải dạng prior.
- Thước đo: SD giữa các seed của Old trên pipeline (0.477) lớn hơn B (0.376); thước cho Old cần xem lại trước thí nghiệm sau. Ở đây không đổi kết luận, vì mọi nhánh không nhãn có ΔOld từ −3.702 trở xuống.
- Bước kế: thiết kế (1) và (2) song song. Đích của (1): Nc, ΔS5_All +2.004, ΔOld +1.889. (1) mở đầu bằng phân tích vì sao E12.02, E09.08, E14.08 hỏng, trước khi đề xuất phương pháp.
