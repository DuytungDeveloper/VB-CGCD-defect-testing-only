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
- E09.08: VB-GMM cho phiên không nhãn, biến thể `09e-vbgmm` (trải 09–13): trách nhiệm mềm thay lượt F, bộ lọc (D12), lượt S và luật dừng (D14); khởi tạo từ `09a` + `09b`, A3 của E06.04 là mô hình lớp cũ (đề xuất 2026-11-05; chưa tiền đăng ký) — QD27, QD28; Kết quả của E06.04, E09.06
- E10.01: D13, Hungarian khi đánh giá (`--hungarian_on eval`) (2026-09-29 → 30; xong: trung bình gần như không đổi, lệch ở 1/25 phiên) — D13
- E12.01: WSM của BSW (CVPR 2026; OT nới một biên cộng RSQ, chạy trên đặc trưng cố định) ở bước 12 (đề xuất 2026-11-05; chưa tiền đăng ký; thí nghiệm riêng hay ablation chốt khi tiền đăng ký E09.08) — QD27; tài liệu chiến lược 2026-11-05 §3, §7 (ngoài repo)
- E14.01: đường cong δ và phép so ở cùng mức ảnh cũ (cánh sườn S–P / PMB), chỉ đọc pickle (đề xuất 2026-11-05; chưa tiền đăng ký) — tài liệu chiến lược 2026-11-05 §3 (ngoài repo)

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
