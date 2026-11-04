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
4. `04-loader` [có, commit `3f6e08e`]: nạp dữ liệu theo phiên, ba mục
   1. kiểm: `rerun` bước 04 trùng từng bit; sáu tập và thành phần lớp;
   2. code: loader t1 của GC10; 5 lần rút RNG, chỉ một lần chọn 50 ảnh cũ vào pool; D16 thấy bằng số (pool sau phiên 1 có 371 hàng);
   3. phiên: các tập test từng hàng; qua 15 ảnh chụp, seed chỉ đổi 50 ảnh cũ được bốc và thứ tự hàng, FT seed không đổi ảnh nào; đổi thứ tự hàng của pool làm đổi k-means ở bước 09 (một hoán vị: ARI 0.73, balanced accuracy lớp mới 0.32 thành 0.56).
5. `05-pca` [có, commit `42c0ca4`]: giảm chiều bằng PCA, fit ở S0, năm mục
   1. kiểm: `rerun` bước 05 trùng từng bit; solver `randomized` (scikit-learn 1.5.2: n = 764 < 10·d), rút RNG toàn cục, ma trận chuyển vị;
   2. code: 09, 10 chạy trên đặc trưng thô 768 chiều, còn 06, 08, 11–14 trong không gian PCA 64 chiều; PCA ngẫu nhiên chỉ xoay 3–4 hướng cuối, nơi các trị riêng sát nhau;
   3. phương sai: không gian con của S0 giữ 0.862 phương sai của S0 nhưng 0.663 của test lớp mới; log det hiệp phương sai mẫu của mỗi lớp cũ từ −35 tới −80, dù 24–32/64 hướng có phương sai trong lớp lớn hơn 1;
   4. lớp mới: giữ 0.53–0.69 phương sai trong lớp nhưng 0.84–0.94 khoảng cách giữa các tâm; ít nhất ở ba cặp crease, rolled_pit, waist_folding;
   5. qua 15 ảnh chụp: mỗi cặp seed khác nhau ở 3–4 hướng (lớn nhất 21.7–79.3°); không seed nào trùng PCA tối ưu.
6. `06-gauss-svi` [có, commit `4112807`]: Gaussian theo lớp, học bằng SVI ở S0. Code dùng lại ở 11, 13. Bốn mục
   1. kiểm: `rerun` bước 06 (SVI trên GPU, chế độ tất định) trùng từng bit; khe 5–9 giữ μ = 0, Σ = I;
   2. code: "SVI" là MLE bằng Adam trên minibatch 128 ảnh, guide rỗng, không prior, không KL (D11); tổng learning rate 0.0334 sau 1000 bước, max|μ| đạt 98.5% con số đó (D23);
   3. params0 so với dữ liệu: r_μ 0.992–0.996, trị riêng Σ 0.82–1.32; độ chính xác định hình trong 100 bước đầu; det Σ < 1 do phần ngoài đường chéo (½‖E‖² 0.42–0.51) và vết âm, det trung bình 0.3707 (Q06.05);
   4. qua 15 ảnh chụp: seed chỉ đổi tối đa 4/241 dự đoán trên test S0 và dưới 1% det; FT seed đổi det (0.37 so với 0.42–0.43), nên đổi ngưỡng dừng.
7. `07-early-stop` [có, commit `0d6d7ad`]: luật dừng sớm theo det Σ; chỉ chạy ở phiên online. Viết đúng thứ tự: ngưỡng của lượt F, lượt S tính từ `params0`; 11 và 13 kiểm, quỹ đạo det theo bước nằm ở 13 (QD24). Ba mục
   1. kiểm: `rerun` bước 07 trùng (chỉ chọn `stop_default`);
   2. code: so det trung bình của 5 lớp mới với det trung bình lớp cũ, sf = 2 ở lượt F và 0.707 ở lượt S, chỉ dừng sau bước 333 và quay lui một bước; `--scaling-factor` và `--early_stop_ratio` là cờ chết (Q07.02); ngưỡng của r0 s4: S 0.524, Eq. 11 0.374, F 0.185;
   3. qua 15 ảnh chụp: lượt F không dừng lần nào, lượt S dừng ở bước 468–620; giữa các FT seed bước dừng đi theo ngưỡng, trong một FT seed đi theo phân hoạch của bước 09.
8. `08-classifier` [có, commit `6cad5fa`]: bộ phân lớp và bảng S0. Code dùng lại ở 12, 14. Năm mục
   1. kiểm: `rerun` bước 08 trùng; cả bốn dòng của bảng S0 là 206/241;
   2. code: argmax trên đủ 10 khe của log-mật độ Gaussian đầy đủ (có log det, tiên nghiệm đều), tức QDA;
   3. cơ chế: tách đúng log-mật độ thành b = −½ log det Σ, q = −½ xᵀ(Σ⁻¹ − I)x, ℓ = μᵀΣ⁻¹x; giữa 5 lớp cũ q trải 11.0, ℓ 0.75, b 0.29; q một mình trùng pipeline ở 240/241 ảnh; Σ − I gần như là ngân sách bước × dấu của hiệp phương sai lớp; bỏ log det (đọc theo paper) không đổi ảnh nào ở r0 s4 (D25);
   4. lỗi S0: 23/35 ảnh sai nằm giữa ba lớp quanh mối hàn; 17/35 bị đoán thành một lớp có hộp trong chính ảnh (Q08.03); camera không phải nguồn lỗi chính;
   5. qua 15 ảnh chụp: q quyết định ở mọi ảnh chụp; bỏ log det đổi 0–3 ảnh; không ảnh nào vào khe 5–9 (D24).
9. `09-cluster` [có, commit `8982942`]: phân cụm pool, không đọc nhãn (QD26). Nguồn cũ: `12_cluster`. Bốn mục
   1. kiểm: `rerun` bước 09 trùng 3/3 khoá; `n_init` 10 của pipeline đúng là lần có inertia nhỏ nhất trong 10 lần khởi tạo đơn rút liên tiếp từ `rng_in` (15/15);
   2. code: `KMeansCluster` gọi `KMeans` của scikit-learn 1.5.2 (k-means++ thử 3 ứng viên mỗi tâm, Lloyd) trên đặc trưng thô 748 × 768, không trên PCA-64;
   3. landscape (`lab/landscape.py`, 10.000 lần khởi tạo đơn mỗi ảnh chụp): 521–1238 nhóm nghiệm ở ARI ≥ 0.95; pipeline ở nhóm sâu nhất 7/15; hai lần chạy pipeline trên cùng pool có E[ARI] 0.75–0.90; các seed của một FT seed rút từ gần như cùng một landscape, nên khác biệt giữa seed ở bước 09 chủ yếu là may rủi khởi tạo; RNG cố định với hàng hoán vị không cùng phân phối với rút khởi tạo mới (sửa QD26);
   4. cụm × camera, cụm × cuộn trên toàn pool: Cramér's V 0.50–0.65 và 0.71–0.78, chưa tách khỏi lớp (để `09d`).
10. `10-naming` [có, commit `b2723cb`]: đặt tên cụm bằng Hungarian trên tập test, chỉ dùng khi chấm. Năm mục
    1. kiểm: `rerun` bước 10 trùng 2/2 khoá; `pi_test` tính lại trùng 15/15;
    2. code: Hungarian cực đại tổng số ảnh đúng trên bảng (lớp × cụm) của 214 ảnh test, cụm lấy từ `predict_raw` của k-means (768 chiều thô), không từ bộ phân lớp; ở r0 s4 hai cụm silk mang tên rolled, crease (0/6, 0/11);
    3. ba cách đặt tên: theo số ảnh (pipeline), tối ưu cho balanced, nhiều–một theo lớp chủ; balanced mức cụm theo pipeline 0.458 / 0.386 / 0.405 (r0 / r1 / r2); cách tối ưu cho balanced bị thổi phồng 2.0–4.4 điểm, kiểm chéo còn +2.2 / +6.7 / −2.4;
    4. sổ mất điểm (`score.ledger`, QD24): cột 09, 10; ở r0 s4 phần mất 09 → 10 chỉ là silk (30/124 ảnh test);
    5. Q10.03: cách chia silk không giữ trên test (tổng biến phân 0.19–0.33, 15/15); ô silk dồn vào ít cuộn co lại (Spearman +0.45 đến +0.77).
11. `11-gauss-f` [có, commit `e46f92d`]: lượt F, SVI trên nhãn cụm. Bốn mục
    1. kiểm: `rerun` bước 11 trùng từng bit (1000 bước SVI, 59 s); 5 khe cũ trùng `params0` từng bit; không dừng sớm;
    2. code: nhãn là id cụm cho cả 748 ảnh pool, khởi tạo từ `params0`, nhánh F của luật dừng (ngưỡng det cũ / 2 = 0.185 ở r0 s4); det mới cuối lượt 0.230, dưới ngưỡng của lượt S (0.524);
    3. `params_f` so với dữ liệu: khe mới dùng đủ ngân sách (max|μ| 0.0315–0.0324); det theo độ tản của cụm (Spearman +0.89 trên 75 khe), cụm silk chặt có det 0.14–0.18;
    4. cột 11 của sổ mất điểm (phản thực tế, dừng sau lượt F): balanced mức cụm giảm ở 15/15 (r0 / r1 / r2: 0.458 → 0.421, 0.386 → 0.311, 0.405 → 0.386); 71–81% ảnh mất sang một khe mới khác, chỉ 19–29% bị khe cũ cướp (chủ yếu inclusion → oil_spot).
12. `12-filter` [có, commit `c331526`]: lọc cũ/mới, giữ ảnh bị đoán là lớp mới. Bốn mục
    1. kiểm: `rerun` bước 12 trùng 1/1 khoá (`keep`); số giữ trùng `S1_novel_kept` ở 15/15 (687–702/748);
    2. code: chỉ lọc, lượt S học id cụm của ảnh được giữ (D12); bộ phân lớp của lượt F trùng id cụm ở 89.6–92.6% pool, chỉ 7–21 ảnh được giữ nằm ở khe mới khác, nên re-labeling của paper đổi ít nhãn lớp mới;
    3. so với nhãn thật: loại 39–48/50 ảnh cũ (lọt chủ yếu oil, water; theo seed) và 5–16/698 ảnh mới (inclusion 4–14/135, silk 0; inclusion → water 80/157 trên 15 ảnh chụp); cụm crease của r0 s2 từ 34/93 ảnh cũ còn 2/59; q quyết định ở 98.4–100% ảnh; bỏ log det (D25) đổi 22 ảnh mới bị loại và 4 ảnh cũ lọt trên 15 ảnh chụp; bộ lọc chỉ dùng ℓ lọt ×1.38, loại ×2.6;
    4. cột 12 của sổ: 80% ảnh mới bị loại nằm ở cụm sai tên; độ sạch nhãn của tập học 0.27–0.45 → 0.29–0.48 (+1.7 đến +3.1 điểm); balanced của pool đổi dưới 1 điểm.
13. `13-gauss-s` [có, commit `a4b9f7e`]: lượt S, SVI lại trên ảnh được giữ, khởi tạo từ tham số S0; Hình 2 của paper trên GC10 (QD24, sửa 2026-11-05). Bốn mục
    1. kiểm: `rerun` bước 13 trùng từng bit; dừng ở bước 526; 5 khe cũ trùng `params0`;
    2. ngân sách: lượt S dùng 48–64% tổng lr; max|μ| của khe mới bám tỷ lệ này trong 1 điểm; det cuối nằm ngay trên ngưỡng S;
    3. quỹ đạo ở r0 s4 (bọc `calculate_metrics_on_covariances` ở mức class, ngay trong notebook): det khe mới giảm đơn điệu từ 1 (Q07.01); lượt F cắt ngưỡng S ở bước 541, det cũ ở 740; lớp mới 0% tới bước 50–100; All đạt đỉnh ở bước 462–484, gần điểm dừng; cùng ngân sách, bộ lọc đổi lớp cũ 0 ảnh, lớp mới +2; dừng ở 525 thay vì 1000 giữ 9 ảnh test cũ, mất 5 ảnh test mới (balanced novel −2.0);
    4. qua 15 ảnh chụp: bước dừng đi theo độ tản của các cụm (Spearman với phương sai trung bình trong cụm +0.90 / +0.90 / +0.56; với det của lượt F +0.90 / +0.90 / +1.00).
14. `14-eval` [chưa có]: đánh giá S1 (All/Old/New/S0test, M_f, M_d).

Biến thể và phần thêm:
- `05a-camera` [có, commit `75e1bf9`]: phân tích cho bước 05 (Q05.03), làm trước 08 (QD25); bốn mục
  1. dữ liệu: gắn mỗi hàng của ảnh chụp với ảnh của nó theo byte (camera, cuộn thép, lớp);
  2. PCA-64, phân tách phương sai (R² hiệu chỉnh) thành lớp / chung / camera / dư: S0 0.304 / 0.145 / 0.070 / 0.481, lớp chiếm pc1–pc4; 698 ảnh lớp mới 0.187 / 0.086 / 0.232 / 0.495;
  3. đặc trưng thô (không gian của k-means): lớp mới 0.146 / 0.063 / 0.198 / 0.593; camera giải thích 0.431 phương sai trong silk_spot, 0.353 trong inclusion, 0.08–0.11 trong crease, rolled_pit, waist_folding; trong silk_spot, cuộn và camera cùng giải thích 0.607;
  4. qua 15 ảnh chụp và `fz`: với lớp mới, camera hơn lớp ở cả 3 FT seed; `fz` có lớp 0.119, camera 0.221.
- `06a-closed-form` [có trong code, chưa có notebook]: E06.03, `--estimator closed`; đã đóng ngày 2/10.
- `09a-overcluster` [có hàm `v09a_overcluster`, chưa có notebook]: phân cụm thừa K' = 20; E09.04, E09.05. Nguồn cũ: `12b_cluster`.
- `09b-overcluster-merge` [chưa có]: gộp cụm con; E09.06, chưa tiền đăng ký. Nguồn cũ: phần Bhattacharyya của `12b_cluster`.
- `09c-multi-init` [chưa có]: E09.07 (tiền đăng ký `1ce3233`), best-of-1000 so với best-of-10 qua 11 lần rút lồng nhau mỗi ảnh chụp; chạy sau 14 (QD26).
- `09d-ownership` [có, commit `abbc3af`; kế hoạch chốt trước khi đọc nhãn ở `d282b37`]: phân tích có nhãn cho bước 09 (QD26). Bảy mục
  1. kiểm: `score09` trùng 26 khoá `k5_*` của E09.05 v2 ở 15/15;
  2. thành phần cụm: xếp sai chủ yếu do cấu trúc (chỉ 4.5% ảnh xếp sai có hộp của lớp chủ cụm); crease và rolled đi theo waist hoặc chung một cụm với ảnh cũ, inclusion lẫn vào silk; CV cỡ lớp 1.17 so với cỡ cụm 0.24–0.57;
  3. sổ Ward: giải phóng một cụm tốn 3645–4748, tách một lớp nhỏ lợi tối đa 1709;
  4. oracle: loại (c) ở 15/15 (ARI với lớp 0.34–0.42); basin cao hơn inertia tốt nhất 0.9–3.6%, best-of-10 không rơi vào;
  5. độ bất ổn theo lớp: hình dạng đổi theo FT seed (r1 dồn vào lớp hiếm, r2 vào inclusion);
  6. Q09.13: trong silk, camera và cuộn giải thích 0.87–0.92 thành viên cụm, mỗi yếu tố có ý nghĩa khi giữ yếu tố kia; inclusion 0.66–0.71;
  7. crease, r0 s2 so với r0 s4: trên cùng pool, cấu hình crease có inertia cao hơn 0.23–0.35% nhưng balanced accuracy tầng 1 0.56–0.57 so với 0.32–0.33.
- `14a-loss-by-step` [chưa có]: phân tích, mất điểm ở khâu nào; Q14.04. Chỉ đọc sổ mất điểm: một hàm trong `score.py`, dựng dần từ notebook 10, mỗi notebook 11–14 thêm cột (QD24). Nguồn cũ: `04_s1_discovery` (cell 9: recall sau đặt tên và ở cuối pipeline, r0 s4).
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
- `score.py`: bộ đo tầng 1, mỗi bước một hàm (`score09`; sau thêm `score10`, `score12`); `tier1_09` chạy K' cụm từ ảnh chụp của một run, lệnh `python lab/score.py 09 --from <run>` qua `exp.py`. Ngoài các lệnh kiểm của `replay.py`, đây là nơi duy nhất trong `lab/` đọc nhãn thật của pool, chỉ để chấm. Thay `pl_cluster.py` (xoá ngày 2026-10-21) sau khi tái lập tầng 1 của E09.05 ở 15/15 cặp (run `gc10_e09_05_v2_*`, commit `b81b78d`). Sổ mất điểm (QD24; commit `3aeb5ae`): `ledger(run)` trả đúng/sai từng ảnh của `test1` và `pool` theo cột bước (hiện 09, 10, 11, 12; cột 11 thêm ở `ae132c1`, cột 12 của tập `pool` ở `4f8166d`; notebook 13–14 thêm cột); `ledger_summary` gộp theo lớp, kèm luồng mất/được giữa hai cột liền nhau và hàng balanced.
- `data_io.py`: dữ liệu thô cho các notebook `00-data-*` (QD23; commit `8910925`, `2f6f1e5`). `images(dataset)`: bảng ảnh của file split, mỗi ảnh một dòng, theo thứ tự hàng của file đặc trưng; vai trò S0, reserve, S1, unused, test (mục 10). `objects(dataset, path)`: hộp lỗi trong XML của một ảnh, cùng quy tắc với `scripts/make_gc10_split.py`. `diagnostics(dataset)`: tính lại mục `diagnostics` của file split, trùng 466/75/6; đọc 2282 file XML trong khoảng 1 s nên không cần bộ nhớ đệm. Chỉ đọc file split và XML, không cần torch, jax. NEU thêm ở commit `beec7b9`: đường dẫn ảnh như `build()` của `feature_extractor/dino-neudet.py`, vai trò như `dataloaders/neudet.py`, XML tra theo tên file trong cả hai thư mục `annotations` (XML của ảnh train `crazing_240` nằm trong `validation/annotations`); `diagnostics("neudet")` cho 123/0/0. Không đặt tên `datasets.py` (sẽ che gói `datasets` của HuggingFace).
- `landscape.py`: landscape của k-means ở bước 09 (QD26; commit `1adc3e2`): nhiều lần khởi tạo k-means++ đơn trên pool của một ảnh chụp, không đọc nhãn (chỉ `pool_x`, `ids`, `inertia`, `rng_in`; khoá `true_*` bỏ khi nạp). `singles(run, n)`: n lần khởi tạo đơn từ một luồng RNG riêng, 1 luồng OpenMP; `permuted(run, m)`: m lần chạy `n_init` 10 từ `rng_in` với hàng hoán vị; `decompose(run)`: tách `n_init` 10 của pipeline thành 10 lần khởi tạo đơn; `run_all`: chạy song song theo ảnh chụp. Cache trên máy ở `runs/<run>/landscape/` (git bỏ qua), khoá theo băm của ảnh chụp, code và tham số. Hàm phân tích không rút RNG: `keep_rule`, `canon`, `ari`, `ari_many`, `modes`, `assign`, `best_of_weights`, `good_turing`, `instability`; `modes_all` chạy `modes` song song (sửa ở commit `62f81a8`).

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
- Notebook mô hình một câu hỏi chính; câu hỏi thứ hai thì tách thành chữ cái mới. Không giới hạn thời gian chạy (QD26; trước đó: Restart & Run All không quá 5 phút): cell chạy lâu in thời gian chạy. Phần tính dùng chung giữa các notebook viết trong `.py` và được lưu cache trên máy, khoá theo băm của đầu vào, code và tham số; Restart & Run All đọc cache có khoá đúng được tính là chạy lại. Notebook dữ liệu được phép dài.
- Cell code đầu tiên (cell 0) in commit và trạng thái dirty, rồi đặt `XLA_FLAGS` ở chế độ GPU tất định (deterministic) trước mọi lần import jax, vì `new_state` của bàn mổ dừng nếu jax đã được import mà thiếu cờ này (mẫu ở mục 11). Cell nạp dữ liệu đầu tiên in đầu vào thật của notebook: run và đường dẫn đặc trưng trong ảnh chụp, hoặc file split kèm sha256. Trước khi commit: Restart & Run All, rồi kiểm năm điều: số chạy của các cell code là 1, 2, …, N; không cell nào lỗi; mỗi cell code kẹp giữa hai ô Markdown; có Summary (EN); commit mà cell 0 in ra trùng commit ghi trong ô Sau của nó (thêm 2026-11-01, sau lỗi ở `00-data-gc10`, sửa ở `75a7086`). Commit kèm output, hình nhỏ.
- Không huấn luyện nặng trong notebook: đọc đặc trưng có sẵn; tối đa chạy SVI cho 1 seed.
- Bộ đo viết một lần trong `.py`, notebook import lại. Xem code trong model: in đúng hàm bằng `inspect.getsource` cạnh đầu vào/đầu ra của nó, không chép code.
- Đánh giá hai tầng: tầng 1 chấm đầu ra của một bước bằng nhãn thật (chỉ để chấm), để lọc nhanh; tầng 2 chạy cả pipeline qua `exp.py`, ít nhất 5 seed. Thứ hạng ở tầng 1 phải được xác nhận ở tầng 2.
- Bối cảnh mặc định: cấu hình paper (`ft`); `fz` là bối cảnh phụ (QD19).
- Ảnh chụp minh hoạ (QD24): hình và giải thích dùng `gc10_v2_ft_r0_s4`; từ bước 09, cell cần cơ chế của một cụm crease dùng `gc10_v2_ft_r0_s2`. Mỗi minh hoạ ghi ảnh chụp đó đứng ở đâu trong 15 ảnh chụp, ví dụ "r0 s4: crease không có cụm riêng ở K = 5, như 9/15 ảnh chụp". Số để kết luận tính trên cả 15 ảnh chụp khi rẻ (đọc pickle); kết luận về biểu diễn lấy FT seed làm đơn vị (n = 3).
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
