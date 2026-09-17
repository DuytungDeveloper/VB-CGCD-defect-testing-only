# Thay đổi code so với upstream

Mốc so sánh: tag `upstream-baseline` (= `7fe9501`, daihao42/VB-CGCD).
Số dòng tính theo **file hiện tại** trong fork. Kiểm tra ngày 2026-09-17 bằng lệnh:

```bash
git diff upstream-baseline -- feature_extractor/ .gitattributes
```

## Tóm tắt

| File | Dòng | Thay đổi | Có đổi kết quả không? |
|---|---|---|---|
| `feature_extractor/dino-cifar100.py` | 17 | PATCH 1: thêm `@torch.no_grad()` | Không |
| `feature_extractor/sl_finetuned_model.py` | 76 | PATCH 2: batch size lấy từ biến môi trường | Có, khi dùng 32×4 |
| `feature_extractor/sl_finetuned_model.py` | 92–123 | PATCH 2: gradient accumulation | Có, khi dùng 32×4 |
| `feature_extractor/sl_finetuned_model.py` | 125–130 | PATCH 5: lưu checkpoint | Không |
| `.gitattributes` | cả file | quy tắc xuống dòng (LF) | Không |

**Code phase 2 không bị sửa:** `main.py`, `classifier/`, `clustering/`, `dataloaders/`, `utils/` giống hệt upstream.

---

## 1. `feature_extractor/dino-cifar100.py`

### Dòng 17: thêm `@torch.no_grad()` (PATCH 1)

```python
@torch.no_grad()
def infer_features_labels(dino, data_loader, features_dir, labels_dir, device, args):
```

**Vì sao:** `dino.eval()` không tắt autograd. Khi trích đặc trưng, PyTorch vẫn xây đồ thị gradient cho toàn mạng, nên batch 128 cần khoảng 15 GB VRAM. Máy chỉ có 8 GB.

**Ảnh hưởng:** không đổi số. Phép tính forward y hệt, chỉ giảm bộ nhớ.

---

## 2. `feature_extractor/sl_finetuned_model.py`

### Dòng 76: batch size lấy từ biến môi trường (PATCH 2)

```python
# upstream
batch_size = 128
# bây giờ
batch_size = int(os.environ.get("FT_BATCH_SIZE", 128))
```

**Vì sao:** upstream cố định batch 128 và không có tham số dòng lệnh. Fine-tune với batch 128 không vừa GPU 8 GB.

### Dòng 92–123: gradient accumulation (PATCH 2)

| Dòng | Code | Việc |
|---|---|---|
| 92 | `accum = int(os.environ.get("FT_ACCUM", 1))` | số batch nhỏ gộp thành 1 bước |
| 94–95 | `optimizer.zero_grad()`, khởi tạo `pending, running, nb` | đầu mỗi epoch |
| 105 | `loss = cls_loss / accum` | chia loss để gradient cộng dồn bằng trung bình |
| 109–114 | `pending += 1`; `if pending == accum: step, zero_grad` | cập nhật trọng số mỗi `accum` batch |
| 116 | `running += cls_loss.item(); nb += 1` | cộng dồn loss để in |
| 118–120 | `if pending > 0: step, zero_grad` | cập nhật nốt các batch dư cuối epoch |
| 122–123 | `print(f"Epoch {epoch}: mean CE = ...")` | in 1 dòng mỗi epoch |

Dòng bị xóa của upstream:
- `optimizer.zero_grad()` trước `loss.backward()` (chuyển xuống sau `step`).
- `print(f"Epoch {epoch}, Loss: {loss.item()}")` in mỗi batch (thay bằng in mỗi epoch).

**Vì sao:** chạy batch 32 × 4 để có batch hiệu dụng 128 và số bước mỗi epoch bằng upstream (196 bước).

**Ảnh hưởng:**
- Khi không đặt biến môi trường (`FT_ACCUM=1`): giống upstream.
- Khi dùng `FT_BATCH_SIZE=32 FT_ACCUM=4` (cấu hình đã chạy): **có đổi số, mức nhỏ**, vì
  1. Dropout của BOFT rút mask mới mỗi lần forward, nên mỗi bước dùng 4 mask thay vì 1.
  2. Rút số ngẫu nhiên nhiều hơn, nên thứ tự mẫu từ epoch 1 có thể khác (suy luận).
  3. Bước cuối mỗi epoch có trọng số khác (0.25/0.25 thay vì 0.8/0.2).

### Dòng 125–130: lưu checkpoint (PATCH 5, thay PATCH 4)

```python
ckpt_path = os.environ.get("FT_CKPT_PATH", "/tmp/ft_ckpt.pt")
os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
ckpt = {n: p.detach().cpu() for n, p in lora_model.named_parameters()
        if p.requires_grad or "pooler.dense" in n}
torch.save({"epoch": epoch, "state": ckpt}, ckpt_path)
```

**Vì sao:** hai lần chạy bị chết giữa chừng mà không còn gì trên đĩa. Lưu tham số huấn luyện được và pooler (khởi tạo ngẫu nhiên, không có trong checkpoint DINO) sau mỗi epoch, vào thư mục của từng run.

**Ảnh hưởng:** không đổi số. Chỉ ghi ra đĩa, không đọc lại, không rút số ngẫu nhiên.

---

## 3. `.gitattributes` (file mới)

Quy định ký tự xuống dòng LF cho `.py`, `.sh`, `.md`, `.txt`, `.json`... và đánh dấu `.npy`, `.pt` là file nhị phân.

**Vì sao:** tránh lỗi CRLF khi làm việc giữa Windows và WSL.

**Ảnh hưởng:** không đổi số.

---

## 4. Khác biệt không do sửa code

**BOFT butterfly factor 2 → 1.** Code khai báo `boft_n_butterfly_factor=2`, nhưng peft không build được extension CUDA `fbd_cuda`, nên tự hạ xuống 1 (chỉ báo `UserWarning`). Xác nhận: `trainable params: 221,184` (factor 2 sẽ là 405,504).

**Ảnh hưởng:** có đổi cấu hình huấn luyện. Paper không nhắc tham số này.

---

## 5. Kết luận

- Nguồn làm đổi số so với upstream: **PATCH 2 (32×4)** và **butterfly factor 1**. Cả hai ở phase 1.
- Kết quả 5 seed: S0 91.90 ± 0.07 (paper 91.68), final_All 79.86 ± 1.68 (paper 81.23). Mức lệch chấp nhận được.
- Chi tiết đầy đủ: `notes/reproducibility-issues.md` (tiếng Anh).