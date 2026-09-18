#!/usr/bin/env python
"""Do VRAM thuc te cua buoc fine-tune BOFT o nhieu batch size.

Dung dung load_model() cua repo (khong sao chep lai cau hinh), chay
forward + backward + optimizer.step() tren du lieu tong hop.

Usage (tu repo root):
    python scripts/vram_probe.py
    python scripts/vram_probe.py 8 16 32 48 64
"""
import sys, os, gc
import torch

sys.path.insert(0, "feature_extractor")   # de `import utils` tro toi
                                          # feature_extractor/utils.py, KHONG
                                          # phai package utils/ o repo root
from sl_finetuned_model import load_model  # noqa: E402
from torch.optim import AdamW              # noqa: E402
import torch.nn.functional as F            # noqa: E402

GB = 1024 ** 3
MODEL = "dino_vitb16"
NUM_CLASSES = 100
IMG = 224


def free_total():
    free, total = torch.cuda.mem_get_info()
    return free / GB, total / GB


def trial(model, bs, device):
    """Mot buoc huan luyen day du. Tra ve peak allocated (GB) hoac None neu OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt = AdamW(model.parameters(), lr=1e-3)
    try:
        x = torch.randn(bs, 3, IMG, IMG, device=device)
        y = torch.randint(0, NUM_CLASSES, (bs,), device=device)
        out, _ = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()                 # AdamW state duoc cap phat o day
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / GB
        return peak
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        for v in ("x", "y", "out", "loss"):
            if v in dir():
                pass
        del opt
        gc.collect()
        torch.cuda.empty_cache()


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [8, 16, 32, 48, 64]
    assert torch.cuda.is_available(), "khong thay CUDA"
    device = torch.device("cuda")

    f0, tot = free_total()
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Tong VRAM      : {tot:.2f} GB")
    print(f"Tu do luc dau  : {f0:.2f} GB  (phan con lai do Windows/tien trinh khac giu)")

    print("\nDang tai model (lan dau se tai ~350 MB tu HuggingFace)...")
    model = load_model(NUM_CLASSES, MODEL).to(device)
    model.train()

    cfg = model.backbone.config
    attn = getattr(cfg, "_attn_implementation", "?")
    print(f"\nattn_implementation = {attn}   "
          f"(sdpa/flash_attention_2 tiet kiem hon eager nhieu)")
    print(f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} patch={cfg.patch_size}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"tham so kha huan luyen: {trainable:,} / {total_p:,} "
          f"({100 * trainable / total_p:.2f}%)")

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated() / GB
    print(f"trong so + head tren GPU: {base:.2f} GB")

    print(f"\n{'batch':>6} | {'peak alloc':>11} | {'activation':>11} | "
          f"{'vua ~':>7} | ket qua")
    print("-" * 62)
    ok = []
    for bs in sizes:
        peak = trial(model, bs, device)
        if peak is None:
            print(f"{bs:>6} | {'—':>11} | {'—':>11} | {'—':>7} | OOM")
            break
        act = peak - base
        head = f0 - peak
        verdict = "OK" if head > 0.8 else ("SAT MEP" if head > 0.2 else "RUI RO")
        if verdict == "OK":
            ok.append(bs)
        print(f"{bs:>6} | {peak:>8.2f} GB | {act:>8.2f} GB | "
              f"{head:>4.2f} GB | {verdict}")

    if ok:
        best = max(ok)
        pow2 = [b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b <= best]
        best = max(pow2) if pow2 else best   # uoc cua 128 -> batch hieu dung khop upstream
        accum = max(1, round(128 / best))
        print(f"\nDe xuat: FT_BATCH_SIZE={best}  FT_ACCUM={accum}  "
              f"-> batch hieu dung {best * accum}")
        if best * accum != 128:
            print(f"  CANH BAO: {best}x{accum} = {best * accum} != 128 (paper). "
                  f"Chon batch la uoc cua 128 de khop chinh xac.")
        print("\nLuu y: day la peak cua MOT buoc. Khau trich dac trung (sau khi")
        print("them @torch.no_grad()) re hon nhieu bac, batch 128 chay duoc.")


if __name__ == "__main__":
    main()