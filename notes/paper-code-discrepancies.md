
## D3 (CONFIRMED). Eq. 2 and Eq. 3 are NOT implemented
Code, feature_extractor/sl_finetuned_model.py, the entire training step:
    cls_loss = F.cross_entropy(output, batch["labels"].to(device))
    loss = cls_loss
No contrastive loss, no lambda, no temperature, no augmented view pairs.

Training transform is deterministic (Resize -> CenterCrop -> ToTensor -> Normalize):
no RandomResizedCrop, no flip, no color jitter. Consistent with the absence of
contrastive learning, which requires two augmented views per image.

Contradicts:
- Eq. 2 (L_con, InfoNCE) — not present.
- Eq. 3 (L_ft = (1-lambda)L_ce + lambda*L_con) — only L_ce present.
- Sec. 3.3 core component (1): "Self-supervised learning based offline fine-tuning"
  — fine-tuning is purely supervised.
- Sec. 3.2.1: "learn label-independent features ... through techniques such as data
  augmentation" — no augmentation applied during fine-tuning.
- Table 1 credits "Offline Fine-Tuning" with addressing Label Bias; with plain CE the
  claimed mechanism (label-independent features) is absent.

Impact: one of four claimed core components does not exist as described. The Table 5
"w/o fine-tuning" ablation removes fine-tuning wholesale and therefore cannot isolate
the contribution of the contrastive term (which was never there).

## D4. Random seed: code uses 42, paper claims 0
feature_extractor/sl_finetuned_model.py:31 -> torch.manual_seed(42)
Paper, Appendix C.3: "All random seeds are set to 0."
TO VERIFY whether main.py sets a different seed for stage 2.

## Additional notes on the fine-tuning implementation
- BOFT config: boft_block_size=4, boft_n_butterfly_factor=2, boft_dropout=0.1,
  bias="boft_only", target_modules=["output.dense","mlp.fc1","mlp.fc2"].
  This part DOES match the paper's "updating only the MLP and output dense layers".
- optimizer: AdamW, lr=1e-3, 10 epochs — matches Appendix C.3.
- MultiCropWrapper docstring describes multi-resolution forward passes, but forward()
  performs a single pass. Vestigial code copied from the DINO repository.
