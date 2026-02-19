# import torch
#
#
# class DPGradClipAdvisor:
#     def __init__(self):
#         self.all_norms = []
#
#     @torch.no_grad()
#     def record(self, model):
#         """
#         Record per-sample gradient norms for this batch.
#         Supports both Tensor and list grad_sample (for complex modules like UNet).
#         """
#         batch_norms = []
#
#         for p in model.parameters():
#             if hasattr(p, "grad_sample") and p.grad_sample is not None:
#                 gs = p.grad_sample
#
#                 if isinstance(gs, list):
#                     # 每个元素 reshape 后 stack
#                     gs = torch.stack([g.reshape(g.size(0), -1) for g in gs], dim=1)
#                     # gs shape: [batch_size, num_parts, flattened_param]
#                     gs = gs.view(gs.size(0), -1)
#                 else:
#                     gs = gs.view(gs.size(0), -1)
#
#                 norms = gs.norm(dim=1)
#                 batch_norms.append(norms)
#
#         if batch_norms:
#             norms = torch.cat(batch_norms)
#             self.all_norms.append(norms.cpu())
#
#     def recommend(self, quantile=0.95):
#         """
#         Recommend a clip norm based on accumulated statistics.
#         """
#         if not self.all_norms:
#             return None
#
#         all_norms = torch.cat(self.all_norms)
#
#         stats = {
#             "mean": all_norms.mean().item(),
#             "std":  all_norms.std().item(),
#             "p50":  all_norms.quantile(0.50).item(),
#             "p90":  all_norms.quantile(0.90).item(),
#             "p95":  all_norms.quantile(0.95).item(),
#             "max":  all_norms.max().item(),
#         }
#
#         C = stats[f"p{int(quantile * 100)}"]
#
#         return stats, C
import torch


class DPGradClipAdvisor:
    def __init__(self, max_batches=10):
        """
        max_batches: 最多保留多少个 batch 的 per-sample norms（防止内存爆炸）。
        """
        self.all_norms = []  # list of 1D CPU tensors, each length = batch_size_of_that_batch
        self.max_batches = max_batches

    @torch.no_grad()
    def record(self, model):
        """
        Record per-sample gradient norms for this batch.
        Supports both Tensor and list grad_sample (for complex modules like UNet).
        This implementation computes per-sample total norm as:
            sqrt( sum_over_params (sum_over_param_dims g^2) )
        """
        per_sample_sq = None  # will be tensor on the same device as grad_sample, shape [B]

        for p in model.parameters():
            gs = getattr(p, "grad_sample", None)
            if gs is None:
                continue

            # gs can be Tensor of shape [B, ...] or list of tensors (for complex modules)
            if isinstance(gs, list):
                # 如果是 list，把每个元素 flatten 并累加平方和
                for part in gs:
                    part_flat = part.view(part.size(0), -1)  # [B, D]
                    sq = (part_flat * part_flat).sum(dim=1)   # [B]
                    if per_sample_sq is None:
                        per_sample_sq = sq
                    else:
                        per_sample_sq = per_sample_sq + sq
            else:
                # 普通 tensor
                flat = gs.view(gs.size(0), -1)  # [B, D]
                sq = (flat * flat).sum(dim=1)   # [B]
                if per_sample_sq is None:
                    per_sample_sq = sq
                else:
                    per_sample_sq = per_sample_sq + sq

        if per_sample_sq is None:
            # 没有 grad_sample（可能没启用 per-sample grad），什么都不记录
            return

        # per_sample_sq 在 GPU（或 param device）上，转换为范数，并搬到 CPU 存储
        per_sample_norm = torch.sqrt(per_sample_sq)

        # 为避免频繁 sync，在这里一次性转到 CPU
        self.all_norms.append(per_sample_norm.detach().cpu())

        # 限制保留的 batch 数量（防止内存用尽）
        if len(self.all_norms) > self.max_batches:
            # 丢弃最早的
            self.all_norms.pop(0)

    def recommend(self, quantile=0.95):
        """
        Recommend a clip norm based on accumulated statistics.
        Returns (stats_dict, recommended_C).
        stats contains mean/std/p50/p90/p95/max.
        """
        if not self.all_norms:
            return None, None

        # 拼接所有保存在 CPU 上的 per-sample norms
        all_norms = torch.cat(self.all_norms)  # 1D tensor

        stats = {
            "mean": all_norms.mean().item(),
            "std":  all_norms.std(unbiased=False).item(),
            "p50":  all_norms.quantile(0.50).item(),
            "p90":  all_norms.quantile(0.90).item(),
            "p95":  all_norms.quantile(0.95).item(),
            "max":  all_norms.max().item(),
            "count": int(all_norms.numel()),
        }

        C = all_norms.quantile(quantile).item()
        return stats, C
