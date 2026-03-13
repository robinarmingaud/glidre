import torch
from gliner.modeling.layers import create_projection_layer
from torch import nn

class RelMarkerV0(nn.Module):
    """
    rel representations
    """

    def __init__(self, hidden_size: int, dropout: float = 0.4, atlop = False, self_rel=False, pooling="mean"):
        super().__init__()
        self.atlop = atlop
        self.self_rel = self_rel
        self.pooling = pooling
        if atlop :
            self.head_extractor = create_projection_layer(hidden_size * 2, dropout, hidden_size)
            self.tail_extractor = create_projection_layer(hidden_size * 2, dropout, hidden_size)
        self.projection = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, rel_idx: torch.Tensor, attentions: torch.Tensor) -> torch.Tensor:
        """
        h: [B, L, D] - Hidden states from the encoder.
        rel_idx: [B, num_rel, 2, n_max_mentions, 2] - Start and end indices of mentions for an entity.
        attentions: [B, H, L, L] - attention values
        Returns:
            span_reps: [B, num_rel, D] - Pooled relation representations.
        """
        B, L, D = h.size()
        _, H, _, _ = attentions.size()
        B, num_rel, num_entities, n_max_mentions, _ = rel_idx.size()
        start_indices = rel_idx[..., 0].view(B, num_rel * num_entities * n_max_mentions)  # [B, num_rel * num_entities * n_max_mentions]
        end_indices = rel_idx[..., 1].view(B, num_rel * num_entities * n_max_mentions)  # [B, num_rel * num_entities * n_max_mentions]
        # ignore -1 padding indices
        valid_mask = (start_indices != -1) & (end_indices != -1) & (start_indices <= L-1) # [B, num_rel * num_entities * n_max_mentions]
        start_indices = torch.clamp(start_indices, min=0, max = L-1) 
        end_indices = torch.clamp(end_indices, min=0, max = L-1)

        lengths = (end_indices - start_indices + 1) * valid_mask  # [B, num_rel * num_entities * n_max_mentions]
        max_span_length = lengths.max().item()

        position_offsets = torch.arange(max_span_length, device=h.device).unsqueeze(0).unsqueeze(0)  # [1, 1, max_span_length]
        start_positions = start_indices.unsqueeze(-1)  # [B, num_rel * num_entities * n_max_mentions, 1]
        positions = start_positions + position_offsets  # [B, num_rel * num_entities * n_max_mentions, max_span_length]

        span_mask = (position_offsets < lengths.unsqueeze(-1)) & valid_mask.unsqueeze(-1)   # [B, num_rel * num_entities * n_max_mentions, max_span_length]
        positions = torch.clamp(positions * span_mask.long(), min=0, max=L - 1)
        positions_flat = positions.view(B, -1)  # [B, num_rel * num_entities * n_max_mentions * max_span_length]

        valid_rel = valid_mask.view(B, num_rel, num_entities, n_max_mentions,1, 1)
        if not self.self_rel:
            positions_structured = positions.view(B, num_rel, num_entities, n_max_mentions,max_span_length)
            valid_head = positions_structured[:,:,0,:,:]
            valid_tail = positions_structured[:,:,1,:,:]
            same_rel_mask = (valid_tail != valid_head).unsqueeze(2).unsqueeze(-1)
            valid_rel = valid_rel*same_rel_mask
        span_lengths = torch.clamp((lengths.float() * valid_mask.float()).view(B, num_rel, num_entities, n_max_mentions, 1), min=0)
        valid_span_mask = span_lengths.sum(-2) > 0
        span_mask = span_mask.view(B, num_rel, num_entities, n_max_mentions, max_span_length, 1)
        M = positions_flat.size(1)
        batch_indices = torch.arange(B, device=h.device).unsqueeze(1).expand(B, M)
        gathered_embeddings = h[batch_indices, positions_flat]
        gathered_embeddings = gathered_embeddings.view(B, num_rel, num_entities, n_max_mentions, max_span_length, D)* span_mask
        if self.pooling == "mean" :
            sum_embeddings = (gathered_embeddings*valid_rel).sum(dim=-2)  # Sum along max_span_length -> [B, num_rel, num_entities, n_max_mentions, D]
            span_reps = sum_embeddings.sum(dim=-2)/(span_lengths.sum(-2) + 1e-7)  # [B, num_rel, num_entities, D]
            span_reps = torch.where(valid_span_mask, span_reps, torch.zeros_like(span_reps))
        elif self.pooling == "logsumexp":
            B, num_rel, num_entities, n_max_mentions, max_span_length, D = gathered_embeddings.shape
            flat_embeddings = gathered_embeddings.view(B, num_rel, num_entities, n_max_mentions * max_span_length, D)
            flat_span_mask = span_mask.view(B, num_rel, num_entities, n_max_mentions * max_span_length)
            masked_embeddings = torch.where(
                flat_span_mask.unsqueeze(-1), 
                flat_embeddings,
                -1e9 
            )
            span_reps = masked_embeddings.logsumexp(dim=-2)
            span_reps = torch.where(valid_span_mask, span_reps, torch.zeros_like(span_reps))
        else :
            print("pooling method not implemented")
        
        #ATLOP-like
        if self.atlop :
            attention_avg = attentions.mean(dim=-3)
            gathered_attentions = attention_avg[batch_indices, positions_flat]
            gathered_attentions = gathered_attentions.view(B, num_rel, num_entities, n_max_mentions, max_span_length, L)* span_mask
            sum_attentions = (gathered_attentions*valid_rel).sum(dim=-2) # Sum attention along max_span_length -> [B, num_rel, num_entities, n_max_mentions, H, L]
            spans_att = sum_attentions.sum(dim=-2)/(span_lengths.sum(-2) + 1e-7) # [B, num_rel, num_entities, L]
            spans_att = torch.where(valid_span_mask, spans_att, torch.zeros_like(spans_att))
            As = spans_att[:, :, 0, :]
            Ao = spans_att[:, :, 1, :]
            attention_weights = (As * Ao)
            attention_weights = attention_weights/(attention_weights.sum(2, keepdim=True) + 1e-5)
            localized_context_embeddings = (attention_weights.unsqueeze(-1) * h.unsqueeze(1)).sum(dim=2)
            head = torch.tanh(self.head_extractor(torch.cat([span_reps[:, :, 0, :], localized_context_embeddings], dim=-1))) # [B, num_rel, 2 * D]
            tail = torch.tanh(self.tail_extractor(torch.cat([span_reps[:, :, 1, :], localized_context_embeddings], dim=-1)))
            rel_reps = self.projection(torch.cat([head, tail], dim=-1))
        else : 
            rel_reps = torch.cat([span_reps[:, :, 0, :], span_reps[:, :, 1, :]], dim=-1)  # [B, num_rel, 2 * D]
            rel_reps = self.projection(rel_reps)  # [B, num_rel, D]
        return rel_reps