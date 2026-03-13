from abc import ABC, abstractmethod
from torch import nn
import torch
import warnings
from typing import Optional
from .encoder import Encoder, BiEncoder
from .loss_functions import focal_loss_with_logits, f1_loss_with_logits, ATLoss, AFLoss
from .rel_rep import RelMarkerV0
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.layers import LstmSeq2SeqEncoder, CrossFuser, create_projection_layer, CrossAttentionBlock, SelfAttentionBlock
from gliner.modeling.base import GLiNERModelOutput, extract_prompt_features_and_word_embeddings

def extract_word_embeddings(token_embeds, attentions, words_mask, attention_mask, 
                                    batch_size, max_text_length, embed_dim, text_lengths):
    words_embedding = torch.zeros(
        batch_size, max_text_length, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    B, H, L1, L2 = attentions.size()

    words_attention = torch.zeros(
        batch_size, H, max_text_length, L1, dtype=attentions.dtype, device=attentions.device
    )

    words_attention_2 = torch.zeros(
        batch_size, H, max_text_length, max_text_length, dtype=attentions.dtype, device=attentions.device
    )

    batch_indices, word_idx = torch.where(words_mask>0)
    
    target_word_idx = words_mask[batch_indices, word_idx]-1


    words_embedding[batch_indices, target_word_idx] = token_embeds[batch_indices, word_idx]
    words_attention[batch_indices, :, target_word_idx, :] = attentions[batch_indices, :, word_idx, :]
    words_attention_2[:,:,:, target_word_idx] = words_attention[:,:,:, word_idx]
    
    aranged_word_idx = torch.arange(max_text_length, 
                                        dtype=attention_mask.dtype, 
                                        device=token_embeds.device).expand(batch_size, -1)
    word_mask = aranged_word_idx < text_lengths.unsqueeze(-1)
    return words_embedding, words_attention_2, word_mask

class BaseModel(ABC, nn.Module):
    def __init__(self, config, from_pretrained = False):
        super(BaseModel, self).__init__()
        self.config = config
        
        if not config.labels_encoder:
            self.token_rep_layer = Encoder(config, from_pretrained)
        else:
            self.token_rep_layer = BiEncoder(config, from_pretrained)
        if self.config.has_rnn:
            self.rnn = LstmSeq2SeqEncoder(config)

        if config.post_fusion_schema:
            self.config.num_post_fusion_layers = 3
            print('Initializing cross fuser...')
            print('Post fusion layer:', config.post_fusion_schema)
            print('Number of post fusion layers:', config.num_post_fusion_layers)

            self.cross_fuser = CrossFuser(self.config.hidden_size,
                                            self.config.hidden_size,
                                            num_heads=self.token_rep_layer.bert_layer.model.config.num_attention_heads,
                                            num_layers=self.config.num_post_fusion_layers,
                                            dropout=config.dropout, 
                                            schema=config.post_fusion_schema)
            
    def features_enhancement(self, text_embeds, labels_embeds, text_mask=None, labels_mask=None):
        labels_embeds, text_embeds = self.cross_fuser(labels_embeds, text_embeds, labels_mask, text_mask)
        return text_embeds, labels_embeds
    
    def _extract_prompt_features_and_word_embeddings(self, token_embeds, input_ids, attention_mask, 
                                                                    text_lengths, words_mask):
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = extract_prompt_features_and_word_embeddings(self.config, 
                                                                                                                       token_embeds, 
                                                                                                                       input_ids, 
                                                                                                                       attention_mask, 
                                                                                                                       text_lengths, 
                                                                                                                       words_mask,
                                                                                                                       self.config.embed_ent_token)
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def get_uni_representations(self, 
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                **kwargs):
        
        token_embeds = self.token_rep_layer(input_ids, attention_mask, **kwargs)

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self._extract_prompt_features_and_word_embeddings(token_embeds, input_ids, attention_mask, 
                                                                                                        text_lengths, words_mask)
        
        if self.config.has_rnn:
            words_embedding = self.rnn(words_embedding, mask)

        return prompts_embedding, prompts_embedding_mask, words_embedding, mask
    
    def get_bi_representations(self, 
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeds: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                **kwargs): 
        if labels_embeds is not None:
            token_embeds = self.token_rep_layer.encode_text(input_ids, attention_mask, **kwargs)
        else:
            token_embeds, labels_embeds = self.token_rep_layer(input_ids, attention_mask,
                                                           labels_input_ids, labels_attention_mask, 
                                                                                            **kwargs) 
        batch_size, sequence_length, embed_dim = token_embeds.shape
        max_text_length = text_lengths.max()
        words_embedding, mask = extract_word_embeddings(token_embeds, words_mask, attention_mask, 
                                    batch_size, max_text_length, embed_dim, text_lengths)
        
        labels_embeds = labels_embeds.unsqueeze(0)
        labels_embeds = labels_embeds.expand(batch_size, -1, -1)
        labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype,
                                                        device = attention_mask.device)

        labels_embeds = labels_embeds.to(words_embedding.dtype)
        if hasattr(self, "cross_fuser"):
            words_embedding, labels_embeds = self.features_enhancement(words_embedding, labels_embeds, text_mask=mask, labels_mask=labels_mask)
        
        return labels_embeds, labels_mask, words_embedding, mask

    def get_representations(self, 
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            labels_embeddings: Optional[torch.FloatTensor] = None,
            labels_input_ids: Optional[torch.FloatTensor] = None,
            labels_attention_mask: Optional[torch.LongTensor] = None,
            text_lengths: Optional[torch.Tensor] = None,
            words_mask: Optional[torch.LongTensor] = None,
            **kwargs):
        if self.config.labels_encoder:
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_bi_representations(
                    input_ids, attention_mask, labels_embeddings, labels_input_ids, labels_attention_mask, 
                                                                        text_lengths, words_mask, **kwargs
            )
        else:
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_uni_representations(
                                    input_ids, attention_mask, text_lengths, words_mask, **kwargs
            )
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask 
    
    @abstractmethod
    def forward(self, x):
        pass

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0, weights=None):

        if self.config.loss == "afloss" :
            loss = AFLoss(gamma_pos=gamma, gamma_neg = gamma)
            all_losses = loss(logits, labels, alpha = alpha, gamma = gamma)
        
        if self.config.loss == "f1" :
        
            all_losses = f1_loss_with_logits(logits, labels,
                                            alpha=alpha,
                                            gamma=gamma,
                                            label_smoothing=label_smoothing,
                                            weights = weights)
            
        if self.config.loss == "both" :
            all_losses = focal_loss_with_logits(logits, labels,
                                            alpha=alpha,
                                            gamma=gamma,
                                            label_smoothing=label_smoothing,
                                            weights = weights) + f1_loss_with_logits(logits, labels,
                                            alpha=alpha,
                                            gamma=gamma,
                                            label_smoothing=label_smoothing,
                                            weights = weights)
        if self.config.loss == "atloss" :
            loss = ATLoss()
            all_losses = loss(logits, labels, alpha = alpha, gamma = gamma)
        else :
            all_losses = focal_loss_with_logits(logits, labels,
                                            alpha=alpha,
                                            gamma=gamma,
                                            label_smoothing=label_smoothing,
                                            weights = weights)
            
        return all_losses

    @abstractmethod
    def loss(self, x):
        pass
    
class SpanModel(BaseModel):
    def __init__(self, config, encoder_from_pretrained):
        super(SpanModel, self).__init__(config, encoder_from_pretrained)
        self.span_rep_layer = SpanRepLayer(span_mode = config.span_mode, 
                                           hidden_size = config.hidden_size, 
                                           max_width = config.max_width,
                                           dropout = config.dropout)

        self.prompt_rep_layer = create_projection_layer(config.hidden_size, config.dropout)


    def forward(self,        
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeddings: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                words_embedding: Optional[torch.FloatTensor] = None,
                mask: Optional[torch.LongTensor] = None,
                prompts_embedding: Optional[torch.FloatTensor] = None,
                prompts_embedding_mask: Optional[torch.LongTensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                span_idx: Optional[torch.LongTensor] = None,
                span_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.FloatTensor] = None,
                **kwargs
                ):

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(input_ids, attention_mask, 
                                                                                labels_embeddings, labels_input_ids, labels_attention_mask, 
                                                                                                                    text_lengths, words_mask)
        print(span_mask)
        span_idx = span_idx*span_mask.unsqueeze(-1)

        span_rep = self.span_rep_layer(words_embedding, span_idx)
        
        prompts_embedding = self.prompt_rep_layer(prompts_embedding)

        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)
        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, span_mask, **kwargs)

        output = GLiNERModelOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output
    
    def loss(self, scores, labels, prompts_embedding_mask, mask_label,
                        alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0, 
                        reduction: str = 'sum', **kwargs):
        
        batch_size = scores.shape[0]
        num_classes = prompts_embedding_mask.shape[-1]

        scores = scores.view(-1, num_classes)
        labels = labels.view(-1, num_classes)
        
        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing)

        masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        mask_label = mask_label.view(-1, 1)
        
        all_losses = all_losses * mask_label.float()

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == 'sum':
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = all_losses.sum()
        return loss
    

class SpanPairModel(SpanModel):
    def __init__(self, config, encoder_from_pretrained):
        super(SpanModel, self).__init__(config, encoder_from_pretrained)
        self.span_rep_layer = RelMarkerV0(hidden_size = config.hidden_size, 
                                           dropout = config.dropout,
                                           atlop = config.atlop,
                                           self_rel = config.self_rel,
                                           pooling = config.pooling)

        self.prompt_rep_layer = create_projection_layer(config.hidden_size, config.dropout)
        if config.cross_attention :
            print("Cross attention layer initialized")
            #self.cross_attention_layer_prompt = CrossAttentionBlock(config.hidden_size, num_heads = 8, dropout = config.dropout)
            self.cross_attention_layer_rel = CrossAttentionBlock(config.hidden_size, num_heads = 8, dropout = config.dropout)
            #self.self_attention_layer_prompt = SelfAttentionBlock(config.hidden_size, num_heads = 8, dropout = config.dropout)

        if config.loss == "atloss" or config.loss == "afloss":
            print("Atlop threshold layer initialized")
            self.threshold_layer = nn.Sequential(nn.Linear(2*config.hidden_size, config.hidden_size),nn.ReLU(),nn.Dropout(self.config.dropout),nn.Linear(config.hidden_size, 1))

    def forward(self,        
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeddings: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                words_embedding: Optional[torch.FloatTensor] = None,
                mask: Optional[torch.LongTensor] = None,
                prompts_embedding: Optional[torch.FloatTensor] = None,
                prompts_embedding_mask: Optional[torch.LongTensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                rel_idx: Optional[torch.LongTensor] = None,
                span_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.FloatTensor] = None,
                **kwargs
                ):
    

        prompts_embedding, prompts_embedding_mask, words_embedding, mask, words_attentions = self.get_representations(input_ids, attention_mask, 
                                                                                labels_embeddings, labels_input_ids, labels_attention_mask, 
                                                                                                                    text_lengths, words_mask)
        rel_rep = self.span_rep_layer(words_embedding, rel_idx, words_attentions)
        if self.config.cross_attention :
            rel_rep = self.cross_attention_layer_rel(rel_rep, prompts_embedding)
            #prompt_embeddings = self.self_attention_layer_prompt(prompts_embedding)
            #prompts_embedding = self.cross_attention_layer_prompt(prompts_embedding, rel_rep)

        prompts_embedding = self.prompt_rep_layer(prompts_embedding)

        scores = torch.einsum("BLD,BCD->BLC", rel_rep, prompts_embedding)
        loss = None
        if self.config.loss == "atloss" or self.config.loss == "afloss":
            threshold_logit = self.threshold_layer(torch.cat([prompts_embedding.mean(1).unsqueeze(1).expand(-1, rel_rep.shape[1], -1),rel_rep], dim = -1))
            scores = torch.einsum("BLD,BCD->BLC", rel_rep, prompts_embedding)
            scores = torch.cat([threshold_logit, scores], dim=-1)
            if labels is not None:
                if labels.dim() == 2:
                    zeros = torch.zeros(labels.size(0), 1, device=labels.device)
                else :
                    zeros = torch.zeros(labels.size(0), labels.size(1), 1, device=labels.device)
                labels = torch.cat([zeros, labels], dim=-1)
        else:
            scores = torch.einsum("BLD,BCD->BLC", rel_rep, prompts_embedding)
            
        if labels is not None:
            if self.config.weights is not None:
                loss = self.loss(scores, labels, prompts_embedding_mask, alpha=self.config.alpha, gamma = self.config.gamma, reduction = self.config.reduction, weights = torch.tensor(self.config.weights, dtype=float).to(labels.device), **kwargs)
            else : 
                loss = self.loss(scores, labels, prompts_embedding_mask, alpha=self.config.alpha, gamma = self.config.gamma, reduction = self.config.reduction, **kwargs)

        output = GLiNERModelOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output
    
    def loss(self, scores, labels, prompts_embedding_mask,
                        alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0, 
                        reduction: str = 'sum',  weights= None, **kwargs):
        
        batch_size = scores.shape[0]
        num_classes = prompts_embedding_mask.shape[-1]
        if  self.config.loss == "atloss" or self.config.loss == "afloss" :
            num_classes += 1

        scores = scores.view(-1, num_classes)
        labels = labels.view(-1, num_classes)
        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing, weights)

        masked_loss = all_losses.view(batch_size, -1, num_classes) #* prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        all_losses = all_losses 

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == 'sum':
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = all_losses.sum()
        return loss
    
    def get_representations(self, 
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeds: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                **kwargs): 
        if labels_embeds is not None:
            token_embeds, attentions = self.token_rep_layer.encode_text(input_ids, attention_mask, **kwargs)
        else:
            token_embeds, attentions, labels_embeds = self.token_rep_layer(input_ids, attention_mask,
                                                           labels_input_ids, labels_attention_mask, 
                                                                                            **kwargs) 
        batch_size, sequence_length, embed_dim = token_embeds.shape
        max_text_length = text_lengths.max()
        words_embedding, words_attentions, mask = extract_word_embeddings(token_embeds, attentions, words_mask, attention_mask, 
                                    batch_size, max_text_length, embed_dim, text_lengths)
        
        labels_embeds = labels_embeds.unsqueeze(0)
        labels_embeds = labels_embeds.expand(batch_size, -1, -1)
        labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype,
                                                        device = attention_mask.device)

        labels_embeds = labels_embeds.to(words_embedding.dtype)
        if hasattr(self, "cross_fuser"):
            words_embedding, labels_embeds = self.features_enhancement(words_embedding, labels_embeds, text_mask=mask, labels_mask=labels_mask)
        
        return labels_embeds, labels_mask, words_embedding, mask, words_attentions
