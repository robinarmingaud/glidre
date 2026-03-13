from gliner.decoding.decoder import BaseDecoder
import torch

class SpanPairDecoder(BaseDecoder):
    def decode(self, tokens, id_to_classes, rel_idx, model_output, threshold=0.5, multi_label=False, mentions=None):
        probs = torch.sigmoid(model_output)  
        relation_pairs = []
        
        for batch_idx, token_batch in enumerate(tokens):
            relation_batch = []
            probs_batch = probs[batch_idx]  
            
            for rel_num, rel_probs in enumerate(probs_batch):
                if self.config.loss == "atloss" or self.config.loss == "afloss":
                    threshold = rel_probs[0]
                    rel_probs = rel_probs[1:]
                    
                entity_1, entity_2 = rel_idx[batch_idx, rel_num] 
                span_1 = [(s[0].item(), s[1].item()) for s in entity_1 if -1 not in s]  
                span_2 = [(s[0].item(), s[1].item()) for s in entity_2 if -1 not in s]  
                
                if multi_label:
                    rel_class_indices = torch.where(rel_probs > threshold)[0]
                    for rel_class in rel_class_indices:
                        relation_type = id_to_classes[rel_class.item()+1]
                        score = rel_probs[rel_class].item()
                        relation_batch.append({
                            "entity_1": span_1,
                            "entity_2": span_2,
                            "relation_type": relation_type,
                            "score": score
                        })
                else:
                    best_score, best_class = torch.max(rel_probs, dim=0)
                    if best_score > threshold:
                        relation_batch.append({
                            "entity_1": span_1,
                            "entity_2": span_2,
                            "relation_type": id_to_classes[best_class.item()+1],
                            "score": best_score.item()
                        })
            relation_pairs.append(relation_batch)
        
        return relation_pairs