from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from gliner.data_processing import SpanBiEncoderProcessor
from typing import Dict, List, Tuple
import random
import torch
from itertools import product
from gliner.data_processing.utils import pad_2d_tensor
import warnings
import torch.nn.functional as F

class SpanPairBiEncoderProcessor(SpanBiEncoderProcessor): 
    @staticmethod
    def get_negatives(batch_list: List[Dict], sampled_neg: int = 5) -> List[str]:
        ent_types = []
        for b in batch_list:
            types = set([el[-2] for el in b['relations']])
            ent_types.extend(list(types))
        ent_types = list(set(ent_types))
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]
    
    def batch_generate_class_mappings(self, batch_list: List[Dict], negatives: List[str]=None) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:
        if negatives is None:
            negatives = self.get_negatives(batch_list, 100)
        #TODO voir si cette version copiée de gliner-bi suffisante (pas de classes négatives) -> sinon s'inspirer de BaseProcessor.batch_generate_class_mappings
        classes = []
        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0
            if "negatives" in b: # manually setting negative types
                negs_i = b["negatives"]
            else: # in-batch negative types
                negs_i = negatives[:len(b["relations"]) * neg_type_ratio] if neg_type_ratio else []
            types = list(set([el[-2] for el in b["relations"]] + negs_i))
            if "label" in b: # labels are predefined
                types = b["label"]
            classes.extend(types)
        random.shuffle(classes)
        classes = list(set(classes))
        class_to_id = {k: v for v, k in enumerate(classes, start=1)}
        id_to_class = {k: v for v, k in class_to_id.items()}
        class_to_ids = [class_to_id for i in range(len(batch_list))]
        id_to_classes = [id_to_class for i in range(len(batch_list))]
        
    
        return class_to_ids, id_to_classes
    
    def get_padding_length(self, batch_list):
        max_mentions = 0
        for batch in batch_list :
            for mentions in batch["mentions"]:
                    if len(mentions["mentions"])>max_mentions:
                        max_mentions = len(mentions["mentions"])
        return max_mentions

    def collate_raw_batch(self, batch_list: List[Dict], entity_types: List[str] = None, negatives: List[str]=None,
                            class_to_ids: Dict=None, id_to_classes: Dict=None) -> Dict:
        
        padding_length = self.get_padding_length(batch_list)
        if entity_types is None and class_to_ids is None:
            class_to_ids, id_to_classes = self.batch_generate_class_mappings(batch_list, negatives)
            batch = [self.preprocess_example(b["tokenized_text"], b["mentions"], class_to_ids[i], relations=b.get("relations"), padding_length=padding_length) for i, b in
                     enumerate(batch_list)]
        else:
            if class_to_ids is None:
                class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}
                id_to_classes = {k: v for v, k in class_to_ids.items()}
            batch = [self.preprocess_example(b["tokenized_text"], b["mentions"], class_to_ids, relations=b.get("relations"), padding_length=padding_length) for b in batch_list]
        return self.create_batch_dict(batch, class_to_ids, id_to_classes)
    def preprocess_example(self, tokens, mentions, classes_to_id, relations, padding_length):
        if len(tokens) == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(f"Sentence of length {len(tokens)} has been truncated to {max_len}")
            tokens = tokens[:max_len]

        span_idx = []
        for mention in mentions :
            mention_idx = []
            for entity in mention["mentions"] :
                mention_idx.append((entity["start"], entity["end"]))
            span_idx.append(mention_idx)
        
        padded_span_idx = [m + [(-1, -1)] * (padding_length - len(m)) for m in span_idx]

        rel_idx = []
        for el1 in padded_span_idx :
            for el2 in padded_span_idx :
                rel_idx.append((el1, el2))

        rel_idx = torch.LongTensor(rel_idx)

        if relations is not None :
            mentions_idx = list(product([mention["id"] for mention in mentions],[mention["id"] for mention in mentions]))
            rel_labels=[]
            for el in mentions_idx : 
                rels = [0 for _ in range(len(classes_to_id))]
                for rel in relations :
                    if rel[0] == el[0] and rel[2] == el[1] :
                        rels[classes_to_id[rel[1]]-1] = 1
                rel_labels.append(rels)
            rel_labels = torch.LongTensor(rel_labels) 
        else : 
            rel_labels = torch.LongTensor([[0 for _ in range(len(classes_to_id))] for el in rel_idx])
                

        return {
            "tokens": tokens,
            "span_idx": span_idx,
            "rel_idx" : rel_idx,
            "rel_label": rel_labels,
            "seq_length": len(tokens),
            "mentions": mentions,
            "relations" : relations
        }
    
    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        tokens = [el["tokens"] for el in batch]
        span_idx = [el["span_idx"] for el in batch]
        rel_idx = pad_sequence([b["rel_idx"] for b in batch], batch_first=True, padding_value=-1)
        rel_label = pad_sequence([el["rel_label"] for el in batch], batch_first=True, padding_value=0)
        seq_length = torch.LongTensor([el["seq_length"] for el in batch]).unsqueeze(-1)
        mentions = [el["mentions"] for el in batch]
        relations = [el["relations"] for el in batch]

        return {
            "tokens": tokens,
            "span_idx": span_idx,
            "rel_idx" : rel_idx,
            "rel_label": rel_label,
            "seq_length": seq_length,
            "mentions": mentions,
            "relations" : relations,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
            }


    def create_labels(self, batch):
        return batch["rel_label"].float()