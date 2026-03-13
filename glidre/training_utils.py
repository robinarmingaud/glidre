from collections import defaultdict
import re
import ast
import copy
import torch
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)


def tokenize_text(text) : 
    tokens = []
    for i, match in enumerate(re.finditer(r'\w+(?:_\w+)*|\S', text)):
        tokens.append(match.group())
    return tokens


def tokenize_and_align_labels(text, entities) :
    tokens = []
    end_matches = 0
    start_matches = 0
    mentions = 0
    new_entities = copy.deepcopy(entities)
    for i, match in enumerate(re.finditer(r'\w+(?:_\w+)*|\S', text)):
        tokens.append(match.group())
        for id_entity, entity in enumerate(entities) :
            for id_mention, mention in enumerate(entity["mentions"]):
                if mention["start"] == match.start():
                    new_entities[id_entity]["mentions"][id_mention]["start"] = i
                    start_matches += 1
                if mention["end"] == match.end():
                    new_entities[id_entity]["mentions"][id_mention]["end"] = i
                    end_matches += 1
    for entity in entities :
        for mention in entity["mentions"]:
            mentions += 1
    assert end_matches == mentions, f"Incorrect number of end matches, end matches = {end_matches}, mentions = {mentions}"
    assert start_matches == mentions, f"Incorrect number of start matches, start matches = {start_matches}, mentions = {mentions}"
    return new_entities

def preprocess(data):
    data['mentions'] = data.apply(lambda row: tokenize_and_align_labels(row['text'], ast.literal_eval(row['entities'])), axis=1)
    data["tokenized_text"] = data.apply(lambda row : tokenize_text(row["text"]), axis=1)
    data["relations"] = data.apply(lambda row : ast.literal_eval(row["relations"]), axis=1)
    return data

def get_labels(train,dev) :
    all_data = train+dev
    labels = set()
    for line in all_data :
        for relation in line["relations"]:
            labels.add(relation[1])

    relation_to_id = {k: v for v, k in enumerate(sorted(labels))}
    return sorted(labels), relation_to_id


def create_optimizer(opt_model, config, adam = True, **optimizer_kwargs):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if config.lr_others is not None:
            encoder_parameters = [name for name, _ in opt_model.named_parameters() if "token_rep_layer" in name]
            # encoder_parameters = [name for name, _ in opt_model.token_rep_layer.named_parameters()]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(config.weight_decay_other),
                    "lr": float(config.lr_others),
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": float(config.lr_others),
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(config.weight_decay_encoder),
                    "lr": float(config.lr_encoder),
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": float(config.lr_encoder),
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(config.weight_decay_encoder),
                    "lr": float(config.lr_encoder),
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": float(config.lr_encoder),
                },
            ]
        if adam : 
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer


def convert_to_target_format(data, threshold=0.3):
    result = []
    for item in data:
        id_1 = item['entity_1'][0]['id']
        id_2 = item['entity_2'][0]['id']
        relation_type = item['relation_type']
        if item["score"] >= threshold:
            result.append([id_1, relation_type, id_2])
    
    return result

def calculate_f1_scores(gold, pred):
    all_true_positives = defaultdict(int)
    all_false_positives = defaultdict(int)
    all_false_negatives = defaultdict(int)
    all_labels = set()
    total_support = defaultdict(int)

    for gold_relations, predicted_relations in zip(gold, pred):
        gold_set = set(tuple(rel) for rel in gold_relations)
        predicted_set = set(tuple(rel) for rel in predicted_relations)

        # Collect all labels from both gold and predicted sets
        for _, rel, _ in gold_set:
            all_labels.add(rel)
        for _, rel, _ in predicted_set:
            all_labels.add(rel)

        # Initialize counters for the current row
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)

        # Compute TP, FP for predicted relations
        for (s1, rel, e1) in predicted_set:
            if (s1, rel, e1) in gold_set:
                true_positives[rel] += 1
            else:
                false_positives[rel] += 1

        # Compute FN for missed gold relations
        for (s1, rel, e1) in gold_set:
            if (s1, rel, e1) not in predicted_set:
                false_negatives[rel] += 1

        # Accumulate results for each label
        for label in all_labels:
            all_true_positives[label] += true_positives[label]
            all_false_positives[label] += false_positives[label]
            all_false_negatives[label] += false_negatives[label]
            total_support[label] += (true_positives[label] + false_negatives[label])

    # Calculate scores for each class
    labels = sorted(all_labels)
    precision = []
    recall = []
    f1_score = []
    support = []

    for label in labels:
        tp = all_true_positives[label]
        fp = all_false_positives[label]
        fn = all_false_negatives[label]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
        support.append(total_support[label])

    # Calculate micro scores
    total_tp = sum(all_true_positives.values())
    total_fp = sum(all_false_positives.values())
    total_fn = sum(all_false_negatives.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Calculate macro scores (average of per-class scores)
    macro_precision = sum(precision) / len(precision) if len(precision) > 0 else 0
    macro_recall = sum(recall) / len(recall) if len(recall) > 0 else 0
    macro_f1 = sum(f1_score) / len(f1_score) if len(f1_score) > 0 else 0

    # Format the results
    evaluation_results = {
        'scores_by_class': {
            'target_names': labels,
            'precision': precision,
            'recall': recall,
            'f_score': f1_score,
            'true_sum': support
        },
        'micro_scores': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f_score': micro_f1
        },
        'macro_scores': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f_score': macro_f1
        }
    }
    
    return evaluation_results

def format_results(evaluation_results):
    scores_by_class = evaluation_results['scores_by_class']
    micro_scores = evaluation_results['micro_scores']
    macro_scores = evaluation_results['macro_scores']

    output_str = f"{'Label':<30}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'Support':<10}\n"
    output_str += "-" * 100 + "\n"

    for label, precision, recall, f_score, support in zip(scores_by_class['target_names'], scores_by_class['precision'], scores_by_class['recall'], scores_by_class['f_score'], scores_by_class['true_sum']):
        output_str += f"{label:<30}{precision:<15.4f}{recall:<15.4f}{f_score:<15.4f}{support:<10}\n"

    output_str += "-" * 100 + "\n"
    output_str += f"{'TOTAL (micro)':<30}{micro_scores['precision']:<15.4f}{micro_scores['recall']:<15.4f}{micro_scores['f_score']:<15.4f}{sum(scores_by_class['true_sum']):<10}\n"
    output_str += f"{'TOTAL (macro)':<30}{macro_scores['precision']:<15.4f}{macro_scores['recall']:<15.4f}{macro_scores['f_score']:<15.4f}{sum(scores_by_class['true_sum']):<10}\n"

    return output_str