import os
import json
import argparse
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import set_seed, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from glidre import GLiDRE
from glidre.collator import DataCollator
from gliner.utils import load_config_as_namespace
from glidre.training_utils import (
    preprocess, 
    get_labels, 
    create_optimizer, 
    calculate_f1_scores, 
    convert_to_target_format, 
    format_results
)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def evaluate(model, data, args, labels) :
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    pred = []
    gold = []
    test_batch_size = config.train_batch_size
    for i in tqdm(range(0, len(data), test_batch_size), desc="Evaluating", leave=False):
        batch_dev = data[i: i + test_batch_size]
        tokenized_texts = [row["tokenized_text"] for row in batch_dev]
        mentions = [row["mentions"] for row in batch_dev]
        batch_predictions = model.batch_predict_tokenized_entities(
            tokenized_texts,
            mentions=mentions,
            labels=labels,
            threshold=args.threshold,
            top_k=100000,
            multi_label=True
        )
        pred.extend(batch_predictions)
        gold.extend([row["relations"] for row in batch_dev])
    best_f1 = 0
    best_th = 0.1
    for th in range(1,10):
        pred_converted  = [convert_to_target_format(prediction,  th/10) for prediction in pred]
        scores = calculate_f1_scores(gold, pred_converted)
        print(th/10, ":", scores['micro_scores']['f_score'])
        if scores['micro_scores']['f_score'] > best_f1 :
            best_th = th/10
            best_f1 = scores['micro_scores']['f_score']
            best_scores = scores
    return best_f1, best_scores, best_th

def train_on_dataset(train_file, dev_file, test_file, config, log_dir, args):
    with open(train_file, "r", encoding="utf8") as f:
        train_data = json.load(f)
    with open(dev_file, "r", encoding="utf8") as f:
        dev_data = json.load(f)
    with open(test_file, "r", encoding="utf8") as f:
        test_data = json.load(f)
    
    
    labels, labels_to_ids = get_labels(train_data, dev_data)
    
    
    model = GLiDRE.from_pretrained(args.checkpoint)

    model.to(args.device)
    
    model.train()
    
    optimizer = create_optimizer(model, config)
    
    
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True, labels=labels)
    train_dataloader = DataLoader(
        train_data, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=config.train_batch_size, 
        num_workers=config.workers, 
        pin_memory=True
    )

    num_epochs = min(30000//len(train_data), 1000)
    config.num_steps = (len(train_dataloader) * num_epochs) // config.accumulation_steps
    
    if config.warmup_ratio < 1:
        num_warmup_steps = int(config.num_steps * config.warmup_ratio)
    else:
        num_warmup_steps = int(config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config.num_steps
    )
    
    iter_train_loader = iter(train_dataloader)
    pbar = tqdm(range(config.num_steps), desc="Training Steps")
    
    
    best_f1 = 0.0
    patience_counter = 0
    best_metrics = None
    eval_every = max(len(train_dataloader)//(config.train_batch_size*config.accumulation_steps), 100//(config.accumulation_steps))
    current_f1=0
    th=0
    for step in pbar:
        for _ in range(config.accumulation_steps):
            try:
                x = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_dataloader)
                x = next(iter_train_loader)

            x = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            
            out = model(**x)  
            loss = out['loss']
            loss = loss / config.accumulation_steps
            loss.backward()
            pbar.set_description(f"Step {step+1}, Training Loss: {loss.item():.4f}, Eval F1: {current_f1:.4f}, Threshold : {th:.2f}")
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        
        if (step + 1) % eval_every == 0:
            current_f1, scores, th = evaluate(model, dev_data, args, labels)
            pbar.set_description(f"Step {step+1}, Eval F1: {current_f1:.4f}, Threshold : {th:.2f}")
            model.train()
            if current_f1 > best_f1:
                best_f1 = current_f1
                test_f1, best_metrics, best_th = evaluate(model, test_data, args, labels)
                patience_counter = 0
                model.save_pretrained(os.path.join(args.log_dir, f"best_model_{os.path.splitext(filename)[0]}"))
            else:
                patience_counter += 1
            
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at step {step+1} (best F1: {best_f1:.4f})")
                break
        
    return best_metrics, best_th

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config_finetuning.yaml", help="Path to the YAML configuration file")
    parser.add_argument('--log_dir', type=str, default='logs/', help="Directory to save evaluation metrics")
    parser.add_argument('--train_dir', type=str, required=True, help="Directory containing training JSON files")
    parser.add_argument('--dev', type=str, required=True, help="Path to the dev JSON file")
    parser.add_argument('--test', type=str, required=True, help="Path to the test JSON file")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint file to load model weights")
    parser.add_argument('--tokenized', action="store_true", help="Flag indicating input data is already tokenized")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--threshold', type=float, default=0.1, help="Threshold for predictions")
    parser.add_argument('--early_stopping_patience', type=int, default=10000, help="Patience steps for early stopping")
    args = parser.parse_args()
    
    set_seed(args.seed)
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = load_config_as_namespace(args.config)
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    
    all_metrics = {}
    for filename in sorted(os.listdir(args.train_dir)):
        if filename.endswith('.json'):
            train_file_path = os.path.join(args.train_dir, filename)
            print(f"\n--- Training on dataset: {train_file_path} ---")
            metrics, th  = train_on_dataset(train_file_path, args.dev, args.test, config, args.log_dir, args)
            all_metrics[filename] = metrics
            metrics_file = os.path.join(args.log_dir, f"metrics_{os.path.splitext(filename)[0]}.json")
            with open(metrics_file, "w", encoding="utf8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            
            
    
    

