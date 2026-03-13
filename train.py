import argparse
import json
import os
import shutil
import pandas as pd 
from glidre import GLiDRE, GLiDREConfig
import torch
from transformers import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from glidre.collator import DataCollator
from transformers import get_cosine_schedule_with_warmup
from glidre.training_utils import preprocess, get_labels, create_optimizer, calculate_f1_scores, convert_to_target_format, format_results
from gliner.utils import load_config_as_namespace
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default= "configs/config_finetuning.yaml")
    parser.add_argument('--log_dir', type=str, default = 'logs/')

    args = parser.parse_args()
    config = load_config_as_namespace(args.config)
    set_seed(config.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    if not config.tokenized : 
        train = pd.read_csv(config.train_data)
        train = preprocess(train)
        train = train.to_dict("records")
        if config.val_data :
            dev = pd.read_csv(config.val_data)
            dev = preprocess(dev)
            dev = dev.to_dict("records")
        if config.test_data :
            test = pd.read_csv(config.test_data)
            test = preprocess(test)
            test = test.to_dict("records")
    
    else :
        with open(config.train_data) as f :
            train = json.load(f)
        if config.val_data :
            with open(config.val_data) as f :
                dev = json.load(f)
        if config.test_data :
            with open(config.test_data) as f :
                test = json.load(f)

    if config.pretrain :
        if config.val_data and config.test_data :
            labels, labels_to_ids = get_labels(dev, test)
        elif config.val_data :
            labels, labels_to_ids = get_labels(dev, [])
        elif config.test_data :
            labels, labels_to_ids = get_labels(test, [])
    else : 
        if config.val_data:
            labels, labels_to_ids = get_labels(train, dev)
        else:
            labels, labels_to_ids = get_labels(train, [])


    if config.checkpoint :
        model = GLiDRE.from_pretrained(config.checkpoint)
        model_config = model.config
    else :
        model_config = GLiDREConfig(model_name = config.model_name,
                                labels_encoder = config.labels_encoder,  
                                max_width= config.max_width,
                                dropout= config.dropout,
                                fuse_layers = config.fuse_layers,
                                fine_tune=config.fine_tune,
                                post_fusion_schema = config.post_fusion_schema,
                                atlop = config.atlop,
                                span_mode= config.span_mode,
                                subtoken_pooling= config.subtoken_pooling,
                                max_len = config.max_len, 
                                hidden_size = config.hidden_size,
                                max_types= config.max_types,
                                max_neg_type_ratio= config.max_neg_type_ratio,
                                cross_attention = config.cross_attention,
                                alpha = config.loss_alpha,
                                loss = config.loss,
                                gamma = config.loss_alpha,
                                reduction = config.loss_reduction,
                                self_rel = config.self_rel,
                                pooling = config.pooling,
                                weights = None if not config.weights else config.weights
                                   )
        model = GLiDRE(model_config)
    model.to(device)
    if config.compile :
        torch.set_float32_matmul_precision('high')
        model.compile_for_training()
    model.train()
    
    optimizer = create_optimizer(model, config)
    
    if config.pretrain:
        data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    else : 
        data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True, labels=labels)

    train_dataloader = DataLoader(train, shuffle=True, collate_fn=data_collator, batch_size=config.train_batch_size, num_workers  = config.workers,pin_memory=True)

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
    pbar = tqdm(range(config.num_steps))
    eval_f1 = 0
    saved_chckpts = []
    best_f1 = 0
    best_global_th = 0.1

    for step in pbar:
        for _ in range(config.accumulation_steps):
            try:
                x = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_dataloader)
                x = next(iter_train_loader)

            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

            if config.pretrain:
                #create bigger batch artificially to get negative samples but solve oom issues
                subbatches = []
                for i in range(config.train_batch_size):
                    micro_batch = {}
                    for k,v in x.items():
                        if "labels_" not in k :
                            micro_batch[k] = v[i].unsqueeze(0)
                        else :
                            micro_batch[k] = v

                    subbatches.append(micro_batch)
                for micro_batch in subbatches:
                    out  = model(**micro_batch)
                    loss = out["loss"]
                    if not torch.isnan(loss):
                        loss = loss / (config.accumulation_steps * config.train_batch_size)
                        loss.backward()

            else :
                out = model(**x)  
                loss = out['loss']
                if not torch.isnan(loss):
                    loss = loss / config.accumulation_steps
                    loss.backward()
        
        pbar.set_description(f"Evaluation F1 at step {step + 1}: {eval_f1} / Training loss {loss}")    
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()  
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % config.eval_every == 0 and config.val_data:
            model.eval()
            with torch.no_grad():
                pred = []
                gold = []
                test_batch_size = config.train_batch_size
                for i in tqdm(range(0, len(dev), test_batch_size)):
                    batch = dev[i : i + test_batch_size]
                    tokenized_texts = [row["tokenized_text"] for row in batch]
                    mentions = [row["mentions"] for row in batch]
                    batch_predictions = model.batch_predict_tokenized_entities(
                        tokenized_texts,
                        mentions = mentions,
                        labels=labels,
                        threshold=0.1,
                        top_k = 100000,
                        multi_label=True
                    )
                    pred.extend(batch_predictions)
                    gold.extend([row["relations"] for row in batch])
            best_th_f1 = 0 
            best_th = 0.1
            
            for th in range(1,10):
                pred_converted  = [convert_to_target_format(prediction,  th/10) for prediction in pred]
                scores = calculate_f1_scores(gold, pred_converted)
                if scores['micro_scores']['f_score'] >= best_th_f1 :
                    best_th = th/10
                    best_th_f1 = scores['micro_scores']['f_score']
                    current_scores = scores
            print("THRESHOLD : ",best_th)
            print(format_results(current_scores))
            eval_f1  = current_scores['micro_scores']["f_score"]
            if eval_f1 > best_f1 :
                best_global_th = best_th
                best_f1 = eval_f1
                model.save_pretrained(config.save_path + "/best_model")
                with open(config.save_path + "/best_model/metrics_dev_th_"+str(best_th)+"_step_"+str(step+1)+".txt", "w") as file:
                    file.write(format_results(current_scores))
            model.train()  

        if (step+1)%config.save_steps == 0 and step != 0:
            saved_chckpts.append(config.save_path + f"/step_{step+1}")
            model.save_pretrained(config.save_path + f"/step_{step+1}")
            if len(saved_chckpts) > config.save_total_limit:
                shutil.rmtree(saved_chckpts.pop(0))
    if config.val_data and config.test_data :
        model = GLiDRE.from_pretrained(config.save_path + "/best_model")
        model.to(device)
        model.eval()
        pred = []
        gold = []
        test_batch_size = config.train_batch_size
        with torch.no_grad():
            for i in tqdm(range(0, len(test), test_batch_size)):
                batch = test[i : i + test_batch_size]
                tokenized_texts = [row["tokenized_text"] for row in batch]
                mentions = [row["mentions"] for row in batch]
                batch_predictions = model.batch_predict_tokenized_entities(
                    tokenized_texts,
                    mentions = mentions,
                    labels=labels,
                    threshold=best_global_th,
                    top_k = 100000,
                    multi_label=True
                )
                pred.extend([convert_to_target_format(prediction,best_global_th) for prediction in batch_predictions])
                gold.extend([row["relations"] for row in batch])
        scores = calculate_f1_scores(gold, pred)
        print(format_results(scores))
        with open(config.save_path + "/best_model/metrics_test.txt", "w") as file:
            file.write(format_results(scores))
                
    
