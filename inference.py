import argparse
import json
import pandas as pd
import torch
import tqdm
from glidre import GLiDRE, GLiDREConfig
from glidre.training_utils import convert_to_target_format, format_results, get_labels, preprocess

def inference_df(text, mentions) :
        if args.tokenized :
            return model.predict_tokenized_entities(tokenized_text = text,
                        labels = labels,  mentions = mentions, threshold = args.threshold, multi_label=True)
        else :
            return model.predict_entities(text = text,
                        labels = labels,  mentions = mentions, threshold = args.threshold, multi_label=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--tokenized', action="store_true")
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--zero_shot', action='store_true', default=False)
    

    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not args.tokenized : 
        test = pd.read_csv(args.test)
        test = preprocess(test)
        train = pd.read_csv(args.train)
        train = preprocess(train)
    
    else :
        with open(args.train) as f :
            train = json.load(f)
        with open(args.test) as f :
            test = json.load(f)
    
    labels, labels_to_ids = get_labels(train, test)
    if args.zero_shot :
        labels = [label.upper().replace(" ", "_") for label in labels]
        print('labels zero shot :', labels)

    model = GLiDRE.from_pretrained(args.model)
    model.to(device)
    model.eval()
    pred_dataset = []
    for row in tqdm.tqdm(test) :
        pred_row = {}
        if args.tokenized :
            pred_row["formatted_predictions"] = inference_df(row["tokenized_text"], row["mentions"])
        else :
            pred_row["formatted_predictions"] = inference_df(row["text"], row["mentions"])
        if args.zero_shot :
            for idx, pred in enumerate(pred_row["formatted_predictions"]) :
                pred_row["formatted_predictions"][idx]["relation_type"] = pred_row["formatted_predictions"][idx]["relation_type"].lower().replace("_", " ")
        pred_dataset.append(pred_row)
    
    with open(args.output, "w") as f:
        json.dump(pred_dataset, f, indent = 1)
