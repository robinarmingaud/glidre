from datasets import load_dataset 
docred = load_dataset("thunlp/docred")
import json
from typing import List, Dict

def convert_docred_to_glidre(docred_data: List[Dict], save_path) -> List[Dict]:
    custom_data = []

    for doc_id, doc in enumerate(docred_data):
        tokenized_text = []
        for sent in doc["sents"]:
            tokenized_text+=sent

        entities = []
        for entity_id, entity_mentions in enumerate(doc["vertexSet"]):
            mentions = [
                {
                    "value": mention["name"],
                    "start": mention["pos"][0]+sum(len(doc["sents"][i]) for i in range(mention["sent_id"])),
                    "end": mention["pos"][1]+sum(len(doc["sents"][i]) for i in range(mention["sent_id"])) - 1
                }
                for mention in entity_mentions
            ]
            entities.append({
                "id": entity_id,
                "mentions": mentions,
                "type": entity_mentions[0]["type"] if entity_mentions else ""
            })

        relations = []
        labels = doc.get("labels", [])
        for head, relation_text, tail in zip(labels["head"], labels["relation_text"], labels["tail"]):
            relations.append([
                head,
                relation_text,
                tail
            ])

        custom_data.append({
            "id": doc_id,
            "tokenized_text": tokenized_text,
            "mentions": entities,
            "relations": relations
        })
    with open(split+".json", "w", encoding='utf8') as f :
        json.dump(custom_data, f, ensure_ascii=False, indent= 4)

for split in docred :
    convert_docred_to_glidre(docred[split], "DocRED/" + split)