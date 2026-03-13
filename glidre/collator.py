from glidre.processor import SpanPairBiEncoderProcessor 

class DataCollator:
    def __init__(self, config, tokenizer=None, words_splitter=None, data_processor=None, 
                        return_tokens: bool = False,
                        return_id_to_classes: bool = False,
                        return_entities: bool = False,
                        prepare_labels: bool = False,
                        entity_types = None,
                        labels = None):
        self.config=config
        if data_processor is None:
            self.data_processor=SpanPairBiEncoderProcessor(config, tokenizer, words_splitter)
        else:
            self.data_processor = data_processor
        self.prepare_labels = prepare_labels
        self.return_tokens = return_tokens
        self.return_id_to_classes = return_id_to_classes
        self.return_entities = return_entities
        self.entity_types = entity_types
        self.labels=labels

    def __call__(self, input_x):
        if self.labels is not None  :
            labels = list(dict.fromkeys(self.labels))
            class_to_ids = {k: v for v, k in enumerate(labels, start=1)}
            id_to_classes = {k: v for v, k in class_to_ids.items()}
            raw_batch = self.data_processor.collate_raw_batch(input_x, entity_types = self.entity_types, class_to_ids=class_to_ids, id_to_classes=id_to_classes)
        else :
            raw_batch = self.data_processor.collate_raw_batch(input_x, entity_types = self.entity_types)
        model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=self.prepare_labels)
        model_input.update(
        {
            "rel_idx": raw_batch.get("rel_idx", None),  
            "rel_labels": raw_batch.get("rel_labels", None),  
            "text_lengths": raw_batch["seq_length"],  
        }
        )
        if self.return_tokens:
            model_input['tokenized_text'] = raw_batch['tokenized_text']
        if self.return_id_to_classes:
            model_input['id_to_classes'] = raw_batch['id_to_classes']
        if self.return_entities:
            model_input['mentions'] = raw_batch['mentions']
        model_input = {k:v for k, v in model_input.items() if v is not None}
        return model_input