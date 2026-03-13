import json
import os
from pathlib import Path
import re
import warnings
from typing import Dict, List, Optional, Union
import torch
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from torch import nn
from transformers import AutoTokenizer
from .config import GLiDREConfig
from gliner.data_processing import SpanProcessor, TokenProcessor
from gliner.data_processing.tokenizer import WordsSplitter
from .base_model import BaseModel
from .base_model import SpanPairModel
from .processor import SpanPairBiEncoderProcessor
from .decoder import SpanPairDecoder
from safetensors.torch import save_file

class GLiDRE(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        config: GLiDREConfig,
        model: Optional[Union[BaseModel]] = None,
        tokenizer: Optional[Union[str, AutoTokenizer]] = None,
        words_splitter: Optional[Union[str, WordsSplitter]] = None,
        data_processor: Optional[Union[SpanProcessor, TokenProcessor]] = None,
        encoder_from_pretrained: bool = True,
    ):
        """
        Initialize the GLiDRE model.

        Args:
            config (GLiDREConfig): Configuration object for the GLiDRE model.
            model (Optional[Union[BaseModel, BaseORTModel]]): GLiDRE model to use for predictions. Defaults to None.
            tokenizer (Optional[Union[str, AutoTokenizer]]): Tokenizer to use. Can be a string (path or name) or an AutoTokenizer instance. Defaults to None.
            words_splitter (Optional[Union[str, WordsSplitter]]): Words splitter to use. Can be a string or a WordsSplitter instance. Defaults to None.
            data_processor (Optional[Union[SpanProcessor, TokenProcessor]]): Data processor - object that prepare input to a model. Defaults to None.
            encoder_from_pretrained (bool): Whether to load the encoder from a pre-trained model or init from scratch. Defaults to True.
        """
        super().__init__()
        self.config = config

        if tokenizer is None and data_processor is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, add_prefix_space=True)

        if words_splitter is None and data_processor is None:
            words_splitter = WordsSplitter(config.words_splitter_type)

        if model is None:
            self.model = SpanPairModel(config, encoder_from_pretrained)
        else:
            self.model = model
        if data_processor is None:
            if config.labels_encoder is not None:
                labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder, add_prefix_space=True)
                self.data_processor = SpanPairBiEncoderProcessor(config, tokenizer, words_splitter, labels_tokenizer)
            else:
                self.data_processor = SpanProcessor(config, tokenizer, words_splitter)
        else:
            self.data_processor = data_processor
        self.decoder = SpanPairDecoder(config)

        if config.vocab_size != -1 and config.vocab_size != len(
            self.data_processor.transformer_tokenizer
        ):
            warnings.warn(f"""Vocab size of the model ({config.vocab_size}) does't match length of tokenizer ({len(self.data_processor.transformer_tokenizer)}). 
                            You should to consider manually add new tokens to tokenizer or to load tokenizer with added tokens.""")
        # to suppress an AttributeError when training
        self._keys_to_ignore_on_save = None

    def forward(self, *args, **kwargs):
        """Wrapper function for the model's forward pass."""
        output = self.model(*args, **kwargs)
        return output

    @property
    def device(self):
        device = next(self.model.parameters()).device
        return device

    def prepare_texts(self, texts: List[str], mentions):
        """
        Prepare inputs for the model.

        Args:
            texts (str): The input text or texts to process.
            labels (str): The corresponding labels for the input texts.
        """
        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []
        start_text_to_token_idx = {}
        end_text_to_token_idx = {}

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            i = 0
            for token, start, end in self.data_processor.words_splitter(text):
                start_text_to_token_idx[start] = i
                end_text_to_token_idx[end] = i
                i += 1
                tokens.append(token)
                start_token_idx_to_text_idx.append(start)
                end_token_idx_to_text_idx.append(end)
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)

        input_x = [{"tokenized_text": tk, "mentions": None} for tk in all_tokens]

        if mentions is not None :
            for i, x in enumerate(input_x):
                mentions_x = []
                for mention in mentions[i]:
                    mention_indexed = {"id" : mention["id"], "mentions": []}
                    for ref in mention["mentions"]:
                        mention_indexed["mentions"].append({
                            "start": start_text_to_token_idx[ref["start"]],
                            "end": end_text_to_token_idx[ref["end"]],
                            "value": ref["value"]
                        })
                    mentions_x.append(mention_indexed)
                x['mentions'] = mentions_x

        return input_x, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx


    def prepare_model_inputs(self, texts: List[str], labels: List[str], mentions, prepare_entities: bool = True):
        """
        Prepare inputs for the model.

        Args:
            texts (str): The input text or texts to process.
            labels (str): The corresponding labels for the input texts.
        """
        # preserving the order of labels
        labels = list(dict.fromkeys(labels))

        class_to_ids = {k: v for v, k in enumerate(labels, start=1)}
        id_to_classes = {k: v for v, k in class_to_ids.items()}

        input_x, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_texts(texts, mentions)

        raw_batch = self.data_processor.collate_raw_batch(input_x, labels,
                                                        class_to_ids = class_to_ids,
                                                        id_to_classes = id_to_classes)
        raw_batch["all_start_token_idx_to_text_idx"] = all_start_token_idx_to_text_idx
        raw_batch["all_end_token_idx_to_text_idx"] = all_end_token_idx_to_text_idx

        model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=False,
                                                        prepare_entities=prepare_entities)
        model_input.update(
        {
            "rel_idx": raw_batch.get("rel_idx", None),  
            "rel_labels": raw_batch.get("rel_labels", None),  
            "text_lengths": raw_batch["seq_length"],  
        }
        )

        device = self.device
        for key in model_input:
            if model_input[key] is not None and isinstance(
                model_input[key], torch.Tensor
            ):
                model_input[key] = model_input[key].to(device)

        return model_input, raw_batch

    def predict_entities(
        self, text, labels, threshold=0.5, multi_label=False, mentions = None
    ):
        """
        Predict entities for a single text input.

        Args:
            text: The input text to predict entities for.
            labels: The labels to predict.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per entity. Defaults to False.

        Returns:
            The list of entity predictions.
        """
        return self.batch_predict_entities(
            [text],
            labels,
            threshold=threshold,
            multi_label=multi_label,
            mentions = [mentions]
        )[0]

    @torch.no_grad()
    def batch_predict_entities(
        self, texts, labels,threshold=0.5, multi_label=False, mentions = None
    ):
        """
        Predict entities for a batch of texts.

        Args:
            texts (List[str]): A list of input texts to predict entities for.
            labels (List[str]): A list of labels to predict.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per token. Defaults to False.

        Returns:
            The list of lists with predicted entities.
        """

        model_input, raw_batch = self.prepare_model_inputs(texts, labels, mentions)
        model_output = self.model(**model_input)[0]
        if not isinstance(model_output, torch.Tensor):
            model_output = torch.from_numpy(model_output)

        outputs = self.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            raw_batch["rel_idx"],
            model_output,
            threshold=threshold,
            multi_label=multi_label,
            mentions = mentions
        )

        all_relations = []
        for i, relation_output in enumerate(outputs):  
            start_token_idx_to_text_idx = raw_batch["all_start_token_idx_to_text_idx"][i]
            end_token_idx_to_text_idx = raw_batch["all_end_token_idx_to_text_idx"][i]
            text = texts[i]  
            mentions_data = mentions[i] 
            
            relations = []
            for relation in relation_output:
                entity_1_spans = []
                entity_2_spans = []
                def find_entity_id(span, mentions_data):
                    for entity in mentions_data:
                        if any((m["start"], m["end"]) == span for m in entity["mentions"]):
                            return entity["id"]
                    return None  
                
                for start_token_idx, end_token_idx in relation["entity_1"]:
                    start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                    end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                    span_text = text[start_text_idx:end_text_idx]
                    span = (start_text_idx, end_text_idx)
                    entity_id = find_entity_id(span, mentions_data)
                    
                    entity_1_spans.append({
                        "id": entity_id,
                        "start": start_text_idx,
                        "end": end_text_idx,
                        "text": span_text
                    })
                
                for start_token_idx, end_token_idx in relation["entity_2"]:
                    start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                    end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                    span_text = text[start_text_idx:end_text_idx]
                    span = (start_text_idx, end_text_idx)
                    entity_id = find_entity_id(span, mentions_data)
                    
                    entity_2_spans.append({
                        "id": entity_id,
                        "start": start_text_idx,
                        "end": end_text_idx,
                        "text": span_text
                    })
                
                relations.append({
                    "entity_1": entity_1_spans,
                    "entity_2": entity_2_spans,
                    "relation_type": relation["relation_type"],
                    "score": relation["score"]
                })

            all_relations.append(relations)

        return all_relations
    
    def predict_tokenized_entities(self, tokenized_text, labels, threshold=0.5, multi_label=False, mentions = None) :
        return self.batch_predict_tokenized_entities(
            [tokenized_text],
            labels,
            threshold=threshold,
            multi_label=multi_label,
            mentions = [mentions]
        )[0]


    def check_for_constraint(self, constraint, entity_1, entity_2) :
        for rel in constraint :
            if entity_1[0]["id"] == rel[0] and entity_2[0]["id"] == rel[1] :
                return True
        return False

    
    @torch.no_grad()
    def batch_predict_tokenized_entities(
        self, tokenized_texts, labels, threshold=0.5, multi_label=False, mentions = None, top_k = 100000, constraints=None
    ):
        labels = list(dict.fromkeys(labels))

        class_to_ids = {k: v for v, k in enumerate(labels, start=1)}
        id_to_classes = {k: v for v, k in class_to_ids.items()}

        input_x = [{"tokenized_text": tk, "mentions": mention} for tk, mention in zip(tokenized_texts, mentions)]

        raw_batch = self.data_processor.collate_raw_batch(input_x, labels,
                                                        class_to_ids = class_to_ids,
                                                        id_to_classes = id_to_classes)

        model_input = self.data_processor.collate_fn(raw_batch)
        model_input.update(
        {
            "text_lengths": raw_batch["seq_length"],  
            "rel_idx": raw_batch.get("rel_idx", None)
        }
        )

        device = self.device
        for key in model_input:
            if model_input[key] is not None and isinstance(
                model_input[key], torch.Tensor
            ):
                model_input[key] = model_input[key].to(device)

        model_output = self.model(**model_input).logits

        if not isinstance(model_output, torch.Tensor):
            model_output = torch.from_numpy(model_output)

        outputs = self.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            raw_batch["rel_idx"],
            model_output,
            threshold=threshold,
            multi_label=multi_label,
            mentions=mentions
        )

        all_relations = []
        for i, relation_output in enumerate(outputs):  
            tokenized_text = tokenized_texts[i]  
            mentions_data = mentions[i] 
            relations = []
            for relation in relation_output:
                entity_1_spans = []
                entity_2_spans = []
                def find_entity_id(span, mentions_data):
                    for entity in mentions_data:
                        if any((m["start"], m["end"]) == span for m in entity["mentions"]):
                            return entity["id"]
                    return None  
                
                for start_token_idx, end_token_idx in relation["entity_1"]:
                    start_text_idx = start_token_idx
                    end_text_idx = end_token_idx
                    span_text = tokenized_text[start_text_idx:end_text_idx+1]
                    span = (start_token_idx, end_token_idx)
                    entity_id = find_entity_id(span, mentions_data)
                    
                    entity_1_spans.append({
                        "id": entity_id,
                        "start": start_text_idx,
                        "end": end_text_idx,
                        "text": span_text
                    })
                
                for start_token_idx, end_token_idx in relation["entity_2"]:
                    start_text_idx = start_token_idx
                    end_text_idx = end_token_idx
                    span_text = tokenized_text[start_text_idx:end_text_idx+1]
                    span = (start_token_idx, end_token_idx)
                    entity_id = find_entity_id(span, mentions_data)
                    
                    entity_2_spans.append({
                        "id": entity_id,
                        "start": start_text_idx,
                        "end": end_text_idx,
                        "text": span_text
                    })
                if entity_1_spans and entity_2_spans :
                    if (not constraints) or self.check_for_constraint(constraints[i], entity_1_spans, entity_2_spans):
                        relations.append({
                            "entity_1": entity_1_spans,
                            "entity_2": entity_2_spans,
                            "relation_type": relation["relation_type"],
                            "score": relation["score"]
                        })

            relations = sorted(relations, key=lambda x: x["score"], reverse=True)[:top_k]
            all_relations.append(relations)

        return all_relations
    
    def compile_for_training(self):
        print("Compiling transformer encoder...")
        self.model.token_rep_layer = torch.compile(self.model.token_rep_layer, backend="aot_eager")
        print("Compiling RNN...")
        self.model.rnn = torch.compile(self.model.rnn, backend="aot_eager")
        if hasattr(self.model, "span_rep_layer"):
            print("Compiling span representation layer...")
            self.model.span_rep_layer = torch.compile(self.model.span_rep_layer, backend="aot_eager")
        if hasattr(self.model, "prompt_rep_layer"):
            print("Compiling prompt representation layer...")
            self.model.prompt_rep_layer = torch.compile(self.model.prompt_rep_layer, backend="aot_eager")
        if hasattr(self.model, "scorer"):
            print("Compiling scorer...")
            self.model.scorer = torch.compile(self.model.scorer, backend="aot_eager")
        if hasattr(self.model, "labels_encoder"):
            print("Compiling labels encoder...")
            self.model.scorer = torch.compile(self.model.scorer, backend="aot_eager")

    def prepare_state_dict(self, state_dict):
        """
        Prepare state dict in the case of torch.compile
        """
        new_state_dict = {}
        for key, tensor in state_dict.items():
            key = re.sub(r"_orig_mod\.", "", key)
            new_state_dict[key] = tensor
        return new_state_dict
    
    def compile(self):
        self.model = torch.compile(self.model)

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[GLiDREConfig] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        safe_serialization = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            safe_serialization (`bool`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        model_state_dict = self.prepare_state_dict(self.model.state_dict())
        # save model weights using safetensors
        if safe_serialization:
            save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))
        else:
            torch.save(
                model_state_dict,
                os.path.join(save_directory, "pytorch_model.bin"),
            )

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            config.to_json_file(save_directory / "glidre_config.json")

        self.data_processor.transformer_tokenizer.save_pretrained(save_directory)
        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None


    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        load_tokenizer: Optional[bool] = False,
        compile_torch_model: Optional[bool] = False,
        **model_kwargs,
    ):
        """
        Load a pretrained model from a given model ID.

        Args:
            model_id (str): Identifier of the model to load.
            revision (Optional[str]): Specific model revision to use.
            cache_dir (Optional[Union[str, Path]]): Directory to store downloaded models.
            force_download (bool): Force re-download even if the model exists.
            proxies (Optional[Dict]): Proxy configuration for downloads.
            resume_download (bool): Resume interrupted downloads.
            local_files_only (bool): Use only local files, don't download.
            token (Union[str, bool, None]): Token for API authentication.
            map_location (str): Device to map model to. Defaults to "cpu".
            strict (bool): Enforce strict state_dict loading.
            load_tokenizer (Optional[bool]): Whether to load the tokenizer. Defaults to False.
            compile_torch_model (Optional[bool]): Compile the PyTorch model. Defaults to False.
            **model_kwargs: Additional keyword arguments for model initialization.

        Returns:
            An instance of the model loaded from the pretrained weights.
        """
        # Use "pytorch_model.bin" and "glidre_config.json"
        model_dir = Path(model_id)  # / "pytorch_model.bin"
        if not model_dir.exists():
            model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        model_file = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "pytorch_model.bin")
        config_file = Path(model_dir) / "glidre_config.json"

        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
            else:
                tokenizer = None
        config_ = json.load(open(config_file))
        config = GLiDREConfig(**config_)

        glidre = cls(config, tokenizer=tokenizer, encoder_from_pretrained=False)
        state_dict = torch.load(model_file, map_location=torch.device(map_location), weights_only=True)
        glidre.model.load_state_dict(state_dict, strict=strict)
        glidre.model.to(map_location)
        if compile_torch_model and "cuda" in map_location:
            print("Compiling torch model...")
            glidre.compile()
        elif compile_torch_model:
            warnings.warn(
                "It's not possible to compile this model putting it to CPU, you should set `map_location` to `cuda`."
            )
        glidre.eval()
        return glidre