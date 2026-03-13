from typing import Optional
from transformers import PretrainedConfig, AutoConfig
from transformers.models.auto import CONFIG_MAPPING

class GLiDREConfig(PretrainedConfig):
    model_type = "glidre"
    is_composition = True
    def __init__(self, 
                 model_name: str = "microsoft/deberta-v3-small",
                 labels_encoder: str = "BAAI/bge-base-en-v1.5",
                 labels_encoder_config: Optional[dict] = None,
                 name: str = "glidre",
                 max_width: int = 12,
                 hidden_size: int = 512,
                 dropout: float = 0.4,
                 fine_tune: bool = True,
                 subtoken_pooling: str = "first",
                 span_mode: str = "markerV0",
                 post_fusion_schema: str = '', #l2l-l2t-t2t
                 num_post_fusion_layers: int = 1, 
                 vocab_size: int = -1,
                 max_neg_type_ratio: int = 1,
                 max_types: int = 25,
                 max_len: int = 512,
                 words_splitter_type: str = "whitespace",
                 has_rnn: bool = True,
                 fuse_layers: bool = False,
                 embed_ent_token: bool = True,
                 class_token_index: int = -1,
                 encoder_config: Optional[dict] = None,
                 ent_token: str = "<<ENT>>",
                 sep_token:str = "<<SEP>>",
                 atlop:bool=True,
                 alpha:float = 0.25,
                 loss:str = "",
                 gamma:float= 2,
                 reduction:str= "mean",
                 self_rel:bool=True,
                 pooling:str = "mean",
                 weights=None,
                 cross_attention=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.encoder_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.labels_encoder_config = AutoConfig.from_pretrained(labels_encoder, trust_remote_code=True) 
        self.model_name = model_name
        self.labels_encoder = labels_encoder
        self.name = name
        self.max_width = max_width
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fine_tune = fine_tune
        self.subtoken_pooling = subtoken_pooling
        self.span_mode = span_mode
        self.post_fusion_schema = post_fusion_schema
        self.num_post_fusion_layers = num_post_fusion_layers
        self.vocab_size = vocab_size
        self.max_neg_type_ratio = max_neg_type_ratio
        self.max_types = max_types
        self.max_len = max_len
        self.words_splitter_type = words_splitter_type
        self.has_rnn = has_rnn
        self.fuse_layers = fuse_layers
        self.class_token_index = class_token_index
        self.embed_ent_token = embed_ent_token
        self.ent_token = ent_token
        self.sep_token = sep_token
        self.atlop = atlop
        self.alpha = alpha
        self.loss = loss
        self.gamma = gamma
        self.reduction = reduction
        self.self_rel = self_rel
        self.pooling = pooling
        self.weights = weights
        self.cross_attention= cross_attention

# Register the configuration
from transformers import CONFIG_MAPPING
CONFIG_MAPPING.update({"gliner": GLiDREConfig})