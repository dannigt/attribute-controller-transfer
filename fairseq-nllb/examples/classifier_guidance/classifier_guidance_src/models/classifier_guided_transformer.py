# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from typing import Optional

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    base_architecture,
)

from fairseq.models.transformer import TransformerModel
from ..modules.classifier import ClassificationLayer

import math
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor


@register_model("classifier_guided_transformer")
class ClassifierGuidedTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--classifier-middle-layer-size",
            default=256,
            type=int,
        )
        parser.add_argument(
            "--classifier-output-size",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--classifier-input",
            default='dec-meanpool',
            type=str,
            help="What the classifier operates on, dec-meanpool | dec-cumsum | dec-all",
        )

    @classmethod
    def build_decoder(cls, args, src_dict, embed_tokens):
        return ClassifierGuidedTransformerDecoder(args, src_dict, embed_tokens)

    def load_state_dict(
            self,
            state_dict,
            strict=True,
            model_cfg=None,
            args=None,
    ):
        # Setting strict to False due to newly added parameters (classifier)
        return super().load_state_dict(state_dict, strict=False)


class ClassifierGuidedTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

        self.output_dim = args.classifier_output_size + 1

        self.attribute_classifier = ClassificationLayer(args=args,
                                                        input_dim=args.decoder_embed_dim,
                                                        middle_dim=args.classifier_middle_layer_size,
                                                        output_dim=self.output_dim,
                                                        )
        self.classifier_input = args.classifier_input

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        forced_decoding_prefix: bool = False,
    ):
        # x is B x T x C
        # normal forward pass
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if self.classifier_input == 'dec-meanpool':
            if not forced_decoding_prefix:  # check training or eval
                # https://github.com/uber-research/PPLM/blob/e236b8989322128360182d29a79944627957ad47/run_pplm_discrim_train.py#L78-L92
                padded = (prev_output_tokens == 1).unsqueeze(-1)  # B x T
                y = x.masked_fill(padded, 0.0) #.sum()  # B x T x C
                x_mean = torch.sum(y, dim=1, keepdim=True) / (~padded).sum(dim=1, keepdim=True) # B x T x C
                classifier_out = self.attribute_classifier(x_mean.transpose(0, 1))
                extra["classification_out"] = classifier_out  # logit
            else:   # test; another forward for prefix
                x_prefix, _ = self.extract_features(
                    prev_output_tokens[:, :-1],
                    encoder_out=encoder_out,
                    full_context_alignment=full_context_alignment,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                )
                extra["prefix_dec_hidden"] = x_prefix

                logging.info(f"*** Prefix hidden shape {extra['prefix_dec_hidden'].shape}")
        elif self.classifier_input == 'dec-all':
            # transpose B x T x C --> T x B x C; output T x B x |V|
            # when decoding, T=1
            lang_classifier_out = self.attribute_classifier(x.transpose(0, 1))
            extra["classification_out"] = lang_classifier_out
        elif self.classifier_input == 'dec-cumsum':
            if not forced_decoding_prefix:  # training or dev
                # do cumsum and average
                x_cumsum = torch.cumsum(x, dim=1)   # B x T x C
                # do masking, need length
                padded = (prev_output_tokens == 1).unsqueeze(-1)  # B x T   # TODO: hardcoded PAD check using token ID = 1
                divisor = torch.arange(1, prev_output_tokens.shape[-1] + 1).to(dtype=torch.float16).to(x.device).unsqueeze(0).unsqueeze(-1)  # 1 x T x 1
                x_cumsum = (x_cumsum / divisor).masked_fill_(padded, 0.0)   # mask out padded
                classifier_out = self.attribute_classifier(x_cumsum.transpose(0, 1))    # B x T x C --> T x B x C
                extra["classification_out"] = classifier_out  # logit

            else:   # test; another forward for prefix
                x_prefix, _ = self.extract_features(
                    prev_output_tokens[:, :-1],
                    encoder_out=encoder_out,
                    full_context_alignment=full_context_alignment,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                )
                extra["prefix_dec_hidden"] = x_prefix
                # logging.info(f"*** Prefix hidden shape {extra['prefix_dec_hidden'].shape}")

        else:
            raise NotImplementedError

        # projection to vocab size
        if not features_only:
            x = self.output_layer(x)

        return x, extra


@register_model_architecture(
    "classifier_guided_transformer", "classifier_guided_transformer"
)
def classifier_guided_transformer(args):
    base_architecture(args)
