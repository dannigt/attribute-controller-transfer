# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq import search
from fairseq.sequence_generator_classifier_guidance import SequenceGeneratorClassifierGuidance
import torch
import logging


@register_task("classifier_guidance")
class ClassifierGuidanceTask(TranslationMultiSimpleEpochTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationMultiSimpleEpochTask.add_args(parser)

        parser.add_argument(
            "--actual-vocab-size",
            default=256000,
            type=int,
            help="Actual vocab size without language tokens",
        )
        # For inference
        parser.add_argument(
            "--wanted-label",
            type=int,
            help="Index of desired label, starting from 1",
        )
        parser.add_argument(
            "--step-size",
            default=0.02,
            type=float,
            help="Step size when decoding with classifier guidance, like learning rate",
        )
        parser.add_argument(
            "--num-iter",
            default=3,
            type=int,
            help="Number of steps to update when decoding with classifier guidance",
        )
        parser.add_argument(
            "--kl-scale",
            default=0.0,
            type=float,
            help="KL scale when decoding with classifier guidance",
        )
        parser.add_argument(
            "--gm-scale",
            default=1.0,
            type=float,
            help="Fusion weight when decoding with classifier guidance, default no fusion with original states",
        )
        parser.add_argument(
            "--deactivate-classifier-guidance",
            action="store_true",
            help="Turn off classifier guidance at inference. For debugging only.",
        )
        parser.add_argument(
            "--only-update-decoder-activation",
            action="store_true",
            help="Only update decoder activations instead of both encoder and decoder activations"
                 "when decoding with classifier guidance",
        )

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)

        self.vocab_size = args.actual_vocab_size

        self.kl_scale = args.kl_scale
        self.wanted_label = args.wanted_label
        self.step_size = args.step_size
        self.num_iter = args.num_iter
        self.gm_scale = args.gm_scale
        self.deactivate_classifier_guidance = args.deactivate_classifier_guidance
        self.only_update_decoder_activation = args.only_update_decoder_activation

        self.args = args

    def build_model(self, args):
        model = super().build_model(args)
        print(type(model.encoder))
        print(type(model.decoder))

        for name, param in model.named_parameters():
            # Freeze everything apart from language classifier
            if "attribute_classifier" in name:
                param.requires_grad = True
                print("===== Train:", name)
            else:
                param.requires_grad = False
        return model

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample,
                                                          vocab_size=self.vocab_size,
                                                          )
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample,
                                                          vocab_size=self.vocab_size,
                                                          )
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        # with torch.no_grad(): need gradient for classifier guidance
        _, tgt_langtok_spec = self.args.langtoks["main"]
        if not self.args.lang_tok_replacing_bos_eos:
            if prefix_tokens is None and tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                src_tokens = sample["net_input"]["src_tokens"]
                bsz = src_tokens.size(0)
                prefix_tokens = (
                    torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                )
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )
        else:
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                bos_token=self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                if tgt_langtok_spec
                else self.target_dictionary.eos(),
            )
        # return super(ClassifierGuidanceTask, self).inference_step(generator, models, sample, prefix_tokens, constraints)

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        # From TranslationMultiSimpleEpochTask(LegacyFairseqTask)
        if not getattr(args, "keep_inference_langtok", False):
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

        # From LegacyFairseqTask
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )
        try:
            from fairseq.fb_sequence_generator import FBSequenceGenerator
        except ModuleNotFoundError:
            pass

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}

        return SequenceGeneratorClassifierGuidance(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            wanted_label=self.wanted_label,   # Added param
            step_size=self.step_size,   # Added param
            num_iter=self.num_iter, # Added param
            kl_scale=self.kl_scale,  # Added param
            gm_scale=self.gm_scale, # Added param
            only_update_decoder_activation=self.only_update_decoder_activation,
            deactivate_classifier_guidance=self.deactivate_classifier_guidance, # Added param
            **extra_gen_cls_kwargs,
        )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        criterion.__class__.reduce_metrics(logging_outputs)
