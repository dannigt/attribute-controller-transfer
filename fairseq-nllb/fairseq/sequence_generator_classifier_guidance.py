# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel

from torch.autograd import Variable
import numpy as np
from operator import add
import copy

SMALL_CONST = 1e-15

class SequenceGeneratorClassifierGuidance(SequenceGenerator):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=0,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        wanted_label=None,
        kl_scale=0.01,
        step_size=0.02,
        num_iter=3,
        gm_scale=1.0,
        only_update_decoder_activation=False,
        deactivate_classifier_guidance=False,
        # use_lm_classifier_guidance=False,
    ):
        logging.info("Entering classifier-guided generator")

        super(SequenceGeneratorClassifierGuidance, self).__init__(models, tgt_dict, beam_size, max_len_a, max_len_b,
                                                                  max_len, min_len, normalize_scores,
                                                                  len_penalty, unk_penalty,
                                                                  temperature, match_source_len, no_repeat_ngram_size,
                                                                  search_strategy, eos, symbols_to_strip_from_output,
                                                                  lm_model, lm_weight)

        self.wanted_label = wanted_label
        self.kl_scale = kl_scale
        self.step_size = step_size
        self.gm_scale = gm_scale
        self.num_iter = num_iter
        self.deactivate_classifier_guidance = deactivate_classifier_guidance
        self.only_update_decoder_activation = only_update_decoder_activation
        # self.use_lm_classifier_guidance = use_lm_classifier_guidance

        logging.info(f"Wanted label: {self.wanted_label}")
        logging.info(f"KL scale: {self.kl_scale}")
        logging.info(f"Step size/LR: {self.step_size}")

        self.model = EnsembleModelClassifierGuidance(models)

        # if self.use_lm_classifier_guidance:
        #     assert self.lm_model is not None

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def to_var(self, x, requires_grad=False, volatile=False, device='cuda'):
        # if torch.cuda.is_available() and device == 'cuda':
        #     x = x.cuda().to(dtype=torch.float16)
        # elif device != 'cuda':
        x = x.to(device).to(dtype=torch.float16)
        x.requires_grad = True
        # res = Variable(x, requires_grad=requires_grad, volatile=volatile)
        # logging.info(f"========= {x.dtype}")
        # return Variable(x, requires_grad=requires_grad, volatile=volatile)
        return x #.float()

    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        # get incremental states (H_t from paper)
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        classifier_correct, classifier_predicted = 0, 0

        # for every decoding time step
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            logging.info(f"===================================== Current step: {step}")

            with torch.no_grad():
                # save a copy of unperturbed h_t at this time step before doing any forward
                h_t = incremental_states[0]
                # make deep copy from h_t; only gets modified for final forward pass of this timestep
                h_t_unpert = {}
                for guid in h_t:
                    for key in h_t[guid].keys():
                        if guid not in h_t_unpert:
                            h_t_unpert[guid] = {}
                        if key not in h_t_unpert[guid]:
                            h_t_unpert[guid][key] = ""
                        h_t_unpert[guid][key] = h_t[guid][key].clone().detach() if h_t[guid][key] is not None else None

                # first forward with unpert, increase size of incremental states
                lprobs, avg_attn_scores, classifier_out = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )

            # logging.info(f"Shape of lprobs: {lprobs.shape}")
            # logging.info(f"Shape of logits: {logits.shape}")

            if step > 0:    # and classifier_out.argmax(dim=-1) != self.wanted_label:
                # h_t = incremental_states[0]  # first model from ensemble

                curr_h_t = h_t_unpert
                grad_accumulator = []   # should not be tensors requiring grad!

                for iter_idx in range(self.num_iter):
                    logging.info(f"iter: {iter_idx}")
                    # ================ Prep delta H
                    # make deep copy from curr_h_t to h_t_perturbed
                    h_t_perturbed = {}
                    for guid in curr_h_t:
                        for key in curr_h_t[guid].keys():
                            if guid not in h_t_perturbed:
                                h_t_perturbed[guid] = {}
                            if key not in h_t_perturbed[guid]:
                                h_t_perturbed[guid][key] = ""
                            h_t_perturbed[guid][key] = curr_h_t[guid][key].clone().detach() if curr_h_t[guid][key] is not None else None
                    curr_perturbation = []

                    # add (delta h_t) to h_t
                    cnt_ht_items_orig = 0
                    for guid in curr_h_t:    # logging.info(f"incremental state {cnt} has ID: {k}")
                        for key in ['prev_key', 'prev_value']:  # keys are prev_key, prev_value, prev_key_padding_mask
                            if curr_h_t[guid][key] is not None:  # shape B*beamsize x 16 (att. head) x cur_timestep or longest enc. seq x 32
                                # if h_t[guid][key].shape[2] == step:  # only take decoder incremental states
                                if iter_idx == 0:  # first iter., create elements in grad_accumulator
                                    grad_ = np.zeros(curr_h_t[guid][key].shape).astype("float32")
                                    grad_accumulator.append(grad_)
                                else:
                                    grad_ = grad_accumulator[cnt_ht_items_orig]

                                # zero out updates for cross-attention
                                if self.only_update_decoder_activation:
                                    if h_t[guid][key].shape[2] == step:
                                        grad_.fill(0.0)

                                curr_perturbation.append(self.to_var(torch.from_numpy(grad_),
                                                                     requires_grad=True,
                                                                     device=curr_h_t[guid][key].device))
                                # logging.info(f"{h_t_perturbed[guid][key].shape} vs {curr_perturbation[-1].shape}")
                                h_t_perturbed[guid][key] += curr_perturbation[-1]   # add the accumulated gradient
                                cnt_ht_items_orig += 1

                    # run forward decoder with h_t + delta h_t; increase size of incremental states
                    lprobs_cur_pert, _, classifier_out = self.model.forward_decoder(
                        tokens[:, : step + 1],
                        encoder_outs,
                        [h_t_perturbed],  # this should be a set of different incremental state (calling forward adds stuff to incremental_states)
                        self.temperature,
                        # dec_hidden_sum=dec_hidden_sum,
                    )
                    # logging.info(f"Shape of classification distribution: {classifier_out.shape}")   # bsz x # classes
                    # logging.info(f"Finished # of hypos: {sum(finished)}")
                    # first do an argmax as sanity check
                    if classifier_out is not None:
                        pred = classifier_out.argmax(dim=-1)
                        if torch.all(pred == self.wanted_label):   # break if everything (all batch, all beams) is correct
                            break
                    uniq, cnt = pred.unique(return_counts=True)
                    logging.info(f"unique classes: {uniq}")
                    logging.info(f"counts: {cnt}")
                    # classifier_correct += (pred == self.wanted_label).sum()
                    # classifier_predicted += pred.numel()
                    # logging.info(f"correct rate by classifier: {classifier_correct / classifier_predicted}")

                    if not self.deactivate_classifier_guidance:
                        ce_loss = nn.CrossEntropyLoss(ignore_index=0) #, reduction='sum')   #reduction='') #label_smoothing=0.3
                        class_target = (torch.ones(classifier_out.shape[0]) * self.wanted_label).long().to(lprobs.device)

                        assert not torch.isnan(classifier_out).any()
                        loss = ce_loss(classifier_out, class_target)

                        assert not torch.isnan(loss).any()
                        logging.info(f"discrim loss: {loss}")

                        if self.kl_scale > 0:
                            kl_loss = self.kl_scale * (lprobs_cur_pert.exp() * (lprobs - lprobs_cur_pert)).sum()
                            logging.info(f"kl loss: {kl_loss}")
                            loss += kl_loss #* classifier_out.shape[0]

                        loss.backward()

                        # denominator of second term in eq. 3
                        grad_norms = []
                        for index, p_ in enumerate(curr_perturbation):
                            # fix this norm scaling with beam and batch size
                            bb_, n_att_, t_, _ = p_.shape
                            # for matrix of shape a x b: if a = a * N, norm scales by sqrt(N)
                            # n_ = torch.linalg.matrix_norm(p_.grad, keepdim=True) / math.sqrt(p_.shape[0]) + SMALL_CONST  # B*beamsize x 16 (att. head) x T x 32
                            # torch.linalg.norm: flattened to 1D and the 2-norm of the resulting vector will be computed.
                            n_ = torch.linalg.norm(p_.grad) / math.sqrt(bb_ + n_att_ + t_) + SMALL_CONST

                            # T = (cur_timestep or longest enc. seq)
                            grad_norms.append(n_)
                            # assert p_.grad.shape[0] == 1

                            # logging.info(f"Grad shape: {p_.grad.shape}, norm: {n_}")

                        # normalize gradients; second term in eq. 3
                        gamma = 1.0
                        grad = [
                            -self.step_size *
                            (p_.grad / grad_norms[index] ** gamma)  #.data.cpu().numpy()
                            for index, p_ in enumerate(curr_perturbation)
                        ]

                        # mask_ = torch.repeat_interleave(n_sample_correct, beam_size)
                        #
                        # accumulate grad
                        for index, p_ in enumerate(grad):
                            grad_accumulator[index] += p_.data.cpu().detach().numpy()

                        # reset gradients, just to make sure
                        for p_ in curr_perturbation:
                            p_.grad.data.zero_()

                # Finished iteration; accumulate gradients onto h_t_unpert
                with torch.no_grad():
                    cnt_ht_items_pert = 0
                    for guid in h_t_unpert:
                        for key in ['prev_key', 'prev_value']:
                            if h_t_unpert[guid][key] is not None:
                                grad_ = grad_accumulator[cnt_ht_items_pert]
                                h_t_unpert[guid][key] += self.to_var(torch.from_numpy(grad_),
                                                                     requires_grad=True,
                                                                     device=curr_h_t[guid][key].device)
                                cnt_ht_items_pert += 1
                    assert cnt_ht_items_orig == cnt_ht_items_pert

                # run forward decoder with perturbed h_t
                lprobs_pert, avg_attn_scores_pert, classifier_out_pert = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    [h_t_unpert],
                    self.temperature,
                )

                if self.gm_scale > 0:
                    if self.gm_scale == 1:
                        lprobs = lprobs_pert
                    else:
                        # postnorm simple fusion
                        # log version of https://github.com/uber-research/PPLM/blob/master/run_pplm.py#L633
                        # log ((pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale)))
                        # = log(pert_probs ** gm_scale) + log(unpert_probs ** (1 - gm_scale))
                        # = log(pert_probs) * gm_scale + log(unpert_probs) * (1 - gm_scale)
                        lprobs = lprobs_pert * self.gm_scale + lprobs * (1 - self.gm_scale)
                        lprobs = F.log_softmax(lprobs.exp(), dim=-1)
                        # assert torch.all(lprobs == F.log_softmax(lprobs_pert.exp(), dim=-1))  this does not work!
                        # due to temperature difference?
                        # logging.info(f"Softmax========={lprobs.sum(dim=-1)}")

            # if self.lm_model is not None and step >= 2:
            #     lm_out = self.lm_model(tokens[:, 2: step + 1])  # exclude BOS and langID
            #     #                lm_out[0][:, :, :4] = 0.0  # normalize excl. EOS and unk according to https://arxiv.org/pdf/1503.03535.pdf doens't really make a diff
            #     probs = self.lm_model.get_normalized_probs(
            #         lm_out, log_probs=True, sample=None
            #     )  # beam*batch x T x V
            #
            #     missing_dim = lprobs.shape[-1] - probs.shape[-1]
            #     logging.info(f"LM: padding {missing_dim} dimensions")
            #     pad_probs = torch.ones(probs.shape[0], probs.shape[1], missing_dim).to(probs.device) * -math.inf
            #
            #     probs = torch.cat((probs, pad_probs), dim=-1)
            #
            #     probs = probs[:, -1, :] * self.lm_weight
            #     lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized


class EnsembleModelClassifierGuidance(EnsembleModel):

    def __init__(self, models):
        super(EnsembleModelClassifierGuidance, self).__init__(models)

        if not self.has_incremental:
            raise NotImplementedError('Classifier guidance only implemented with incremental decoder')

        if self.models_size > 1:
            raise NotImplementedError("Classifier-guided decoding not implemented for ensemble")

    @torch.jit.export
    def forward_decoder(
            self,
            tokens,
            encoder_outs: List[Dict[str, List[Tensor]]],
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            temperature: float = 1.0,
    ):
        # for a single time step
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        n_steps = tokens.shape[-1]

        for i, model in enumerate(self.models):
            if hasattr(model.decoder, 'classifier_input'):
                do_forced_decoding_prefix = model.decoder.classifier_input in ['dec-cumsum', 'dec-meanpool'] and n_steps > 1
            else:
                do_forced_decoding_prefix = False

            if self.has_encoder():
                encoder_out = encoder_outs[i]

            # decode each model
            # decoder_out[0]: x
            # decoder_out[1]: extra
            # if self.has_incremental_states():
            cur_dec_hidden, extra = model.decoder.forward(
                tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states[i],
                features_only=True,     # only take hidden state
                forced_decoding_prefix=do_forced_decoding_prefix,
            )
            # compared to superclass: reconstruct tuple by 1) projecting to vocab size; 2) extra
            decoder_out = [model.decoder.output_layer(cur_dec_hidden), extra]
            #     decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)

            if hasattr(model.decoder, 'classifier_input'):
                if model.decoder.classifier_input == 'dec-all':
                    # [1] to take extra
                    # sft = torch.softmax(dim=0)
                    discr_out = extra['classification_out'][0, :, :]   # "1 x B(active)" x beamsize x Classes
                    # keep beam dim
                    # assert torch.equal(discr_out[0], discr_out[1])
                    # discr_out = discr_out[::5, :]
                elif model.decoder.classifier_input in ['dec-cumsum', 'dec-meanpool']:
                    # logging.info(f"*** USING dec-cumsum, n_steps {n_steps}")
                    cur_dec_hidden = cur_dec_hidden.squeeze(1)
                    if n_steps > 1:
                        avg_dec_hidden = (extra["prefix_dec_hidden"].sum(axis=1) + cur_dec_hidden) / n_steps
                    else:
                        avg_dec_hidden = cur_dec_hidden

                    discr_out = model.decoder.attribute_classifier(avg_dec_hidden)

                else:
                    raise NotImplementedError
            else:
                discr_out = None

            # sft = torch.softmax(discr_out, dim=-1)
            # logging.info(f"discrim out: {discr_out.shape}")
            # logging.info(f"discrim out: {sft.max(dim=-1)}")

            # logging.info(discr_out.shape)
            # decoder_out[0].shape: B(active) * beamsize x 1 x |V|

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)

            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]

            # case w/o ensemble
            return probs, attn, discr_out


