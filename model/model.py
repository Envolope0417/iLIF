# coding: UTF-8

import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from .CGE import CGEConv
from torch.nn.utils.rnn import pad_sequence
embedding_size = 768

class focal_loss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += (1 - alpha)
            self.alpha[1:] += alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=-1)
        preds_logsoft = F.log_softmax(preds, dim=-1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class bertCSRModel(nn.Module):
    def __init__(self, args):
        super(bertCSRModel, self).__init__()
        self.device = args.device
        self.pretrained_model = BertModel.from_pretrained(args.model_name)
        self.mlp = nn.Sequential(nn.Linear(1 * args.n_last + 2 * embedding_size, args.mlp_size),
                                  nn.ReLU(), nn.Dropout(args.mlp_drop),
                                  nn.Linear(args.mlp_size, args.no_of_classes))
        self.rate = args.rate
        self.w = args.w
        self.max_iteration = args.max_iteration + 1
        self.threshold = args.threshold
        self.min_iteration = args.min_iteration
        self.CGE = CGEConv(in_channels=args.in_channels,
                            out_channels=args.n_last,
                            metadata=args.metadata,
                            heads=args.num_heads,
                            beta_intra=args.beta_intra,
                            beta_inter=args.beta_inter
                            )
        self.focal_loss = focal_loss(gamma=args.gamma, num_classes=args.no_of_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_type = args.loss_type
        # --- adaptive confidence threshold (iLIF improvement) ---
        self.use_dynamic = getattr(args, 'use_dynamic', False)
        self.use_persistence = getattr(args, 'use_persistence', False)
        self.w_max = getattr(args, 'w_max', 0.70)
        self.w_min = getattr(args, 'w_min', 0.40)
        self.ema_alpha = getattr(args, 'ema_alpha', 0.5)
        self.persist_lambda = getattr(args, 'persist_lambda', 0.05)
        self.persist_base = getattr(args, 'persist_base', 0.5)
        self.w_floor = getattr(args, 'w_floor', 0.35)
        self.use_infer_anneal = getattr(args, 'use_infer_anneal', False)
        self.anneal_start = getattr(args, 'anneal_start', 0.65)
        self.anneal_step = getattr(args, 'anneal_step', 0.05)
        self.anneal_floor = getattr(args, 'anneal_floor', 0.45)
        self.anneal_direction = getattr(args, 'anneal_direction', 'down')
        self.anneal_ceiling = getattr(args, 'anneal_ceiling', 0.65)
        self._debug_omega = []
        self._debug_nedge = []

    def forward(self, enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs):
        sent_emb = self.pretrained_model(enc_input_ids, enc_mask_ids)[0].to(self.device)
        sent_list = []
        for idx, event in enumerate(node_event):
            sent_id = int(event[0])
            event_pid = event[1]
            e_emb = self.extract_event(sent_emb[sent_id], event_pid).to(self.device)
            if idx == 0:
                event_embed = e_emb
            else:
                event_embed = torch.cat((event_embed, e_emb))
            if sent_id not in sent_list:
                sent_list.append(sent_id)
        if self.max_iteration > sent_list.__len__():
            max_iteration = sent_list.__len__()
        else:
            max_iteration = self.max_iteration

        pad_event1_pos = pad_sequence([torch.tensor(pos) for pos in t1_pos]).t().to(self.device).long()
        pad_event2_pos = pad_sequence([torch.tensor(pos) for pos in t2_pos]).t().to(self.device).long()
        event1 = torch.index_select(event_embed, 0, pad_event1_pos[0])
        event2 = torch.index_select(event_embed, 0, pad_event2_pos[0])
        event_pair_embed = torch.cat([event1, event2], dim=-1)

        loss = 0.0
        iteration = 0
        difference = self.threshold
        event_embed = self.CGE.proj['event'](event_embed)
        if len(event_pairs) > 1:
            target = torch.cat([torch.tensor(t) for t in target], dim=0).to(self.device)
        else:
            target = torch.tensor(target[0]).to(self.device)

        # --- adaptive threshold state (per document) ---
        num_pairs = len(event_pairs)
        if self.use_dynamic:
            cur_w = self.w_max
        elif self.use_infer_anneal and not self.training:
            cur_w = self.anneal_start
        else:
            cur_w = self.rate if self.training else self.w
        C = torch.zeros(num_pairs, device=self.device) if self.use_persistence else None
        d_bar = None
        d_ref = 1.0
        edge_set_prev = None
        self._debug_omega = []
        self._debug_nedge = []

        while iteration < max_iteration and (iteration < self.min_iteration or difference > self.threshold):
            if len(event_pairs) > 1:
                event_diff = torch.cat([event_embed[[pair[0]]] - event_embed[[pair[1]]] for pair in event_pairs])
            else:
                event_diff = event_embed[[event_pairs[0][0]]] - event_embed[[event_pairs[0][1]]]
            event_pair_pre = torch.cat((event_pair_embed, event_diff), dim=1)
            prediction = self.mlp(event_pair_pre)
            if self.loss_type == 'focal':
                loss += (1/(iteration + 1)) * self.focal_loss(prediction, target)
            else:
                loss += (1/(iteration + 1)) * self.ce_loss(prediction, target)

            # per-edge confidence thresholds
            if self.use_dynamic:
                if self.use_persistence:
                    rates = torch.clamp(cur_w - self.persist_lambda * C, min=self.w_floor)
                else:
                    rates = cur_w
            else:
                rates = cur_w
            self._debug_omega.append(float(cur_w))
            graphedge_index, edge_set = self.get_graphedge_index(prediction, event_pairs, rel_type, rates)
            self._debug_nedge.append(len(edge_set))

            # structural difference -> adapt the global threshold for the next iteration
            # (ratchet: only ever relax, never tighten -- prevents oscillation)
            if self.use_dynamic and iteration >= 1:
                d_l = len(edge_set ^ edge_set_prev) if edge_set_prev is not None else len(edge_set)
                if d_bar is None:
                    d_bar = float(d_l)
                    d_ref = max(d_bar, 1.0)
                else:
                    d_bar = self.ema_alpha * d_l + (1 - self.ema_alpha) * d_bar
                r = min(max(d_bar / d_ref, 0.0), 1.0)
                cur_w = min(cur_w, self.w_min + (self.w_max - self.w_min) * r)

            # inference-only fixed anneal (per-iteration threshold step)
            if self.use_infer_anneal and not self.training:
                if self.anneal_direction == 'up':
                    cur_w = min(cur_w + self.anneal_step, self.anneal_ceiling)
                else:
                    cur_w = max(cur_w - self.anneal_step, self.anneal_floor)

            # temporal-persistence counter update for the next iteration
            if self.use_persistence:
                pred_soft = torch.softmax(prediction, dim=1)
                pred_max, pred_arg = pred_soft.max(dim=1)
                eligible = (pred_arg != 0) & (pred_max >= self.persist_base)
                C = torch.where(eligible, C + 1.0, torch.zeros_like(C))

            edge_set_prev = edge_set

            if graphedge_index.__len__() > 0:
                node_length = event_embed.size()[0]
                self_loop = torch.cat((torch.arange(node_length).unsqueeze(0), torch.arange(node_length).unsqueeze(0)),dim=0).to(self.device)
                if ('event', 'intra', 'event') in graphedge_index:
                    graphedge_index[('event', 'intra', 'event')] = torch.cat((graphedge_index[('event', 'intra', 'event')], self_loop), dim=-1).long()
                else:
                    graphedge_index[('event', 'intra', 'event')] = self_loop.long()
                event_han = {'event': event_embed}
                event_han = self.CGE(event_han, graphedge_index)
                event_embed = event_han['event']
            else:
                event_embed = self.CGE.proj['event'](event_embed)
            if iteration > 0:
                difference = self.Contrast_pre(prediction, prediction_1)
            iteration += 1
            prediction_1 = prediction
        return loss, prediction

    # Extract the representation vector of the event
    def extract_event(self, embed, event_pid):
        e_1 = int(event_pid[0])
        e_2 = int(event_pid[1])
        e1_embed = torch.zeros(1, embedding_size).to(self.device)
        length = e_2 - e_1
        for j in range(e_1,e_2):
            e1_embed += embed[j]
        event_embed = e1_embed / length
        return event_embed

    # Construct adjacency matrix (and return the admitted edge set for structural difference)
    def get_graphedge_index(self, prediction, event_pair, rel_type, rates):
        pred_soft = torch.softmax(prediction, dim=1)
        pred_max, pred_arg = pred_soft.max(dim=1)          # [num_pairs]
        rel0 = (torch.as_tensor(rel_type[0], device=self.device) == 0)
        rates_t = torch.as_tensor(rates, dtype=pred_max.dtype, device=self.device)
        keep = (pred_arg != 0) & (pred_max > rates_t)      # causal + above threshold
        keep1 = keep & (pred_arg == 1)                     # CAUSE: pair[0] -> pair[1]
        keep2 = keep & (pred_arg == 2)                     # EFFECT: pair[1] -> pair[0]

        pairs = torch.tensor([[p[0], p[1]] for p in event_pair], device=self.device).long()  # [num_pairs, 2]

        intra1 = pairs[keep1 & rel0]
        intra2 = pairs[keep2 & rel0]
        inter1 = pairs[keep1 & (~rel0)]
        inter2 = pairs[keep2 & (~rel0)]

        intra_index = torch.cat([
            torch.stack([intra1[:, 0], intra1[:, 1]], dim=0),
            torch.stack([intra2[:, 1], intra2[:, 0]], dim=0),
        ], dim=-1)

        inter_index = torch.cat([
            torch.stack([inter1[:, 0], inter1[:, 1]], dim=0),
            torch.stack([inter2[:, 1], inter2[:, 0]], dim=0),
        ], dim=-1)

        graphedge_index = {}
        if intra_index.size(1) > 0:
            graphedge_index[('event', 'intra', 'event')] = intra_index.long()
        if inter_index.size(1) > 0:
            graphedge_index[('event', 'inter', 'event')] = inter_index.long()

        edge_set = set()
        if intra_index.size(1) > 0:
            edge_set.update(tuple(e) for e in intra_index.t().tolist())
        if inter_index.size(1) > 0:
            edge_set.update(tuple(e) for e in inter_index.t().tolist())
        return graphedge_index, edge_set

    # Compare the structural differences before and after iteration
    def Contrast_pre(self, prediction, prediction_last):
        if self.training:
            rate = self.rate
        else:
            rate = self.w
        if rate != 0:
            pred_soft = torch.softmax(prediction, dim=1)
            pred_last_soft = torch.softmax(prediction_last, dim=1)
            pre_list = torch.tensor([]).to(self.device)
            pre_last_list = torch.tensor([]).to(self.device)
            max_pro, pred_t = torch.max(pred_soft, dim=1)
            for idx, pre in enumerate(max_pro):
                if pre > self.rate:
                    pre_list = torch.cat((pre_list, pred_t[[idx]]), dim=-1)
                else:
                    pre_list = torch.cat((pre_list, torch.tensor([0]).to(self.device)), dim=-1)
            max_pro, pred_t = torch.max(pred_last_soft, dim=1)
            for idx, pre in enumerate(max_pro):
                if pre > self.rate:
                    pre_last_list = torch.cat((pre_last_list, pred_t[[idx]]), dim=-1)
                else:
                    pre_last_list = torch.cat((pre_last_list, torch.tensor([0]).to(self.device)), dim=-1)
        else:
            pre_list = torch.argmax(prediction, dim=1)
            pre_last_list = torch.argmax(prediction_last, dim=1)
        different = (pre_list != pre_last_list).sum().item()
        return different
