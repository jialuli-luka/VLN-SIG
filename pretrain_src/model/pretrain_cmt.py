from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead
from .vilmodel import NavPreTrainedModel


class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class NextActionRegression(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 3))

    def forward(self, x):
        return self.net(x)

class SpatialRelRegression(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 2))

    def forward(self, x):
        return self.net(x)

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ItmPrediction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)


class MimPrediction(nn.Module):
    " for Mim(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class ActionPredictionWithImageGeneration(nn.Module):
    " for action prediction(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class MppPrediction(nn.Module):
    " for Mpp(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class WeightedActionPredictionWithImageGeneration(nn.Module):
    def __init__(self, hidden_size, label_dim, num_patches):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))
        self.map_weight = nn.Sequential(nn.Linear(num_patches, hidden_size),
                                        nn.ReLU(),
                                        BertLayerNorm(hidden_size, eps=1e-12),
                                        nn.Linear(hidden_size, hidden_size))

    def forward(self, input_, weight):
        weight = self.map_weight(weight.squeeze(-1))
        output = self.net(input_ + weight)
        return output


class WeightedMaskedImageModeling(nn.Module):
    def __init__(self, hidden_size, label_dim, num_patches):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))
        self.map_weight = nn.Sequential(nn.Linear(num_patches, hidden_size),
                                        nn.ReLU(),
                                        BertLayerNorm(hidden_size, eps=1e-12),
                                        nn.Linear(hidden_size, hidden_size))

    def forward(self, input_, weight):
        weight = self.map_weight(weight.squeeze(-1))
        output = self.net(input_ + weight)
        return output


class WeightedMppPrediction(nn.Module):
    " for Mpp(-kl)"
    def __init__(self, hidden_size, label_dim, num_patches):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))
        self.map_weight = nn.Sequential(nn.Linear(num_patches, hidden_size),
                                        nn.ReLU(),
                                        BertLayerNorm(hidden_size, eps=1e-12),
                                        nn.Linear(hidden_size, hidden_size))

    def forward(self, input_, weight):
        weight = self.map_weight(weight.squeeze(-1))
        output = self.net(input_ + weight)
        return output


class MultiStepNavCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = NavPreTrainedModel(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'sap' in config.pretrain_tasks:
            self.next_action = NextActionPrediction(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'sar' in config.pretrain_tasks:
            self.regress_action = NextActionRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'sprel' in config.pretrain_tasks:
            self.sprel_head = SpatialRelRegression(self.config.hidden_size, self.config.pred_head_dropout_prob)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
        if 'itm' in config.pretrain_tasks:
            self.itm_head = ItmPrediction(self.config.hidden_size)
        if 'mim' in config.pretrain_tasks:
            print("Using d_vae classes:", self.config.dvae_classes)
            if self.config.mim_weighted:
                self.mim_head = WeightedMaskedImageModeling(self.config.hidden_size, self.config.dvae_classes, self.config.patch_nums)
            else:
                self.mim_head = MimPrediction(self.config.hidden_size, self.config.dvae_classes)
        if 'apwig' in config.pretrain_tasks:
            print("Using d_vae classes:", self.config.dvae_classes_apwig)
            self.apwig_head = ActionPredictionWithImageGeneration(self.config.hidden_size, self.config.dvae_classes_apwig)
        if 'mpp' in config.pretrain_tasks:
            print("Using d_vae classes:", self.config.dvae_classes_mpp)
            if self.config.mpp_weighted:
                self.mpp_head = WeightedMppPrediction(self.config.hidden_size, self.config.dvae_classes_mpp, self.config.patch_nums)
            else:
                self.mpp_head = MppPrediction(self.config.hidden_size, self.config.dvae_classes_mpp)
        if 'mapwig' in config.pretrain_tasks:
            self.mapwig_head = WeightedActionPredictionWithImageGeneration(self.config.hidden_size, self.config.dvae_classes_apwig, self.config.patch_nums)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True, d_vae=None, dy_filter=None, pos_input=None):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(batch['txt_ids'], batch['txt_masks'], 
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['txt_labels'], compute_loss)
        elif task == 'sap':
            return self.forward_sap(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['ob_action_viewindex'], compute_loss)
        elif task.startswith('sar'):
            return self.forward_sar(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['ob_action_angles'], batch['ob_progress'], compute_loss)
        elif task.startswith('sprel'):
            return self.forward_sprel(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], 
                                    batch['ob_nav_types'], batch['ob_masks'],
                                    batch['sp_anchor_idxs'], batch['sp_targets'], 
                                    compute_loss)
        elif task.startswith('mrc'):
            return self.forward_mrc(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['hist_mrc_masks'], batch['hist_img_probs'], compute_loss)
        elif task.startswith('itm'):
            return self.forward_itm(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'], 4, compute_loss)
        elif task.startswith('mim'):
            return self.forward_mim(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['hist_mrc_masks'], batch['hist_images_dvae'],
                                    compute_loss, d_vae=d_vae, filter=batch['dvae_filter'], dy_filter=dy_filter)
        elif task.startswith('apwig'):
            return self.forward_apwig(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'],
                                    batch['ob_nav_types'], batch['ob_masks'], compute_loss,
                                    batch['ob_action_image'], d_vae=d_vae, filter=batch['dvae_filter'], dy_filter=dy_filter)
        elif task.startswith('mpp'):
            return self.forward_mpp(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'], batch['ob_nav_types'], batch['ob_masks'], batch['ob_pano_images'], batch['ob_mpp_masks'],
                                    compute_loss, d_vae=d_vae, filter=batch['dvae_filter'], dy_filter=dy_filter)
        elif task.startswith('mapwig'):
            return self.forward_mapwig(batch['txt_ids'], batch['txt_masks'],
                                    batch['hist_img_fts'], batch['hist_ang_fts'],
                                    batch['hist_pano_img_fts'], batch['hist_pano_ang_fts'], batch['hist_masks'],
                                    batch['ob_img_fts'], batch['ob_ang_fts'],
                                    batch['ob_nav_types'], batch['ob_masks'], compute_loss,
                                    batch['ob_action_image'], d_vae=d_vae, filter=batch['dvae_filter'], dy_filter=dy_filter, pos_input=pos_input)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    txt_labels, compute_loss):
        txt_embeds, _, _ = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            None, None, None, None)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(prediction_scores, 
                                        txt_labels[txt_labels != -1], 
                                        reduction='none')
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _compute_masked_hidden_image(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(hidden)
        hidden_masked = torch.masked_select(hidden, mask)
        return hidden_masked

    def forward_sap(self, txt_ids, txt_masks,
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks,
                    act_labels, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks,
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)
        
        # combine text and visual to predict next action
        prediction_scores = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
        prediction_scores.masked_fill_(ob_nav_types == 0, -float('inf'))

        if compute_loss:
            act_loss = F.cross_entropy(prediction_scores, act_labels, reduction='none')
            return act_loss
        else:
            return prediction_scores

    def forward_sar(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, 
                    ob_act_angles, ob_progress, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)

        prediction_scores = self.regress_action(txt_embeds[:, 0])   # [CLS] token

        if compute_loss:
            act_targets = torch.cat([ob_act_angles, ob_progress.unsqueeze(1)], dim=1)
            act_loss = F.mse_loss(prediction_scores, act_targets, reduction='none')
            return act_loss
        else:
            return prediction_scores

    def forward_sprel(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, 
                    sp_anchor_idxs, sp_targets, compute_loss):
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)

        # img_embeds: (batch, views, dim), sp_anchor_idxs: (batch)
        anchor_ob_embeds = torch.gather(ob_embeds, 1, 
            sp_anchor_idxs.unsqueeze(1).unsqueeze(2).repeat(1, 36, ob_embeds.size(-1)))
        # (batch, 1, dim)
        cat_ob_embeds = torch.cat([anchor_ob_embeds, ob_embeds[:, :-1]], -1)
        
        prediction_scores = self.sprel_head(cat_ob_embeds) # (batch, 36, 2)

        if compute_loss:
            sprel_loss = F.mse_loss(prediction_scores, sp_targets, reduction='none')
            return sprel_loss
        else:
            return prediction_scores

    def forward_mrc(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    hist_mrc_masks, hist_img_probs, compute_loss=True):
        txt_embeds, hist_embeds, _ = self.bert(txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            None, None, None, None)
        # print("1")
        # only compute masked regions for better efficient=cy
        hist_embeds = hist_embeds[:, 1:] # remove global embedding
        masked_output = self._compute_masked_hidden(hist_embeds, hist_mrc_masks)
        # print("2")
        prediction_soft_labels = self.image_classifier(masked_output)
        # print("3")
        hist_mrc_targets = self._compute_masked_hidden(hist_img_probs, hist_mrc_masks)

        if compute_loss:
            prediction_soft_labels = F.log_softmax(prediction_soft_labels, dim=-1)
            mrc_loss = F.kl_div(prediction_soft_labels, hist_mrc_targets, reduction='none').sum(dim=1)
            return mrc_loss
        else:
            return prediction_soft_labels, hist_mrc_targets

    def forward_itm(self, txt_ids, txt_masks, 
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks, 
                    num_neg_trajs, compute_loss):
        # (batch_size, 1+num_negs, dim)
        fused_embeds = self.bert.forward_itm(
            txt_ids, txt_masks, 
            hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
            num_neg_trajs=num_neg_trajs)

        prediction_scores = self.itm_head(fused_embeds).squeeze(2) # (batch, 1+num_negs, 1)
        # The first is positive
        itm_targets = torch.zeros(fused_embeds.size(0), dtype=torch.long).to(self.device)

        if compute_loss:
            sprel_loss = F.cross_entropy(prediction_scores, itm_targets, reduction='none')
            return sprel_loss
        else:
            return prediction_scores, itm_targets

    def forward_mim(self, txt_ids, txt_masks, hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
                    hist_mrc_masks, hist_images_dvae, compute_loss=True, d_vae=None, filter=None, dy_filter=None):
        txt_embeds, hist_embeds, _ = self.bert(txt_ids, txt_masks,
                                               hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts,
                                               hist_masks,
                                               None, None, None, None)
        with torch.no_grad():
            B, T, C, H, W = hist_images_dvae.shape
            hist_images_dvae = hist_images_dvae.view(-1, C, H, W)
            output_probs = d_vae.get_codebook_probs(hist_images_dvae)   # (B*T, class_number, patches, patches)
            if self.config.mim_weighted:
                BT, C, patch, _ = output_probs.shape
                if self.config.weighted_method == "middle":
                    alpha = np.ones((patch, patch))
                    alpha[3:12, 3:12] = 2
                    alpha[5:9, 5:9] = 3
                    alpha = alpha.reshape(-1)
                    weights = torch.from_numpy(np.random.dirichlet(alpha, BT)).to(txt_ids.get_device()).unsqueeze(
                        -1).float()
                elif self.config.weighted_method == "block":
                    alpha = np.ones((patch, patch))
                    block_size = random.randint(1, 7)
                    start_location = random.randint(0, patch-block_size-1)
                    alpha[start_location:start_location+block_size, start_location: start_location+block_size] = 5
                    alpha = alpha.reshape(-1)
                    weights = torch.from_numpy(np.random.dirichlet(alpha, BT)).to(txt_ids.get_device()).unsqueeze(
                        -1).float()
                else:
                    weights = torch.from_numpy(np.random.dirichlet(np.ones(patch * patch), BT)).to(
                    txt_ids.get_device()).unsqueeze(-1).float()
                output_probs = torch.transpose(output_probs.view(BT, C, -1), 1, 2)
                output_probs = torch.sum(torch.mul(output_probs, weights), dim=1)
                target = output_probs.view(B, T, -1)
            else:
                target = torch.mean(output_probs, dim=(-1,-2))
                target = target.view(B, T, -1)

            if filter is not None:
                target = torch.index_select(target, dim=-1, index=filter)

            target_all = target.clone()
            if dy_filter is not None:
                dy_filter = torch.from_numpy(dy_filter).to(txt_ids.get_device())
                # target_all = target.clone()
                target = torch.index_select(target, dim=-1, index=dy_filter)

        hist_embeds = hist_embeds[:, 1:]  # remove global embedding
        masked_output = self._compute_masked_hidden(hist_embeds, hist_mrc_masks)
        if self.config.mim_weighted:
            masked_weights = self._compute_masked_hidden(weights.reshape(B, T, -1), hist_mrc_masks)
            prediction_soft_labels = self.mim_head(masked_output, masked_weights)
        else:
            prediction_soft_labels = self.mim_head(masked_output)
        hist_mrc_targets = self._compute_masked_hidden(target, hist_mrc_masks)

        prediction_sf = F.softmax(prediction_soft_labels, dim=-1)
        target_all = self._compute_masked_hidden(target_all, hist_mrc_masks)
        prediction_diff = torch.abs(target_all - prediction_sf)

        if compute_loss:
            if dy_filter is not None:
                prediction_soft_labels = torch.index_select(prediction_soft_labels, dim=-1, index=dy_filter)
            prediction_soft_labels = F.log_softmax(prediction_soft_labels, dim=-1)
            mrc_loss = F.kl_div(prediction_soft_labels, hist_mrc_targets, reduction='none').sum(dim=1)
            return mrc_loss, target_all.detach().cpu().numpy(), prediction_diff.detach().cpu().numpy()
        else:
            if dy_filter is not None:
                prediction_soft_labels = torch.index_select(prediction_soft_labels, dim=-1, index=dy_filter)
            return prediction_soft_labels, hist_mrc_targets

    def forward_apwig(self, txt_ids, txt_masks,
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, compute_loss, ob_action_image, d_vae=None, filter=None, dy_filter=None):

        if not self.config.apwig_dy:
            dy_filter = None
        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks,
                                                       hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts,
                                                       hist_masks,
                                                       ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)

        with torch.no_grad():
            B, C, H, W = ob_action_image.shape
            output_probs = d_vae.get_codebook_probs(ob_action_image)   # (B*T, class_number, patches, patches)
            target = torch.mean(output_probs, dim=(-1,-2))
            target = target.view(B, -1)

            if filter is not None:
                target = torch.index_select(target, dim=-1, index=filter)
            target_all = target.clone()
            if dy_filter is not None:
                dy_filter = torch.from_numpy(dy_filter).to(ob_embeds.get_device())
                # target_all = target.clone()
                target = torch.index_select(target, dim=-1, index=dy_filter)

        prediction_soft_labels = self.apwig_head(txt_embeds[:, 0])

        prediction_sf = F.softmax(prediction_soft_labels, dim=-1)
        prediction_diff = torch.abs(target_all - prediction_sf)

        if compute_loss:
            if dy_filter is not None:
                prediction_soft_labels = torch.index_select(prediction_soft_labels, dim=-1, index=dy_filter)
            prediction_soft_labels = F.log_softmax(prediction_soft_labels, dim=-1)
            apwig_loss = F.kl_div(prediction_soft_labels, target, reduction='none').sum(dim=1)
            if dy_filter is not None:
                return apwig_loss, target_all.detach().cpu().numpy(), prediction_diff.detach().cpu().numpy()
            else:
                return apwig_loss
        else:
            if dy_filter is not None:
                prediction_soft_labels = torch.index_select(prediction_soft_labels, dim=-1, index=dy_filter)
            return prediction_soft_labels, target


    def forward_mpp(self, txt_ids, txt_masks, hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, ob_pano_images, ob_mpp_masks, compute_loss=True, d_vae=None, filter=None, dy_filter=None):

        if not self.config.mpp_dy:
            dy_filter = None

        with torch.no_grad():
            B, T, C, H, W = ob_pano_images.shape
            ob_pano_images = ob_pano_images.view(-1, C, H, W)
            ob_pano_images_masked = torch.masked_select(ob_pano_images, ob_mpp_masks.view(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            ob_pano_images_masked = ob_pano_images_masked.view(-1, W)
            ob_pano_images_masked = ob_pano_images_masked.view(-1, H, W)
            ob_pano_images_masked = ob_pano_images_masked.view(-1, C, H, W)
            output_probs = d_vae.get_codebook_probs(ob_pano_images_masked)  # (B*36, class_number, patches, patches)
            if self.config.mpp_weighted:
                BV, C, patch, _ = output_probs.shape
                if self.config.weighted_method == "middle":
                    alpha = np.ones((patch, patch))
                    alpha[3:12, 3:12] = 2
                    alpha[5:9, 5:9] = 3
                    alpha = alpha.reshape(-1)
                    weights = torch.from_numpy(np.random.dirichlet(alpha, BV)).to(txt_ids.get_device()).unsqueeze(
                        -1).float()
                elif self.config.weighted_method == "block":
                    alpha = np.ones((patch, patch))
                    block_size = random.randint(1, 7)
                    start_location = random.randint(0, patch - block_size - 1)
                    alpha[start_location:start_location + block_size, start_location: start_location + block_size] = 5
                    alpha = alpha.reshape(-1)
                    weights = torch.from_numpy(np.random.dirichlet(alpha, BV)).to(txt_ids.get_device()).unsqueeze(
                        -1).float()
                else:
                    weights = torch.from_numpy(np.random.dirichlet(np.ones(patch * patch), BV)).to(
                    txt_ids.get_device()).unsqueeze(-1).float()
                output_probs = torch.transpose(output_probs.view(BV, C, -1), 1, 2)
                output_probs = torch.sum(torch.mul(output_probs, weights), dim=1)
                target = output_probs
            else:
                target = torch.mean(output_probs, dim=(-1, -2))

            if filter is not None:
                target = torch.index_select(target, dim=-1, index=filter)
            target_all = target.clone()
            if dy_filter is not None:
                dy_filter = torch.from_numpy(dy_filter).to(txt_ids.get_device())
                # target_all = target.clone()
                target = torch.index_select(target, dim=-1, index=dy_filter)

        _, _, ob_pano_embeds = self.bert(txt_ids, txt_masks,
                                               hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts,
                                               hist_masks,
                                               ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)

        masked_output = self._compute_masked_hidden(ob_pano_embeds, ob_mpp_masks)

        if self.config.mpp_weighted:
            prediction_soft_labels = self.mpp_head(masked_output, weights)
        else:
            prediction_soft_labels = self.mpp_head(masked_output)
        ob_mpp_targets = target

        prediction_sf = F.softmax(prediction_soft_labels, dim=-1)
        prediction_diff = torch.abs(target_all - prediction_sf)

        if compute_loss:
            if dy_filter is not None:
                prediction_soft_labels = torch.index_select(prediction_soft_labels, dim=-1, index=dy_filter)
            prediction_soft_labels = F.log_softmax(prediction_soft_labels, dim=-1)
            mrc_loss = F.kl_div(prediction_soft_labels, ob_mpp_targets, reduction='none').sum(dim=1)
            if dy_filter is not None:
                return mrc_loss, target_all.detach().cpu().numpy(), prediction_diff.detach().cpu().numpy()
            else:
                return mrc_loss
        else:
            if dy_filter is not None:
                prediction_soft_labels = torch.index_select(prediction_soft_labels, dim=-1, index=dy_filter)
            return prediction_soft_labels, ob_mpp_targets

    def forward_mapwig(self, txt_ids, txt_masks,
                    hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts, hist_masks,
                    ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks, compute_loss, ob_action_image, d_vae=None, filter=None, dy_filter=None, pos_input=None):

        if not self.config.apwig_dy:
            dy_filter = None

        with torch.no_grad():
            output_probs = d_vae.get_codebook_probs(ob_action_image)   # (B, class_number, patches, patches)
            B, C, patch, _ = output_probs.shape

            if self.config.weighted_method == "middle":
                alpha = np.ones((patch,patch))
                alpha[3:12, 3:12] = 2
                alpha[5:9, 5:9] = 3
                alpha = alpha.reshape(-1)
                weights = torch.from_numpy(np.random.dirichlet(alpha, B)).to(txt_ids.get_device()).unsqueeze(-1).float()
            elif self.config.weighted_method == "block":
                alpha = np.ones((patch, patch))
                block_size = random.randint(1, 7)
                start_location = random.randint(0, patch - block_size - 1)
                alpha[start_location:start_location + block_size, start_location: start_location + block_size] = 5
                alpha = alpha.reshape(-1)
                weights = torch.from_numpy(np.random.dirichlet(alpha, B)).to(txt_ids.get_device()).unsqueeze(-1).float()
            else:
                weights = torch.from_numpy(np.random.dirichlet(np.ones(patch*patch), B)).to(txt_ids.get_device()).unsqueeze(-1).float()
            if pos_input is not None:
                weights = pos_input.to(txt_ids.get_device()).float()
            output_probs = torch.transpose(output_probs.view(B, C, -1), 1, 2)
            output_probs = torch.sum(torch.mul(output_probs, weights), dim=1)
            target = output_probs

            if filter is not None:
                target = torch.index_select(target, dim=-1, index=filter)
            target_all = target.clone()
            if dy_filter is not None:
                dy_filter = torch.from_numpy(dy_filter).to(txt_ids.get_device())
                # target_all = target.clone()
                target = torch.index_select(target, dim=-1, index=dy_filter)

        txt_embeds, hist_embeds, ob_embeds = self.bert(txt_ids, txt_masks,
                                                       hist_img_fts, hist_ang_fts, hist_pano_img_fts, hist_pano_ang_fts,
                                                       hist_masks,
                                                       ob_img_fts, ob_ang_fts, ob_nav_types, ob_masks)

        prediction_soft_labels = self.mapwig_head(txt_embeds[:, 0], weights)

        prediction_sf = F.softmax(prediction_soft_labels, dim=-1)
        prediction_diff = torch.abs(target_all - prediction_sf)

        if compute_loss:
            if dy_filter is not None:
                prediction_soft_labels = torch.index_select(prediction_soft_labels, dim=-1, index=dy_filter)
            prediction_soft_labels = F.log_softmax(prediction_soft_labels, dim=-1)
            apwig_loss = F.kl_div(prediction_soft_labels, target, reduction='none').sum(dim=1)
            if dy_filter is not None:
                return apwig_loss, target_all.detach().cpu().numpy(), prediction_diff.detach().cpu().numpy()
            else:
                return apwig_loss
        else:
            if dy_filter is not None:
                prediction_soft_labels = torch.index_select(prediction_soft_labels, dim=-1, index=dy_filter)
            return prediction_soft_labels, target