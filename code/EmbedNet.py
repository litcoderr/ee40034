#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, pdb, sys
import time, importlib
from pathlib import Path
from PIL import Image
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class EmbedNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerClass, **kwargs):
        super(EmbedNet, self).__init__();

        ## __E__ is the embedding model
        EmbedNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__E__ = EmbedNetModel(**kwargs);

        ## __C__ is the classifier plus the loss function
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__C__ = LossFunction(**kwargs);

        ## Number of examples per identity per batch
        self.nPerClass = nPerClass

    def forward(self, data, label=None):

        data    = data.reshape(-1,data.size()[-3],data.size()[-2],data.size()[-1])
        outp    = self.__E__.forward(data)

        if label == None:
            return outp

        else:
            outp    = outp.reshape(self.nPerClass,-1,outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss = self.__C__.forward(outp,label)
            return nloss


class ModelTrainer(object):

    def __init__(self, embed_model, optimizer, scheduler, **kwargs):

        self.__model__  = embed_model

        ## Optimizer (e.g. Adam or SGD)
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        ## Learning rate scheduler
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        assert self.lr_step in ['epoch', 'iteration']
        self.global_step = 0

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, wandb_run=None, epoch=None):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        loss    = 0

        with tqdm(loader, unit="batch") as tepoch:
        
            for data, label in tepoch:

                tepoch.total = tepoch.__len__()

                data    = data.transpose(1,0)

                ## Reset gradients
                self.__optimizer__.zero_grad()

                ## Forward pass and compute loss
                nloss = self.__model__(data.cuda(), label.cuda())
                ## Backward pass
                nloss.backward()
                ## Optimizer step
                self.__optimizer__.step()

                ## Keep cumulative statistics
                loss    += nloss.item()
                counter += 1;

                # Print statistics to progress bar
                tepoch.set_postfix(loss=loss/counter)

                self.global_step += 1

                if wandb_run is not None:
                    current_lr = max(x['lr'] for x in self.__optimizer__.param_groups)
                    metrics = {"train/loss": nloss.item(), "lr": current_lr}
                    if epoch is not None:
                        metrics["epoch"] = epoch
                    wandb_run.log(metrics, step=self.global_step)

                if self.lr_step == 'iteration': self.__scheduler__.step()

            if self.lr_step == 'epoch': self.__scheduler__.step()
        
        return (loss/counter)


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, transform, print_interval=100, num_eval=10, **kwargs):
        
        self.__model__.eval();

        attn_map_enabled     = kwargs.get('attn_map', False)
        tta                  = int(kwargs.get('TTA', 1) or 1)
        use_tta              = tta > 1
        n_attn_map           = int(kwargs.get('n_attn_map', 0) or 0)
        attn_map_save_root   = kwargs.get('attn_map_save_path', "")
        attn_map_save_root   = Path(attn_map_save_root) if attn_map_save_root not in ["", None] else Path(kwargs.get('save_path', 'exps')) / 'attn_maps'
        attn_buffers         = []
        attn_handles         = []
        attn_saved           = 0
        save_all_attn        = n_attn_map <= 0

        need_attn_maps = attn_map_enabled or use_tta

        if need_attn_maps:
            attn_buffers, attn_handles = self._setup_attn_hooks()
        
        feats       = {}

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split(',')[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, transform=transform, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )

        print('Generating embeddings')

        ## Extract features for every image
        for data in tqdm(test_loader):
            image_tensor        = data[0][0]
            inp1                = image_tensor.cuda()

            if need_attn_maps:
                attn_buffers.clear()

            with torch.no_grad():
                ref_feat        = self.__model__(inp1).detach().cpu()

            attn_for_base = list(attn_buffers) if need_attn_maps else []
            prob_map = None
            if use_tta:
                prob_map = self._build_probability_map(attn_for_base, image_tensor)
                ref_feat = self._apply_tta(image_tensor, ref_feat, prob_map, tta, attn_buffers if need_attn_maps else None)

            feats[data[1][0]]   = ref_feat

            if attn_map_enabled and (save_all_attn or attn_saved < n_attn_map) and len(attn_for_base) > 0:
                self._save_attention_maps(image_tensor, attn_for_base, attn_map_save_root, data[1][0])
                attn_saved += 1

            if attn_map_enabled and (not save_all_attn) and attn_saved >= n_attn_map and len(attn_handles) > 0:
                if not use_tta:
                    for handle in attn_handles:
                        handle.remove()
                    attn_handles = []
                attn_map_enabled = False

            if need_attn_maps:
                attn_buffers.clear()

        for handle in attn_handles:
            handle.remove()

        all_scores = [];
        all_labels = [];
        all_trials = []

        print('Computing similarities')

        ## Read files and compute all scores
        for line in tqdm(lines):

            data = line.strip().split(',');

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]

            ## Find cosine similarity score
            if getattr(self.__model__.__C__, 'test_normalize', False):
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            score = torch.mean(F.cosine_similarity(ref_feat, com_feat))

            all_scores.append(score.item());  
            all_labels.append(int(data[0]));
            all_trials.append(data[1] + "," + data[2])

        return (all_scores, all_labels, all_trials)

    def _setup_attn_hooks(self):
        
        attn_buffers = []
        handles = []

        def make_hook(idx):
            def hook(module, input, output):
                attn_buffers.append((idx, torch.sigmoid(output.detach().cpu())))
            return hook

        idx = 0
        for _, module in self.__model__.__E__.named_modules():
            mask_conv = getattr(module, 'mask_conv', None)
            if mask_conv is not None:
                handles.append(mask_conv.register_forward_hook(make_hook(idx)))
                idx += 1

        if idx == 0:
            print('No attention-bearing modules found; skipping attention map saving.')

        return attn_buffers, handles

    def _save_attention_maps(self, image_tensor, attn_buffers, save_root, image_relpath):

        if len(attn_buffers) == 0:
            return

        save_root = Path(save_root)
        rel_path = Path(image_relpath).with_suffix("")
        base_dir = save_root / rel_path
        base_dir.mkdir(parents=True, exist_ok=True)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image_tensor.detach().cpu()
        image = image * std + mean
        image = image.clamp(0, 1)
        image_np = image.permute(1, 2, 0).numpy()

        for idx, mask in sorted(attn_buffers, key=lambda x: x[0]):

            if mask.dim() == 4:
                mask = mask[0, 0, :, :]
            elif mask.dim() == 3:
                mask = mask[0, :, :]

            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=image.shape[1:],
                mode='bilinear',
                align_corners=False,
            ).squeeze().clamp(0, 1)

            mask_np = mask_resized.numpy()
            heatmap_rgb = self._apply_colormap(mask_np)
            alpha = 0.55
            overlay = (1.0 - alpha) * image_np + alpha * heatmap_rgb
            overlay = np.clip(overlay, 0.0, 1.0)

            outfile = base_dir / f"{idx:02d}.png"
            Image.fromarray((overlay * 255).astype(np.uint8)).save(outfile)

    def _apply_colormap(self, mask_np):

        mask_flat = mask_np.astype(np.float32).reshape(-1)
        # Color stops inspired by inferno/magma but with a brighter mid/high end
        stops = np.array([0.0, 0.2, 0.45, 0.7, 1.0], dtype=np.float32)
        colors = np.array(
            [
                [0.05, 0.05, 0.10],
                [0.23, 0.08, 0.32],
                [0.60, 0.20, 0.48],
                [0.93, 0.56, 0.35],
                [1.00, 0.88, 0.70],
            ],
            dtype=np.float32,
        )

        r = np.interp(mask_flat, stops, colors[:, 0])
        g = np.interp(mask_flat, stops, colors[:, 1])
        b = np.interp(mask_flat, stops, colors[:, 2])
        colored = np.stack([r, g, b], axis=1).reshape(mask_np.shape + (3,))
        return colored

    def _build_probability_map(self, attn_buffers, image_tensor):

        if len(attn_buffers) == 0:
            h, w = image_tensor.shape[1:]
            return torch.ones((h, w)) / float(h * w)

        h, w = image_tensor.shape[1:]
        accum = None
        for _, mask in attn_buffers:
            if mask.dim() == 4:
                mask = mask[0, 0, :, :]
            elif mask.dim() == 3:
                mask = mask[0, :, :]
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            ).squeeze()
            accum = mask_resized if accum is None else accum + mask_resized

        if accum is None:
            accum = torch.ones((h, w))

        accum = accum.clamp(min=0)
        total = accum.sum()
        if total <= 0:
            accum = torch.ones_like(accum) / float(h * w)
        else:
            accum = accum / total

        return accum

    def _apply_tta(self, image_tensor, base_feat, prob_map, tta, attn_buffers=None):

        h, w = image_tensor.shape[1:]
        min_side = min(h, w)
        side = max(1, int(round(0.8 * min_side)))
        side = min(side, min_side)
        grid_size = max(1, tta)
        probs = prob_map if prob_map is not None else torch.ones((h, w)) / float(h * w)

        feats = [base_feat.squeeze(0)]
        weights = [1.0]

        if grid_size > 1:
            centers_y = np.linspace(side / 2, h - side / 2, grid_size)
            centers_x = np.linspace(side / 2, w - side / 2, grid_size)

            for cy in centers_y:
                for cx in centers_x:
                    top = int(round(cy - side / 2))
                    left = int(round(cx - side / 2))
                    top = max(0, min(top, h - side))
                    left = max(0, min(left, w - side))

                    crop = image_tensor[:, top:top + side, left:left + side]
                    crop = F.interpolate(
                        crop.unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)

                    if attn_buffers is not None:
                        attn_buffers.clear()

                    with torch.no_grad():
                        tta_feat = self.__model__(crop.cuda()).detach().cpu().squeeze(0)

                    region_weight = probs[top:top + side, left:left + side].sum().item()
                    weights.append(region_weight)
                    feats.append(tta_feat)

        feat_stack = torch.stack(feats, dim=0)
        weight_tensor = torch.tensor(weights, dtype=feat_stack.dtype).unsqueeze(1)
        weight_sum = weight_tensor.sum() + 1e-8
        weighted_feat = (feat_stack * weight_tensor).sum(dim=0, keepdim=True) / weight_sum

        return weighted_feat


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.__model__.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                if name not in self_state:
                    print(f'{origname} is not in the model.');
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print(f"Wrong parameter length: {origname}, model: {self_state[name].size()}, loaded: {loaded_state[origname].size()}");
                continue;

            self_state[name].copy_(param);
