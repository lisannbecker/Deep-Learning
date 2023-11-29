################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

"""Defines the VisualPrompting model (based on CLIP)"""
from pprint import pprint
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from clip import clip
from vp import (
    PadPrompter,
    FixedPatchPrompter,
)
#my additions
import torch.nn.functional as F

PROMPT_TYPES = {
    "padding": PadPrompter,
    "fixed_patch": FixedPatchPrompter,
}


def load_clip_to_cpu(cfg):
    """Loads CLIP model to CPU."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class VisualPromptCLIP(nn.Module):
    """Modified CLIP module to support prompting."""
    def __init__(self, args, dataset, template="This is a photo of {}"):
        super(VisualPromptCLIP, self).__init__()
        classnames = dataset.classes

        print(f"Loading CLIP (backbone: {args.arch})")
        clip_model = self.load_clip_to_cpu(args)
        clip_model.to(args.device)

        # Hack to make model as float() (This is a CLIP hack)
        if args.device == "cpu":
            clip_model = clip_model.float()

        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        print("List of prompts:")
        print(prompts)

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Write code to compute text features.
        # Hint: You can use the code from clipzs.py here!

        # Instructions:
        # - Given a list of prompts, compute the text features for each prompt.
        tokenized_text = clip.tokenize(prompts)

        text_features = clip_model.encode_text(tokenized_text.to(args.device))
        text_features = (text_features - text_features.mean()) / text_features.std() 

        # - Return a tensor of shape (num_prompts, 512).
        #print(f"Shape text features: {text_features.size()} - should be (num_prompts, 512)")

        #######################
        # END OF YOUR CODE    #
        #######################

        self.text_features = text_features
        self.clip_model = clip_model
        self.logit_scale = self.clip_model.logit_scale.exp().detach()

        assert args.method in PROMPT_TYPES, f"{args.method} is not supported :)!"
        self.prompt_learner = PROMPT_TYPES[args.method](args)

        if args.visualize_prompt:
            self.visualize_prompt(args.method)

    def forward(self, image):
        """Forward pass of the model."""

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Implement the forward function. This is not exactly the same as
        # the model_inferece function in clipzs.py! Please see the steps below.

        # Steps:
        # - [!] Add the prompt to the image using self.prompt_learner.
        prompted_img = self.prompt_learner(image)#[0].cpu()

        # - Compute the image features using the CLIP model.
        img_features = self.clip_model.encode_image(image)

        # - Normalize the image features.
        normalized_img_features = (img_features - img_features.mean()) / img_features.std()

        # - Compute similarity logits between the image features and the text features.
        similarity_logits = F.cosine_similarity(normalized_img_features.unsqueeze(1), self.text_features.unsqueeze(0), dim=-1)

        # - You need to multiply the similarity logits with the logit scale (clip_model.logit_scale).
        similarity_logits *= self.clip_model.logit_scale
        
        # - Return logits of shape (batch size, number of classes).
        return similarity_logits

        #######################
        # END OF YOUR CODE    #
        #######################

    def load_clip_to_cpu(self, args):
        """Loads CLIP model to CPU."""
        backbone_name = args.arch
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, args.root)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())
        return model

    @torch.no_grad()
    def visualize_prompt(self, method):
        """Visualizes the prompt."""
        fake_img = torch.ones(1, 3, 224, 224)
        prompted_img = self.prompt_learner(fake_img)[0].cpu()
        prompted_img = torch.clamp(prompted_img, 0, 1)

        print("Visualizing prompt...")
        plt.imsave(f"prompt_{method}.png", prompted_img.permute(1, 2, 0).numpy())
