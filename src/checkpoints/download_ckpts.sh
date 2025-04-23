#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the URLs for model checkpoints
MaskRCNN_URL="https://huggingface.co/kew1046/MaskRCNN-resnet50FPN/resolve/main/maskrcnn-resnet50fpn.pt"

# checkpoints
echo "Downloading maskrcnn-resnet50fpn.pt checkpoint..."
$CMD $MaskRCNN_URL || { echo "Failed to download checkpoint from $MaskRCNN_URL"; exit 1; }

echo "All checkpoints are downloaded successfully."
