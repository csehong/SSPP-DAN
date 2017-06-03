#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
DIR="$(cd "$(dirname "$SCRIPT")"; pwd -P)"
cd "$DIR"

echo "Downloading..."

wget -c http://www.vlfeat.org/matconvnet/models/vgg-face.mat
