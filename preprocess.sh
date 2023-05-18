#!/bin/bash

# Check if video folder path and output folder path are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 video_folder_path output_folder_path"
    exit 1
fi

# Create features and clustering folders inside output folder
mkdir -p "$2/embeddings" "$2/clustering"

# Run feature extraction script with arguments
python extractor.py \
--video-folder "$1" \
--embedding-folder "$2/embeddings" \
--frame-rate 4 \
--representation cls

# Run clustering script with arguments
python clustering.py \
--embedding-folder "$2/embeddings" \
--clustering-folder "$2/clustering" \
--method dbscan \
--num-clusters 10 \
--window-size 10 \
--min-seg-length 10 \
--distance jensenshannon
