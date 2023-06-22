import os

distance = "euclidean"
window_size = 3
min_seg_length = 5
modulation = 1e-3
embedding_dim = 3

embedding_folder = "ablation/dino-brute-force/embeddings"
context_folder = "ablation/dino-brute-force/contexts/{distance}/pca-tsne/embedding-{embedding_dim}/intermediate-{intermediate_dim}"

for i in range(4, 129):
    intermediate_dim = i
    tmp_context_folder = context_folder.format(distance=distance, embedding_dim=embedding_dim, intermediate_dim=intermediate_dim)
    if not os.path.exists(tmp_context_folder):
        os.mkdir(tmp_context_folder)
    os.system("python scripts/extraction.py --embedding-folder {embedding_folder} --context-folder {context_folder} --method ours --distance {distance} --embedding-dim {embedding_dim} --window-size {window_size} --min-seg-length {min_seg_length} --modulation {modulation} --intermediate-components {intermediate_dim}".format(
        embedding_folder=embedding_folder,
        context_folder=tmp_context_folder,
        distance=distance,
        embedding_dim=embedding_dim,
        window_size=window_size,
        min_seg_length=min_seg_length,
        modulation=modulation,
        intermediate_dim=intermediate_dim
    ))

# import os

# distance = "euclidean"
# kf_mod = "middle-ends"
# embedding_dim = 3

# embedding_folder = "ablation/dino-brute-force/embeddings"
# context_folder = "ablation/dino-brute-force/contexts/{distance}/pca-tsne/embedding-{embedding_dim}/intermediate-{intermediate_dim}"
# summary_folder="ablation/dino-brute-force/summaries/{distance}/pca-tsne/embedding-{embedding_dim}/intermediate-{intermediate_dim}"

# for i in range(2, 8):
#     intermediate_dim = 1 << i
#     tmp_context_folder = context_folder.format(distance=distance, embedding_dim=embedding_dim, intermediate_dim=intermediate_dim)
#     tmp_summary_folder = summary_folder.format(distance=distance, embedding_dim=embedding_dim, intermediate_dim=intermediate_dim)
#     os.mkdir(tmp_summary_folder)
    
#     os.system("python scripts/summarization.py --embedding-folder {embedding_folder} --context-folder {context_folder} --summary-folder {summary_folder} --scoring-mode uniform --kf-mode middle-ends --reduced-emb --bias -1 --max-len 0".format(
#         embedding_folder=embedding_folder,
#         context_folder=tmp_context_folder,
#         summary_folder=tmp_summary_folder
#     ))