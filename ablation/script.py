import os

model = 'dino-b16'
representation = 'cls'
distance = 'cosine'
final_reducer = 'pca'
ext = ''

window_size = 3
min_seg_length = 5
modulation = 1e-3


embedding_folder = 'ablation/{model}/embeddings/{representation}'.format(
    model=model,
    representation=representation,
)
context_folder = 'ablation/{model}/contexts/{representation}/{distance}/{final_reducer}{ext}/{embedding_dim}'
summary_folder = 'ablation/{model}/summaries/{representation}/{distance}/{final_reducer}{ext}/{kf_mode}/{embedding_dim}'

for i in range(2, 159):
    current_context_folder = context_folder.format(
        model=model,
        representation=representation,
        distance=distance,
        final_reducer=final_reducer,
        ext=ext,
        embedding_dim=i,
    )
    if not os.path.exists(current_context_folder):
        os.mkdir(current_context_folder)
        
    os.system("python scripts/extraction.py --embedding-folder {embedding_folder} --context-folder {context_folder} --method ours --max-len 60 --distance {distance} --embedding-dim {embedding_dim} --window-size {window_size} --min-seg-length {min_seg_length} --modulation {modulation} --intermediate-components {intermediate_dim} --final-reducer {final_reducer}".format(
        embedding_folder=embedding_folder,
        context_folder=current_context_folder,
        distance=distance,
        embedding_dim = i,
        window_size=window_size,
        min_seg_length=min_seg_length,
        modulation=modulation,
        intermediate_dim=-1,
        final_reducer=final_reducer,
    ))
    
    mean_summary_folder = summary_folder.format(
        model=model,
        representation=representation,
        distance=distance,
        final_reducer=final_reducer,
        ext=ext,
        embedding_dim=i,
        kf_mode = 'mean',
    )
    
    if not os.path.exists(mean_summary_folder):
        os.mkdir(mean_summary_folder)
    
    os.system("python scripts/summarization.py --embedding-folder {embedding_folder} --context-folder {context_folder} --summary-folder {summary_folder} --scoring-mode {scoring_mode} --kf-mod {kf_mode} --reduced-emb --bias 0.5".format(
        embedding_folder=embedding_folder,
        context_folder=current_context_folder,
        summary_folder=mean_summary_folder,
        scoring_mode='uniform',
        kf_mode='mean',
    ))
    
    middle_ends_summary_folder = summary_folder.format(
        model=model,
        representation=representation,
        distance=distance,
        final_reducer=final_reducer,
        ext=ext,
        embedding_dim=i,
        kf_mode = 'middle-ends',
    )
    
    if not os.path.exists(middle_ends_summary_folder):
        os.mkdir(middle_ends_summary_folder)
    
    os.system("python scripts/summarization.py --embedding-folder {embedding_folder} --context-folder {context_folder} --summary-folder {summary_folder} --scoring-mode {scoring_mode} --kf-mod {kf_mode} --reduced-emb --bias -1".format(
        embedding_folder=embedding_folder,
        context_folder=current_context_folder,
        summary_folder=middle_ends_summary_folder,
        scoring_mode='uniform',
        kf_mode='middle-ends',
    ))