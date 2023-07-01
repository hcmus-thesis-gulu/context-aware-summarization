import os
import gradio as gr
import argparse

from scripts.application import VidSum


parser = argparse.ArgumentParser()
parser.add_argument('--example-folder', type=str, default='examples',
                    help='Path to the example folder')
parser.add_argument('--example-extension', type=str, default='examples',
                    choices=['mp4', 'webm', 'avi'],
                    help='Path to the example folder')
parser.add_argument('--output-folder', type=str, default='output',
                    help='Path to the output folder')
cli_args = parser.parse_args()
vs = VidSum()


def summarize_video(*args):
    # Create output folder
    if not os.path.exists(cli_args.output_folder):
        os.makedirs(cli_args.output_folder)
    
    for arg in args:
        print(type(arg))
    
    vs.set_params(*args[1:])
    summary = vs.summarize(args[0], cli_args.output_folder)
    return summary


video = gr.Video(label="Upload your video or select an example")


input_frame_rate = gr.Dropdown(choices=[1, 2, 4, 8, 16], default=4,
                               label="Input Frame Rate (fps) to Sample Features")
method = gr.Dropdown(choices=['kmeans', 'dbscan', 'gaussian', 'ours', 'agglo'],
                     default='ours',
                     label="Clustering Method for Information Propation")
distance = gr.Dropdown(choices=['euclidean', 'cosine'], default='cosine',
                       label="Distance used for Clustering")
max_length = gr.Slider(minimum=5, maximum=180, step=1, default=30,
                       label="Maximum Length of Video Summary (seconds)")
modulation = gr.Slider(minimum=-10, maximum=-1, step=0.1, default=-3,
                       label="Modulation Exponent ($10^x$) for Cluster Numbers")
intermediate_components = gr.Slider(minimum=2, maximum=128, step=1, default=50,
                                    label="Number of Intermediate Components")
window_size = gr.Slider(minimum=1, maximum=9, step=2, default=3,
                        label="Window Size for Smoothing")
min_seg_length = gr.Slider(minimum=1, maximum=5, step=1, default=3,
                           label='Minimum Segment Length')


reduced_emb = gr.Checkbox(label='Use Reduced Embeddings')
scoring_mode = gr.Dropdown(choices=['mean', 'middle', 'uniform'], default='uniform',
                           label='Method for Calculating Importances on Segments')
kf_mode = gr.CheckboxGroup(choices=['mean', 'middle', 'ends'],
                           default=['middle', 'ends'],
                           label='Method for Selecting Keyframes from Segments')
bias = gr.Slider(minimum=-1, maximum=1, step=0.1, default=-1,
                 label='Bias for Frames near Keyframes (0: No Bias)')


output_frame_rate = gr.Dropdown(choices=['auto', 8, 16, 24, 30, 32], default=4,
                                label="Output Frame Rate (fps) of Video Summary")
sum_rate = gr.Dropdown(choices=['10%', '15%', '20%', '25%', '30%'], default='20%',
                       label="Ratio of Video Summary to Original Video Length")
extension = gr.Dropdown(choices=['mp4', 'webm', 'avi'], default='mp4',
                        label="Extension of Video Summary (for storage)")


inputs = [video, input_frame_rate, method, distance, max_length, modulation,
          intermediate_components, window_size, min_seg_length, reduced_emb,
          scoring_mode, kf_mode, bias, output_frame_rate, sum_rate, extension]
outputs = [gr.Video(label="Video Summary")]


examples = []
for example in os.listdir(cli_args.example_folder):
    if example.endswith(cli_args.example_extension):
        example_file = os.path.join(cli_args.example_folder, example)
        example = [example_file] + [None] * (len(inputs) - 1)
        examples.append(example_file)


demo = gr.Interface(summarize_video,
                    inputs=inputs,
                    outputs=outputs,
                    examples=[examples],
                    cache_examples=True,
                    live=False
                    )


def app():
    demo.launch(share=True,
                debug=True)


if __name__ == "__main__":
    app()
