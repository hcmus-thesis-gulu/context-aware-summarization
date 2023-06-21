import gradio as gr
from scripts.summarizer import VidSum


vs = VidSum()


def summarize(vid, hp1, hp2):
    vs.change_param(hyperparam1=hp1, hyperparam2=hp2)
    summary = vs.summarize(vid)
    return summary


video = gr.inputs.Video(label="Upload Video")
hyperparam1 = gr.inputs.Slider(minimum=0, maximum=10, default=5,
                               label="Hyperparameter 1")
hyperparam2 = gr.inputs.Slider(minimum=0, maximum=10, default=5,
                               label="Hyperparameter 2")

iface = gr.Interface(fn=summarize, inputs=[video, hyperparam1, hyperparam2],
                     outputs="video", live=False)
iface.launch()
