import os
import gradio as gr


def video_identity(video):
    print(type(video))
    return video


examples = ['Jumps.mp4', 'Jumps.webm']
example_files = [os.path.join(os.path.dirname(__file__),
                              f"examples/{example}")
                 for example in examples]


demo = gr.Interface(video_identity,
                    gr.Video(),
                    "playable_video",
                    examples=example_files
                    )

if __name__ == "__main__":
    demo.launch(share=True,
                debug=True)
