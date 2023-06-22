import gradio as gr


def video_identity(video):
    print(type(video))
    return video


demo = gr.Interface(video_identity,
                    gr.Video(),
                    "playable_video",
                    share=True,
                    )

if __name__ == "__main__":
    demo.launch()
