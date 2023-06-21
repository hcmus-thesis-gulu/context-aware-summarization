# Video Summarization with Gradio

This project provides a Gradio interface for video summarization. Users can upload their videos and adjust several hyperparameters before or after uploading their original video. Afterward, they can press a button to start the summarizer.

## Installation

To install the required dependencies for this project, run the following command:

```
pip install -r requirements.txt
```

## Usage

To launch the Gradio interface, navigate to the `demo` folder and run the following command:

```
python app.py
```

This will launch the Gradio interface in your web browser. To use the interface, follow these steps:

1. Upload your video by clicking on the "Upload Video" button and selecting your video file.
2. Adjust the hyperparameters by moving the sliders for "Hyperparameter 1" and "Hyperparameter 2".
3. Click on the "Submit" button to start the summarization process.
4. The summarized video will be displayed below the "Submit" button.

## Project Introduction

This project uses a pre-trained video summarization model to generate summaries of user-uploaded videos. The model is implemented as a `VidSum` object, which is placed inside the `summarizer.py` file of the `scripts` folder inside the same project folder. The files for the Gradio application are placed in a folder called `demo` inside the general folder of the project.
