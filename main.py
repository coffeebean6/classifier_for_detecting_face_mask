import gradio as gr
from infer import infer_pic


def infer(image_path):
    pred = infer_pic(image_path)
    return pred


with gr.Blocks() as gui:
    image_path = gr.Image(label="Upload an image", type="filepath")
    infer_button = gr.Button("Determine if wearing a face mask")
    infer_output = gr.Textbox(label="Output")
    
    infer_button.click(infer, inputs=image_path, outputs=infer_output)
        
gui.launch()

