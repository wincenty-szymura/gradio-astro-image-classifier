# import relevant libraries
from fastai.vision.all import *
import gradio as gr

# load the model
learn = load_learner('AstroVisionModel.pkl')

# define output categories
categories = ("Gravitational lensing", "No lensing")

# define function that gradio is going to call
def classify_image(image):
    # make predictions
    prediction, index, probabilities = learn.predict(image)
    # return predictions in the format expected by gradio
    return dict(zip(categories, map(float, probabilities)))

# define input, output and examples
image = gr.Image(height=224, width=224)
label = gr.Label()
examples = ['test_n.jpg', 'test_u2.jpg', 'test_y.jpg']

# define and launch the interface
interface = gr.Interface(
    fn=classify_image, 
    inputs=image, 
    outputs=label, 
    examples=examples,
    title="Image classifier spotting gravitational lensing"
)
interface.launch(inline=False)