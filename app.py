# import relevant libraries
from fastai.vision.all import *
#from pathlib import Path
from safetensors.torch import load_file
import gradio as gr
import timm

# build minimal dataloaders with transformations reflecting training
dls = ImageDataLoaders.from_folder(
    'example_data',
    valid_pct=0.5,
    item_tfms=Resize(224, method='squish'),
    bs=1
)

# recreate the training architecture
learn = vision_learner(dls, 'convnext_tiny.fb_in22k')

# load trained model weights
state = load_file('astro_vision_model.safetensors')
learn.model.load_state_dict(state, strict=True)
learn.model.eval()

# define output categories
categories = ['Gravitational Lensing', 'No Lensing']

# define function that gradio is going to call
def classify_image(image):
    # make predictions
    prediction, index, probabilities = learn.predict(image)
    # return predictions in the format expected by gradio
    return dict(zip(categories, map(float, probabilities)))

# define input, output and examples
image = gr.Image(height=224, width=224)
label = gr.Label()
examples = [
    'example_data/gravitational_lensing/test_y.jpg',
    'example_data/no_lensing/test_n.jpg',
    'example_data/no_lensing/test_u2.jpg',
]

# define and launch the interface
interface = gr.Interface(
    fn=classify_image, 
    inputs=image, 
    outputs=label, 
    examples=examples,
    title="Image classifier spotting gravitational lensing"
)
interface.launch(inline=False)