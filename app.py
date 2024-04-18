from io import BytesIO 
import base64 
import torchvision.transforms as transforms 
import torchvision.transforms.v2 as transforms_v2
import torch 
import numpy
import json 
from PIL import Image 
from model.vitbnv1 import ViTBN


from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

@app.route('/') 
def root(): 
    return render_template("index2.html")

@app.route('/predict', methods=['POST']) 
def predictions_endpoint(): 
   
    if request.method == 'POST': 
        
        file = request.data

        predicted_class, confidence = make_predictions(file)
        
        response_1 = predicted_class.item()
        response_2 = confidence.item()
        
        
        
        return jsonify(response_1, response_2) 
    


if __name__ == "__main__": 

    host = "127.0.0.1"
    port_number = 8080 

    app.run(host, port_number)


def transform_image(image): 

    transformation = transforms.Compose([
	transforms.Resize(28),
        transforms.ToTensor()
    ]) 
    data = image.split(b',')[-1]
    image_data = base64.decodebytes(data) 
    pil_image = Image.open(BytesIO(image_data))
    processed_image = transformation(pil_image) 

    return processed_image

    
    
model = ViTBN(
                image_size = 28,
                patch_size = 7,
                num_classes = 10,
                channels =1,
                dim = 64,
                depth = 6,
                heads = 8,
                mlp_dim = 128,
                pool = 'cls',
                dropout = 0.0,
                emb_dropout = 0.0,
                pos_emb ='learn'
    )

model.load_state_dict(torch.load("model100epoch_mnist.pth"))
model.eval()


def make_predictions(image_file): 
    
    input_tensor = transform_image(image_file) 
    input_tensor = torch.reshape(input_tensor[3], [1, 1, 28, 28])
    outputs = model(input_tensor) 
    softmax_layer = torch.nn.Softmax(1)
    outputs = softmax_layer(outputs)
    x_hat, y_hat = torch.max(outputs, 1)
    predicted_class = y_hat
    confidence = x_hat
    
    return predicted_class, confidence 




