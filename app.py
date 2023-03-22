from flask import Flask, request,  render_template
from PIL import Image
import torch
import torchvision.transforms as transforms


app = Flask(__name__)

@app.route('/')
def home():
    # return 'welcome to the pytorch flask app!'
    return render_template('home.html')




if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)