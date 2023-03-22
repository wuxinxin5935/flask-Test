from flask import Flask, request,  render_template
from PIL import Image
import torch
import torchvision.transforms as transforms


app = Flask(__name__)

# 加载模型
model = torch.jit.load('model/best.pt')

# 类别数组
class_names = ['apple', 'banana']

# 图像预处理
def process_image(image):
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transformation(image).unsqueeze(0)

    return image_tensor

# 添加Home路由
@app.route('/')
def home():
    # return 'welcome to the pytorch flask app!'
    return render_template('home.html')

# 添加predict路由
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_tensor = process_image(Image.open(image))
    output = model(image_tensor)

    probs = torch.nn.functional.softmax(output, dim=1)
    probs = probs.detach().numPy()[0]
    class_index = probs.argmax()

    predicted_class = class_names[class_index]
    prob = probs[class_index]

    class_probs = list(zip(class_names, probs))
    class_probs.sort(key=lambda x:x[1], reverse=True)

    return render_template('predict.html', class_probs=class_probs, predicted_class=predicted_class, prob=prob)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=5000)