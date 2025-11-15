from PIL import Image
import matplotlib.pyplot as plt
import torch

def predict_image(img_path, model, class_names, device, image_transform):
    '''
    Predict the class of a single image.

    Args:
    img_path: A string path to an image file.
    model: A trained PyTorch model.
    class_names: A list of class names.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    image_transform: A transform to apply to the image.
    '''

    img = Image.open(img_path)
    img_transformed = image_transform(img).to(device)
    model.eval()
    with torch.inference_mode():
        pred_logit = model(img_transformed.unsqueeze(0).to(device))
        pred_probs = torch.softmax(pred_logit, dim=1)[0]
        pred_classes = pred_logit.argmax(1)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[pred_classes]} | Probability: {pred_probs[pred_classes].item()*100:.2f} %")
    print(f"Prediction: {class_names[pred_classes]} | Probability: {pred_probs[pred_classes].item()*100:.2f} %")
    plt.axis("off")    
