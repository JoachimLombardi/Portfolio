from PIL import Image
import matplotlib.pyplot as plt
import torch
import random

def predict_image(img_path, model, class_names, device, image_transform, h_size=5, l_size=5):
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
        print(pred_logit.size())
    pred_probs = torch.softmax(pred_logit, dim=1)[0]
    pred_classes = pred_logit.argmax(1)
    fig = plt.figure(figsize=(l_size, h_size))
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[pred_classes]} | Probability: {pred_probs[pred_classes].item()*100:.2f} %")
    plt.axis("off")   


def plot_predictions(device, test_dataloader, model, denorm=None, h_size=5, l_size=5, rows=3, cols=3, h_space=0.3, l_space=0.3, k=9):
    fig = plt.figure(figsize=(l_size,h_size))
    class_names = test_dataloader.dataset.classes
    indices = random.sample(range(len(test_dataloader.dataset)), k=9)
    model.eval()
    with torch.inference_mode():
        for i, indice in enumerate(indices):
            image, label = test_dataloader.dataset[indice]
            image = image.to(device)
            logit = model(image.unsqueeze(0)).to(device)
            logit = logit.cpu()
            image = image.cpu()
            probs = torch.softmax(logit, dim=1)
            preds = torch.argmax(probs, dim=1)
            fig.add_subplot(rows, cols, i+1)
            if denorm:
                img = denorm(image)
            else:
                img = image.permute(1,2,0)
            plt.imshow(img)
            plt.subplots_adjust(
            wspace=l_space,
            hspace=h_space
            )
            title = f"Prediction: {class_names[preds]} | Probability: {probs[0][preds].item()*100:.2f} % | Truth: {class_names[label]}"
            if preds == label:
                plt.title(title, fontsize=10, c="g")
            else:
                plt.title(title, fontsize=10, c="r")
            plt.axis("off")
