import torch
from torchvision import transforms
from PIL import Image
from GtsrbCNN import GtsrbCNN
from dataloader import GTSRBSequenceDataset

class GTSRBClassifier:
    def __init__(self, best_model_path, img_size=128, n_class=43):
        """
        Initializes the classifier by loading the trained model.
        :param best_model_path: Path to the saved model weights.
        :param img_size: Expected image size (default 128).
        :param n_class: Number of classes (default 43).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GtsrbCNN(n_class=n_class, img_size=img_size).to(self.device)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])


    def predict_sequence(self, rgb_images, filenames):
        """
        Predicts the class for each image in a sequence.
        :param rgb_images: List of RGB images as numpy arrays (H, W, 3) or PIL Images.
        :param filenames: List of filenames (to be returned along with prediction).
        :return: List of tuples: (filename, predicted_class, confidence)
        """
        tensors = [self.transform(img) for img in rgb_images]
        batch = torch.stack(tensors, dim=0)
        with torch.no_grad():
            outputs = self.model(batch.to(self.device))
            probs = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, dim=1)
        predictions = [(fname, int(pred.item()), float(confidence.item())) for fname, pred, confidence in zip(filenames, preds, confidence)]
        return predictions

