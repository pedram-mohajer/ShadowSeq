import os
import torch
from torchvision import transforms
from PIL import Image
import csv
from tqdm import tqdm
from GtsrbCNN import GtsrbCNN 

best_model_path = "models/gtsrb_cnn/best_model.pth"
image_dir = "images"     
output_csv = "predictions.csv"   

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GtsrbCNN(n_class=43, img_size=128).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

predictions = []

with torch.no_grad():
    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue
        
        img_tensor = transform(img).unsqueeze(0)  
        
        output = model(img_tensor.to(device))
        _, pred = torch.max(output, 1)
        predictions.append((filename, int(pred.item())))

with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["imagename", "predicted_class"])
    for fname, pred_class in predictions:
        writer.writerow([fname, pred_class])

print(f"Predictions written to {output_csv}")

