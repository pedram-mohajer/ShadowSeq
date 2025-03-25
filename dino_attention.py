import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as pth_transforms
import util.vision_transformer as vits
import json
import gc
from tqdm import tqdm


class DINOAttentionExtractor:
    def __init__(self, pretrained_weights, meta_dir, patch_size=16, arch="vit_small", image_size=(2048, 2048), device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.image_size = image_size
        self.meta_dir = meta_dir

        self.model = vits.__dict__[arch](patch_size=self.patch_size, num_classes=0)
        self.model.eval().to(self.device)

        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)

        self.transform = pth_transforms.Compose([
            pth_transforms.Resize(self.image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def extract_attention_maps(self):
        attention_maps = {}
        for fname in tqdm(sorted(os.listdir(self.meta_dir)), desc="Extracting attention maps"):
            if not fname.endswith(".png"):
                continue
            class_id = os.path.splitext(fname)[0]
            img_path = os.path.join(self.meta_dir, fname)
            img = Image.open(img_path).convert("RGB").resize(self.image_size)
            img_tensor = self.transform(img)
            w, h = img_tensor.shape[1] - img_tensor.shape[1] % self.patch_size, img_tensor.shape[2] - img_tensor.shape[2] % self.patch_size
            img_tensor = img_tensor[:, :w, :h].unsqueeze(0).to(self.device)

            with torch.no_grad():
                w_featmap = img_tensor.shape[-2] // self.patch_size
                h_featmap = img_tensor.shape[-1] // self.patch_size
                attentions = self.model.get_last_selfattention(img_tensor)
                nh = attentions.shape[1]
                attn_map = attentions[0, :, 0, 1:].reshape(nh, w_featmap, h_featmap).mean(0)

                attn_map -= attn_map.min()
                attn_map /= attn_map.max() + 1e-6
                attn_map_np = (attn_map.detach().cpu().numpy() * 255).astype(np.uint8)
                attn_map_rgb = np.stack([attn_map_np] * 3, axis=-1)
                attention_maps[class_id] = attn_map_rgb

            del img_tensor, attn_map, attn_map_rgb
            torch.cuda.empty_cache()
            gc.collect()
        return attention_maps


    def extract_attention_map_image(self, image):
        """
        Computes the attention map for a single image.
        
        :param image: A PIL Image (if a numpy array is passed, it is converted to PIL).
        :return: The attention map as an RGB numpy array (values 0-255).
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        image = image.convert("RGB").resize(self.image_size)
        
        img_tensor = self.transform(image)
        w = img_tensor.shape[1] - img_tensor.shape[1] % self.patch_size
        h = img_tensor.shape[2] - img_tensor.shape[2] % self.patch_size
        img_tensor = img_tensor[:, :w, :h].unsqueeze(0).to(self.device)

        with torch.no_grad():
            w_featmap = img_tensor.shape[-2] // self.patch_size
            h_featmap = img_tensor.shape[-1] // self.patch_size
            attentions = self.model.get_last_selfattention(img_tensor)
            nh = attentions.shape[1]
            attn_map = attentions[0, :, 0, 1:].reshape(nh, w_featmap, h_featmap).mean(0)
            attn_map -= attn_map.min()
            attn_map /= (attn_map.max() + 1e-6)
            attn_map_np = (attn_map.detach().cpu().numpy() * 255).astype(np.uint8)
            attn_map_rgb = np.stack([attn_map_np] * 3, axis=-1)
            
        return attn_map_rgb


    def extract_attention_map_images(self, images, filenames):
        """
        Computes attention maps for a list of images.
        
        :param images: List of images (each either a PIL Image or a numpy array).
        :param filenames: List of filenames corresponding to the images.
        :return: A dictionary mapping each filename to its attention map (RGB numpy array).
        """
        attn_dict = {}
        for img, fname in tqdm(list(zip(images, filenames)), desc="Extracting attention maps", total=len(images)):
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB").resize(self.image_size)
            img_tensor = self.transform(img)
            w = img_tensor.shape[1] - img_tensor.shape[1] % self.patch_size
            h = img_tensor.shape[2] - img_tensor.shape[2] % self.patch_size
            img_tensor = img_tensor[:, :w, :h].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                w_featmap = img_tensor.shape[-2] // self.patch_size
                h_featmap = img_tensor.shape[-1] // self.patch_size
                attentions = self.model.get_last_selfattention(img_tensor)
                nh = attentions.shape[1]
                attn_map = attentions[0, :, 0, 1:].reshape(nh, w_featmap, h_featmap).mean(0)
                
                attn_map -= attn_map.min()
                attn_map /= (attn_map.max() + 1e-6)
                attn_map_np = (attn_map.detach().cpu().numpy() * 255).astype(np.uint8)
                attn_map_rgb = np.stack([attn_map_np] * 3, axis=-1)
            
            attn_dict[fname] = attn_map_rgb
            del img_tensor, attn_map, attn_map_rgb
            torch.cuda.empty_cache()
        return attn_dict



