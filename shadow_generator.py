
import os
import numpy as np
import cv2
from PIL import Image
from dataloader import GTSRBSequenceDataset



class ShadowGenerator:
    def __init__(self):
        pass

    def generate_shadow_mask_from_points(self, image_shape, points):
        H, W = image_shape[:2]
        shadow_mask = np.zeros((H, W, 3), dtype=np.uint8)
        polygon = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(shadow_mask, [polygon], (255, 255, 255))
        return shadow_mask

    def apply_shadow_effects(self, image_rgb, full_mask, opacity):
        lab_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        mask_gray = cv2.cvtColor(full_mask, cv2.COLOR_RGB2GRAY)
        shadow_region = mask_gray > 0
        lab_img[:, :, 0] = np.where(shadow_region, lab_img[:, :, 0] * (1 - opacity), lab_img[:, :, 0])
        lab_img = self.edge_blur(lab_img, full_mask, coefficient=5)
        lab_img = self.motion_blur(lab_img, size=5, angle=45)
        lab_img = self.adjust_brightness(lab_img, factor=0.8)
        rgb_out = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
        return rgb_out

    def edge_blur(self, lab_img, shadow_mask, coefficient=5):
        mask_gray = cv2.cvtColor(shadow_mask, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(mask_gray, 100, 200)
        blurred = cv2.GaussianBlur(lab_img, (coefficient, coefficient), 0)
        lab_img[edges > 0, 0] = blurred[edges > 0, 0]
        return lab_img

    def motion_blur(self, lab_img, size=5, angle=45):
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[(size-1)//2, :] = np.ones(size, dtype=np.float32)
        M = cv2.getRotationMatrix2D((size/2 - 0.5, size/2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        kernel = kernel / np.sum(kernel)
        lab_img[:, :, 0] = cv2.filter2D(lab_img[:, :, 0], -1, kernel)
        return lab_img

    def adjust_brightness(self, lab_img, factor=0.8):
        lab_img[:, :, 0] = np.clip(lab_img[:, :, 0] * factor, 0, 255)
        return lab_img

    def process_sequence(self, images, shadow_params):
        n = len(images)
        points = shadow_params["points"]
        opacity = shadow_params["opacity"]

        base_mask = self.generate_shadow_mask_from_points(images[0].shape, points)
        shadowed_images = []

        min_scale = 0.6
        max_scale = 1.0

        for i, img in enumerate(images):
            H, W, _ = img.shape
            scale_factor = min_scale + (max_scale - min_scale) * (i / (n - 1)) if n > 1 else 1.0
            new_W = int(W * scale_factor)
            new_H = int(H * scale_factor)
            resized_mask = cv2.resize(base_mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
            full_mask = np.zeros((H, W, 3), dtype=np.uint8)
            x_offset = (W - new_W) // 2
            y_offset = (H - new_H) // 2
            full_mask[y_offset:y_offset+new_H, x_offset:x_offset+new_W] = resized_mask

            processed_img = self.apply_shadow_effects(img, full_mask, opacity)
            shadowed_images.append(processed_img)

        return shadowed_images, shadow_params



