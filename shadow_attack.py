# shadow_attack.py
import torch
import numpy as np
from copy import deepcopy
import random
from PIL import Image
import os
from GTSRBClassifier import GTSRBClassifier
from tqdm import tqdm


THRESHOLD = 27
LAMBDA_ATTENTION = 0.5

class ShadowAttack:
    def __init__(self, gtsrb_model, dino_model, shadow_generator, population_size=5, generations=5,shape_shadow="polygon"):
        """which
        Initializes the attack with the given models and parameters.
        :param gtsrb_model: An instance of GTSRBClassifier (loaded with your CNN weights).
        :param dino_model: The DINO attention extractor model.
        :param shadow_generator: The shadow generator instance.
        :param population_size: Number of candidates in each generation.
        :param generations: Number of generations to run.
        """
        self.GtsrbCNN = gtsrb_model
        self.dino = dino_model
        self.shadow_gen = shadow_generator
        self.population_size = population_size
        self.generations = generations
        self.meta_attn_maps = self.dino.extract_attention_maps()
        self.shape_shadow = shape_shadow
        print("shape_shadow", self.shape_shadow)



    def compute_loss(self, shadowed_imgs, preds, real_class, target_attn, attn_images_shadow, attn_dict, filenames):
        """
        Computes the multi-objective loss for a candidate set.
        For each image:
        - Classification loss: The confidence for the true class if the image is correctly classified,
            and 0 if misclassified.
        - Attention loss: Mean squared error between the candidate attention map (retrieved via filename)
            and the target attention map.
        - Total loss for the image: cls_loss - LAMBDA_ATTENTION * attn_loss.
        Returns the average loss over the sequence.
        
        :param shadowed_imgs: List of shadowed RGB images.
        :param preds: List of tuples (filename, predicted_class, confidence) from the classifier.
        :param real_class: True class label for the sequence.
        :param target_attn: Target attention map for the true class.
        :param attn_images_shadow: Dictionary mapping each filename to its candidate attention map.
        :param filenames: List of filenames corresponding to the images.
        :return: Average total loss.
        """
        total_loss = 0.0
        for i, (fname, pred_class, confidence) in enumerate(preds):
            cls_loss = 1.0 - confidence if pred_class == real_class else confidence

            attn_pred = attn_images_shadow[fname]
            attn_clean = attn_dict[fname]

            # Normalize to [0, 1]
            attn_pred = attn_pred / (np.max(attn_pred) + 1e-8)
            attn_clean = attn_clean / (np.max(attn_clean) + 1e-8)

            attn_loss = np.mean((attn_pred - attn_clean) ** 2)

            total_loss += cls_loss - LAMBDA_ATTENTION * attn_loss
        return total_loss / len(preds)



    def random_shadow_params(self, H, W,shape_shadow):
        center = (W // 2, H // 2) 
        max_dev = 50
        min_dev = 25

        if shape_shadow == "polygon":
            print("shadow shape: ", shape_shadow)
            points = [
                (np.random.randint(0, W // 2), np.random.randint(0, H // 2)),
                (np.random.randint(W // 2, W), np.random.randint(0, H // 2)),
                (np.random.randint(W // 2, W), np.random.randint(H // 2, H)),
                (np.random.randint(0, W // 2), np.random.randint(H // 2, H))
            ]
        elif shape_shadow == "triangle":
            pt_top = (int(center[0] + np.random.uniform(-max_dev, max_dev)),
                    int(center[1] + np.random.uniform(-max_dev, -min_dev)))
            pt_bottom_left = (int(center[0] + np.random.uniform(-max_dev, -min_dev)),
                            int(center[1] + np.random.uniform(min_dev, max_dev)))
            pt_bottom_right = (int(center[0] + np.random.uniform(min_dev, max_dev)),
                            int(center[1] + np.random.uniform(min_dev, max_dev)))
            points = [pt_top, pt_bottom_left, pt_bottom_right]
        else:
            raise ValueError("Invalid SHAPE specified. Use 'polygon' or 'triangle'.")
            
        return {
            "opacity": np.random.uniform(0.1, 0.7),  
            "points": points
        }


    def mutate(self, params, H, W):
        mutated = deepcopy(params)

        mutated["opacity"] = np.clip(mutated["opacity"] + np.random.uniform(-0.1, 0.1), 0.1, 0.7)
        mutated["points"] = [
            (
                np.clip(x + np.random.randint(-10, 11), 0, W - 1),
                np.clip(y + np.random.randint(-10, 11), 0, H - 1)
            ) for (x, y) in params["points"]
        ]
        return mutated

    def crossover(self, p1, p2):
        return {
            "opacity": (p1["opacity"] + p2["opacity"]) / 2,
            "points": [random.choice(pair) for pair in zip(p1["points"], p2["points"])]
        }

    def attack_sequence(self, rgb_images, filenames, real_class, attn_dict):
        """
        Runs the genetic algorithm-based attack on a sequence of images using a multi-objective framework.
        :param rgb_images: List of RGB images (numpy arrays, 128x128).
        :param filenames: List of image filenames.
        :param real_class: The true class label for the sequence.
        :return: (best_shadowed_images, best_shadow_params)
        """
        H, W, _ = rgb_images[0].shape
        target_attn = self.meta_attn_maps[str(real_class)]

        population = [self.random_shadow_params(H, W,self.shape_shadow) for _ in range(self.population_size)]
        best_params = None
        best_shadowed = None
        best_score = float('inf')
        print("For Clean Images: ")
        for gen in tqdm(range(self.generations), desc="Generations", unit="gen"):
            scored_population = []
            for params in tqdm(population, desc=f"Gen {gen+1}: Evaluating candidates", leave=False):
                shadowed_imgs, _ = self.shadow_gen.process_sequence(deepcopy(rgb_images), params)
                preds = self.GtsrbCNN.predict_sequence(shadowed_imgs, filenames)
                print("For Shadow Images: ")
                attn_maps_shadowed = self.dino.extract_attention_map_images(shadowed_imgs, filenames)
                misclassified_count = sum(1 for _, pred_class, _ in preds if pred_class != real_class)
                print("misclassified_count : ",misclassified_count)
                if misclassified_count >= THRESHOLD:
                    tqdm.write(f"Threshold met in generation {gen+1}: {misclassified_count} misclassified images.")
                    return shadowed_imgs, params

                total_loss = self.compute_loss(shadowed_imgs, preds, real_class, target_attn, attn_maps_shadowed,attn_dict,filenames)
                scored_population.append((total_loss, params, shadowed_imgs))

                tqdm.write(f"Candidate loss: {total_loss:.3f}")

                if total_loss < best_score:
                    best_score = total_loss
                    best_params = params
                    best_shadowed = shadowed_imgs


            scored_population.sort(key=lambda x: x[0], reverse=True)
            top_half = [p for _, p, _ in scored_population[:self.population_size // 2]]


            new_population = top_half[:]
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(top_half, 2)
                child = self.crossover(parent1, parent2)
                mutated = self.mutate(child, H, W)
                new_population.append(mutated)

            population = new_population

        return None, None
