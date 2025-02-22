import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, resnet50
import torch.nn.functional as F
import io
import time
from collections import Counter

class SpeciesMonitoringSystem:
    def __init__(self):
        # Initialize models
        self.detection_model = efficientnet_b0(pretrained=True)
        self.detection_model.eval()
        
        # Define species classes (example classes)
        self.species_classes = [
                'deer', 'elk', 'moose', 'bear', 'wolf', 'mountain lion', 'bobcat', 
                'lynx', 'bighorn sheep', 'bison', 'wild boar', 'caribou', 'antelope',
                'coyote', 'jaguar', 'leopard', 'tiger', 'lion', 'gorilla', 'chimpanzee',
                'fox', 'raccoon', 'beaver', 'badger', 'otter', 'wolverine', 'porcupine',
                'skunk', 'opossum', 'armadillo', 'wild cat', 'jackal', 'hyena',
                'marten', 'fisher', 'weasel', 'mink', 'coati', 'monkey', 'lemur',
                'rabbit', 'squirrel', 'chipmunk', 'rat', 'mouse', 'vole', 'mole',
                'shrew', 'bat', 'hedgehog', 'gopher', 'prairie dog', 'muskrat',
                'hamster', 'guinea pig', 'ferret', 'chinchilla', 'dormouse',
                'eagle', 'hawk', 'falcon', 'owl', 'vulture', 'condor', 'crow', 'raven',
                'woodpecker', 'duck', 'goose', 'swan', 'heron', 'crane', 'stork',
                'pelican', 'flamingo', 'penguin', 'ostrich', 'emu', 'kiwi', 'peacock',
                'pheasant', 'quail', 'grouse', 'turkey', 'cardinal', 'bluejay',
                'sparrow', 'finch', 'warbler', 'thrush', 'swallow', 'hummingbird',
                'snake', 'lizard', 'turtle', 'tortoise', 'alligator', 'crocodile',
                'iguana', 'gecko', 'monitor lizard', 'chameleon', 'python', 'cobra',
                'viper', 'rattlesnake', 'boa', 'anaconda', 'skink', 'bearded dragon',
                'frog', 'toad', 'salamander', 'newt', 'axolotl', 'caecilian',
                'tree frog', 'bullfrog', 'fire salamander', 'spotted salamander',
                'salmon', 'trout', 'bass', 'pike', 'catfish', 'carp', 'perch',
                'tuna', 'swordfish', 'marlin', 'shark', 'ray', 'eel', 'sturgeon',
                'barracuda', 'grouper', 'snapper', 'cod', 'halibut', 'flounder',
                'whale', 'dolphin', 'porpoise', 'seal', 'sea lion', 'walrus',
                'orca', 'narwhal', 'beluga', 'manatee', 'dugong', 'sea otter',
                'butterfly', 'moth', 'beetle', 'ant', 'bee', 'wasp', 'spider',
                'scorpion', 'centipede', 'millipede', 'crab', 'lobster', 'shrimp',
                'octopus', 'squid', 'jellyfish', 'starfish', 'sea urchin', 'coral',
                'snail', 'slug', 'earthworm', 'leech'
            ]
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def detect_species(self, image):
        # Transform image for model input
        img_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.detection_model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        # Get top predictions
        top_prob, top_class = torch.topk(probabilities, 3)
        results = []
        
        for i in range(3):
            species = self.species_classes[top_class[0][i] % len(self.species_classes)]
            confidence = top_prob[0][i].item() * 100
            results.append((species, confidence))
            
        return results

    def count_population(self, image):
        # Simplified population counting using object detection
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the image
        img_with_contours = np.array(image).copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
        
        return len(contours), Image.fromarray(img_with_contours)

    def assess_health(self, image):
        # Enhanced health assessment based on color and texture analysis
        img_array = np.array(image)
        
        # Calculate average color values for each channel
        avg_color = np.mean(img_array, axis=(0, 1))
        
        # Calculate texture features
        texture_measure = np.std(img_array)
        color_variation = np.std(avg_color)
        
        # Enhanced health scoring
        color_score = np.mean(avg_color) / 255 * 100
        texture_score = min(100, texture_measure / 2)
        variation_score = min(100, color_variation * 2)
        
        # Weighted health score
        health_score = (color_score * 0.4 + texture_score * 0.3 + variation_score * 0.3)
        
        # Determine status with more detailed categories
        if health_score > 80:
            status = "Excellent"
        elif health_score > 60:
            status = "Good"
        elif health_score > 40:
            status = "Fair"
        else:
            status = "Poor"
            
        # Additional health indicators
        indicators = {
            "Color Vibrancy": color_score,
            "Texture Complexity": texture_score,
            "Pattern Variation": variation_score
        }
            
        return status, health_score, indicators
