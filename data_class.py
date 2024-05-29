import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class LandmarksDataset(Dataset):
    def __init__(self, tps_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.data = self.read_tps(tps_file)
        self.transform = transform
        self.resize_transform = transforms.Resize((244, 244))
        self.to_tensor = transforms.ToTensor()
    
    def read_tps(self, file_path):
        with open(file_path, 'r') as file:
            data = []
            current_entry = {}
            coordinates = []

            for line in file:
                line = line.strip()
                if line.startswith("LM"):
                    if current_entry:  # save previous entry if exists
                        current_entry['coordinates'] = coordinates
                        data.append(current_entry)
                    current_entry = {}
                    coordinates = []
                    current_entry['LM'] = int(line.split('=')[1])
                elif line.startswith("IMAGE"):
                    current_entry['IMAGE'] = line.split('=')[1]
                elif line.startswith("ID"):
                    current_entry['ID'] = int(line.split('=')[1])
                else:
                    # Add coordinates
                    x, y = map(float, line.split())
                    coordinates.append((x, y))
            
            # Append the last entry
            if current_entry:
                current_entry['coordinates'] = coordinates
                data.append(current_entry)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_file = entry['IMAGE']
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        if self.transform:
            image = self.transform(image)
        else:
            image = self.resize_transform(image)

        image_tensor = self.to_tensor(image)
        
        # Scale landmarks
        coordinates = entry['coordinates']
        x_coords, y_coords = zip(*coordinates)
        
        # Calculate scaling factors
        x_scale = 244 / original_size[0]
        y_scale = 244 / original_size[1]
        
        # Scale the coordinates
        scaled_coords = [(x * x_scale, y * y_scale) for x, y in coordinates]

        # Reflect the y-coordinates
        reflected_coords = [(x, 244 - y) for x, y in scaled_coords]
        
        # Convert to tensor
        landmarks_tensor = torch.tensor(reflected_coords, dtype=torch.float32)
        
        return image_tensor, landmarks_tensor

