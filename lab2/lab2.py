import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torchattacks
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- 1. DEFINICJA MODELU ---
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. SETUP I ŁADOWANIE MODELU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wczytanie wag (upewnij się, że plik model.pth jest w tym samym folderze)
try:
    checkpoint = torch.load("model.pth", map_location=device)
    num_classes = checkpoint['classifier.3.weight'].shape[0]
    model = SmallCNN(num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model załadowany pomyślnie. Liczba klas: {num_classes}")
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku model.pth!")
    # Dla celów demonstracyjnych, jeśli brak pliku, tworzymy losowy model
    model = SmallCNN(10).to(device)
    model.eval()

# --- 3. PRZYGOTOWANIE DANYCH ---
transform = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Podzbiór 100 obrazków do testów
subset_indices = list(range(100))
subset_loader = DataLoader(Subset(test_set, subset_indices), batch_size=10, shuffle=False)

# --- 4. FUNKCJE POMOCNICZE I METRYKI ---

def custom_fgsm(model, image, label, epsilon):
    """ Samodzielna implementacja FGSM """
    if epsilon == 0: return image
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    adv_image = image + epsilon * image.grad.data.sign()
    return torch.clamp(adv_image, 0, 1)

def calculate_batch_metrics(orig_tensor, adv_tensor):
    """ Oblicza L-inf oraz SSIM dla całego bacha """
    # L-infinity
    l_inf = torch.max(torch.abs(orig_tensor - adv_tensor)).item()
    
    # SSIM (średnia dla bacha)
    ssim_values = []
    orig_np = orig_tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)
    adv_np = adv_tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)
    
    for i in range(orig_np.shape[0]):
        # win_size=3 dla małych obrazków 32x32
        s = ssim(orig_np[i], adv_np[i], channel_axis=2, data_range=1.0)
        ssim_values.append(s)
    
    return l_inf, np.mean(ssim_values)

# --- 5. ANALIZA WPŁYWU EPSILONA Z METRYKAMI ---

def analyze_epsilons(model, loader, device):
    epsilons = [0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 0.5, 0.75]
    accuracies = []
    l_infs = []
    ssims = []
    
    print("\n" + "="*50)
    print(f"{'Epsilon':<10} | {'Acc [%]':<10} | {'L-inf':<10} | {'SSIM':<10}")
    print("-" * 50)
    
    for eps in epsilons:
        correct = 0
        total = 0
        batch_l_infs = []
        batch_ssims = []
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = custom_fgsm(model, images.clone(), labels, eps)
            
            with torch.no_grad():
                outputs = model(adv_images)
                correct += (outputs.argmax(1) == labels).sum().item()
            
            total += labels.size(0)
            
            # Oblicz metryki jakościowe
            li, ss = calculate_batch_metrics(images, adv_images)
            batch_l_infs.append(li)
            batch_ssims.append(ss)
        
        acc = 100 * correct / total
        avg_li = np.mean(batch_l_infs)
        avg_ss = np.mean(batch_ssims)
        
        accuracies.append(acc)
        l_infs.append(avg_li)
        ssims.append(avg_ss)
        
        print(f"{eps:<10.4f} | {acc:<10.2f} | {avg_li:<10.4f} | {avg_ss:<10.4f}")

    # Rysowanie wykresu
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Accuracy [%]', color='red')
    ax1.plot(epsilons, accuracies, 'o-', color='red', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('SSIM (Structural Similarity)', color='blue')
    ax2.plot(epsilons, ssims, 's--', color='blue', label='SSIM')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title("Wpływ Epsilona na Skuteczność (Accuracy) i Jakość (SSIM)")
    fig.tight_layout()
    plt.show()

# --- 6. PORÓWNANIE IMPLEMENTACJI ---

def run_comparison(model, loader, epsilon, device):
    total = 0
    correct_custom = 0
    correct_lib = 0
    
    atk_lib = torchattacks.FGSM(model, eps=epsilon)
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        
        adv_custom = custom_fgsm(model, images.clone(), labels, epsilon)
        adv_lib = atk_lib(images, labels)
        
        with torch.no_grad():
            correct_custom += (model(adv_custom).argmax(1) == labels).sum().item()
            correct_lib += (model(adv_lib).argmax(1) == labels).sum().item()

    print("\n--- PORÓWNANIE IMPLEMENTACJI (EPS=8/255) ---")
    print(f"Accuracy (Twój FGSM): {100*correct_custom/total:.2f}%")
    print(f"Accuracy (Biblioteka): {100*correct_lib/total:.2f}%")
    if abs(correct_custom - correct_lib) <= 0:
        print("[SUKCES] Twoja implementacja jest identyczna z biblioteczną!")

# --- 7. PORÓWNANIE FGSM vs PGD ---

def compare_attacks(model, loader, epsilon, device):
    atk_fgsm = torchattacks.FGSM(model, eps=epsilon)
    atk_pgd = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10)
    
    correct_fgsm = 0
    correct_pgd = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        
        adv_fgsm = atk_fgsm(images, labels)
        adv_pgd = atk_pgd(images, labels)
        
        with torch.no_grad():
            correct_fgsm += (model(adv_fgsm).argmax(1) == labels).sum().item()
            correct_pgd += (model(adv_pgd).argmax(1) == labels).sum().item()
            
    print("\n--- PORÓWNANIE SIŁY ATAKÓW ---")
    print(f"Accuracy po FGSM: {100*correct_fgsm/total:.2f}%")
    print(f"Accuracy po PGD:  {100*correct_pgd/total:.2f}%")

# --- 8. URUCHOMIENIE ---

eps_default = 8/255

# 1. Porównanie implementacji
run_comparison(model, subset_loader, eps_default, device)

# 2. Pełna analiza epsilonów z metrykami
analyze_epsilons(model, subset_loader, device)

# 3. Porównanie FGSM vs PGD
compare_attacks(model, subset_loader, eps_default, device)

# 4. Atak celowany (krótki test)
img, lbl = next(iter(subset_loader))
img, lbl = img[0:1].to(device), lbl[0:1].to(device)
target = torch.tensor([3]).to(device)
atk_target = torchattacks.FGSM(model, eps=eps_default)
atk_target.set_mode_targeted_by_label()
adv_target = atk_target(img, target)
pred = model(adv_target).argmax(1).item()
print(f"\n--- TEST ATAKU CELOWANEGO ---")
print(f"Cel: Klasa 3 | Wynik: Klasa {pred}")