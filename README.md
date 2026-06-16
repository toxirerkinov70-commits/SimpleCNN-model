# Image Classification with SimpleCNN (PyTorch)

6 toifali tasvir tasniflash modeli — PyTorch yordamida qurilgan o'ziga xos CNN arxitekturasi.

## Loyiha haqida

Bu loyihada sahnalarni tasvir bo'yicha 6 sinfga ajratuvchi **SimpleCNN** modeli yaratilgan:

| Sinf | Ma'no |
|------|-------|
| `buildings` | Binolar |
| `forest` | O'rmon |
| `glacier` | Muzlik |
| `mountain` | Tog' |
| `sea` | Dengiz |
| `street` | Ko'cha |

## Natijalar

| Ko'rsatkich | Qiymat |
|-------------|--------|
| Train Accuracy | **83.04%** |
| Test Accuracy | **84.40%** |
| Epochlar soni | 10 |
| Optimizer | Adam (lr=0.001) |
| Loss funksiyasi | CrossEntropyLoss |

## Model Arxitekturasi

```
SimpleCNN(
  (conv1): Conv2d(3, 16, kernel_size=3, padding=1)
  (pool):  MaxPool2d(2, 2)
  (conv2): Conv2d(16, 32, kernel_size=3, padding=1)
  (pool):  MaxPool2d(2, 2)
  (fc1):   Linear(32768, 128)
  (fc2):   Linear(128, 6)
)
```

Kirish: `128×128` piksellik RGB tasvirlar → Chiqish: 6 sinfdan biri

## Ma'lumotlar To'plami

**Intel Image Classification** dataset:
- Train: **14,034** tasvir
- Test:  **3,000** tasvir
- Papka tuzilishi: `Amaliy/seg_train/` va `Amaliy/seg_test/`

### Data Augmentation (Train uchun)

```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
transforms.RandomResizedCrop(128, scale=(0.8, 1.0))
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

## O'rnatish

```bash
pip install torch torchvision matplotlib
```

## Ishga Tushirish

1. Dataset papkasini loyiha ichiga qo'ying:
```
ML_ikkinchi/
├── Amaliy/
│   ├── seg_train/
│   │   ├── buildings/
│   │   ├── forest/
│   │   └── ...
│   └── seg_test/
│       ├── buildings/
│       ├── forest/
│       └── ...
└── Amaliy_N2.ipynb
```

2. Jupyter Notebook ni oching va barcha katakchalarni ketma-ket ishga tushiring.

## Texnologiyalar

- Python 3.x
- PyTorch
- torchvision
- matplotlib

## Muallif

Amaliy mashg'ulot №2 — Machine Learning kursi
