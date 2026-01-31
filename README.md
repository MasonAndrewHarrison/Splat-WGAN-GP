## Make sure you have python version <=3.11.
   `~/.pyenv/versions/3.11.11/bin/python -m venv .venv`

## Installation
1. Clone this repository:\
   `git clone https://github.com/MasonAndrewHarrison/Splat-WGAN-GP.git`

2. Change Directory:\
   `cd Splat-WGAN-GP`
      
4. Create virtual environment:\
   `python -m venv venv`
   
5. Activate it:\
   (Linux)`source venv/bin/activate`\
   (Windows CMD)`venv\Scripts\activate.bat`\
   (Windows Power Shell)`venv\Scripts\Activate.ps1`
   
7. Install PyTorch:\
   (For CUDA)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`\
   (For CPU)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

8. Install PyTorch Extentions (works with: cpu, cu126, cu128, and cu129):\
   `pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git`
   `pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_sparse.git`
   `pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_cluster.git`
   `pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_spline_conv.git`

9. Install dependencies:\
   `pip install -r requirements.txt`

10. Download Dataset:\
  `python create_image_dataset.py`
  `python create_point_cloud_dataset.py`
   
12. Run:\
  `python main.py`
