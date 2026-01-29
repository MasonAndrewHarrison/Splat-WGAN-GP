## Make sure you have python version <=3.12.11.
   `~/.pyenv/versions/3.12.11/bin/python -m venv .venv`

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
   (For CUDA)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121`\
   (For CPU)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

8. Install PyG:\
   (For CUDA)`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \-f https://data.pyg.org/whl/torch-2.5.0+cu121.html`\
   (For CPU)`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \-f https://data.pyg.org/whl/torch-2.5.0+cpu.html`

9. Install dependencies:\
   `pip install -r requirements.txt`

10. Download Dataset:\
  `python create_dataset.py`
   
12. Run:\
  `python main.py`
