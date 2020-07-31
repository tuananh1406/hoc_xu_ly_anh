#Tai anaconda https://www.anaconda.com/distribution/#download-section
#Tao moi truong ao
conda create --name opencv-env python=3.6
#Kich hoat moi truong ao
activate opencv-env
#Cai dat dlib
pip install numpy scipy matplotlib scikit-learn jupyter
pip install opencv-contrib-python
pip install cmake
pip install dlib
pause