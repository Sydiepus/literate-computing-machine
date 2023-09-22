# Remove background

***Model accuracy is about 75%***
Image      | Background removed
:---------:|:---------:
![](test_imgs/photo_2023-09-21_19-52-52.jpg)  | ![](test_imgs/photo_2023-09-21_19-52-52_back_removed.png)

### How to use
You can clone the repo which is 201.95 MiB in size or download the model from google drive 90.9MB in size.

1. clone repo (201.95 MiB) and move into the cloned repo:
```
git clone https://github.com/Sydiepus/literate-computing-machine.git
cd literate-computing-machine
```
- You can proceed to install the dependencies.
2. If you don't want to clone the repo, download the model from [here](https://drive.google.com/file/d/1Ed_m1x7k0m0rWpP9zBugVB22ySGPDVoe/view?usp=sharing) and place it in a folder named `DUTS_set_model`, then download the script
  ```
  wget https://github.com/Sydiepus/literate-computing-machine/raw/main/rm_back.py
  ```
  - The script should be in the same place as the folder caintaining the model not inside it.
      - This is the path of the model that the script expect : `./DUTS_set_model/DUTS_set_unet_semantic_segmentation_512_512_+4_epochs.keras`
  - You can proceed to install the dependencies.

- Install dependencies : 

``` 
pip install tensorflow numpy Pillow
```

Run `rm_back.py` :

``` 
python rm_back.py
```

This produces an image with a black background,
It will infinitely ask for an image path, predict and save the output image to same path with `_back_removed` added to the ended of the image name.

To stop just use `ctrl + C`. 

# Limitations
Due to hardware, and connection limitations I am unable to train/develope the model on my machine, all the development was done on Google Collab as much as it allows before it yanks the GPU.

***The models are not well trained***
# Development 
The models where developed on Google Collab.

I'm using MobileNetV2 as base model for semantic image segmentation.

2 models are available that can detect objects somewhat accurately.
- cocoset_seg
- DUTS_set_model

Many versions of the models exists the best one is : `DUTS_set_model/DUTS_set_unet_semantic_segmentation_512_512_+4_epochs.keras`

A jupyter notebook for each model is available
In order to get started install requirements-model.txt, 

``` 
pip install -r requirements-model.txt
```

then you can run all the commands in the notebook it will download and extract the dataset and start training the model.

## DUTS_set_model
this model is trained using the DUTS dataset, and resize images to 512x512, the quality is pretty much left intact.
## cocoset_seg

This is the model resize images to 128x128 hence the quality will be degraded.

It is trained using coco2017 dataset.

No class filters where used, all available classes where used for training and a binary mask was applied.

## oxf_pet_ds_mdl

Contains the first model it is trained on the oxford pet dataset, this was used for initial testing and for me to 
learn about semantic segmentation.
