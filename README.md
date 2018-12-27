# :milky_way: ConvNet-Vis

ConvNet-Vis helps to visualize the Deep Convolutional Neural Networks with following methods.

- Activation of image from each layer
- Deconvolution
- Deep-Dream

## Requirements:
* Tensorflow
* Keras
* numpy
* scipy
* h5py
* wget
* six
* scikit-image

## Run:
Very first time it will download the weights of the model you pick, so it requires an internet connection. Also you can pass custom model.

```python
import vis

# For pretrained model visualization
convnet_vis = vis.ConvNet_Vis(image_path="cat.jpg")

# For custom model visualization
# model=<path-to-model> OR Keras Model obect
convnet_vis = vis.ConvNet_Vis(image_path="cat.jpg", model="final_model.hdf5")
```

> **NOTE:** Results will be stored in **vis** directory.

> **Google-Colab** support added. 

## For Tensorboard:

TensorBoard gives you flexibility to visualize all the test image on same model with brightness and contrast adustment.

```
tensorboard --logdir=<LOG-PATH (layerwise)>
```

## Todo:

- [ ] Deep-dream support for all the model. **(Current Support only for InceptionV3)**

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author

[<img src="https://avatars3.githubusercontent.com/u/24426731?s=460&v=4" width="70" height="70" alt="Ghanshyam_Chodavadiya">](https://github.com/CG1507)

## Acknowledgement

:green_heart: [tfcnn_vis](https://github.com/InFoCusp/tf_cnnvis)
