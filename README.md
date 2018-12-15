# :milky_way: ConvNet-Vis

ConvNet-Vis helps to visualize the Deep Convolutional Neural Networks with following methds.

- Activation image from each layer
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
convnet_vis = vis.ConvNet_Vis(image_path="cat.jpg", model="final_model.hdf5")
```

## For Tensorboard:

TensorBoard gives you flexibility to visualize all the test image on same model with brightness and contrast adustment.

```
tensorboard --logdir=<LOG-PATH (layerwise)>
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author

[Ghanshyam Chodavadiya](https://cg1507.github.io)

## Acknowledgement

:green_heart: [tfcnn_vis](https://github.com/InFoCusp/tf_cnnvis)
