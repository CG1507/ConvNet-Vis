# :milky_way: ConvNet-Vis

ConvNet visualization methods

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
Very first time it will download the weights of the model you pick, so it requires an internet connection.
```
python vis.py
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
