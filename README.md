# Disentangled representation learning in cardiac image analysis

Implementation of the **SDNet** model to perform disentanglement of anatomical and modality information in medical images. For further details please see our [paper], accepted in [Medical Image Analysis].

The structure of this project is the following:

* **configuration**: package containing configuration parameters for running an experiment.
* **layers**: package with custom Keras layers
* **loaders**: package with data loaders
* **models**: package with the SDNet model and other Keras models
* **model_executors**: package with scripts for running an experiment
* **callbacks**: package with Keras callbacks for printing images and losses during training

To define a new data loader, extend class `base_loader.Loader`, and register the loader in `loader_factory.py`. The datapath is specified in `parameters.py`.

To run an experiment, execute `experiment.py`, passing the configuration filename and the split number as runtime parameters:
```
python experiment.py --config myconfiguration --split 0
```

The code is written in [Keras] version 2.1.6 with [tensorflow] 1.4.0 and experiments were run with a Titan-X GPU.

A tensorflow implementation is uploaded in https://github.com/GabrieleValvano/SDNet.

## Citation

If you use this code for your research, please cite our paper:

```
@article{CHARTSIAS2019101535,
title = "Disentangled representation learning in cardiac image analysis",
journal = "Medical Image Analysis",
volume = "58",
pages = "101535",
year = "2019",
issn = "1361-8415",
doi = "https://doi.org/10.1016/j.media.2019.101535",
url = "http://www.sciencedirect.com/science/article/pii/S1361841519300684",
author = "Agisilaos Chartsias and Thomas Joyce and Giorgos Papanastasiou and Scott Semple and Michelle Williams and David E. Newby and Rohan Dharmakumar and Sotirios A. Tsaftaris",
keywords = "Disentangled representation learning, Cardiac magnetic resonance imaging, Semi-supervised segmentation, Multitask learning"
}
```
 
[paper]: https://www.sciencedirect.com/science/article/abs/pii/S1361841519300684
[Keras]: https://keras.io/
[tensorflow]: https://www.tensorflow.org/
[Medical Image Analysis]: https://www.sciencedirect.com/journal/medical-image-analysis
