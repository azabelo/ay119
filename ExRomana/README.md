# ExRomana
Unclean astro images for classification

The cnn_classifier.py shows how to read the training_samples.py file.
It has 64x64 cutouts of two types, TP (True Positives), and FP (False Positives).
The basic CNN shows how to read the data, and run the model after splitting the set
into train, test, validation set.

It trains quickly (in about 5 epochs) and clearly overfits.

If using a CNN, your task is: Improve it.

Possible things to look at:
(1) model architecture
(2) hyperparameters (training rate, batch size and all those things)
(3) alternate normalizing?
Plus many possibilities

- Start by doing some visualization of a few TP and few FP.
- You may want to do data augmentation
- Save models
- Plot confision matrix
- Visualize misclassifications
- Use other/more channels (only one is used by default)

And correct minor issues in the python code.

If you are doing clustering, use the 64x64 images, or inner 32x32 images.
Use t-SNE, UMAP, PCA, KMEANS, whatever. Use the labels to color and check, but not to tweak.
If you want to use it to tweak, use a small subset of labels. (Why?)

The dataset cane be found here: https://sites.astro.caltech.edu/~aam/exromana_training_samples.npy
