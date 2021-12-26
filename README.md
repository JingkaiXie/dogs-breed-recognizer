# dogs-breed-recognizer
dataset: http://vision.stanford.edu/aditya86/ImageNetDogs/

REQUIREMENTS:
Pytorch, Numpy, PIL, Tensorboard, GPU(for training)


To train: 
1) put all images in "../dog breed data/images/all_images" folder. and 
2) run command line: python -m codes.train --log_dir log.
3) point tensorboard to log folder to check loss value.

To evaluate:
1) find a dog's picture, and put it in the main folder, and name it "dog.jpg"
2) run command line: python -m codes.evaluation
