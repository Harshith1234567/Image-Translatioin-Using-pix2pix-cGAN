# Image-Translatioin-Using-pix2pix-cGAN

Based on paper https://tcwang0509.github.io/pix2pixHD/

Image-to-Image Translation is a task in computer vision and machine learning where the goal is to learn a mapping between an input image and an output image, such that the output image can be used to perform a specific task, such as style transfer, data augmentation, or image restoration.

It is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image. It can be applied to a wide range of applications, such as collection style transfer, object transfiguration, season transfer and photo enhancement.

In the pix2pix cGAN, we condition on input images and generate corresponding output images. cGANs were first proposed in Conditional Generative Adversarial Nets (Mirza and Osindero, 2014)
![image](https://tcwang0509.github.io/pix2pixHD/images/teaser_720.gif)

The architecture of your network will contain:

A generator with a U-Net-based architecture.
A discriminator represented by a convolutional PatchGAN classifier

As described in the pix2pix paper, we need to apply random jittering and mirroring to preprocess the training set.

Define several functions that:

Resize each 256 x 256 image to a larger height and width—286 x 286.
Randomly crop it back to 256 x 256.
Randomly flip the image horizontally i.e. left to right (random mirroring).
Normalize the images to the [-1, 1] range.

![image](https://github.com/Harshith1234567/Image-Translatioin-Using-pix2pix-cGAN/assets/53342028/a05971e3-3d66-4e89-b4d7-fb8042c372a0)

The architecture of our network will contain:

- A generator with a U-Net-based architecture.
- A discriminator represented by a convolutional PatchGAN classifier

<b> Generator: </b>

The generator of your pix2pix cGAN is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). (You can find out more about it in the Image segmentation tutorial and on the U-Net project website.)

- Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU
- Each block in the decoder is: Transposed convolution -> Batch normalization -> Dropout (applied to the first 3 blocks) -> ReLU
- There are skip connections between the encoder and decoder (as in the U-Net).

<b> Generator Loss: </b>

GANs learn a loss that adapts to the data, while cGANs learn a structured loss that penalizes a possible structure that differs from the network output and the target image, as described in the pix2pix paper.

- The generator loss is a sigmoid cross-entropy loss of the generated images and an array of ones.
- The pix2pix paper also mentions the L1 loss, which is a MAE (mean absolute error) between the generated image and the target image.
- This allows the generated image to become structurally similar to the target image.
- The formula to calculate the total generator loss is gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. - This value was decided by the authors of the paper.

<b> Discriminator: </b>

The discriminator in the pix2pix cGAN is a convolutional PatchGAN classifier—it tries to classify if each image patch is real or not real, as described in the pix2pix paper.

- Each block in the discriminator is: Convolution -> Batch normalization -> Leaky ReLU.
- The shape of the output after the last layer is (batch_size, 30, 30, 1).
- Each 30 x 30 image patch of the output classifies a 70 x 70 portion of the input image.
- The discriminator receives 2 inputs:
    - The input image and the target image, which it should classify as real.
    - The input image and the generated image (the output of the generator), which it should classify as fake.
    - Use tf.concat([inp, tar], axis=-1) to concatenate these 2 inputs together.
    

<b> Discriminator Loss: </b>

- The discriminator_loss function takes 2 inputs: real images and generated images.
- real_loss is a sigmoid cross-entropy loss of the real images and an array of ones(since these are the real images).
- generated_loss is a sigmoid cross-entropy loss of the generated images and an array of zeros (since these are the fake images).
- The total_loss is the sum of real_loss and generated_loss.

Interpreting the logs is more subtle when training a GAN (or a cGAN like pix2pix) compared to a simple classification or regression model. Things to look for:

- Check that neither the generator nor the discriminator model has "won". If either the gen_gan_loss or the disc_loss gets very low, it's an indicator that this model is dominating the other, and you are not successfully training the combined model.
- The value log(2) = 0.69 is a good reference point for these losses, as it indicates a perplexity of 2 - the discriminator is, on average, equally uncertain about the two options.
- For the disc_loss, a value below 0.69 means the discriminator is doing better than random on the combined set of real and generated images.
- For the gen_gan_loss, a value below 0.69 means the generator is doing better than random at fooling the discriminator.
- As training progresses, the gen_l1_loss should go down.

![image](https://github.com/Harshith1234567/Image-Translatioin-Using-pix2pix-cGAN/assets/53342028/b7e67980-e659-423e-9546-426a2353caee)
