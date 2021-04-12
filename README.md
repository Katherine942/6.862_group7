# CNN Residual Learning Method for Artifacts and Noise Removal in MRI
6.862 Group7 Project | Team Members: Katherine Li & Ngoc La & Tommy Shi
## Description
Magnetic resonance imaging (MRI) with its non-invasive characteristics is a promising method for studying human organs, diagnosing diseases, and staging. However, MR images usually come with noises and artifact. In this project, we would like to propose using machine learning (ML) to improve the quality of MRI through a reduction of Rician noise. Specifically, we used the supervised Convolution Network model of 17 layers to learn the noise from 480 synthetic noisy images, which were combination of brain MR ground truth images and Rician noise model. The learning model was tested with 100 MR images of the same source. Quantitatively, we found the average peak signal to noise ratio (PSNR) of 28.40 dB, mean squared error (MSE) of 94.92, and root mean squared error (RMSE) of 9.72. Qualitatively, comparing the restored images with their original ground truth and their noise-added version, we found that the model worked relatively good and it could be improved by increasing epochs and training size. In future, we will implement the block matching and 3D filtering method (BM3D) to compare the results with our learning model.

The actual implementation codes for the project can be found in folder **src/denoise_cnn**

