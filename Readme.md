# Table Edge Detection

This project implements a table edge detection algorithm using the Sobel edge detection method, non-maximum suppression, and threshold hysteresis. The project also includes an extended line drawing feature using the Hough Transform to detect lines and draw them across the entire image.

## Table of Contents

- [Code Overview](#code-overview)
- [Results](#results)


## Code Overview

### `sobel_edge_detection(image)`

This function performs Sobel edge detection on the input image. It returns the gradient magnitude and gradient direction.

### `non_max_suppression(img, theta)`

This function performs non-maximum suppression on the gradient magnitude image to thin out the edges.

### `threshold_hysterisis(img, high, low)`

This function applies double thresholding and hysteresis to finalize the edge detection.

### Main Script

The main script performs the following steps:
1. Reads the input image.
2. Applies Sobel edge detection.
3. Performs non-maximum suppression.
4. Applies threshold hysteresis.
5. Detects lines using the Hough Transform.
6. Draws the detected lines extended across the entire image.

## Results

After running the script, the results will be saved as images in the project directory:
- `Edge_for_table.png`: Detected edges using Sobel operator.
- `Table_edge.png`: Detected and extended lines drawn on the original image.

### Example Output

![Table Edge](https://github.com/lalitkarthik/Save-Luna/blob/main/Table%20Edge%20Detection/Table_edge.png)





# Stereo Vision Depth Map Construction

This project demonstrates the construction of a depth map from stereo image pairs using Sobel edge detection, padding, and disparity calculation. The disparity map is then used to create a visual depth map.

## Table of Contents

- [Code Overview](#code-overview)

## Code Overview

### `sobel_processing(image)`

This function performs Sobel edge detection on the input grayscale image and returns the gradient magnitude.

### `add_padding(image, kernel_size)`

This function adds padding to the input image based on the specified kernel size.

### `disparity_construction(left_img, right_img, kernel_size, max_disparity=30)`

This function calculates the disparity map between the left and right images using a specified kernel size and maximum disparity.

### `depth_map_construction(disparity_map)`

This function constructs a depth map from the disparity map.

### Main Script

The main script performs the following steps:
1. Reads the stereo images.
2. Converts the images to grayscale and applies Gaussian blur.
3. Applies Sobel edge detection to both images.
4. Constructs the disparity map using the padded images.
5. Generates the depth map from the disparity map.
6. Saves and displays the depth map.

## Results

After running the script, the result will be saved as `Depth_Map.png`.




