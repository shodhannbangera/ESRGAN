import cv2
import numpy as np
import matplotlib.pyplot as plt

def psnr(img1, img2):
    # Resize img2 to match the shape of img1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    mse = np.mean((img1 - img2_resized) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite for identical images
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def display_images(img1, img2, title1='Image 1', title2='Image 2'):
    plt.figure(figsize=(12, 6))

    # Display the first image
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray' if len(img1.shape) == 2 else None)
    plt.title(title1)
    plt.axis('off')

    # Display the second image
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray' if len(img2.shape) == 2 else None)
    plt.title(title2)
    plt.axis('off')

    # Resize img2 to match the shape of img1 for the absolute difference
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Display the absolute difference between the images
    difference = cv2.absdiff(img1, img2_resized)
    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap='gray' if len(difference.shape) == 2 else None)
    plt.title('Absolute Difference')
    plt.axis('off')

    plt.show()

# Load PNG images
image1 = cv2.imread('results/baboon_rlt.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel
image2 = cv2.imread('results_st/baboon_rlt_structured.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Convert images to grayscale if needed
if len(image1.shape) == 3:
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
else:
    gray_image1 = image1

if len(image2.shape) == 3:
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
else:
    gray_image2 = image2

# Calculate PSNR
psnr_value = psnr(gray_image1, gray_image2)
print(f'PSNR Value: {psnr_value:.2f} dB')

# Display images and difference
display_images(image1, image2)
