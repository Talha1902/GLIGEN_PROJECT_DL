import matplotlib.pyplot as plt
import numpy as np
import os

def show_images_with_graph(normal_image_path, defected_image_path, actual_defected_image_path, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Load images
    normal_image = plt.imread(normal_image_path)
    defected_image = plt.imread(defected_image_path)
    actual_defected_image = plt.imread(actual_defected_image_path)

    # Display images
    axs[0].imshow(normal_image)
    axs[0].set_title('Normal Image')
    axs[0].axis('off')

    axs[1].imshow(defected_image)
    axs[1].set_title('Generated Defected Image')
    axs[1].axis('off')

    axs[2].imshow(actual_defected_image)
    axs[2].set_title('Actual Defected Image')
    axs[2].axis('off')

    # Generate random data for the graph
    num_points = 10
    x = np.arange(num_points)
    y_normal = np.random.rand(num_points)
    y_generated_defected = np.random.rand(num_points)
    y_actual_defected = np.random.rand(num_points)

    # Plot the graph
    axs[3].plot(x, y_normal, label='Normal', marker='o')
    axs[3].plot(x, y_generated_defected, label='Generated Defect', marker='o')
    axs[3].plot(x, y_actual_defected, label='Actual Defect', marker='o')
    axs[3].set_title('Defect Metrics')
    axs[3].set_xlabel('Sample Index')
    axs[3].set_ylabel('Defect Metric')
    axs[3].legend()
    axs[3].grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as '{save_path}'.")

# Base directory containing the dataset
base_dir = 'path_to_dataset'
# Directory where generated defected images are stored
output_dir = 'path_to_generated_defected_images'

# Iterate over train and test directories
for split in ['train', 'test']:
    image_dir = os.path.join(base_dir, split)

    for file_name in os.listdir(image_dir):
        # Process only original images with the _GT suffix
        if '_GT.jpg' in file_name:
            # Remove the _GT suffix to get the base name
            base_name = file_name.replace('_GT.jpg', '')

            # Get paths for the images
            normal_image_path = os.path.join(image_dir, file_name)
            defected_image_path = os.path.join(output_dir, f"{base_name}.jpg")
            actual_defected_image_path = os.path.join(image_dir, f"{base_name}.jpg")

            # Define save path for the plot
            save_path = os.path.join(image_dir, f"{base_name}_comparison.png")

            # Check if all paths exist
            if os.path.exists(normal_image_path) and os.path.exists(defected_image_path) and os.path.exists(actual_defected_image_path):
                show_images_with_graph(normal_image_path, defected_image_path, actual_defected_image_path, save_path)
            else:
                print(f"Missing images for {base_name}")
