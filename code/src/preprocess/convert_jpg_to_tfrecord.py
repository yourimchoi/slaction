"""
Video Frame to TFRecord Conversion Pipeline

This module converts video frame sequences (JPG images) to TensorFlow TFRecord format
for efficient data loading during model training. The pipeline processes sleep study
video data with customizable cropping, resizing, and frame standardization.

Key Features:
- Processes video frames from organized directory structures
- Applies configurable cropping and resizing transformations
- Converts RGB images to grayscale for sleep action recognition
- Handles variable frame counts by padding to 150 frames
- Generates TFRecord files with embedded video tensors
- Supports parallel processing for improved performance
- Provides verification utilities for generated TFRecord files

Directory Structure Expected:
    data/demo_frames/
    ├── train/
    │   └── A2019-EM-01-0001/
    │       ├── 0001/
    │       ├── 0002/
    │       └── ...
    ├── valid/
    └── test/

Output Structure:
    results/tfrecord/
    ├── train/
    │   └── A_center/
    │       └── A2019-EM-01-0001/
    │           ├── 0001.tfrecord
    │           ├── 0002.tfrecord
    │           └── ...
    ├── valid/
    └── test/
"""

import argparse
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt


# Set TensorFlow logging level to suppress CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage for preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow CUDA messages

def read_and_process_image(filename, crop_top=50, crop_left=140, crop_height=400, crop_width=400):
    """
    Read and preprocess a single image file for video analysis.
    
    Applies consistent preprocessing pipeline including cropping, grayscale conversion,
    and resizing to prepare images for neural network input.
    
    Args:
        filename (str): Path to the input JPEG image file
        crop_top (int, optional): Top pixel offset for cropping. Defaults to 50.
        crop_left (int, optional): Left pixel offset for cropping. Defaults to 140.
        crop_height (int, optional): Height of cropped region. Defaults to 400.
        crop_width (int, optional): Width of cropped region. Defaults to 400.
    
    Returns:
        numpy.ndarray: Processed grayscale image as uint8 array with shape [256, 256, 1]
    """
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img_gray = tf.image.rgb_to_grayscale(img)
    img_gray = tf.image.crop_to_bounding_box(img_gray, crop_top, crop_left, crop_height, crop_width)
    img_gray = tf.image.resize(img_gray, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
    img_gray = tf.cast(img_gray, tf.uint8)
    return img_gray.numpy()

def create_tfrecord_for_video(image_folder, output_folder, output_filename, crop_top=50, crop_left=140, crop_height=400, crop_width=400):
    """
    Convert a sequence of video frames to TFRecord format.
    
    Processes all JPEG images in a folder, applies preprocessing transformations,
    and serializes the video sequence as a TFRecord file. Ensures consistent
    frame count by padding shorter sequences to 150 frames.
    
    Args:
        image_folder (str): Path to folder containing JPEG frame images
        output_folder (str): Directory path for output TFRecord file
        output_filename (str): Name of the output TFRecord file
        crop_top (int, optional): Top pixel offset for cropping. Defaults to 50.
        crop_left (int, optional): Left pixel offset for cropping. Defaults to 140.
        crop_height (int, optional): Height of cropped region. Defaults to 400.
        crop_width (int, optional): Width of cropped region. Defaults to 400.
    
    Note:
        - Only processes sequences with 146+ frames (pads to 150)
        - Output tensor shape: [150, 1, 256, 256] as uint8
        - TFRecord contains 'video' (serialized tensor) and 'length' features
    """
    os.makedirs(output_folder, exist_ok=True)
    tfrecord_path = os.path.join(output_folder, output_filename)
    
    image_files = [os.path.join(image_folder, filename) for filename in sorted(os.listdir(image_folder)) if filename.endswith('.jpg')]
    
    frames = [read_and_process_image(file, crop_top, crop_left, crop_height, crop_width) for file in image_files]
    
    # Ensure sequence has exactly 150 frames by padding with last frame if needed
    if len(frames) >= 146:
        while len(frames) < 150:
            frames.append(frames[-1])
    
    video_array = np.stack(frames, axis=0)
    
    # Reshape to [num_frames, 1, height, width] format expected by model
    video_array = np.expand_dims(video_array, axis=1)
    
    video_tensor = tf.convert_to_tensor(video_array, dtype=tf.uint8)
    video_bytes = tf.io.serialize_tensor(video_tensor).numpy()
    
    feature = {
        'video': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_bytes])),
        'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(frames)])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        writer.write(example.SerializeToString())

def process_directory_structure(split_dir, num_folders=None):
    """
    Analyze directory structure and identify valid video folders for processing.
    
    Scans the dataset split directory to find video folders with sufficient frames
    and generates statistics about frame count distribution across the dataset.
    
    Args:
        split_dir (str): Path to dataset split directory (train/valid/test)
        num_folders (int, optional): Unused parameter for potential future filtering.
                                   Defaults to None.
    
    Returns:
        tuple: A tuple containing:
            - folders_to_process (list): List of (subject_path, video_folder_name) tuples
              for folders with 146+ frames that will be processed
            - frame_counts (Counter): Statistics of frame counts across all folders
    
    Note:
        Only folders with 146+ frames are included in processing list to ensure
        adequate sequence length for padding to 150 frames.
    """
    print(f"Processing directory structure for {split_dir}...")
    folders_to_process = []
    frame_counts = Counter()
    
    # Process each subject directory in the split (train/valid/test)
    for subject_name in sorted(os.listdir(split_dir)):
        subject_path = os.path.join(split_dir, subject_name)
        if os.path.isdir(subject_path):
            for video_folder_name in sorted(os.listdir(subject_path)):
                video_folder_path = os.path.join(subject_path, video_folder_name)
                if os.path.isdir(video_folder_path):
                    num_frames = len([filename for filename in os.listdir(video_folder_path) if filename.endswith('.jpg')])
                    if num_frames > 0:
                        frame_counts[num_frames] += 1
                        # Filter for sequences with adequate length (146+ frames can be padded to 150)
                        if num_frames == 150 or num_frames >= 146:
                            folders_to_process.append((subject_path, video_folder_name))
    return folders_to_process, frame_counts

def process_folder(root, dir_name, base_input_dir, base_output_dir, crop_top=50, crop_left=140, crop_height=400, crop_width=400):
    """
    Process a single video folder and convert it to TFRecord format.
    
    Handles the conversion of one video folder containing frame images to a TFRecord file.
    Automatically organizes output by medical center based on subject naming convention.
    
    Args:
        root (str): Path to the subject directory containing the video folder
        dir_name (str): Name of the video folder (e.g., '0001', '0002')
        base_input_dir (str): Base input directory path for computing relative paths
        base_output_dir (str): Base output directory for TFRecord files
        crop_top (int, optional): Top pixel offset for cropping. Defaults to 50.
        crop_left (int, optional): Left pixel offset for cropping. Defaults to 140.
        crop_height (int, optional): Height of cropped region. Defaults to 400.
        crop_width (int, optional): Width of cropped region. Defaults to 400.
    
    Note:
        - Extracts medical center from subject name (first character: A, B, D, etc.)
        - Creates center-specific subdirectories (e.g., 'A_center', 'B_center')
        - Output filename matches video folder name with .tfrecord extension
    """
    image_folder = os.path.join(root, dir_name)
    relative_path = os.path.relpath(image_folder, base_input_dir)
    
    # Extract medical center identifier from subject name for organized output structure
    subject_name = os.path.basename(os.path.dirname(relative_path))
    center_name = subject_name[0]  # Extract first character (A, B, D, etc.)
    center_prefix = f"{center_name}_center"
    
    # Construct output path with center-specific organization
    path_parts = relative_path.split(os.sep)
    if len(path_parts) >= 2:
        # path_parts[0] is subject_name, path_parts[1] is video_folder
        output_folder = os.path.join(base_output_dir, center_prefix, path_parts[0])
    else:
        output_folder = os.path.join(base_output_dir, os.path.dirname(relative_path))
    
    output_filename = f"{dir_name}.tfrecord"
    create_tfrecord_for_video(image_folder, output_folder, output_filename, crop_top, crop_left, crop_height, crop_width)

import random

def test(split_dir, crop_top, crop_left, crop_height, crop_width, num_frames_to_show=2):
    """
    Test the preprocessing pipeline by visualizing sample processed frames.
    
    Randomly selects a video folder from the dataset and displays processed frames
    to verify that the image preprocessing pipeline is working correctly.
    
    Args:
        split_dir (str): Path to dataset split directory to test
        crop_top (int): Top pixel offset for cropping
        crop_left (int): Left pixel offset for cropping  
        crop_height (int): Height of cropped region
        crop_width (int): Width of cropped region
        num_frames_to_show (int, optional): Number of frames to display. Defaults to 2.
    
    Note:
        - Displays grayscale processed frames using matplotlib
        - Useful for validating preprocessing parameters before full conversion
        - Selects frames from the beginning of a random video sequence
    """
    # Process directory structure to get folders to process
    folders_to_process, _ = process_directory_structure(split_dir)
    
    if folders_to_process:
        # Select a random folder
        root, dir_name = random.choice(folders_to_process)
        image_folder = os.path.join(root, dir_name)
        image_files = [os.path.join(image_folder, filename) for filename in sorted(os.listdir(image_folder)) if filename.endswith('.jpg')]
        
        # Ensure num_frames_to_show does not exceed available frames
        num_frames_to_show = min(num_frames_to_show, len(image_files))
        
        # Select the specified number of frames from the start
        selected_files = image_files[:num_frames_to_show]
        
        frames = [read_and_process_image(file, crop_top, crop_left, crop_height, crop_width) for file in selected_files]
        
        for i, frame in enumerate(frames):
            plt.figure()
            plt.imshow(frame.squeeze(), cmap='gray')
            plt.title(f'Frame {i+1} from random video')
            plt.show()

def verify_tfrecord(tfrecord_path):
    """
    Verify and visualize the contents of a generated TFRecord file.
    
    Loads a TFRecord file, parses its contents, and displays sample frames
    to verify that the conversion process completed successfully.
    
    Args:
        tfrecord_path (str): Path to the TFRecord file to verify
    
    Note:
        - Prints file size and video sequence length information
        - Displays first 2 frames as grayscale images using matplotlib
        - Useful for quality control after TFRecord generation
    """
    # Check file size
    file_size = os.path.getsize(tfrecord_path)
    print(f"TFRecord file size: {file_size} bytes")
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    feature_description = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'length': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)

    for parsed_record in parsed_dataset:
        video_bytes = parsed_record['video'].numpy()
        length = parsed_record['length'].numpy()
        video_tensor = tf.io.parse_tensor(video_bytes, out_type=tf.uint8)
        video_tensor = tf.reshape(video_tensor, (length, 1, 256, 256))
        
        print(f"Video length: {length}")
        for i in range(min(length, 2)):  # Display first two frames for verification
            frame = video_tensor[i, 0, :, :].numpy()
            plt.figure()
            plt.imshow(frame, cmap='gray')
            plt.title(f'Frame {i+1}')
            plt.show()
        break  # Process only the first record for verification

def process_split(split_dir, output_split_dir, crop_top, crop_left, crop_height, crop_width, workers):
    """
    Process an entire dataset split using parallel workers.
    
    Coordinates the conversion of all valid video folders in a dataset split
    (train/valid/test) to TFRecord format using multiprocessing for efficiency.
    
    Args:
        split_dir (str): Path to the dataset split directory
        output_split_dir (str): Path to the output directory for TFRecord files
        crop_top (int): Top pixel offset for cropping
        crop_left (int): Left pixel offset for cropping
        crop_height (int): Height of cropped region
        crop_width (int): Width of cropped region
        workers (int): Number of parallel worker processes to use
    
    Note:
        - Prints frame count statistics before processing
        - Uses ProcessPoolExecutor for parallel processing
        - Shows progress bar during conversion
    """
    folders_to_process, frame_counts = process_directory_structure(split_dir)
    
    print(f"Frame count statistics for {os.path.basename(split_dir)}:")
    for frame_count, num_folders in sorted(frame_counts.items()):
        print(f"Number of frames: {frame_count}, Number of folders: {num_folders}")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_folder, root, dir_name, split_dir, output_split_dir, crop_top, crop_left, crop_height, crop_width)
            for root, dir_name in folders_to_process
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(split_dir)} folders"):
            future.result()

def parse_args():
    """
    Parse command line arguments for TFRecord conversion configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - demo_frames_dir: Input directory path for video frames
            - output_dir: Output directory path for TFRecord files  
            - workers: Number of parallel processing workers
            - crop_top, crop_left, crop_height, crop_width: Image cropping parameters
    """
    parser = argparse.ArgumentParser(description="Process videos and convert them to TFRecord format.")
    parser.add_argument('--demo_frames_dir', type=str, default='data/demo_frames', help="Path to the demo_frames directory.")
    parser.add_argument('--output_dir', type=str, default='results/tfrecord', help="Path to the output directory for TFRecord files.")
    parser.add_argument('--workers', type=int, default=32, help="Number of parallel workers for processing.")
    parser.add_argument('--crop_top', type=int, default=0, help="Top crop value.")
    parser.add_argument('--crop_left', type=int, default=100, help="Left crop value.")
    parser.add_argument('--crop_height', type=int, default=480, help="Crop height.")
    parser.add_argument('--crop_width', type=int, default=480, help="Crop width.")
    return parser.parse_args()

def main(args):
    """
    Main execution function for video frame to TFRecord conversion pipeline.
    
    Orchestrates the complete conversion process by setting up workspace paths,
    processing all dataset splits (train/valid/test), and converting video frames
    to TFRecord format with the specified preprocessing parameters.
    
    Args:
        args (argparse.Namespace): Command line arguments containing configuration
                                 for input/output paths and processing parameters
    
    Note:
        - Automatically resolves workspace root directory from script location
        - Processes train, valid, and test splits sequentially
        - Skips missing split directories with warning messages
    """
    # Get the base directory (workspace root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.join(script_dir, '..', '..', '..')
    workspace_root = os.path.abspath(workspace_root)
    
    demo_frames_dir = os.path.join(workspace_root, args.demo_frames_dir)
    output_dir = os.path.join(workspace_root, args.output_dir)
    
    print(f"Demo frames directory: {demo_frames_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process each split (train, valid, test)
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = os.path.join(demo_frames_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        
        if os.path.exists(split_dir):
            print(f"\nProcessing {split} split...")
            process_split(split_dir, output_split_dir, args.crop_top, args.crop_left, 
                         args.crop_height, args.crop_width, args.workers)
        else:
            print(f"Warning: {split_dir} does not exist, skipping...")

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
    # Uncomment the following line to run the test function for preprocessing validation
    # test(args.demo_frames_dir, args.crop_top, args.crop_left, args.crop_height, args.crop_width, num_frames_to_show=1)
    
    # Uncomment the following lines to verify a specific TFRecord file
    # tfrecord_path = '/path/to/specific/tfrecord/file.tfrecord'
    # verify_tfrecord(tfrecord_path)