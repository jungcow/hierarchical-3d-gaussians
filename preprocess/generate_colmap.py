#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, sys, shutil
import subprocess
import argparse
from read_write_model import read_images_binary,write_images_binary, Image
import time, platform
from tqdm import tqdm

def replace_images_by_masks(images_file, out_file):
    """Replace images.jpg to images.png in the colmap images.bin to process masks the same way as images."""
    images_metas = read_images_binary(images_file)
    out_images_metas = {}
    for key in images_metas:
        in_image_meta = images_metas[key]
        out_images_metas[key] = Image(
            id=key,
            qvec=in_image_meta.qvec,
            tvec=in_image_meta.tvec,
            camera_id=in_image_meta.camera_id,
            name=in_image_meta.name[:-3]+"png",
            xys=in_image_meta.xys,
            point3D_ids=in_image_meta.point3D_ids,
        )
    
    write_images_binary(out_images_metas, out_file)

def setup_dirs(project_dir):
    """Create the directories that will be required."""
    if not os.path.exists(project_dir):
        print("creating project dir.")
        os.makedirs(project_dir)
    
    if not os.path.exists(os.path.join(project_dir, "camera_calibration/aligned")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/aligned/sparse/0"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/rectified")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/rectified"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/unrectified")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified"))
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified", "sparse"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/unrectified", "sparse")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified", "sparse"))


def find_images_names(root_dir):
    image_files_by_subdir = []

    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):

        # Filter for image files (you can add more extensions if needed), sort images
        image_files = sorted([f for f in filenames if f.lower().endswith(('.png', '.jpg', '.JPG', '.PNG'))])

        # If there are image files in the current directory, add them to the list
        if image_files:
            image_files_by_subdir.append({
                'dir': os.path.basename(dirpath) if dirpath != root_dir else "",
                'images': image_files
            })

    return image_files_by_subdir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True)
    parser.add_argument('--images_dir', default="", help="Will be set to project_dir/inputs/images if not set")
    parser.add_argument('--masks_dir', default="", help="Will be set to project_dir/inputs/masks if exists and not set")
    args = parser.parse_args()
    
    if args.images_dir == "":
        args.images_dir = os.path.join(args.project_dir, "inputs/images")
    if args.masks_dir == "":
        args.masks_dir = os.path.join(args.project_dir, "inputs/masks")
        args.masks_dir = args.masks_dir if os.path.exists(args.masks_dir) else ""

    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"
    start_time = time.time()

    print(f"Project will be built here ${args.project_dir} base images are available there ${args.images_dir}.")

    setup_dirs(args.project_dir)

    ## Feature extraction, matching then mapper to generate the colmap.
    print("extracting features ...")
    colmap_feature_extractor_args = [
        colmap_exe, "feature_extractor",
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--image_path", f"{args.images_dir}",
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.default_focal_length_factor", "0.5",
        "--ImageReader.camera_model", "OPENCV_FISHEYE",
        ]
    try:
        subprocess.run(colmap_feature_extractor_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap feature_extractor: {e}")
        sys.exit(1)

    #! save database.db
    shutil.copy(f"{args.project_dir}/camera_calibration/unrectified/database.db", 
                f"{args.project_dir}/camera_calibration/database.db")

    # Build Vocab Tree. Check the manual in colmap vocab_tree_builder -h
    print("making vocab tree ...")
    make_vocab_tree_args = [
        "python", f"{args.project_dir}/../../scripts/make_vocab_tree.py",
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--image_path", f"{args.images_dir}",
        "--vocab_tree_path", f"{args.project_dir}/../configs/",
        ]
    try:
        subprocess.run(make_vocab_tree_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap feature_extractor: {e}")
        sys.exit(1)
    
    ## Sequential matching and loop detection using vocab tree
    colmap_sequential_matcher_args = [
        colmap_exe, "sequential_matcher",
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--SequentialMatching.overlap", "16",
        "--SequentialMatching.quadratic_overlap", "14",
        "--SequentialMatching.loop_detection", "1",
        "--SequentialMatching.loop_detection_period", "16",
        "--SequentialMatching.loop_detection_num_images", "64", # default (=50)
        "--SequentialMatching.vocab_tree_path", f"{args.project_dir}/../configs/vocab_tree_1602_by_builder.bin",
    ]
    try:
        subprocess.run(colmap_sequential_matcher_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap sequential_matcher: {e}")
        sys.exit(1)

    # print("making custom matches...")
    # make_colmap_custom_matcher_args = [
    #     "python", f"preprocess/make_colmap_custom_matcher.py",
    #     "--image_path", f"{args.images_dir}",
    #     "--output_path", f"{args.project_dir}/camera_calibration/unrectified/matching.txt",
    #     "--n_seq_matches_per_view", "8",
    #     "--n_quad_matches_per_view", "12",
    # ]

    # print('make_colmap custom cmd args: ', make_colmap_custom_matcher_args)
    # try:
    #     subprocess.run(make_colmap_custom_matcher_args, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error executing make_colmap_custom_matcher: {e}")
    #     sys.exit(1)

    # ## Feature matching
    # print("matching features...")
    # colmap_matches_importer_args = [
    #     colmap_exe, "matches_importer",
    #     "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
    #     "--match_list_path", f"{args.project_dir}/camera_calibration/unrectified/matching.txt"
    #     ]
    # try:
    #     subprocess.run(colmap_matches_importer_args, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error executing colmap matches_importer: {e}")
    #     sys.exit(1)

    # ## set known intrinsic parameters to database.db
    # print("set camera parameters to database.db")
    # set_intrinsic_to_database_args = [
    #     "python", f"{args.project_dir}/scripts/set_intrinsic_to_database.py",
    #     "--intrinsic_path", f"{args.project_dir}/intrinsic",
    #     "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db"
    #     ]
    # try:
    #     subprocess.run(set_intrinsic_to_database_args, check=True)
    # except subprocess.CalledProcessError as e:
    #     print("Error executing python set_intrinsic_to_database.db: {e}")
    #     sys.exit(1)


    ## Generate sfm pointcloud
    print("generating sfm point cloud...")
    colmap_hierarchical_mapper_args = [
        colmap_exe, "hierarchical_mapper",
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--image_path", f"{args.images_dir}",
        "--output_path", f"{args.project_dir}/camera_calibration/unrectified/sparse",
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_local_num_images", "15",
        "--image_overlap", "50",
        "--ba_local_max_num_iterations", "25", # default: 25
        ]
    try:
        subprocess.run(colmap_hierarchical_mapper_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap hierarchical_mapper: {e}")
        sys.exit(1)

    # # preserve the original sparse reconstruction for debugging
    # if not os.path.exists(f"{args.project_dir}/camera_calibration/unrectified/sparse/original"):
    #     os.makedirs(f"{args.project_dir}/camera_calibration/unrectified/sparse/original")
    # shutil.copy(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/cameras.bin", 
    #             f"{args.project_dir}/camera_calibration/unrectified/sparse/original/cameras.bin")
    # shutil.copy(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/points3D.bin", 
    #             f"{args.project_dir}/camera_calibration/unrectified/sparse/original/points3D.bin")
    # shutil.copy(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/images.bin", 
    #             f"{args.project_dir}/camera_calibration/unrectified/sparse/original/images.bin")

    ## Two times more BA
    # print("running triangulator and BA two times more...")
    # for _ in range(2):
    #     colmap_point_triangulator_args = [
    #         colmap_exe, "point_triangulator",
    #         "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
    #         "--image_path", f"{args.images_dir}",
    #         "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0",
    #         "--output_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0",
    #         "--Mapper.ba_global_function_tolerance", "0.000001",
    #         "--Mapper.ba_global_max_num_iterations", "30",
    #         "--Mapper.ba_global_max_refinements", "3",
    #         "--Mapper.ba_refine_focal_length", "0",
    #         "--Mapper.ba_refine_extra_params", "0",
    #         "--Mapper.ba_local_num_images", "30",
    #         ]
    #     try:
    #         subprocess.run(colmap_point_triangulator_args, check=True)
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error executing colmap point_triangulator: {e}")
    #         sys.exit(1)

    #     colmap_bundle_adjuster_args = [
    #         colmap_exe, "bundle_adjuster",
    #         "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0",
    #         "--output_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0",
    #         "--BundleAdjustment.refine_extra_params", "0",
    #         "--BundleAdjustment.refine_focal_length", "0",
    #         "--BundleAdjustment.function_tolerance", "0.000001",
    #         "--BundleAdjustment.max_linear_solver_iterations", "100",
    #         "--BundleAdjustment.max_num_iterations", "50",
    #         ]
    #     try:
    #         subprocess.run(colmap_bundle_adjuster_args, check=True)
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error executing colmap bundle_adjuster: {e}")
    #         sys.exit(1)

    ## Simplify images so that everything takes less time (reading colmap usually takes forever)
    simplify_images_args = [
        "python", f"preprocess/simplify_images.py",
        "--base_dir", f"{args.project_dir}/camera_calibration/unrectified/sparse/0"
    ]
    try:
        subprocess.run(simplify_images_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing simplify_images: {e}")
        sys.exit(1)

    ## Post processing: rig_bundle_adjustment
    print("postprocessing for multi-camera rig constraint BA...")
    colmap_rig_bundle_adjuster_args = [
        colmap_exe, "rig_bundle_adjuster",
        "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0",
        "--output_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0",
        "--rig_config_path", f"{args.project_dir}/../configs/rig_config.json",
        "--BundleAdjustment.refine_focal_length", "0",
        "--BundleAdjustment.refine_principal_point", "0",
        "--BundleAdjustment.refine_extra_params", "0",
        "--estimate_rig_relative_poses", "0",
        "--RigBundleAdjustment.refine_relative_poses", "1",
        ]
    try:
        for _ in range(3):
            subprocess.run(colmap_rig_bundle_adjuster_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap rig_bundle_adjuster: {e}")
        sys.exit(1)

    ## Undistort images
    #  colmap image_undistorter --input_path ./rectified_rig_ba/1/ --output_path rectified --blank_pixel=0 --min_scale=0 --max_image_size=2048 --image_path /root/dataset/haedong/project_1hz/303_outdoor/
    print(f"undistorting images from {args.images_dir} to {args.project_dir}/camera_calibration/rectified images...")
    colmap_image_undistorter_args = [
        colmap_exe, "image_undistorter",
        "--image_path", f"{args.images_dir}",
        "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0", 
        "--output_path", f"{args.project_dir}/camera_calibration/rectified/",
        "--output_type", "COLMAP",
        "--max_image_size", "2048",
        "--blank_pixels", "1",
        "--max_scale", "1.0",
        ]
    try:
        subprocess.run(colmap_image_undistorter_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing image_undistorter: {e}")
        sys.exit(1)

    if not args.masks_dir == "":
        # create a copy of colmap as txt and replace jpgs with pngs to undistort masks the same way images were distorted
        if not os.path.exists(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks"):
            os.makedirs(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks")

        shutil.copy(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/cameras.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/cameras.bin")
        shutil.copy(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/points3D.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/points3D.bin")
        replace_images_by_masks(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/images.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/images.bin")

        print("undistorting masks aswell...")
        colmap_image_undistorter_args = [
            colmap_exe, "image_undistorter",
            "--image_path", f"{args.masks_dir}",
            "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks", 
            "--output_path", f"{args.project_dir}/camera_calibration/tmp/",
            "--max_image_size", "2048",
            "--blank_pixels", "1",
            "--max_scale", "1.0",
            ]
        try:
            subprocess.run(colmap_image_undistorter_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing image_undistorter: {e}")
            sys.exit(1)
        
        make_mask_uint8_args = [
            "python", f"preprocess/make_mask_uint8.py",
            "--in_dir", f"{args.project_dir}/camera_calibration/tmp/images",
            "--out_dir", f"{args.project_dir}/camera_calibration/rectified/masks"
        ]
        try:
            subprocess.run(make_mask_uint8_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing make_colmap_custom_matcher: {e}")
            sys.exit(1)

        # remove temporary dir containing undistorted masks
        shutil.rmtree(f"{args.project_dir}/camera_calibration/tmp")

    # re-orient + scale colmap
    print(f"re-orient and scaling scene to {args.project_dir}/camera_calibration/aligned/sparse/0")
    reorient_args = [
            "python", f"preprocess/auto_reorient.py",
            "--input_path", f"{args.project_dir}/camera_calibration/rectified/sparse",
            "--output_path", f"{args.project_dir}/camera_calibration/aligned/sparse/0"
        ]
    try:
        subprocess.run(reorient_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing auto_orient: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"Preprocessing done in {(end_time - start_time)/60.0} minutes.")
