
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

#******************************************************************************
#* For facilitating the hyperparameter tuning test
#* extractor    - SIFT / DSP-SIFT(domain_size_pooling)
#* matcher      - vocab_tree_matcher | sequential_matcher | custom_matcher
#*                                     (loop_detection O/X)
#* mapper       - mapper | hierarchical_mapper (N, O)
#*                         (local ba: num_images N)
#*                         (overlap: O)
#* 


import os, sys, shutil
import subprocess
import argparse

PROJECT_PATH = "/root/dataset/haedong/project_1hz/outdoor/incremental/camera_calibration"

DATABASE_PATH = f"{PROJECT_PATH}/unrectified/database.db"
# DATABASE_PATH = f"{PROJECT_PATH}/unrectified/dsp-database.db"
IMAGE_PATH = "/root/dataset/haedong/project_1hz/303_outdoor"
# IMAGE_PATH = "/root/dataset/haedong/project_1hz/303_front_only"

CONFIG_PATH = "/root/dataset/haedong/project_1hz/configs"
# VOCAB_TREE_indoor_by_buider = "vocab_tree_indoor_by_builder.bin"
VOCAB_TREE_outdoor_by_buider = "vocab_tree_outdoor_by_builder.bin"
# VOCAB_TREE_pretrained_256K = "vocab_tree_flickr100K_words256K.bin"


def create_sequential_matcher_args(overlap=16,
                                quadratic_overlap=1,
                                loop_detection=0,
                                loop_detection_period=10,
                                loop_detection_num_images=50,
                                vocab_tree_file=None):
    args = [
        "colmap", "sequential_matcher",
        "--database_path", DATABASE_PATH,
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.quadratic_overlap", str(quadratic_overlap),
    ]

    if loop_detection:
        if vocab_tree_file is None:
            print("[sequential_matcher] loop detection mode need to pass the vocab_tree_file")
            exit(0)
        args += [
            "--SequentialMatching.loop_detection", "true",
            "--SequentialMatching.loop_detection_period", str(loop_detection_period),
            "--SequentialMatching.loop_detection_num_images", str(loop_detection_num_images),
            "--SequentialMatching.vocab_tree_path", os.path.join(CONFIG_PATH, vocab_tree_file)
        ]
    return args


def create_custom_matcher_args(n_seq_matches_per_view=0, n_quad_matches_per_view=10):
    args = [
        "python", "preprocess/make_colmap_custom_matcher.py",
        "--image_path", IMAGE_PATH,
        "--output_path", f"{PROJECT_PATH}/unrectified/matching.txt",
        # "--config_path", f"{CONFIG_PATH}",
        "--n_seq_matches_per_view", str(n_seq_matches_per_view),
        "--n_quad_matches_per_view", str(n_quad_matches_per_view),
        "--loop_matches", "45", "361", "53", "385", "81", "326"
    ]
    return args


def create_vocab_tree_matcher_args(vocab_tree_file=None):
    if vocab_tree_file is None:
        print("[vocab_tree_matcher] Must need to have vocab_tree_path.")
    args = [
        "colmap", "vocab_tree_matcher",
        "--database_path", DATABASE_PATH,
        "--VocabTreeMatching.vocab_tree_path", os.path.join(CONFIG_PATH, vocab_tree_file),
    ]
    return args


def create_hierarchical_mapper_args(ba_local_num_images=6, image_overlap=50):
    args = [
        "colmap", "hierarchical_mapper",
        "--database_path", f"{PROJECT_PATH}/unrectified/database.db",
        "--image_path", IMAGE_PATH,
        "--output_path", f"{PROJECT_PATH}/unrectified/sparse",
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_local_num_images", str(ba_local_num_images),
        "--image_overlap", str(image_overlap),
    ]
    return args
 
def create_mapper_args(ba_local_num_images=6):
    args = [
        "colmap", "mapper",
        "--database_path", f"{PROJECT_PATH}/unrectified/database.db",
        "--image_path", IMAGE_PATH,
        "--output_path", f"{PROJECT_PATH}/unrectified/sparse",
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_local_num_images", str(ba_local_num_images),
    ]
    return args


def create_glomap_mapper_args():
    # glomap mapper --database_path DATABASE_PATH --output_path OUTPUT_PATH --image_path IMAGE_PATH
    args = [
        "glomap", "mapper",
        "--database_path", f"{PROJECT_PATH}/unrectified/database.db",
        "--image_path", IMAGE_PATH,
        "--output_path", f"{PROJECT_PATH}/unrectified/sparse",
    ]
    return args


def setup_dirs(project_dir):
    """Create the directories that will be required."""
    if not os.path.exists(project_dir):
        print("creating project dir.")
        os.makedirs(project_dir)
    
    if not os.path.exists(os.path.join(project_dir, "aligned")):
        os.makedirs(os.path.join(project_dir, "aligned/sparse/0"))

    if not os.path.exists(os.path.join(project_dir, "rectified")):
        os.makedirs(os.path.join(project_dir, "rectified"))

    if not os.path.exists(os.path.join(project_dir, "unrectified")):
        os.makedirs(os.path.join(project_dir, "unrectified"))
        os.makedirs(os.path.join(project_dir, "unrectified", "sparse"))

    if not os.path.exists(os.path.join(project_dir, "unrectified", "sparse")):
        os.makedirs(os.path.join(project_dir, "unrectified", "sparse"))


def main():
    #!##################################################################
    #! Matcher test set 
    #!##################################################################
    colmap_matcher_exe_set = [
        # create_custom_matcher_args(n_seq_matches_per_view=8, n_quad_matches_per_view=12),
        create_custom_matcher_args(n_seq_matches_per_view=32, n_quad_matches_per_view=12),
        create_custom_matcher_args(n_seq_matches_per_view=64, n_quad_matches_per_view=12),
        create_custom_matcher_args(n_seq_matches_per_view=96, n_quad_matches_per_view=12),
        # create_custom_matcher_args(n_seq_matches_per_view=200, n_quad_matches_per_view=12),
        # create_sequential_matcher_args(overlap=16,
        #                     quadratic_overlap=1,
        #                     loop_detection=1,
        #                     loop_detection_period=16,
        #                     loop_detection_num_images=64,
        #                     vocab_tree_file=VOCAB_TREE_outdoor_by_buider),
        # create_sequential_matcher_args(overlap=32,
        #                     quadratic_overlap=1,
        #                     loop_detection=1,
        #                     loop_detection_period=16,
        #                     loop_detection_num_images=128,
        #                     vocab_tree_file=VOCAB_TREE_outdoor_by_buider),
        create_vocab_tree_matcher_args(vocab_tree_file=VOCAB_TREE_outdoor_by_buider)
    ]

    #!##################################################################
    #! Mapper test set 
    #!##################################################################
    colmap_mapper_exe_set = [
        create_glomap_mapper_args(),
        # create_hierarchical_mapper_args(ba_local_num_images=6, image_overlap=50),
        # create_hierarchical_mapper_args(ba_local_num_images=6, image_overlap=200),
        # create_hierarchical_mapper_args(ba_local_num_images=30, image_overlap=50),
        # create_hierarchical_mapper_args(ba_local_num_images=30, image_overlap=200),
        # create_mapper_args(ba_local_num_images=6),
        # create_mapper_args(ba_local_num_images=30),
    ]

    colmap_exe = "colmap"
    colmap_exe_combination = [(matcher, mapper) for matcher in colmap_matcher_exe_set 
                                                for mapper in colmap_mapper_exe_set]

    #!##################################################################
    #! Do tests of combination case 
    #!##################################################################
    for idx, exe_comb in enumerate(colmap_exe_combination, start=0):
        print(f"{idx}_test started...")
        setup_dirs(PROJECT_PATH)

        # just copy pre-extracted features
        if os.path.isfile(f"{PROJECT_PATH}/../database.db"):
            shutil.copy(f"{PROJECT_PATH}/../database.db", f"{DATABASE_PATH}")
        else:
            colmap_feature_extractor_args = [
                colmap_exe, "feature_extractor",
                "--database_path", DATABASE_PATH,
                "--image_path", IMAGE_PATH,
                "--ImageReader.single_camera_per_folder", "1",
                "--ImageReader.default_focal_length_factor", "0.5",
                "--ImageReader.camera_model", "OPENCV_FISHEYE",
            ]
            try:
                subprocess.run(colmap_feature_extractor_args, check=True)
                shutil.copy(f"{DATABASE_PATH}", f"{PROJECT_PATH}/../database.db")
            except subprocess.CalledProcessError as e:
                print(f"Error executing feature extractor: {e}")
                sys.exit(1)
 
        matcher_exe_args, mapper_exe_args = exe_comb
        print("[matcher_args]\n", matcher_exe_args)
        print("[mapper_args]\n", mapper_exe_args)

        #?##################################################################
        #? Matcher
        #?##################################################################
        # try:
        #     subprocess.run(matcher_exe_args, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error executing colmap matcher: {e}")
        #     sys.exit(1)

        # For more test faster
        try:
            if matcher_exe_args[1] == "sequential_matcher":
                matched_database_name = f"database-seq{matcher_exe_args[5]}-img{matcher_exe_args[-3]}-matched.db"
                if not os.path.isfile(f"{PROJECT_PATH}/../{matched_database_name}"):
                    subprocess.run(matcher_exe_args, check=True)
                    shutil.copy(DATABASE_PATH, f"{PROJECT_PATH}/../{matched_database_name}")
                else:
                    os.remove(DATABASE_PATH)
                    shutil.copy(f"{PROJECT_PATH}/../{matched_database_name}", DATABASE_PATH)
            elif matcher_exe_args[1] == "vocab_tree_matcher": # vocab_tree_matcher
                if not os.path.isfile(f"{PROJECT_PATH}/../database-vocab-matched.db"):
                    subprocess.run(matcher_exe_args, check=True)
                    shutil.copy(DATABASE_PATH, f"{PROJECT_PATH}/../database-vocab-matched.db")
                else:
                    os.remove(DATABASE_PATH)
                    shutil.copy(f"{PROJECT_PATH}/../database-vocab-matched.db", DATABASE_PATH)
            else:
                subprocess.run(matcher_exe_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap matcher: {e}")
            sys.exit(1)


        #*##################################################################
        #* Only for custom matcher
        #*##################################################################
        print("matcher name: ", matcher_exe_args[1])
        print("type: ", type(matcher_exe_args[1]))
        if matcher_exe_args[1] == "preprocess/make_colmap_custom_matcher.py":
            print("matching features...")
            colmap_matches_importer_args = [
                "colmap", "matches_importer",
                "--database_path", f"{PROJECT_PATH}/unrectified/database.db",
                "--match_list_path", f"{PROJECT_PATH}/unrectified/matching.txt"
            ]
            try:
                subprocess.run(colmap_matches_importer_args, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing colmap matches_importer: {e}")
                sys.exit(1)

        #*##################################################################
        #* Put known intrinsic into database
        #*##################################################################
        print("set camera parameters to database.db")
        set_intrinsic_to_database_args = [
            "python", f"{CONFIG_PATH}/scripts/set_intrinsic_to_database.py",
            "--intrinsic_path", f"{CONFIG_PATH}/intrinsic",
            "--database_path", f"{PROJECT_PATH}/unrectified/database.db"
        ]
        try:
            subprocess.run(set_intrinsic_to_database_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing python set_intrinsic_to_database.db: {e}")
            sys.exit(1)

        #?##################################################################
        #? Mapper
        #?##################################################################
        try:
            subprocess.run(mapper_exe_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap mapper: {e}")
            sys.exit(1)

        #*##################################################################
        #* When mapper result in multiple models, it should be merged
        #*##################################################################
        # if mapper_exe_args[0] == "colmap" and mapper_exe_args[1] == "mapper":
        #     files = os.listdir(f"{PROJECT_PATH}/unrectified/sparse")
        #     print("[mapper] more than 1 sparse model. need to merge models...")
        #     for model in range(len(files)-1) :
        #         try:
        #             colmap_model_merger_args = [
        #                 "colmap", "model_merger",
        #                 "--input_path1", f"{PROJECT_PATH}/unrectified/sparse/0",
        #                 "--input_path2", f"{PROJECT_PATH}/unrectified/sparse/{model+1}",
        #                 "--output_path", f"{PROJECT_PATH}/unrectified/sparse/0",
        #             ]
        #             subprocess.run(colmap_model_merger_args, check=True)
        #         except subprocess.CalledProcessError as e:
        #             print(f"Error executing colmap merger: {e}")
        #             sys.exit(1)

        #*##################################################################
        #* Simplified models.
        #*##################################################################
        simplify_images_args = [
            "python", f"preprocess/simplify_images.py",
            "--base_dir", f"{PROJECT_PATH}/unrectified/sparse/0"
        ]
        try:
            subprocess.run(simplify_images_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing simplify_images: {e}")
            sys.exit(1)

        #?##################################################################
        #? Rig Bundle Adjuster
        #?##################################################################
        # print("postprocessing for multi-camera rig constraint BA...")
        # colmap_rig_bundle_adjuster_args = [
        #     "colmap", "rig_bundle_adjuster",
        #     "--input_path", f"{PROJECT_PATH}/unrectified/sparse/0",
        #     "--output_path", f"{PROJECT_PATH}/unrectified/sparse/0",
        #     "--rig_config_path", f"{CONFIG_PATH}/rig_config.json",
        #     "--BundleAdjustment.refine_focal_length", "0",
        #     "--BundleAdjustment.refine_principal_point", "0",
        #     "--BundleAdjustment.refine_extra_params", "0",
        #     "--estimate_rig_relative_poses", "0",
        #     "--RigBundleAdjustment.refine_relative_poses", "1",
        #     ]
        # try:
        #     for _ in range(3):
        #         subprocess.run(colmap_rig_bundle_adjuster_args, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error executing colmap rig_bundle_adjuster: {e}")
        #     sys.exit(1)

        os.rename(PROJECT_PATH, f"{PROJECT_PATH}_{idx+1:02}")



if __name__ == "__main__":
    main()