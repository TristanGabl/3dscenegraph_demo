import argparse
import multiprocessing as mp
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from scenegraph3d import SceneGraph3D


def get_parser():
    parser = argparse.ArgumentParser(description="3dscenegraph pipeline using mask2former")
    parser.add_argument(
        "--config-file",
        default="Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
        metavar="FILE",
        help="path to config file (default set in main.py)",
    )
    
    parser.add_argument(
        "--input",
        nargs=1,
        required=True,
        help="A directory of a scan from '3D Scanner App'; "
        "The directory should be a full export of a scan, containing images and json files etc.",
    )
    parser.add_argument(
        "--output",
        default="output", 
        help="A file or directory to save output visualizations."
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs (default set in main.py); "
        "example: MODEL.WEIGHTS /path/to/model_checkpoint.pkkl",
        default=["MODEL.WEIGHTS", "model_weights/model_final_f07440.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


DEBUG = False
SAVE_OBJECTS = True
FORCE_MASK2FORMER = False # if True, the mask2former model will be run even if the processed images already exist
SAVE_VIZ = False
SKIP_PROJECTION_VIZ = False # if True, generating frame_XXXXX_projections.jpg will be skipped
SKIP_FUSED_VOTES_VIZ = False # if True, generating frame_XXXXX_fused_votes.jpg will be skipped

def main():
    args = get_parser().parse_args() # we first parse the args, see example use above
    # SceneGraph3D can be called as a class instance    
    pipeline = SceneGraph3D(args, DEBUG, SAVE_OBJECTS, FORCE_MASK2FORMER, SKIP_PROJECTION_VIZ, SKIP_FUSED_VOTES_VIZ)
    pipeline.generate_3d_scene_graph() # this will run the whole pipeline and save the results in the output folder, additionally we have the result within the class instance


if __name__ == "__main__":
    main()