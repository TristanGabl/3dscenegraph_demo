import sys
import os
import glob
import multiprocessing as mp
import pickle
import re
import trimesh
import json
import cv2
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Mask2Former')) 
from mask2former import add_maskformer2_config

from utils.scenegraph3d_objects import Objects
from utils.setup_logger import setup_logger
from plot.plot_labeled_pointcloud import plot_labeled_pointcloud
from relationship_prediction.relation_net import generate_edge_relationship, load_model
# from utils.open_clip_ import compute_similarity
# from utils.find_clusters import find_best_kmeans_clusters


# the whole generation is doine within this class
class SceneGraph3D:
    def __init__(
        self,
        args, 
        DEBUG: bool,
        SAVE_OBJECTS: bool,
        FORCE_MASK2FORMER: bool,
        SKIP_PROJECTION_VIZ: bool,
        SKIP_FUSED_VOTES_VIZ: bool,
        SAVE_VIZ: bool = True,
    ):
        mp.set_start_method("spawn", force=True)
        self.logger = setup_logger(DEBUG)
        self.DEBUG = DEBUG
        self.logger.info("SceneGraph3D logger initialized")
        self.args = args
        self.logger.info("Arguments: " + str(args))
        self.SAVE_VIZ = SAVE_VIZ
        self.logger.info("SAVE_VIZ: " + str(SAVE_VIZ))
        self.SAVE_OBJECTS = SAVE_OBJECTS
        self.logger.info("SAVE_OBJECTS: " + str(SAVE_OBJECTS))
        self.FORCE_MASK2FORMER = FORCE_MASK2FORMER
        self.logger.info("FORCE_MASK2FORMER: " + str(FORCE_MASK2FORMER))
        self.SKIP_PROJECTION_VIZ = SKIP_PROJECTION_VIZ
        self.logger.info("SKIP_PROJECTION_VIZ: " + str(SKIP_PROJECTION_VIZ))
        self.SKIP_FUSED_VOTES_VIZ = SKIP_FUSED_VOTES_VIZ
        self.logger.info("SKIP_FUSED_VOTES_VIZ: " + str(SKIP_FUSED_VOTES_VIZ))

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.logger.info("Device: " + str(self.device))
        args.opts += ["MODEL.DEVICE", str(self.device)]
        self.config = self.setup_config(args)

        self.mask2former_predictor = DefaultPredictor(self.config)
        self.metadata = self.mask2former_predictor.metadata
        
        self.input_frames, self.input_scan_path = self.generate_input_frames()
        self.input_folder_name = self.input_scan_path.split('/')[-2] if self.input_scan_path.endswith('/') else self.input_scan_path.split('/')[-1]
       
        output_scan_path = os.path.join(self.args.output, self.input_folder_name)
        self.full_output_scan_path = os.path.join(output_scan_path, 'full')
        self.plot_output_scan_path = os.path.join(output_scan_path, 'plot')
        self.result_output_scan_path = os.path.join(output_scan_path, 'result')
        self.object_output_scan_path = os.path.join(output_scan_path, 'objects')
        self.number_input_image_paths = len(self.input_frames)
        self.logger.info("Output path: " + output_scan_path)
        self.logger.info("Number image frames: " + str(len(self.input_frames)))

        # two data scan input types are supported: 
        if re.match(r"scene\d{4}_\d{2}", self.input_folder_name):
            self.scan_type = "scannet"
        else:
            self.scan_type = "3dscannerapp"
        self.logger.info("Scan type: " + self.scan_type)
            
    # this is the main function to trigger the whole pipeline
    def generate_3d_scene_graph(self): # main function
        self.processed_frame_paths = self.run_mask2former() # first use Mask2Former to get the panoptic segmentations
        
        # all classes
        self.all_categories = np.unique([segment_info['category_id'] for path in self.processed_frame_paths for segment_info in pickle.load(open(path + '.pkl', 'rb'))[1]])
    
        # load the mesh vertices and faces from input scan
        self.mesh_vertices = np.array(trimesh.load_mesh(os.path.join(self.input_scan_path, 'export_refined.obj')).vertices)
        self.mesh_faces = np.array(trimesh.load_mesh(os.path.join(self.input_scan_path, 'export_refined.obj')).faces)

        if os.path.exists(os.path.join(self.result_output_scan_path, 'semantic_segmentation.npy')) and os.path.exists(os.path.join(self.result_output_scan_path, 'mesh_vertices_frame_observations.npy')) and True:
            # progess loading
            self.mesh_vertices_classes = np.load(os.path.join(self.result_output_scan_path, 'semantic_segmentation.npy'))
            mesh_vertices_frame_observations = np.load(os.path.join(self.result_output_scan_path, 'mesh_vertices_frame_observations.npy'), allow_pickle=True)
            self.logger.info("Loaded mesh_vertices_classes progress")
        else:
            # distribute the panoptic segmentations from the images to the mesh vertices and remember in which frames a vertex was observed
            mesh_vertices_votes_global, mesh_vertices_frame_observations = self.distribute_panoptic_segmentations()

            # this is the final class assignment for each mesh vertex, -1 corresponds to background
            self.mesh_vertices_classes = np.apply_along_axis(lambda row: self.all_categories[np.argmax(row)] if np.any(row) else -1, 1, mesh_vertices_votes_global)
    
            # progess saving
            os.makedirs(self.result_output_scan_path, exist_ok=True)
            np.save(os.path.join(self.result_output_scan_path, 'semantic_segmentation.npy'), self.mesh_vertices_classes)
            np.save(os.path.join(self.result_output_scan_path, 'mesh_vertices_frame_observations.npy'), mesh_vertices_frame_observations)
            self.logger.info("Exported mesh_vertices_classes and mesh_vertices_frame_observations")

        #  metadata pulled from Mask2Former for class names and colors
        self.id_to_class = {i: name for i, name in enumerate(self.metadata.stuff_classes)}
        self.id_to_class[-1] = "background"
        self.id_to_class_color = {i: color for i, color in enumerate(self.metadata.stuff_colors)}
        self.id_to_class_color[-1] = [0, 0, 0] # black

        with open("label_mapping/coco_id_to_name.json", "w") as json_file:
            json.dump(self.id_to_class, json_file, indent=4)

        # create connected graph from the mesh vertices 
        self.mesh_edges = self.create_graph_edges()

        # get edge boarders according to naive semantic segmentation oriented instance segmentation, neighbors of same class are assigned neighbors in instance segmentation step
        self.edges_boarders = self.mesh_edges[np.logical_and(self.mesh_vertices_classes[self.mesh_edges[:, 0]] != self.mesh_vertices_classes[self.mesh_edges[:, 1]], 
                                              np.logical_and(self.mesh_vertices_classes[self.mesh_edges[:, 0]] != -1, self.mesh_vertices_classes[self.mesh_edges[:, 1]] != -1))]
        
        # inital creation of 3d scene graph objects
        self.objects = self.create_3dscenegraph_objects()


        if os.path.exists(os.path.join(self.result_output_scan_path, 'post_double_check_objects.json')) and True:
            self.logger.info("Loading post double check objects progress")
            # progress loading
            with open(os.path.join(self.result_output_scan_path, 'post_double_check_objects.json'), 'r') as f:
                self.objects = [Objects(**obj) for obj in json.load(f)]
        else:
            # best-perspective frame instance segmentation technique
            self.objects = self.duplicate_double_check_mask2former(self.objects, mesh_vertices_frame_observations)
            # progress saving
            with open(os.path.join(self.result_output_scan_path, 'post_double_check_objects.json'), 'w') as f:
                json.dump([obj.__dict__ for obj in self.objects], f, indent=4)
    
        # failed attempt to use kmeans clustering for instance segmentation
        # self.objects = self.duplicate_double_check_kmeans(self.objects)

        # assign lost vertices to nearest object by using BFS
        self.assign_lost_vertices_to_nearest_object(self.objects, self.mesh_edges, self.mesh_vertices_classes)

        # after finishing the full instance segmentation, we can update the neighbors of the objects
        self.update_neighbors(self.objects, self.edges_boarders)

        # option to save the objects as .ply files
        if self.SAVE_OBJECTS:
            self.save_object_vertices(self.objects)
        
        
        coco_name_to_name_simplified = json.load(open("label_mapping/coco_name_to_name_simplified.json", 'r'))
        relation_net_model, name2idx, label2idx, idx2label = load_model("relationship_prediction/relation_model.pth", self.device)
        self.edge_relationships = [["" for _ in range(len(self.objects))] for _ in range(len(self.objects))]
        for object1 in self.objects:
            for object2 in self.objects:
                if object1.object_id in set(object2.neighbors):
                    obj1 = object1.__class__(**object1.__dict__.copy())
                    obj2 = object2.__class__(**object2.__dict__.copy())
                    obj1_name_with_number = obj1.name[:]
                    obj2_name_with_number = obj2.name[:]
                    obj1.name = coco_name_to_name_simplified[obj1.name.split(" #")[0]]
                    obj2.name = coco_name_to_name_simplified[obj2.name.split(" #")[0]]
                    edge_forward = generate_edge_relationship(obj1, obj2, relation_net_model, name2idx, label2idx, idx2label)
                    edge_backward = generate_edge_relationship(obj2, obj1, relation_net_model, name2idx, label2idx, idx2label)
                    self.edge_relationships[object1.object_id][object2.object_id] = edge_forward
                    self.edge_relationships[object2.object_id][object1.object_id] = edge_backward                    
                    obj1.name = obj1_name_with_number
                    obj2.name = obj2_name_with_number
                

    
        self.logger.info("Finished generating edge relationships")


        # saving objects into a json file
        if not os.path.exists(self.result_output_scan_path):
            os.makedirs(self.result_output_scan_path, exist_ok=True)
        self.objects_json = [{k: v for k, v in obj.__dict__.items() if v != ''} for obj in self.objects]
        with open(os.path.join(self.result_output_scan_path, 'objects.json'), 'w') as f:
            json.dump(self.objects_json, f, indent=4)
       
        # saving edge relationships into a json file
        self.edge_relationships_json = [[edge for edge in row] for row in self.edge_relationships]
        with open(os.path.join(self.result_output_scan_path, 'edge_relationships.json'), 'w') as f:
            json.dump(self.edge_relationships_json, f, indent=4)

        # plot everything
        self.save_segmented_pointcloud()

        return self.objects, self.edge_relationships

    # this function is inspired by the Mask2Former demo script
    def run_mask2former(self): 
        pbar = tqdm.tqdm(
            total=self.number_input_image_paths,
            unit="images"
        )

        processed_frame_paths = [] # saved with no suffix
        for frame in self.input_frames:
            frame_path = os.path.join(self.input_scan_path, frame)
            image_path = frame_path + '.jpg'
            image_info_path = frame_path + '.json'
            output_image_path = os.path.join(self.full_output_scan_path, frame)
            inference_output_path = output_image_path + '.pkl'

            image = read_image(image_path, format="BGR")

            if not os.path.exists(inference_output_path) or self.FORCE_MASK2FORMER:
                pbar.set_description(f"Running Mask2Former on image: {frame}")
                pbar.update()

                image_info = json.load(open(image_info_path, 'r'))

                if self.scan_type == "scannet":
                    # in unlikly case some data is missing, we skip the image and remove it from the input frames
                    if not any(key in image_info.keys() for key in ["calibrationColorIntrinsic", "calibrationDepthIntrinsic", "Pose", "depthShift", "depthWidth", "depthHeight", "colorWidth", "colorHeight"]):
                        self.logger.warning(f"{frame}: Frame not in input frames, skipping this image")
                        self.input_frames.remove(frame)
                        continue
                else: # "3dscannerapp"
                    if not any(key in image_info.keys() for key in ["cameraPoseARFrame", "projectionMatrix"]):
                        self.logger.warning(f"{frame}: None of cameraPoseARFrame, projectionMatrix, or mvp found in image info json file, skipping this image")
                        self.input_frames.remove(frame)
                        continue

                
                os.makedirs(self.full_output_scan_path, exist_ok=True)

                # main inference step of Mask2Former
                predictions = self.mask2former_predictor(image)

                # BGR to RGB
                image = image[:, :, ::-1]

                # save inference output
                with open(inference_output_path, 'wb') as f:
                    pickle.dump((predictions["panoptic_seg"][0].to(torch.device("cpu")), predictions["panoptic_seg"][1]), f)
                
                # saving image info
                if self.scan_type == "scannet":
                    image_info_relevant = {key: image_info[key] for key in ['calibrationColorIntrinsic', 'calibrationDepthIntrinsic', 'Pose', 'depthShift', 'depthWidth', 'depthHeight', 'colorWidth', 'colorHeight']}
                    with open(output_image_path + '.json', 'w') as f:
                        json.dump(image_info_relevant, f)
                else: # "3dscannerapp" 
                    image_info_relevant = {key: image_info[key] for key in ['cameraPoseARFrame', 'projectionMatrix']}
                    with open(output_image_path + '.json', 'w') as f:
                        json.dump(image_info_relevant, f)
        
                if self.SAVE_VIZ:
                    panoptic_seg, panoptic_seg_info = predictions["panoptic_seg"]
                    visualizer = Visualizer(image, 
                                        MetadataCatalog.get(self.config.DATASETS.TEST[0] if len(self.config.DATASETS.TEST) else "__unused"), 
                                        instance_mode=ColorMode.IMAGE
                                        )
                    vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(torch.device("cpu")), panoptic_seg_info)
                    vis_output.save(output_image_path + '.jpg')
                    self.logger.debug(f"Saved visualization to: {output_image_path}.jpg")
                    
            else:
                pbar.set_description(f"Skipping image (inference already saved): {frame}")
                pbar.update()

            # if everything is successful, add the path to the list of processed images
            processed_frame_paths.append(output_image_path)
        pbar.close()

        self.logger.important("Finished running Mask2former on images")
        return processed_frame_paths

    # main method for best-persective frame instance segmentation 
    def duplicate_double_check_mask2former(self, objects, mesh_vertices_frame_observations):
        pbar = tqdm.tqdm(
            total=len(objects),
            unit="objects"
        )

        new_objects = []
        new_objects_id = 0  
        for obj in objects:
            if any(keyword in obj.name.lower().replace('-', ' ').split() for keyword in ["floor", "wall", "table", "ceiling"]):
                print("Skipping floor, wall, table, ceiling")
                obj.object_id = new_objects_id
                obj.name = obj.name + " #" + str(new_objects_id)
                new_objects_id += 1
                new_objects.append(obj)
                pbar.update()
                continue

            # all frame indices where some part of the object was observed
            frames_tmp = np.unique([frame for idx in obj.index_set for frame in mesh_vertices_frame_observations[idx]])

            best_perspective_frame = None
            object_count_best = -1
            vertices_count_best = -1
            for frame in frames_tmp:
                image_path = os.path.join(self.full_output_scan_path, frame)
                panoptic_seg, panoptic_seg_info = pickle.load(open(image_path + '.pkl', 'rb'))
                object_count = np.sum(segment_info['category_id'] == obj.class_id for segment_info in panoptic_seg_info)

                image_path = os.path.join(self.full_output_scan_path, frame)
                panoptic_seg, panoptic_seg_info = pickle.load(open(image_path + '.pkl', 'rb'))
                image_info = json.load(open(image_path + '.json', 'r'))

                # creatimg mask for object, need no depth image since the old index set is already filtered by depth 
                dummy_image = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1], 3), dtype=np.uint8)
                if self.scan_type == "scannet":
                    projections_filtered, projections_filtered_mask = self.project_pointcloud_scannet(image_info, dummy_image, self.mesh_vertices[obj.index_set])
                else:
                    projections_filtered, projections_filtered_mask = self.project_pointcloud_3dscannerapp(image_info, dummy_image, self.mesh_vertices[obj.index_set])

                point_values = panoptic_seg[projections_filtered[:, 1], projections_filtered[:, 0]]

                if object_count > object_count_best or (object_count == object_count_best and len(point_values) > vertices_count_best):
                    best_perspective_frame = frame
                    object_count_best = object_count
                    vertices_count_best = len(point_values)

                    
            # not clean but loading image segmentation again, from best perspective frame
            image_path = os.path.join(self.full_output_scan_path, best_perspective_frame)
            panoptic_seg, panoptic_seg_info = pickle.load(open(image_path + '.pkl', 'rb'))
            image_info = json.load(open(image_path + '.json', 'r'))

            local_class_values = [i for i, _ in enumerate(panoptic_seg_info) if panoptic_seg_info[i]['category_id'] == obj.class_id]
            if len(local_class_values) == 1:
                print("Skipping floor, wall, table, ceiling, or single object")
                obj.object_id = new_objects_id
                obj.name = obj.name + " #" + str(new_objects_id)
                new_objects_id += 1
                new_objects.append(obj)
                pbar.update()
                continue
            pbar.set_description(f"Checking for duplicates (mask2former): {obj.name}")
            pbar.update()    

            dummy_image = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1], 3), dtype=np.uint8) # just for the size, to use the same function
            if self.scan_type == "scannet":
                projections_filtered, projections_filtered_mask = self.project_pointcloud_scannet(image_info, dummy_image, self.mesh_vertices[obj.index_set])
            else:
                projections_filtered, projections_filtered_mask = self.project_pointcloud_3dscannerapp(image_info, dummy_image, self.mesh_vertices[obj.index_set])        


           
            # get values at the projected points
            point_values = panoptic_seg[projections_filtered[:, 1], projections_filtered[:, 0]]
            
            # If there are multiple unique values for the object's class, split the object
            self.logger.info(f"Object {obj.name} has multiple segments in the panoptic segmentation.")
            new_object_batch = []
            for value in local_class_values:
                new_index_set = np.array(obj.index_set)[np.where(np.isin(point_values, value+1))[0]]
                if len(new_index_set) == 0:
                    continue

                object_id = new_objects_id
                object_class = self.id_to_class[self.mesh_vertices_classes[new_index_set[0]]] + " #" + str(new_objects_id)
                class_id = obj.class_id
                center = np.mean(self.mesh_vertices[new_index_set], axis=0)
                min_coords = np.min(self.mesh_vertices[new_index_set], axis=0)
                max_coords = np.max(self.mesh_vertices[new_index_set], axis=0)
                size_x = max_coords[0] - min_coords[0]
                size_y = max_coords[1] - min_coords[1]
                size_z = max_coords[2] - min_coords[2]

                new_object = Objects(
                    name=object_class, 
                    object_id=object_id, 
                    class_id=class_id, 
                    x=center[0], 
                    y=center[1], 
                    z=center[2],
                    size_x=size_x,
                    size_y=size_y,
                    size_z=size_z,
                    index_set=new_index_set,
                    neighbors=obj.neighbors,
                    relations=obj.relations, 
                    best_perspective_frame=best_perspective_frame
                )
                new_object_batch.append(new_object)
                new_objects_id += 1
            
            # since we split the object, we need to update the neighbors
            for i, obj1 in enumerate(new_object_batch):
                obj1.neighbors = [o.object_id for j, o in enumerate(new_object_batch) if j != i]
            new_objects.extend(new_object_batch)
        
        
        pbar.close()
        return new_objects

    
    def assign_lost_vertices_to_nearest_object(self, objects, mesh_edges, vertices_classes):
        ids = np.unique([obj.class_id for obj in objects])
        object_sets = {obj_class: set() for obj_class in ids}
        

        [object_sets[obj.class_id].add(obj.object_id) for obj in objects]

        # we go over object class to object class
        for objs in object_sets:
            class_vert_idxs = np.where(vertices_classes == objs)[0]

            mask0 = np.isin(mesh_edges[:, 0], class_vert_idxs)
            mask1 = np.isin(mesh_edges[:, 1], class_vert_idxs)
            sub_edges = mesh_edges[mask0 & mask1]

            G = nx.Graph()
            G.add_edges_from(sub_edges)

            # keep track of assigned vertices to object_ids
            assignment = {}
            queue = deque()

            for obj in [objects[i] for i in object_sets[objs]]:
                obj_id = obj.object_id
                for v in obj.index_set:
                    if v in assignment:  # Skip if already assigned
                        continue
                    if v in G:  # Only if v appears in the subgraph
                        assignment[v] = obj_id
                        queue.append((v, obj_id))


            # we grow out the instance proposals using BFS to create instances
            while queue:
                v, oid = queue.popleft()
                for nbr in G.neighbors(v):
                    if nbr not in assignment:
                        assignment[nbr] = oid
                        queue.append((nbr, oid))
            
            
            for obj in [objects[i] for i in object_sets[objs]]:
                # collecting all vertices that are assigned to the object after BFS growing
                obj.index_set = [int(v) for v, owner in assignment.items() if owner == obj.object_id]

                    

    def distribute_panoptic_segmentations(self):
        pbar = tqdm.tqdm(
            total=self.number_input_image_paths,
            unit="images",
        )

        # final votes to be returned
        mesh_vertices_votes_global = np.zeros((self.mesh_vertices.shape[0], len(self.all_categories)), dtype=int)

        # storage to record frames in which each mesh vertex was observed
        mesh_vertices_frame_observations = [set() for _ in range(self.mesh_vertices.shape[0])]

        # iterate over the processed images and project the 3d points into the image and apply panoptic segmentation
        for frame in self.processed_frame_paths:
            pbar.set_description(f"Projecting points into image and applying panoptic segmentation: {os.path.basename(frame)}")
            
            panoptic_seg, panoptic_seg_info = pickle.load(open(frame + '.pkl', 'rb'))
            image_info = json.load(open(frame + '.json', 'r'))

            # in some cases frame seemed to be mssing, we skip
            if not os.path.exists(frame + '.jpg'):
                self.logger.error(f"Image not found: {frame + '.jpg'}")
                self.processed_frame_paths.remove(frame)
                continue

            image = cv2.imread(frame + '.jpg')
            depth_map_path = os.path.join(self.input_scan_path, frame.split('/')[-1].replace("frame", "depth") + ".png")
            depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED) / 1000 # convert to meters
            if self.scan_type == "scannet":
                # depth_map = depth_map
                depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) # resize to image size
            else:
                depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) # resize to image size

            # project the 3d point cloud into the image and filter out points that are not in the image
            if self.scan_type == "scannet":
                projections_filtered, projections_filtered_mask = self.project_pointcloud_scannet(image_info, image)
            else:
                projections_filtered, projections_filtered_mask = self.project_pointcloud_3dscannerapp(image_info, image)

            if len(projections_filtered) == 0:
                # we skip the frame if no points were projected
                continue

            # remember what image a vertex was observed in
            [mesh_vertices_frame_observations[idx].add(frame.split("/")[-1]) for idx in np.where(projections_filtered_mask)[0]]
            
            # depth values at the projected points
            depth_array = depth_map[projections_filtered[:, 1].astype(int), projections_filtered[:, 0].astype(int)]

            # for debugging projected points
            if self.SAVE_VIZ and not self.SKIP_PROJECTION_VIZ:
                img_projected = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # add grey border around the image for debugging
                border_offset = 50
                img_projected = cv2.copyMakeBorder(img_projected, border_offset, border_offset, border_offset, border_offset, cv2.BORDER_CONSTANT, value=[128, 128, 128])

                for i, point in enumerate(projections_filtered[:,:2]): # not efficient but just for debugging
                    if abs(projections_filtered[i, 2] - depth_array[i]) <= 0.03:
                        cv2.circle(img_projected, tuple(point.ravel().astype(int) + border_offset), 2, (0, 0, 255), -1)
                    else:
                        cv2.circle(img_projected, tuple(point.ravel().astype(int) + border_offset), 2, (228, 228, 228), -1)

                cv2.imwrite(frame + '_projections.jpg', img_projected) 
                self.logger.debug(f"saved projections to {frame}_projections.jpg")
            

            # we use panoptic_seg to get votes for object classes for each 3d point
            local_class_ids = np.array([segment_info['id'] for segment_info in panoptic_seg_info])
            # similar to the mesh_vertices_votes_global but locally
            projections_class_votes_local = np.zeros((len(projections_filtered), len(local_class_ids)))

            # distribute the class votes to the mesh vertices
            projections_class_votes_local, mesh_vertices_votes_global = self.distribute_class_votes(mesh_vertices_votes_global, projections_filtered, projections_filtered_mask, panoptic_seg, panoptic_seg_info, depth_array, projections_class_votes_local)
            
            # fuse votes to the vote that is most frequent
            projections_class_votes_local = np.apply_along_axis(lambda row: local_class_ids[np.argmax(row)] if np.any(row) else -1, 1, projections_class_votes_local)
            
            # for debugging point classes
            if self.DEBUG:
                number_of_classes = len(local_class_ids)
                number_of_class_points = np.sum(projections_class_votes_local != 0)
                self.logger.debug(f"number_of_classes: {number_of_classes}")
                self.logger.debug(f"number_of_class_points: {number_of_class_points}")
                if self.SAVE_VIZ and not self.SKIP_FUSED_VOTES_VIZ: 
                    # slow, can be skipped by setting SKIP_FUSED_VOTES_VIZ to True in main.py
                    image_fused_votes = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                    # add grey border around the image for debugging
                    border_offset = 50
                    image_fused_votes = cv2.copyMakeBorder(image_fused_votes, border_offset, border_offset, border_offset, border_offset, cv2.BORDER_CONSTANT, value=[128, 128, 128])

                    
                    colors = np.zeros((len(projections_filtered), 3))
                    colormap = plt.cm.get_cmap("tab20", np.max(local_class_ids) + 1)


                    # Map each class to a color
                    for i, class_id in enumerate(local_class_ids):
                        if class_id == -1:
                            colors[projections_class_votes_local == class_id] = (0, 0, 0)
                        else:
                            colors[projections_class_votes_local == class_id] = colormap(class_id)[:3]
                    
                    for i, point in enumerate(projections_filtered[:,:2]): # leave out the depth
                        cv2.circle(image_fused_votes, tuple(point.ravel().astype(int) + border_offset), 4, (colors[i] * 255), -1)

                    cv2.imwrite(frame + '_fused_votes.jpg', image_fused_votes)
                    self.logger.debug(f"saved projections with votes to {frame}_fused_votes.jpg")

            pbar.update()
        pbar.close()

        self.logger.important("Finished distributing panoptic segmentations to mesh vertices")
        return mesh_vertices_votes_global, mesh_vertices_frame_observations


    def distribute_class_votes(self, mesh_vertices_votes_global, projections_filtered, projections_filtered_mask, panoptic_seg, panoptic_seg_info, depth_array, projections_class_votes_local):
        for category_id_local, category_id_global in enumerate([seg["category_id"] for seg in panoptic_seg_info], start=1):
            # filter out for current local category
            mask = panoptic_seg == category_id_local 
            if mask.sum() == 0:
                continue
        
            # transform to a boolean mask for the filtered points
            mask = mask[projections_filtered[:, 1], projections_filtered[:, 0]]
            # adding a criteria for depth (+-0.05 meters)
            mask = mask & (np.abs(depth_array - projections_filtered[:, 2]) < 0.05)

            # add the votes to the global mesh vertices according to the number of classes in the panoptic segmentation 
            # (if there are more classes, the image has more of an overview of the scene -> better segmentation)
            weight_factor = len(panoptic_seg_info) * len(panoptic_seg_info)
            mesh_vertices_votes_global[projections_filtered_mask, np.where(self.all_categories == category_id_global)[0][0]] += mask.numpy() * weight_factor
            # add the votes to the local mesh projections
            projections_class_votes_local[:, category_id_local-1] = mask.numpy()
        
        return projections_class_votes_local, mesh_vertices_votes_global
        
    # handles the case of Scannet input scan
    def project_pointcloud_scannet(self, image_info, image, vertices=None):
        pose = np.array(image_info['Pose']).reshape((4, 4))
        pose = np.linalg.inv(pose)
        # Convert 3D points to homogeneous coordinates
        if vertices is None:
            points_homogeneous = np.hstack((self.mesh_vertices, np.ones((self.mesh_vertices.shape[0], 1)))) 
        else:
            points_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        points_camera = np.dot(points_homogeneous, pose.T)

        K = np.array(image_info['calibrationColorIntrinsic'])[:3, :3]

        # performing projection into image space

        points_2d = (K @ points_camera[:, :3].T).T  # [N, 3]

        points_2d[:, 0] /= points_2d[:, 2]
        points_2d[:, 1] /= points_2d[:, 2]
        points_2d[:, :2] = np.round(points_2d[:, :2]).astype(int) 

        projections_filtered_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0]) &
            (points_2d[:, 2] > 0)  # Ensure points are in front of the camera
        )

        # Filter points
        projections_filtered = points_2d[projections_filtered_mask]
        

        if self.DEBUG:
            self.logger.debug(f"number_points_projected: {projections_filtered}")

        return projections_filtered, projections_filtered_mask

    # 3D Scanner App case
    def project_pointcloud_3dscannerapp(self, image_info, image, vertices=None):
        # compute the model view projection matrix

        pose = np.array(image_info['cameraPoseARFrame']).reshape((4, 4))
        projection_matrix = np.array(image_info['projectionMatrix']).reshape((4, 4))
        view_matrix = np.linalg.inv(pose)
        mvp = np.dot(projection_matrix, view_matrix)
        
        # project the 3d point cloud and filter out points that are not in the image
        if vertices is None:
            projections = self.project_points_to_image(self.mesh_vertices, mvp, image.shape[1], image.shape[0])
        else:
            projections = self.project_points_to_image(vertices, mvp, image.shape[1], image.shape[0])
        projections_2d = np.round(projections[:,:2]).astype(int) # round to nearest pixel
        projections[:, :2] = projections_2d # store the z coordinate
        projections_filtered_mask = (projections[:, 0] >= 0) & (projections[:, 0] < image.shape[1]) & (projections[:, 1] >= 0) & (projections[:, 1] < image.shape[0])
        projections_filtered = projections[projections_filtered_mask]

        if self.DEBUG:
            self.logger.debug(f"number_points_projected: {projections_filtered}")

        return projections_filtered, projections_filtered_mask
    
    # creates edges from mesh faces, this would be done differently if input scan would not include a mesh
    def create_graph_edges(self):
        edges = np.vstack([self.mesh_faces[:, [0, 1]], self.mesh_faces[:, [1, 2]], self.mesh_faces[:, [2, 0]]])
        
        # remove double edges 
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        return edges
    
    # initally creating objects from "islands of vertices" in the mesh, basically naively connecting vertices of the same category
    def create_3dscenegraph_objects(self):

        G = nx.Graph()
        # removing edges that connect two vertices of same category
        edges_single_classes = self.mesh_edges[self.mesh_vertices_classes[self.mesh_edges[:, 0]] == self.mesh_vertices_classes[self.mesh_edges[:, 1]]]
        G.add_edges_from(edges_single_classes)  # Add edges to the graph, also adds the vertices
        blobs = list(nx.connected_components(G))

        # remove small blobs and blobs corresponding to background
        blobs = [list(blob) for blob in blobs if np.all(self.mesh_vertices_classes[list(blob)] != -1) and len(blob) > 30]
         
        self.logger.info(f"Number of objects found in Graph: {len(blobs)}")

        # create a object for each island of vertices
        objects = [] 
        for i, blob in enumerate(blobs):
            object_id = i
            object_class = self.id_to_class[self.mesh_vertices_classes[blob[0]]]

            class_id = self.mesh_vertices_classes[blob[0]]
            center = np.mean(self.mesh_vertices[blob], axis=0)
            min_coords = np.min(self.mesh_vertices[blob], axis=0)
            max_coords = np.max(self.mesh_vertices[blob], axis=0)
            size_x = max_coords[0] - min_coords[0]
            size_y = max_coords[1] - min_coords[1]
            size_z = max_coords[2] - min_coords[2]

            objects.append(Objects(name=object_class, 
                                   object_id=object_id, 
                                   class_id=class_id, 
                                   x=center[0], 
                                   y=center[1], 
                                   z=center[2],
                                   size_x=size_x,
                                   size_y=size_y,
                                   size_z=size_z,
                                   index_set=blob))
            
        return objects
    

    def update_neighbors(self, objects, edges_borders):
        self.logger.info("Updating neighbors...")

        index_to_obj = {}
        for obj_id, obj in enumerate(objects):
            for idx in obj.index_set:
                index_to_obj[idx] = obj_id

        for a, b in edges_borders:
            object_id_0 = index_to_obj.get(a)
            object_id_1 = index_to_obj.get(b)
            if object_id_0 is None or object_id_1 is None:
                continue

            # add each other as neighbors
            if object_id_1 not in objects[object_id_0].neighbors:
                objects[object_id_0].neighbors.append(object_id_1)
            if object_id_0 not in objects[object_id_1].neighbors:
                objects[object_id_1].neighbors.append(object_id_0)
        
        # option to expand neigbors to all objects that are close to each other
        if (False):
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i != j:
                        distance = np.linalg.norm(
                        np.array([obj1.x, obj1.y, obj1.z]) - np.array([obj2.x, obj2.y, obj2.z])
                        )
                        if distance <= 1.0:
                            if obj2.object_id not in obj1.neighbors:
                                obj1.neighbors.append(obj2.object_id)
                            if obj1.object_id not in obj2.neighbors:
                                obj2.neighbors.append(obj1.object_id)

        self.logger.info("Updated neighbors!")

    # not pretty but works
    def save_segmented_pointcloud(self):
        path_full = os.path.join(self.full_output_scan_path, self.input_folder_name)
        path_plot = os.path.join(self.plot_output_scan_path, self.input_folder_name)
        np.save(path_full + '_pointcloud_classes.npy', self.mesh_vertices_classes)

        os.makedirs(self.plot_output_scan_path, exist_ok=True)
        np.save(path_plot + '_pointcloud_classes.npy', self.mesh_vertices_classes)

        self.logger.info("saved pointcloud_classes.npy")

        os.system("cp " + os.path.join(self.input_scan_path, 'export_refined.obj') + " " + path_full + '_pointcloud_classes.obj')
        os.system("cp " + os.path.join(self.input_scan_path, 'export_refined.obj') + " " + path_plot + '_pointcloud_classes.obj')
        self.logger.info("copied export_refined.obj")
    
        if self.SAVE_VIZ:
            # Load the point cloud
            name = path_plot + '_pointcloud_classes'
            
            edges_single_classes = self.mesh_edges[self.mesh_vertices_classes[self.mesh_edges[:, 0]] == self.mesh_vertices_classes[self.mesh_edges[:, 1]]]
            fig = plot_labeled_pointcloud(self, name, self.mesh_vertices_classes, self.mesh_vertices, edges_single_classes, self.edge_relationships, self.objects, self.id_to_class, self.id_to_class_color)

            # saving the whole graph as a finished backed pointcloud visualization into a html file
            fig.write_html(name + '.html')

            self.logger.info("saved pointcloud visualization html")

    # only for debugging, was used to look object individual vertices
    def save_object_vertices(self, objects):
        # Save each object as a separate .obj file

        os.makedirs(self.object_output_scan_path, exist_ok=True)

        # prevent a mess
        for file in os.listdir(self.object_output_scan_path):
            if file.endswith(".obj"):
                os.remove(os.path.join(self.object_output_scan_path, file))
                self.logger.debug(f"Removed file: {file}")

        for obj in objects:
            if len(obj.index_set) != 0:
                obj_vertices = self.mesh_vertices[obj.index_set]

                obj_file_path = os.path.join(self.object_output_scan_path, f"{obj.name.replace(' ', '_')}_{len(obj.index_set)}_vertices.obj")
                with open(obj_file_path, 'w') as obj_file:
                    for vertex in obj_vertices:
                        obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                self.logger.debug(f"Saved object {obj.name} to {obj_file_path}")
            else:
                self.logger.debug(f"Object {obj.name} has no vertices, skipping saving.")
            
                
    # triggered in main.py to evaluate settings
    def setup_config(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # disable model outputs we don't need, having issues with memory otherwise
        # we both those included in panoptic segmentation
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        
        cfg.freeze()
        return cfg
    
    # generates the input scope for the pipeline, this list is potentially shortened on inference
    def generate_input_frames(self):
        assert len(self.args.input) == 1, "Only one input directory should be provided"

        if os.path.isdir(self.args.input[0]):
            paths = sorted(glob.glob(os.path.join(self.args.input[0], 'frame_*.jpg')))
            input_frames = [os.path.basename(path).removesuffix('.jpg') for path in paths]
            assert input_frames, f"Provided input directory does not contain any images, check if it is a directory of a scan from '3D Scanner App', the folder only contrains: {os.listdir(self.args.input[0])}"
            
        else:
            raise ValueError("Input does not exist or is not a directory")
        
        return input_frames, self.args.input[0]

    # inspired by code from Dr. Dániel Béla Baráth, thank you!
    def project_points_to_image(self, p_in, mvp, image_width, image_height):
        p0 = np.concatenate([p_in, np.ones([p_in.shape[0], 1])], axis=1)
        e0 = np.dot(p0, mvp.T)
        pos_z = e0[:, 2]
        e0 = (e0.T / e0[:, 3]).T
        pos_x = e0[:, 0]
        pos_y = e0[:, 1]
        projections = np.zeros([p_in.shape[0], 3])
        projections[:, 0] = (0.5 + (pos_x) * 0.5) * image_width
        projections[:, 1] = (1.0 - (0.5 + (pos_y) * 0.5)) * image_height
        projections[:, 2] = pos_z  # Store the z coordinate
        return projections
    
    