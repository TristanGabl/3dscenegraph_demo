import numpy as np
import trimesh
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
from plyfile import PlyData, PlyElement
import json



# used for benchmarking against scannet ground truth
def create_ply_file_in_scannet_format(vertices, labels, output_path):
    # Convert labels to scannet ids
    with open('label_mapping/coco_id_to_scannet_id.json', 'r') as f:
        coco_to_scannet = json.load(f)
    labels = [coco_to_scannet[f"{label}"] for label in labels]  # Default to 0 if not found

    vertex_data = np.array(
        [(v[0], v[1], v[2], l) for v, l in zip(vertices, labels)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'i4')]
    )

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element]).write(output_path)

def plot_labeled_pointcloud(self, name, ids, vertices, edges, edge_relationships, objects, ids_to_class, ids_to_class_color):


    create_ply_file_in_scannet_format(vertices, ids, name + "_with_scannet_ids.ply")


    # Invert the x-axis and switch the y and z axes due to different format
    if self.scan_type == '3dscannerapp':
        vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
        vertices[:, 0] = -vertices[:, 0]
        for obj in objects:
            obj.x = -obj.x
            obj.y, obj.z = obj.z, obj.y

    # object to color
    num_objects = len(objects)
    np.random.seed(42)  # For reproducibility
    colors = [f'rgb({np.random.randint(0, 256)},{np.random.randint(0, 256)},{np.random.randint(0, 256)})' for _ in range(num_objects)]
    dict_object_to_color = {obj.name: colors[i] for i, obj in enumerate(objects)}
    dict_object_to_color['background'] = 'rgb(0,0,0)'
    
    # id to object
    dict_id_to_object = {}
    for obj in objects:
        if hasattr(obj, 'index_set'):
            for id_ in obj.index_set:
                dict_id_to_object[id_] = obj.name
        else:
            print(f"Warning: Object {obj.name} is missing 'index_set' attribute.")


    # add vertices
    df = pd.DataFrame(vertices, columns=['x', 'y', 'z'])
    df['id'] = ids
    df['labels'] = [dict_id_to_object[i] if i in dict_id_to_object else 'background' for i,x in enumerate(ids)]
    df['color'] = [dict_object_to_color[dict_id_to_object[i]] if i in dict_id_to_object else 'rgb(0,0,0)' for i in ids]
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='labels', 
                        color_discrete_map=dict_object_to_color,
                        hover_data={'x': True, 'y': True, 'z': True, 'id': True, 'labels': True, 'color': True}
                        )
    

    # add edges
    edge_x = []
    edge_y = []
    edge_z = []

    coords = df[['x', 'y', 'z']].to_numpy()
    edge_coords = coords[np.array(edges)]  # shape: (num_edges, 2, 3)

    edge_x = edge_coords[:, :, 0].flatten()
    edge_y = edge_coords[:, :, 1].flatten()
    edge_z = edge_coords[:, :, 2].flatten()

    # found somewhere, not sure why write like this
    insert_indices = np.arange(2, edge_x.size + 1, 2)
    edge_x = np.insert(edge_x, insert_indices, None)
    edge_y = np.insert(edge_y, insert_indices, None)
    edge_z = np.insert(edge_z, insert_indices, None)

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='black', width=1), hoverinfo='none')
    edge_trace.name = 'edges'
    fig.add_trace(edge_trace)

    # add objects
    objects_for_df = [[obj.name + ' center',
                       obj.x, obj.y, obj.z,
                       ] 
                       for obj in objects]
    obj_df = pd.DataFrame(objects_for_df, columns=['name', 'x', 'y', 'z'])
    
    dict_name_to_magenta = {obj.name + ' center': f'rgb(255,0,255)' for obj in objects}

    obj_points = px.scatter_3d(obj_df, x='x', y='y', z='z', color='name',
                    color_discrete_map=dict_name_to_magenta,
                    hover_data={'x': True, 'y': True, 'z': True, 'name': True},
                    )
    
    for trace in obj_points.data:
        fig.add_trace(trace)


    # add relationships edges between object centers
    edge_x = []
    edge_y = []
    edge_z = []
    midpoints = []
    midpoint_texts = []

    for i, obj1 in enumerate(objects):
        for j in range(i + 1, len(objects)):
            if edge_relationships[i][j] == "":
                continue
            obj2 = objects[j]
            x0, y0, z0 = obj1.x, obj1.y, obj1.z
            x1, y1, z1 = obj2.x, obj2.y, obj2.z

            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]
            
            text = f"{obj1.name} {edge_relationships[i][j]} {obj2.name} <|> {obj2.name} {edge_relationships[j][i]} {obj1.name}"
            
            mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
            midpoints.append((mx, my, mz))
            midpoint_texts.append(text)

    # Add relationship edges for visual orientation
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, 
        mode='lines', 
        line=dict(color='red', width=4), 
    )
    edge_trace.name = 'relationships'
    fig.add_trace(edge_trace)

    # Add actual relationship text in middle
    if midpoints:
        midpoint_x, midpoint_y, midpoint_z = zip(*midpoints)
        midpoint_trace = go.Scatter3d(
            x=midpoint_x, y=midpoint_y, z=midpoint_z,
            mode='markers',
            marker=dict(size=10, color='red'),
            text=midpoint_texts,
            textposition='top center',
            hoverinfo='text'
        )
        midpoint_trace.name = 'relationships'
        fig.add_trace(midpoint_trace)

    # somehow keeping all axis spacings the same
    fig.update_layout(title=name, legend=dict(itemsizing='constant'))  
    fig.update_scenes(aspectratio=dict(x=(df['x'].max() - df['x'].min()) / 2, 
                                    y=(df['y'].max() - df['y'].min()) / 2, 
                                    z=(df['z'].max() - df['z'].min()) / 2
                                    ),
                    xaxis_autorange=False, 
                    yaxis_autorange=False, 
                    zaxis_autorange=False,
                    xaxis_range=[df['x'].min(), df['x'].max()], 
                    yaxis_range=[df['y'].min(), df['y'].max()], 
                    zaxis_range=[df['z'].min(), df['z'].max()]
                    )

    fig.update_traces(marker=dict(size=3.5))
    
    return fig

# debugging
if __name__ == '__main__':
    name = "/teamspace/studios/this_studio/3dscenegraph/output/small_living_room/plot/small_living_room_pointcloud_classes"
    fig = plot_labeled_pointcloud(name)
    fig.write_html("/teamspace/studios/this_studio/plot_labeled_pointcloud.html")