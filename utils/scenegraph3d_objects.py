
class Objects:
    def __init__(self,
                    name: str,
                    object_id: int, # index in collection of objects from scene
                    class_id: int,  # class id from metadata of Mask2Former
                    x: float,
                    y: float, 
                    z: float,
                    size_x: float = 0.0,
                    size_y: float = 0.0,
                    size_z: float = 0.0,
                    index_set: list = [], # list of indices of pixels in the image that belong to this object 
                    neighbors: list = [], # set of object_ids that touch this object
                    relations: list = [], # was not filled in objects at the end
                    best_perspective_frame: list = None):
        self.name = str(name)
        self.object_id = int(object_id)
        self.class_id = int(class_id)
        self.index_set = [int(i) for i in index_set]
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.size_z = float(size_z)
        self.neighbors = [int(i) for i in neighbors]
        self.relations = [str(i) for i in relations]
        self.best_perspective_frame = best_perspective_frame if best_perspective_frame is not None else None
