import math

class BoundingBoxList:
    """
    BoundingBoxList. Class that defines all the possible grasp locations
    for an image. grasp locations are defined as a rectangle of 4 points
    Arguments:
        bbox_list (list, string): a list of strings, where each string is
            has the box coordinates
    """

    def __init__(self, bbox_list):
        irecs = [
            [self.get_coord(bbox_list[i]),
            self.get_coord(bbox_list[i+1]),
            self.get_coord(bbox_list[i+2]),
            self.get_coord(bbox_list[i+3])]
                for i in range(0, len(bbox_list), 4)
        ]
        self.irecs = irecs

    def get_coord(self, f):
        """
        given a string containing coordinates, convert them to
            a tuple of floats. The angle is calculated as an offset
            from a vertical (either oriented up or down, it does not
            matter so much) normalized vector, using cosine similarity.
        """

        ln = f.split()
        return (float(ln[0]), float(ln[1]))
