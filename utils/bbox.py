import math

class BoundingBoxList:
    """
    BoundingBoxList. Class that defines all the possible grasp locations
    for a given image. The grasps are defined as an Nx3 tensor (x, y, a).

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
        ibboxes = [self.get_tuple(rec) for rec in irecs]
        self.ibboxes = ibboxes

    def get_coord(self, f):
        """
        given a string containing coordinates, convert them to
            a tuple of floats
        """

        ln = f.split()
        return (float(ln[0]), float(ln[1]))

    def get_tuple(self, rec):
        """
        given a rectangle defined by its four verticies, extract the
        tuple (x, y, a)
        """

        x = float(sum([point[0] for point in rec]))/4
        y = float(sum([point[1] for point in rec]))/4
        if rec[0][0] < rec[1][0]:
            xhat = rec[1][0] - rec[0][0]
            yhat = rec[1][1] - rec[0][1]
        else:
            xhat = rec[0][0] - rec[1][0]
            yhat = rec[0][1] - rec[1][1]
        ang = math.atan2(-1*yhat, xhat)
        return (x, y, ang)
