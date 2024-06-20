from abc import ABC, abstractmethod

class GeometricShape(ABC):

    @abstractmethod
    def point_relative_center_position(self, point):
        """Calculates the relative position from the shape to a point"""

    @abstractmethod
    def is_inside(self,point):
        """Determine whether a point is inside the shape"""

    @abstractmethod
    def distance_object_to_point(self, point):
        """Calculates the relative position from the shape to a point"""

    # @abstractmethod
    # def report_state(self):
    #     """report state for being part of RL state"""


class Geometric3D(GeometricShape):
    @abstractmethod
    def point_relative_center_position(self, point, direction):
        """Calculates the relative position from the shape to a point"""

class Geometric2D(GeometricShape):
    pass