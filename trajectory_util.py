from aiml_virtual.simulated_object.mocap_object.mocap_object import MocapObject
from typing import Optional, cast, Type
import xml.etree.ElementTree as ET
import numpy as np


class Trajectory_Marker(MocapObject):
    """
    Mocap object to display a piece of paper with the airport sign on it.
    """

    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "trajectory_marker"
    def __init__(self, source = None, mocap_name = None, x = [0], y = [0]):
        super().__init__(source, mocap_name)
        self.x = x
        self.y = y
    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")


        object = ET.Element("body", name="marker")
        n = np.shape(self.x)[0]
        for i in range(n):
            ET.SubElement(object,"geom", type = "sphere", name = f"_marker{i}", contype="0" ,conaffinity="0", pos = f"{self.x[i]} {self.y[i]} 0.02", size = f"{0.05}")
        ret = {"worldbody" : [object]}
        return ret
