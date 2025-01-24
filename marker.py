from aiml_virtual.simulated_object.simulated_object import SimulatedObject
from aiml_virtual.mocap.mocap_source import MocapSource
from aiml_virtual.utils import utils_general
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import hooked_bumblebee
from aiml_virtual.simulated_object.mocap_object.mocap_object import MocapObject
from xml.etree import ElementTree as ET
from typing import Optional

class Marker(MocapObject):
    """
    Mocap object to display a piece of paper with the parking lot sign on it.
    """

    def __init__(self, N = 10):
        self.N = N
        self.name = "mpcc_marker"
        self.update_frequency = 100
        self.source = None
    @classmethod
    def get_identifier(cls) -> Optional[str]:
        return "marker"

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:
        markers = []
        size= "0.1"
        ret = {"worldbody": [],
                "contact": []}
        for i in range(self.N):
            name = f"mpcc_{i}"
            x,y,z = 0+i*1, 0, 1

            markers.append(ET.Element("body", name = name,mocap = "true", pos = f"{x} {y} {z}", quat = quat))

            m = ET.SubElement(markers[i], "geom", name = f"mpcc_marker{i}", type = "sphere",contype="0" ,conaffinity="0", size = f"{size}", pos = "0 0 0", rgba = color)
            #joint = ET.SubElement(markers[i], "joint", type = "free", name = f"marker_{i}_free_joint")
            #markers.append(ET.SubElement(marker_root, "geom", type = "sphere", size = ".05", pos = f"{x} {y} {z}", rgba = color))

            ret["contact"].append(ET.Element("exclude",name= f"f{i}" ,body1 = "Car_0",body2=  name))
        #ET.SubElement(self.contact, "exclude",name = f"marker_exclude{self.trajectory_markers}", body1= "Fleet1Tenth_0",body2=  name)

        #body = ET.Element("body", name=self.name, pos=pos, quat=quat, mocap="true")
        #ET.SubElement(body, "geom", name=self.name, type="plane", pos="0 0 .05",
        #              size="0.105 0.105 .05", material="mat-parking_lot")
        ret["worldbody"] = [*markers]
        return ret