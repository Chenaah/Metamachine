"""
Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from pathlib import Path

import numpy as np
from lxml import etree

from metamachine import METAMACHINE_ROOT_DIR, get_asset_path
from metamachine.utils.visual_utils import vec2string


def fix_model_file_path(root):
    try:
        parts_dir = get_asset_path("parts")
    except (FileNotFoundError, ImportError):
        # Fallback to old method
        parts_dir = os.path.join(METAMACHINE_ROOT_DIR, "assets", "parts")

    mesh_files = [file.name for file in Path(parts_dir).rglob("*") if file.is_file()]
    mesh_file_paths = [
        str(file.resolve()) for file in Path(parts_dir).rglob("*") if file.is_file()
    ]
    for mesh in root.findall(".//mesh") + root.findall(".//hfield"):
        if mesh.get("file") in mesh_files:
            mesh.set("file", mesh_file_paths[mesh_files.index(mesh.get("file"))])
            # print(f"Fixed mesh file path: {mesh.get('file')}")
    return root


def position_to_torque_control(root):
    for position in root.findall(".//position"):
        position.tag = "motor"
        del position.attrib["kp"]
        del position.attrib["kv"]
        position.attrib["ctrlrange"] = position.attrib["forcerange"]
        del position.attrib["forcerange"]
    return root


def torque_to_position_control(root, kp=20, kd=0.5):
    for motor in root.findall(".//motor"):
        motor.tag = "position"
        motor.attrib["kp"] = f"{kp}"
        motor.attrib["kv"] = f"{kd}"
        motor.attrib["forcerange"] = motor.attrib["ctrlrange"]
        del motor.attrib["ctrlrange"]
    return root


def update_xml_timestep(root, timestep) -> None:
    option = root.xpath('//option[@integrator="RK4"]')[0]
    option.attrib["timestep"] = f"{timestep}"
    return root


class XMLCompiler:
    def __init__(self, xml_file) -> None:
        parser = etree.XMLParser(remove_blank_text=True)
        if xml_file.endswith(".xml"):
            if not os.path.exists(xml_file):
                xml_file = os.path.join(
                    METAMACHINE_ROOT_DIR, "assets", "robots", xml_file
                )
            tree = etree.parse(xml_file, parser)
            self.root = tree.getroot()
        else:
            self.root = etree.fromstring(xml_file, parser)

        # Fix mesh file path
        # Fix mesh file paths for compatibility
        self.root = fix_model_file_path(self.root)

    def torque_control(self):
        self.root = position_to_torque_control(self.root)

    def position_control(self, kp=20, kd=0.5):
        self.root = torque_to_position_control(self.root, kp=20, kd=0.5)

    def pyramidal_cone(self):
        option_element = self.root.find(".//option")
        if option_element is not None:
            option_element.set("cone", "pyramidal")

    def update_timestep(self, timestep) -> None:
        self.root = update_xml_timestep(self.root, timestep)

    def get_string(self):
        xml_string = etree.tostring(
            self.root, pretty_print=True, xml_declaration=False, encoding="utf-8"
        ).decode()
        return xml_string

    def save(self, file):
        tree = etree.ElementTree(self.root)
        tree.write(file, pretty_print=True, xml_declaration=False, encoding="utf-8")

    def get_mass_range(self, percentage=0.1):
        """Get mass range for all geoms with name and mass attributes.
        
        Args:
            percentage: Percentage variation for mass range (default 0.1 = ±10%)
            
        Returns:
            dict: {geom_name: [min_mass, max_mass]} for all geoms with mass
        """
        mass_range = {}
        # Find all geoms that have both name and mass attributes
        for geom in self.root.xpath('//geom[@name and @mass]'):
            name = geom.get('name')
            mass = float(geom.get('mass'))
            mass_range[name] = [
                mass * (1 - percentage),
                mass * (1 + percentage),
            ]
        return mass_range

    def update_mass(self, mass_dict) -> None:
        """Update mass for geoms with exact name match.
        
        Args:
            mass_dict: {geom_name: mass} dictionary
        """
        for name, mass in mass_dict.items():
            geoms = self.root.xpath(f'//geom[@name="{name}"]')
            if geoms:
                geoms[0].set("mass", str(mass))

    def update_damping(self, armature, damping) -> None:
        # joints = self.root.find('default').findall('joint')
        # for joint in joints:
        #     joint.set('armature', str(armature))
        #     joint.set('damping', str(damping))
        for joint in self.root.xpath('//joint[starts-with(@name, "joint")]'):
            joint.set("armature", str(armature))
            joint.set("damping", str(damping))
            # print(f"Updated damping and armature of {joint.get('name')} to {damping} and {armature}")

    def add_walls(self, transparent=False, angle=0):
        world_body = self.root.findall("./worldbody")[0]
        wall = etree.SubElement(
            world_body,
            "body",
            name="boundary",
            pos="0 0 0",
            axisangle=f"0 0 1 {angle}",
        )
        etree.SubElement(
            wall,
            "geom",
            name="boundary/right",
            pos="0 1 0.25",
            type="box",
            material="boundary",
            size="25 0.1 0.5",
            **({} if not transparent else {"rgba": "0.1 0.1 0.1 0.0"}),
        )
        etree.SubElement(
            wall,
            "geom",
            name="boundary/left",
            pos="0 -1 0.25",
            type="box",
            material="boundary",
            size="25 0.1 0.5",
            **({} if not transparent else {"rgba": "0.1 0.1 0.1 0.0"}),
        )

    def remove_walls(self):
        world_body = self.root.findall("./worldbody")[0]
        for wall in world_body.findall("./body[@name='boundary']"):
            world_body.remove(wall)

    def reset_hfield(
        self, radius_x, radius_y, elevation_z, base_z, hfield_file=None, size=None
    ):
        """
        If hfield_file is None, model.hfield_data should be provided.
        """
        if hfield_file is not None:
            try:
                hfield_file = get_asset_path("parts", hfield_file)
            except (FileNotFoundError, ImportError):
                hfield_file = os.path.join(
                    METAMACHINE_ROOT_DIR, "assets", "parts", hfield_file
                )
        hfields = self.root.findall(".//hfield")
        for hfield in hfields:
            if hfield_file is not None:
                hfield.set("file", hfield_file)
            else:
                hfield.attrib.pop("file", None)
            hfield.set("size", f"{radius_x} {radius_y} {elevation_z} {base_z}")
        if not hfields:
            world_body = self.root.findall("./worldbody")[0]
            floor_elem = world_body.find("geom[@name='floor']")
            assert floor_elem is not None, "No floor element found in the XML file."
            world_body.remove(floor_elem)

            assets = self.root.findall(".//asset")[0]
            etree.SubElement(
                world_body,
                "geom",
                name="floor",
                pos="0 0 0",
                type="hfield",
                material="hfield",
                conaffinity="1",
                condim="6",
                friction="1.0 .0 .0",
                hfield="rugged",
            )
            if hfield_file is not None:
                etree.SubElement(
                    assets,
                    "hfield",
                    name="rugged",
                    size=f"{radius_x} {radius_y} {elevation_z} {base_z}",
                    file=hfield_file,
                )
            else:
                assert size is not None, "Size of the hfield is not provided."
                etree.SubElement(
                    assets,
                    "hfield",
                    name="rugged",
                    size=f"{radius_x} {radius_y} {elevation_z} {base_z}",
                    nrow=f"{size}",
                    ncol=f"{size}",
                )

    def reset_obstacles(self, terrain_params) -> None:
        world_body = self.root.findall("./worldbody")[0]
        elements = [
            geom
            for geom in world_body.findall("./geom")
            if geom.get("name", "").startswith("obstacle")
        ]
        # assert elements, "No obstacle found in the old model."
        if not elements:
            num_bumps = (
                terrain_params["num_bumps"] if "num_bumps" in terrain_params else 200
            )
            for i in range(num_bumps):
                pos = np.random.uniform(-5, 5, 2)
                angle = np.random.uniform(0, 360)
                height = np.random.uniform(*terrain_params["height_range"])
                width = np.random.uniform(*terrain_params["width_range"])
                # self.builder.add_minibump(pos=pos, angle=angle, height=height, width=width, length=1)
                length = 1
                etree.SubElement(
                    world_body,
                    "geom",
                    name=f"obstacle{i}",
                    pos=f"{pos[0]} {pos[1]} {height/2}",
                    axisangle=f"0 0 1 {angle}",
                    type="box",
                    material="boundary",
                    size=f"{width/2} {length/2} {height/2}",
                )
        else:
            for obstacle in elements:
                pos = np.random.uniform(-5, 5, 2)
                angle = np.random.uniform(0, 360)
                height = np.random.uniform(*terrain_params["height_range"])
                width = np.random.uniform(*terrain_params["width_range"])
                length = 1

                pos = f"{pos[0]} {pos[1]} {height/2}"
                axisangle = f"0 0 1 {angle}"
                size = f"{width/2} {length/2} {height/2}"

                obstacle.set("pos", pos)
                obstacle.set("axisangle", axisangle)
                obstacle.set("size", size)

    def update_mesh(self, mesh_dict, robot_cfg=None) -> None:
        # This function is deprecated. 
        raise DeprecationWarning("This function is deprecated. Please use the new mesh system.")
        # Update all the geom mesh in the model
        world_body = self.root.findall("./worldbody")[0]
        assets = self.root.findall(".//asset")[0]

        # Import the mesh files
        for key, mesh_file in mesh_dict.items():
            if mesh_file.endswith(".obj") or mesh_file.endswith(".stl"):
                meshes = assets.findall(f"./mesh[@name='{key}']")
                for mesh in meshes:
                    assets.remove(mesh)
                etree.SubElement(
                    assets, "mesh", file=mesh_file, name=key, scale="1 1 1"
                )

        if "up" in mesh_dict:
            lefts = [
                geom
                for geom in world_body.findall(".//geom")
                if geom.get("name", "").startswith("left")
            ]
            for left_geom in lefts:
                name = left_geom.get("name")
                color = left_geom.get("rgba")
                mass = left_geom.get("mass")
                parent = left_geom.getparent()
                parent.remove(left_geom)  # Remove the old geom from its parent
                if mesh_dict["up"].endswith(".obj") or mesh_dict["up"].endswith(".stl"):
                    etree.SubElement(
                        parent,
                        "geom",
                        type="mesh",
                        name=name,
                        mesh="up",
                        rgba=color,
                        mass=mass,
                        material="metallic",
                        friction="1.0 .0 .0",
                        priority="2",
                    )
                elif mesh_dict["up"] == "SPHERE":
                    # Draft mode: Use sphere instead of mesh
                    assert robot_cfg is not None, "Robot configuration is not provided."
                    radius = robot_cfg["R"]
                    etree.SubElement(
                        parent,
                        "geom",
                        type="sphere",
                        name=name,
                        size=f"{radius}",
                        rgba=color,
                        mass=mass,
                        friction="1.0 .0 .0",
                        priority="2",
                    )
                else:
                    raise ValueError(
                        "The mesh should be either a .obj file or a SPHERE"
                    )

        if "bottom" in mesh_dict:
            rights = [
                geom
                for geom in world_body.findall(".//geom")
                if geom.get("name", "").startswith("right")
            ]
            for right_geom in rights:
                name = right_geom.get("name")
                color = right_geom.get("rgba")
                mass = right_geom.get("mass")
                parent = right_geom.getparent()
                parent.remove(right_geom)

                if mesh_dict["bottom"].endswith(".obj") or mesh_dict["bottom"].endswith(
                    ".stl"
                ):
                    etree.SubElement(
                        parent,
                        "geom",
                        type="mesh",
                        name=name,
                        mesh="bottom",
                        rgba=color,
                        mass=mass,
                        material="metallic",
                        friction="1.0 .0 .0",
                        priority="2",
                    )
                elif mesh_dict["bottom"] == "SPHERE":
                    assert robot_cfg is not None, "Robot configuration is not provided."
                    radius = robot_cfg["R"]
                    etree.SubElement(
                        parent,
                        "geom",
                        type="sphere",
                        name=name,
                        size=f"{radius}",
                        rgba=color,
                        mass=mass,
                        friction="1.0 .0 .0",
                        priority="2",
                    )
                else:
                    raise ValueError(
                        "The mesh should be either a .obj file or a SPHERE"
                    )

        if "battery" in mesh_dict:
            batteries = [
                geom
                for geom in world_body.findall(".//geom")
                if geom.get("name", "").startswith("battery")
            ]
            assert batteries, "No battery found in the old model."  # TODO
            for battery in batteries:
                name = battery.get("name")
                color = battery.get("rgba")
                mass = battery.get("mass")
                parent = battery.getparent()
                parent.remove(battery)

                if mesh_dict["battery"].endswith(".obj") or mesh_dict[
                    "battery"
                ].endswith(".stl"):
                    etree.SubElement(
                        parent,
                        "geom",
                        type="mesh",
                        name=name,
                        mesh="battery",
                        rgba=color,
                        mass=mass,
                        material="metallic",
                        contype="10",
                        conaffinity="0",
                    )
                elif mesh_dict["battery"] == "NONE":
                    pass
                else:
                    raise ValueError("The mesh should be either a .obj file or NONE")

        if "pcb" in mesh_dict:
            pcbs = [
                geom
                for geom in world_body.findall(".//geom")
                if geom.get("name", "").startswith("pcb")
            ]
            assert pcbs, "No PCB found in the old model."  # TODO
            for pcb in pcbs:
                name = pcb.get("name")
                color = pcb.get("rgba")
                mass = pcb.get("mass")
                parent = pcb.getparent()
                parent.remove(pcb)

                if mesh_dict["pcb"].endswith(".obj") or mesh_dict["pcb"].endswith(
                    ".stl"
                ):
                    etree.SubElement(
                        parent,
                        "geom",
                        type="mesh",
                        name=name,
                        mesh="pcb",
                        rgba="0 0 0 0.5",
                        mass=mass,
                        material="metallic",
                        contype="10",
                        conaffinity="0",
                    )
                elif mesh_dict["pcb"] == "NONE":
                    pass
                else:
                    raise ValueError("The mesh should be either a .obj file or NONE")

        if "motor" in mesh_dict:
            motors = [
                geom
                for geom in world_body.findall(".//geom")
                if geom.get("name", "").startswith("motor")
            ]
            assert motors, "No motor found in the old model."
            for motor in motors:
                name = motor.get("name")
                color = motor.get("rgba")
                mass = motor.get("mass")
                parent = motor.getparent()
                parent.remove(motor)

                if mesh_dict["motor"].endswith(".obj") or mesh_dict["motor"].endswith(
                    ".stl"
                ):
                    etree.SubElement(
                        parent,
                        "geom",
                        type="mesh",
                        name=name,
                        mesh="motor",
                        rgba="1 0 0 0.5",
                        mass=mass,
                        contype="10",
                        conaffinity="0",
                    )
                elif mesh_dict["motor"] == "CYLINDER":
                    etree.SubElement(
                        parent,
                        "geom",
                        type="cylinder",
                        name=name,
                        pos=vec2string([0, 0, -0.015]),
                        quat=vec2string([1, 0, 0, 0]),
                        size=f"{0.05} {0.03/2}",
                        rgba=color,
                        mass=mass,
                        contype="10",
                        conaffinity="0",
                    )
                elif mesh_dict["motor"] == "NONE":
                    pass
                else:
                    raise ValueError(
                        "The mesh should be either a .obj file or CYLINDER or NONE"
                    )

        if "stick" in mesh_dict:
            sticks = [
                geom
                for geom in world_body.findall(".//geom")
                if geom.get("name", "").startswith("stick")
            ]
            assert sticks, "No stick found in the old model."
            for stick in sticks:
                name = stick.get("name")
                color = stick.get("rgba")
                mass = stick.get("mass")
                parent = stick.getparent()
                parent.remove(stick)

                if mesh_dict["stick"].endswith(".obj") or mesh_dict["stick"].endswith(
                    ".stl"
                ):
                    etree.SubElement(
                        parent,
                        "geom",
                        name=name,
                        type="mesh",
                        pos="0 0 0",
                        quat="1 0 0 0",
                        mesh="stick",
                        rgba=color,
                        mass=mass,
                        friction="1.0 .0 .0",
                        priority="2",
                    )
                elif mesh_dict["stick"] == "CYLINDER":
                    assert robot_cfg is not None, "Robot configuration is not provided."
                    radius = robot_cfg["r"]
                    length = robot_cfg["l_"]
                    broken = 0
                    etree.SubElement(
                        parent,
                        "geom",
                        name=name,
                        type="cylinder",
                        pos="0 0 0",
                        quat="1 0 0 0",
                        size=f"{radius} {length/2 *(1-broken)}",
                        rgba=color,
                        mass=mass,
                        friction="1.0 .0 .0",
                        priority="2",
                    )
                elif mesh_dict["stick"] == "CAPSULE":
                    assert robot_cfg is not None, "Robot configuration is not provided."
                    radius = robot_cfg["r"]
                    length = robot_cfg["l_"]
                    broken = 0
                    etree.SubElement(
                        parent,
                        "geom",
                        name=name,
                        type="capsule",
                        pos="0 0 0",
                        quat="1 0 0 0",
                        size=f"{radius} {length/2 *(1-broken)}",
                        rgba=color,
                        mass=mass,
                        friction="1.0 .0 .0",
                        priority="2",
                    )
                else:
                    raise ValueError(
                        "The mesh should be either a .obj file or CYLINDER or CAPSULE"
                    )

    def recolor_floor(self, color, mark_color=".8 .8 .8"):
        floor = self.root.xpath('//texture[@name="texplane"]')[0]
        floor.set("rgb1", color[0])
        floor.set("rgb2", color[1])
        floor.set("markrgb", mark_color)

    def recolor_sky(self, color):
        sky = self.root.xpath('//texture[@type="skybox"]')[0]
        sky.set("rgb1", color[0])
        sky.set("rgb2", color[1])

    def remove_floor(self):
        world_body = self.root.findall("./worldbody")[0]
        floor_elem = world_body.find("geom[@name='floor']")
        assert floor_elem is not None, "No floor element found in the XML file."
        world_body.remove(floor_elem)

    def recolor_robot(self, colors, sphere_only=False):
        # lefts = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith("left")]
        # lefts = [body for body in self.root.findall(".//body") if body.get("name", "").startswith("l")]
        for i in range(len(colors)):
            left_geom = [
                geom
                for geom in self.root.findall(".//geom")
                if geom.get("name", "").startswith(f"left{i}")
            ]
            right_geom = [
                geom
                for geom in self.root.findall(".//geom")
                if geom.get("name", "").startswith(f"right{i}")
            ]
            stick_geom1 = [
                geom
                for geom in self.root.findall(".//geom")
                if geom.get("name", "").startswith(f"stick{i*2}")
            ]
            stick_geom2 = [
                geom
                for geom in self.root.findall(".//geom")
                if geom.get("name", "").startswith(f"stick{i*2+1}")
            ]
            battery_geom = [
                geom
                for geom in self.root.findall(".//geom")
                if geom.get("name", "").startswith(f"battery{i}")
            ]
            pcb_geom = [
                geom
                for geom in self.root.findall(".//geom")
                if geom.get("name", "").startswith(f"pcb{i}")
            ]
            motor_geom = [
                geom
                for geom in self.root.findall(".//geom")
                if geom.get("name", "").startswith(f"motor{i}")
            ]
            imu_site = [
                geom
                for geom in self.root.findall(".//site")
                if geom.get("name", "").startswith(f"imu_site{i}")
            ]
            back_imu_site = [
                geom
                for geom in self.root.findall(".//site")
                if geom.get("name", "").startswith(f"back_imu_site{i}")
            ]
            if left_geom:
                left_geom[0].set("rgba", colors[i])
            if right_geom:
                right_geom[0].set("rgba", colors[i])
            if battery_geom:
                battery_geom[0].set("rgba", colors[i])
            if pcb_geom:
                pcb_geom[0].set("rgba", colors[i])
            if motor_geom:
                motor_geom[0].set("rgba", colors[i])
            if imu_site:
                imu_site[0].set("rgba", colors[i])
            if back_imu_site:
                back_imu_site[0].set("rgba", colors[i])
            if not sphere_only:
                if stick_geom1:
                    stick_geom1[0].set("rgba", colors[i])
                if stick_geom2:
                    stick_geom2[0].set("rgba", colors[i])
        # for right_geom, color in zip(rights, colors):
        #     right_geom.set("rgba", color)
        # sticks = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith("stick")]
        # for i, stick_geom in enumerate(sticks):
        #     stick_geom.set("rgba", colors[int(i/2)])

    def remove_shadow(self):
        quality = self.root.find(".//visual/quality")
        if quality is not None:
            # Set the shadowsize attribute to "0"
            quality.set("shadowsize", "0")
        else:
            # If the <quality> element doesn't exist, create it under <visual>
            visual = self.root.find(".//visual")
            if visual is None:
                # If there's no <visual> element, create one
                visual = etree.SubElement(self.root, "visual")
            quality = etree.SubElement(visual, "quality")
            quality.set("shadowsize", "0")
