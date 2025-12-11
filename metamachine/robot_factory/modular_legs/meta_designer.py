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

import copy
import random

import numpy as np

from metamachine.utils.validation import is_list_like

from .constants import MESH_DICT_FINE, ROBOT_CFG_AIR1S
from .designer import RobotDesigner
from .morphology import DockPosition, ModuleConnection, RobotMorphology


class ModularLegs:
    """
    A meta-designer for asymmetric robot designs that matches real robot configurations.

    This class provides functionality to design and construct asymmetric robot configurations
    by managing the placement and connection of balls and sticks in a modular fashion.

    Args:
        morphology (RobotMorphology, optional): Initial robot morphology. Defaults to None.
        robot_cfg (dict): Robot configuration parameters. Defaults to ROBOT_CFG_AIR1S.
        mesh_dict (dict): Dictionary containing mesh configurations. Defaults to MESH_DICT_FINE.
        allow_overlapping (bool): Whether to allow overlapping components. Defaults to False.
        color (list, optional): RGB color values for components. Defaults to [0.6, 0.6, 0.6].
        broken_mask (list, optional): Mask indicating broken components. Defaults to None.
        mesh_mode (str, optional): Mode for mesh rendering. Defaults to None.
    """

    def __init__(
        self,
        morphology=None,
        robot_cfg=ROBOT_CFG_AIR1S,
        mesh_dict=MESH_DICT_FINE,
        broken_mask=None,
        sim_cfg=None,
        log_dir=None,
    ):
        """Initialize the ModularLegs instance."""
        self.robot_cfg = robot_cfg.copy()
        self.robot_cfg["length"] = robot_cfg["l_"] - (
            robot_cfg["R"] - np.sqrt(robot_cfg["R"] ** 2 - robot_cfg["r"] ** 2)
        )
        self.mesh_dict = mesh_dict
        self.broken_mask = (
            copy.deepcopy(broken_mask) + [0] * 100000
            if broken_mask is not None
            else [0] * 100000
        )
        self.sim_cfg = sim_cfg
        self.log_dir = log_dir if log_dir is not None else "debug"

        # Convert legacy pipeline to RobotMorphology if needed
        if is_list_like(morphology):
            morphology = RobotMorphology.from_sequence(morphology)

        if morphology is not None:
            self.reset()
            for connection in morphology.connections:
                self.add_module(connection)

    def reset(self, init_a_module=True) -> None:
        """Reset the designer to initial state.

        Args:
            init_a_module (bool): Whether to initialize with a single module. Defaults to True.
        """
        self.designer = RobotDesigner(
            init_pipeline=None,
            robot_cfg=self.robot_cfg,
            mesh_dict=self.mesh_dict,
            allow_overlapping=False,
            sim_cfg=self.sim_cfg,
        )
        self.designer.reset(init_a_ball=False)

        self.module_id_dict = {}
        self.module_id_counter = 0
        self.available_pos_dict = {}  # {module_id: [pos_id]}
        self.morphology = RobotMorphology()

        if init_a_module:
            self._init_a_module()

    def _init_a_module(self, broken=None) -> None:
        """Initialize a basic module consisting of a ball and two sticks.

        Args:
            broken (list, optional): List indicating which components are broken. Defaults to None.
        """
        node_ids = []
        node_ids += self.designer._add_a_ball()
        node_ids += self.designer._add_a_stick(
            node_ids[0],
            0,
            0,
            0,
            broken=self.broken_mask.pop(0) if broken is None else broken[0],
        )
        node_ids += self.designer._add_a_stick(
            node_ids[1],
            0,
            0,
            0,
            broken=self.broken_mask.pop(0) if broken is None else broken[1],
        )

        self.module_id_dict[self.module_id_counter] = node_ids
        self.available_pos_dict[self.module_id_counter] = list(range(18))
        self.module_id_counter += 1

    def _init_a_right_stick_module(self, broken=None) -> None:
        node_ids = []
        node_ids += self.designer._add_a_ball()
        node_ids += self.designer._add_a_stick(
            node_ids[0],
            0,
            0,
            0,
            broken=self.broken_mask.pop(0) if broken is None else broken[0],
        )
        node_ids += self.designer._add_a_stick(
            node_ids[1],
            0,
            0,
            0,
            broken=self.broken_mask.pop(0) if broken is None else broken[1],
        )

        self.module_id_dict[self.module_id_counter] = node_ids
        self.available_pos_dict[self.module_id_counter] = list(range(18))

        self.designer.builder.delete_body(f"l{self.module_id_counter}")
        self.designer.builder.delete_body(
            f"passive{self.designer.builder.passive_idx_counter-2}"
        )
        self.designer.builder.delete_body(
            f"passive{self.designer.builder.passive_idx_counter-2}"
        )
        self.designer.builder.delete_sensor(f"imu_quat{self.module_id_counter}")
        self.designer.builder.delete_sensor(f"imu_gyro{self.module_id_counter}")
        self.designer.builder.delete_sensor(f"imu_globvel{self.module_id_counter}")
        self.designer.builder.delete_sensor(f"imu_vel{self.module_id_counter}")
        self.designer.builder.delete_joint(f"joint{self.module_id_counter}")
        self.designer.builder.hind_imu(f"imu_site{self.module_id_counter}")

        self.module_id_counter += 1

    def _add_a_module(self, module_id, pos_a, pos_b, rotation):
        """
        Position [A] on a module:
        0-6   stick on upper ball ; pos 7-1 of the stick
        7-8  upper ball           ; pos 1-2 of the ball
        9-10 lower ball           ; pos 1-2 of the ball
        11-17 stick on lower ball ; pos 1-7 of the stick
        Position [B] on a module:
        0-6   stick on upper ball ; pos 7-1 of the stick
        7-8  upper ball           ; pos 1-2 of the ball
        """
        assert pos_a >= 0 and pos_a < 18, f"Invalid module position: {pos_a}"
        assert pos_b >= 0 and pos_b < 9, f"Invalid module position: {pos_b}"

        # print(f"Adding a module: {module_id}, {pos_a}, {pos_b}, {rotation}")

        # if module_id not in self.module_id_dict:
        #     print(f"Invalid module id: {module_id}; Available module ids: {self.module_id_dict.keys()}")
        #     pdb.set_trace()
        # if pos_a not in self.available_pos_dict[module_id]:
        #     print(f"Invalid module position: {pos_a}; Available positions: {self.available_pos_dict[module_id]}")
        #     pdb.set_trace()
        # if pos_b not in list(range(9)):
        #     print(f"Invalid module position: {pos_b}; Available positions: {list(range(9))}")
        #     pdb.set_trace()
        # if rotation not in self.get_available_rotation_ids(pos_a, pos_b):
        #     print(f"Invalid rotation: {rotation}; Available rotations: {self.get_available_rotation_ids(pos_a, pos_b)}")
        #     pdb.set_trace()

        parent_id = self._module_id_to_parent_id(module_id, pos_a)
        position_id_a = self._module_pos_to_local_pos(pos_a)
        position_id_b = self._module_pos_to_local_pos(pos_b)

        if pos_b <= 6:
            # stick -> module
            ls_id = self.designer._add_a_stick(
                parent_id,
                position_id_a=position_id_a,
                position_id_b=position_id_b,
                rotation_id=rotation,
                broken=self.broken_mask.pop(0),
            )
            ball_ids = self.designer._add_a_ball(ls_id[0], 0, 0, 0)
            # We hardcode the stick orientation here to simplify the encoding
            rs_id = self.designer._add_a_stick(
                ball_ids[1], 0, 0, 0, broken=self.broken_mask.pop(0)
            )
        else:
            # ball -> module
            ball_ids = self.designer._add_a_ball(
                parent_id,
                position_id_a=position_id_a,
                position_id_b=position_id_b,
                rotation_id=rotation,
            )
            ls_id = self.designer._add_a_stick(
                ball_ids[0], 0, 0, 0, broken=self.broken_mask.pop(0)
            )
            rs_id = self.designer._add_a_stick(
                ball_ids[1], 0, 0, 0, broken=self.broken_mask.pop(0)
            )

        # upper ball, lower ball, upper stick, lower stick
        node_ids = ball_ids + ls_id + rs_id

        self.module_id_dict[self.module_id_counter] = node_ids
        self.available_pos_dict[self.module_id_counter] = list(range(18))

        self.available_pos_dict[module_id].remove(pos_a)
        self.available_pos_dict[self.module_id_counter].remove(pos_b)

        self.module_id_counter += 1

    def add_extra_stick(
        self,
        module_id,
        pos_a,
        pos_b,
        rotation,
        broken=0,
        stick_id=None,
        reserve_next_id=False,
    ):

        parent_id = self._module_id_to_parent_id(module_id, pos_a)
        position_id_a = self._module_pos_to_local_pos(pos_a)
        position_id_b = self._module_pos_to_local_pos(pos_b)

        cut_length = self.robot_cfg["l_"] * (broken)

        pos_offset = np.array([0.0, 0, -cut_length])
        if stick_id is not None:
            self.designer.builder.passive_idx_counter = stick_id
        self.designer._add_a_stick(
            parent_id,
            position_id_a=position_id_a,
            position_id_b=position_id_b,
            rotation_id=rotation,
            broken=broken,
            pos_offset=pos_offset,
        )
        if reserve_next_id:
            self.designer.builder.passive_idx_counter += 1

    def add_extra_ball(self, module_id, pos_a, pos_b, rotation):
        parent_id = self._module_id_to_parent_id(module_id, pos_a)
        position_id_a = self._module_pos_to_local_pos(pos_a)
        position_id_b = self._module_pos_to_local_pos(pos_b)

        self.designer._add_a_ball(
            parent_id,
            position_id_a=position_id_a,
            position_id_b=position_id_b,
            rotation_id=rotation,
            passive=True,
        )

    def _module_id_to_parent_id(self, module_id, module_pos):
        """Convert module ID and position to parent component ID.

        Module position mapping:
        - 0-6:   stick on upper ball (pos 7-1 of stick)
        - 7-8:   upper ball (pos 1-2 of ball)
        - 9-10:  lower ball (pos 1-2 of ball)
        - 11-17: stick on lower ball (pos 1-7 of stick)

        Args:
            module_id (int): ID of the module
            module_pos (int): Position on the module (0-17)

        Returns:
            int: Parent component ID

        Raises:
            AssertionError: If invalid module_id or module_pos is provided
        """
        node_ids = self.module_id_dict[module_id]
        if len(node_ids) != 4:
            raise ValueError(f"Invalid node_ids: {node_ids}. Expected 4 components.")
        if not 0 <= module_pos < 18:
            raise ValueError(
                f"Invalid module position: {module_pos}. Must be between 0 and 17."
            )

        # node_ids order: upper ball, lower ball, upper stick, lower stick
        if module_pos <= 6:
            return node_ids[2]  # upper stick
        elif module_pos <= 8:
            return node_ids[0]  # upper ball
        elif module_pos <= 10:
            return node_ids[1]  # lower ball
        else:
            return node_ids[3]  # lower stick

    def _module_pos_to_local_pos(self, module_pos):
        """Convert module position to local component position.

        Args:
            module_pos (int): Position on the module (0-17)

        Returns:
            int: Local position on the component

        Raises:
            ValueError: If invalid module_pos is provided
        """
        if not 0 <= module_pos < 18:
            raise ValueError(
                f"Invalid module position: {module_pos}. Must be between 0 and 17."
            )

        if module_pos <= 6:
            return 7 - module_pos
        elif module_pos <= 8:
            return module_pos - 6
        elif module_pos <= 10:
            return module_pos - 8
        else:
            return module_pos - 10

    def step(self, step):
        """Execute a single step in the design pipeline.

        Args:
            step (list): List containing [module_id, pos_a, pos_b, rotation]

        Raises:
            ValueError: If step length is not 4
        """
        if len(step) != 4:
            raise ValueError(f"Invalid pipeline step length: {len(step)}. Must be 4.")
        self._add_a_module(*step)

    def get_available_module_ids(self):
        """Get list of available module IDs.

        Returns:
            list: List of module IDs
        """
        return list(self.module_id_dict.keys())

    def get_available_posa_ids(self, module_id):
        """Get available position A IDs for a given module.

        Args:
            module_id (int): Module ID

        Returns:
            list: List of available position A IDs
        """
        return self.morphology.get_available_docks(module_id)
        # return self.available_pos_dict[module_id]

    def get_available_posb_ids(self):
        """Get available position B IDs.

        Returns:
            list: List of available position B IDs (0-8)
        """
        return list(range(9))

    def get_available_rotation_ids(self, posa, posb):
        """Get available rotation IDs for given positions.

        Args:
            posa (int): Position A ID
            posb (int): Position B ID

        Returns:
            list: List of available rotation IDs
        """
        stick_side = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
        if posa in stick_side and posb in stick_side:
            return [0, 1]
        return [0, 1, 2]

    def paint(self, color="black"):
        """Paint the robot components with specified color.

        Args:
            color (str): Color name. Currently supports "black". Defaults to "black".
        """
        dark_grey = (0.15, 0.15, 0.15)
        black = (0.1, 0.1, 0.1)
        if color == "black":
            self.designer.builder.change_color_name("l", black)
            self.designer.builder.change_color_name("r", dark_grey)
            self.designer.builder.change_color_name("s", dark_grey)

    def save(self, save_dir=None):
        """Save the robot design to specified directory.

        Args:
            save_dir (str, optional): Directory to save the design. If None, uses self.log_dir.
        """
        if save_dir is None:
            save_dir = self.log_dir
        self.paint("black")
        self.designer.save(save_dir)

    def get_string(self):
        self.paint("black")
        self.save()  # Will use self.log_dir (defaults to "debug" if not set)
        return self.designer.get_string()

    def render(self, fix=False, pos=(0, 0, 0.4), quat=None, joint_pos=None):
        if quat is None:
            quat = [1, 0, 0, 0]
        self.paint("black")
        self.designer.render(fix, pos, quat, joint_pos)

    def add_module(self, connection: ModuleConnection):
        """Add a new module using the specified connection.

        Args:
            connection: ModuleConnection specifying how to connect the new module
        """
        if not self.morphology.validate_connection(connection):
            raise ValueError("Invalid connection configuration")

        self._add_a_module(
            connection.parent_module_id,
            (
                connection.parent_dock
                if isinstance(connection.parent_dock, int)
                else connection.parent_dock.value
            ),
            (
                connection.child_dock
                if isinstance(connection.child_dock, int)
                else connection.child_dock.value
            ),
            connection.orientation,
        )
        self.morphology.add_connection(connection)

    def get_available_docks(self, module_id: int) -> list[DockPosition]:
        """Get available docking positions for a module.

        Args:
            module_id: ID of the module to check

        Returns:
            List of available DockPosition values
        """
        return self.morphology.get_available_docks(module_id)


def test_basic_design():
    """Test basic robot design functionality."""
    # Create a simple robot morphology
    morphology = RobotMorphology()
    robot_designer = ModularLegs(morphology=morphology)

    # Add modules using the new API
    connections = [
        ModuleConnection(
            parent_module_id=0,
            parent_dock=DockPosition.UPPER_STICK_2,
            child_dock=DockPosition.UPPER_STICK_1,
            orientation=0,
        ),
        ModuleConnection(
            parent_module_id=0,
            parent_dock=DockPosition.UPPER_STICK_4,
            child_dock=DockPosition.UPPER_STICK_1,
            orientation=0,
        ),
        ModuleConnection(
            parent_module_id=0,
            parent_dock=DockPosition.LOWER_STICK_4,
            child_dock=DockPosition.UPPER_STICK_1,
            orientation=0,
        ),
        ModuleConnection(
            parent_module_id=0,
            parent_dock=DockPosition.LOWER_STICK_6,
            child_dock=DockPosition.UPPER_STICK_1,
            orientation=0,
        ),
    ]

    for connection in connections:
        robot_designer.add_module(connection)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        robot_designer.save(
            tmpdir, fix=True, pos=(0, 0, 0.4), quat=[1, 0, 0, 0], render=True
        )


def test_pipeline_design():
    """Test robot design with pipeline initialization."""
    # Create morphology using legacy pipeline format
    legacy_pipeline = [0, 1, 0, 0, 0, 3, 0, 0, 0, 13, 0, 0, 0, 15, 0, 0]
    morphology = RobotMorphology.from_sequence(legacy_pipeline)

    robot_designer = ModularLegs(morphology=morphology)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        robot_designer.save(
            tmpdir, fix=True, pos=(0, 0, 0.4), quat=[1, 0, 0, 0], render=True
        )


def test_random_design():
    """Test random robot design generation."""
    robot_designer = ModularLegs()
    robot_designer.reset()

    # Create random connections using the new API
    for _ in range(3):
        module_id = random.choice(robot_designer.get_available_module_ids())
        available_docks = robot_designer.get_available_docks(module_id)
        if not available_docks:
            continue

        parent_dock = random.choice(available_docks)
        child_dock = random.choice(
            list(DockPosition)[0:9]
        )  # Only use first 9 positions for child
        orientation = random.choice(
            robot_designer.get_available_rotation_ids(
                parent_dock.value, child_dock.value
            )
        )

        connection = ModuleConnection(
            parent_module_id=module_id,
            parent_dock=parent_dock,
            child_dock=child_dock,
            orientation=orientation,
        )

        if robot_designer.morphology.validate_connection(connection):
            robot_designer.add_module(connection)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        robot_designer.save(
            tmpdir, fix=True, pos=(0, 0, 1), quat=[1, 0, 0, 0], render=True
        )


if __name__ == "__main__":
    test_random_design()
