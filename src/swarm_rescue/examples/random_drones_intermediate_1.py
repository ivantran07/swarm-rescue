"""
This program can be launched directly.
"""

import math
import random
import sys
from pathlib import Path

# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maps.map_intermediate_01 import MyMapIntermediate01
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.utils.utils import normalize_angle


class MyDroneRandom(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.counterStopStraight = random.uniform(10, 30)
        self.isTurningLeft = False
        self.isTurningRight = False

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an
        obstacle is hit
        """
        if self._misc_data.max_timestep_limit is not None:
            perc_timestep = self.elapsed_timestep / self._misc_data.max_timestep_limit * 100
        else:
            perc_timestep = None

        if self._misc_data.max_walltime_limit is not None:
            perc_walltime = self.elapsed_walltime / self._misc_data.max_walltime_limit * 100
        else:
            perc_walltime = None

        if perc_timestep is not None and perc_walltime is not None:
            print(f"% walltime = {perc_walltime:.1f}% and % timestep = {perc_timestep:.1f}%")

        command_straight = {"forward": 1.0,
                            "rotation": 0.0}
        command_turn_left = {"forward": 0.0,
                             "rotation": 1.0}
        command_turn_right = {"forward": 0.0,
                              "rotation": -1.0}

        self.counterStraight += 1

        if (not self._is_turning() and
                self.counterStraight > self.counterStopStraight):
            self.angleStopTurning = random.uniform(-math.pi, math.pi)
            diff_angle = normalize_angle(self.angleStopTurning -
                                         self.measured_compass_angle())
            if diff_angle > 0:
                self.isTurningLeft = True
            else:
                self.isTurningRight = True

        diff_angle = normalize_angle(self.angleStopTurning -
                                     self.measured_compass_angle())
        if self._is_turning() and abs(diff_angle) < 0.2:
            self.isTurningLeft = False
            self.isTurningRight = False
            self.counterStraight = 0
            self.counterStopStraight = random.uniform(10, 30)

        if self.isTurningLeft:
            return command_turn_left
        elif self.isTurningRight:
            return command_turn_right
        else:
            return command_straight

    def _is_turning(self):
        return self.isTurningLeft or self.isTurningRight


def main():
    my_map = MyMapIntermediate01()
    my_playground = my_map.construct_playground(drone_type=MyDroneRandom)

    gui = GuiSR(playground=my_playground,
                the_map=my_map,
                use_keyboard=False,
                use_mouse_measure=True,
                enable_visu_noises=False,
                )
    gui.run()


if __name__ == '__main__':
    main()
