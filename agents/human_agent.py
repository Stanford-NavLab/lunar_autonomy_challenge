#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Instructions:

    W and S                  : change the IPEx linear speed
    A and D                  : change the IPEx angular speed

    X and C                  : open / close the radiator cover
    F and V                  : change the front drums speed
    G and B                  : change the front arm angle
    H and N                  : change the back arms angle
    J and M                  : change the back drums speed
    T                        : toggle the semantic segmentation camera

    1 to 8                   : select a sensor position (marked by the '*' symbol)
    Tab                      : (de)activate the selected camera
    O and P                  : decrease/increase the selected light's intensity

    ESC                      : quit
"""

import collections
import datetime
import os

import numpy as np
import pygame
import pygame.locals as pykeys

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent


def get_entry_point():
    return 'HumanAgent'

class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, width, height, controller, scale):
        self._width = width
        self._height = height
        self._controller = controller
        self._scale = scale
        self._last_image = np.zeros([self._width, self._height])
        self._max_power = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width * scale, self._height * scale), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")
        self.set_black_screen()

        # Font
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)

        self.help = HelpText(pygame.font.Font(mono, 16), width * scale, height * scale)


    def run_interface(self, sensor_data, semantic_data, vehicle_data, imu_data, refresh=True):
        """
        Run the GUI
        """
        # Process camera data
        image = None
        if sensor_data is not None:
            if self._controller.semantic_active:
                image = semantic_data
            else:
                image = sensor_data
                image = np.stack([image, image, image], axis = -1)
            image = image.swapaxes(0, 1)
        elif refresh:
            image = np.zeros([self._width, self._height])
        else:
            image = self._last_image

        self._last_image = image

        if self._max_power is None:
            self._max_power = self._controller.agent.get_current_power()

        self._surface = pygame.surfarray.make_surface(image)
        self._surface = pygame.transform.scale(self._surface, (self._width * self._scale, self._height * self._scale))
        self._display.blit(self._surface, (0, 0))

        # Render only the Help if active
        if self._controller.render_help:
            self.help.render(self._display)
            pygame.display.flip()
            return

        transform = vehicle_data.transform

        # Update HUD
        radiator_state = "open" if self._controller._radiator_cover_state == carla.RadiatorCoverState.Open else "closed"
        info_text = [
            'Mission time:       %11s' % (str(datetime.timedelta(seconds=int(self._controller.agent._mission_time)))),
            '',
            'IMU data:',
            'Accelerometer:   %4.1f   %4.1f   %4.1f' % (imu_data[0], imu_data[1], imu_data[2]),
            'Gyroscope:       %4.1f   %4.1f   %4.1f' % (imu_data[3], imu_data[4], imu_data[5]),
            'Vehicle control:',
            ' Linear speed:      %5.2f   m/s' % (self._controller.linear_target_speed),
            ' Angular speed:     %5.2f   rad/s' % (self._controller.angular_target_speed),
            '',
            'Components   Current  Target',
            ' Front Drum:   %4.2f    %4.2f  rad/s' % (vehicle_data.front_drums_speed, self._controller.front_drums_target_speed),
            ' Front Arm:    %4.2f    %4.2f    rad' % (vehicle_data.front_arm_angle, self._controller.front_arm_target_angle),
            ' Back Arm:     %4.2f    %4.2f    rad' % (vehicle_data.back_arm_angle, self._controller.back_arm_target_angle),
            ' Back Drum:    %4.2f    %4.2f  rad/s' % (vehicle_data.back_drums_speed, self._controller.back_drums_target_speed),
            ' Radiator:     %4.2f   %6s' % (abs(vehicle_data.radiator_cover_angle), radiator_state),
            '',
            'Sensors:              %1s    %1s' % ("C", "L") ,
            '%s(1)Front:           %3s  %3d%%' % self._controller.get_sensor_info(carla.SensorPosition.Front),
            '%s(2)FrontLeft:       %3s  %3d%%' % self._controller.get_sensor_info(carla.SensorPosition.FrontLeft),
            '%s(3)FrontRight:      %3s  %3d%%' % self._controller.get_sensor_info(carla.SensorPosition.FrontRight),
            '%s(4)Left:            %3s  %3d%%' % self._controller.get_sensor_info(carla.SensorPosition.Left),
            '%s(5)Right:           %3s  %3d%%' % self._controller.get_sensor_info(carla.SensorPosition.Right),
            '%s(6)BackLeft:        %3s  %3d%%' % self._controller.get_sensor_info(carla.SensorPosition.BackLeft),
            '%s(7)BackRight:       %3s  %3d%%' % self._controller.get_sensor_info(carla.SensorPosition.BackRight),
            '%s(8)Back:            %3s  %3d%%' % self._controller.get_sensor_info(carla.SensorPosition.Back),
            '',
            'Status:',
            ' Current power:      %7.0f Wh' % (self._controller.agent.get_current_power()),
            ' Battery percentage:  %8.1f%%' % (self._controller.agent.get_current_power() / self._max_power*100.),
            ' Location:   % 18s' % ('(%5.2f, %5.2f)' % (transform.location.x, transform.location.y)),
            ' Height:  % 18.0f m' % transform.location.z,
            '',
            'Press I for instructions',
            ]

        info_surface = pygame.Surface((285, self._height))
        info_surface.set_alpha(100)
        self._display.blit(info_surface, (0, 0))
        v_offset, bar_h_offset, bar_width = 4, 100, 106
        for item in info_text:
            if v_offset + 18 > self._height:
                break
            if isinstance(item, list):
                if len(item) > 1:
                    points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                    pygame.draw.lines(self._display, (255, 136, 0), False, points, 2)
                item = None
                v_offset += 18
            elif isinstance(item, tuple):
                if isinstance(item[1], bool):
                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                    pygame.draw.rect(self._display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(self._display, (255, 255, 255), rect_border, 1)
                    f = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                    pygame.draw.rect(self._display, (255, 255, 255), rect)
                item = item[0]
            if item:  # At this point has to be a str.
                surface = self._font_mono.render(item, True, (255, 255, 255))
                self._display.blit(surface, (8, v_offset))
            v_offset += 18

        pygame.display.flip()

    def set_black_screen(self):
        """Set the surface to black"""
        black_array = np.zeros([self._width, self._height])
        self._surface = pygame.surfarray.make_surface(black_array)
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def quit(self):
        pygame.quit()


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.3 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def render(self, display):
        display.blit(self.surface, self.pos)


class KeyboardControl(object):
    """
    Keyboard control for the human agent
    """
    KEY2SENSOR = {
        pykeys.K_1: carla.SensorPosition.Front,
        pykeys.K_2: carla.SensorPosition.FrontLeft,
        pykeys.K_3: carla.SensorPosition.FrontRight,
        pykeys.K_4: carla.SensorPosition.Left,
        pykeys.K_5: carla.SensorPosition.Right,
        pykeys.K_6: carla.SensorPosition.BackLeft,
        pykeys.K_7: carla.SensorPosition.BackRight,
        pykeys.K_8: carla.SensorPosition.Back,
    }

    def __init__(self, agent):
        self.agent = agent
        self._initial_tick = True
        self._pygame_window_active = True

        self.linear_target_speed = 0
        self.angular_target_speed = 0
        self.front_drums_target_speed = 0
        self.front_arm_target_angle = 1.5
        self.back_arm_target_angle = 0.3
        self.back_drums_target_speed = 0

        self._linear_speed_increase = 0.02
        self._angular_speed_increase = 0.02
        self._drum_speed_increase = 0.01
        self._arm_angle_increase = 0.1

        self._max_linear_speed = 0.4
        self._max_angular_speed = 0.5
        self._max_arm_angle = 2.4
        self._max_drum_speed = 0.5

        self.render_help = False
        self.semantic_active = False

        self._radiator_cover_state = carla.RadiatorCoverState.Close

        self._sensor_active = carla.SensorPosition.Front
        self._sensors_status = collections.OrderedDict()
        for k, v in self.agent.sensors().items():
            self._sensors_status[k] = [v['camera_active'], v['light_intensity']]

    def get_sensor_info(self, sensor_position):
        return (
            "*" if self._sensor_active == sensor_position else " ",
            "on" if self._sensors_status[sensor_position][0] else "off",
            self._sensors_status[sensor_position][1] * 100.0
        )

    def parse_events(self, delta_seconds):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        return self._parse_vehicle_keys(pygame.key.get_pressed(), delta_seconds*1000)

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        # Set the initial position
        if self._initial_tick:
            self.agent.set_front_drums_target_speed(self.front_drums_target_speed)
            self.agent.set_front_arm_angle(self.front_arm_target_angle)
            self.agent.set_back_arm_angle(self.back_arm_target_angle)
            self.agent.set_back_drums_target_speed(self.back_drums_target_speed)
            self._initial_tick = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._pygame_window_active = False
            elif event.type == pygame.KEYUP:
                # Mission complete
                if event.key == pykeys.K_ESCAPE:
                    self.agent.mission_complete()

                # Help menu
                if event.key == pykeys.K_i:
                    self.render_help = not self.render_help

                if event.key == pykeys.K_t:
                    self.semantic_active = not self.semantic_active

                # Open/close radiator cover.
                if event.key == pykeys.K_x:
                    self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Open)
                    self._radiator_cover_state = carla.RadiatorCoverState.Open
                elif event.key == pykeys.K_c:
                    self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Close)
                    self._radiator_cover_state = carla.RadiatorCoverState.Close

                # Change active camera.
                if event.key in KeyboardControl.KEY2SENSOR:
                    self._sensor_active = KeyboardControl.KEY2SENSOR[event.key]

                # Turn on/off camera
                if event.key == pykeys.K_TAB:
                    self._sensors_status[self._sensor_active][0] = not self._sensors_status[self._sensor_active][0]
                    self.agent.set_camera_state(self._sensor_active, self._sensors_status[self._sensor_active][0])

                # Turn on/off lights
                if event.key == pykeys.K_p:
                    current_intensity = self._sensors_status[self._sensor_active][1]
                    target_intensity = min(current_intensity + 0.1, 1.0)
                    self.agent.set_light_state(self._sensor_active, target_intensity)
                    self._sensors_status[self._sensor_active][1] = target_intensity
                elif event.key == pykeys.K_o:
                    current_intensity = self._sensors_status[self._sensor_active][1]
                    target_intensity = max(0.0, current_intensity - 0.1)
                    self.agent.set_light_state(self._sensor_active, target_intensity)
                    self._sensors_status[self._sensor_active][1] = target_intensity

        keys = pygame.key.get_pressed()

        if keys[pykeys.K_UP] or keys[pykeys.K_w]:
            self.linear_target_speed = min(self.linear_target_speed + self._linear_speed_increase, self._max_linear_speed)
        elif keys[pykeys.K_DOWN] or keys[pykeys.K_s]:
            self.linear_target_speed = max(self.linear_target_speed - self._linear_speed_increase, -self._max_linear_speed)
        else:
            self.linear_target_speed = max(self.linear_target_speed - 2 * self._linear_speed_increase, 0.0)

        if keys[pykeys.K_RIGHT] or keys[pykeys.K_d]:
            self.angular_target_speed = max(self.angular_target_speed - self._angular_speed_increase, -self._max_angular_speed)
        elif keys[pykeys.K_LEFT] or keys[pykeys.K_a]:
            self.angular_target_speed = min(self.angular_target_speed + self._angular_speed_increase, +self._max_angular_speed)
        else:
            self.angular_target_speed = max(self.angular_target_speed - 2 * self._angular_speed_increase, 0.0)

        if keys[pykeys.K_f]:
            self.front_drums_target_speed = min(self.front_drums_target_speed + self._drum_speed_increase, self._max_drum_speed)
            self.agent.set_front_drums_target_speed(self.front_drums_target_speed)
        elif keys[pykeys.K_v]:
            self.front_drums_target_speed = max(self.front_drums_target_speed - self._drum_speed_increase, -self._max_drum_speed)
            self.agent.set_front_drums_target_speed(self.front_drums_target_speed)
            
        if keys[pykeys.K_g]:
            self.front_arm_target_angle = min(self.front_arm_target_angle + self._arm_angle_increase, self._max_arm_angle)
            self.agent.set_front_arm_angle(self.front_arm_target_angle)
        elif keys[pykeys.K_b]:
            self.front_arm_target_angle = max(self.front_arm_target_angle - self._arm_angle_increase, -self._max_arm_angle)
            self.agent.set_front_arm_angle(self.front_arm_target_angle)
        if keys[pykeys.K_h]:
            self.back_arm_target_angle = min(self.back_arm_target_angle + self._arm_angle_increase, self._max_arm_angle)
            self.agent.set_back_arm_angle(self.back_arm_target_angle)
        elif keys[pykeys.K_n]:
            self.back_arm_target_angle = max(self.back_arm_target_angle - self._arm_angle_increase, -self._max_arm_angle)
            self.agent.set_back_arm_angle(self.back_arm_target_angle)
        if keys[pykeys.K_j]:
            self.back_drums_target_speed = min(self.back_drums_target_speed + self._drum_speed_increase, self._max_drum_speed)
            self.agent.set_back_drums_target_speed(self.back_drums_target_speed)
        elif keys[pykeys.K_m]:
            self.back_drums_target_speed = max(self.back_drums_target_speed - self._drum_speed_increase, -self._max_drum_speed)
            self.agent.set_back_drums_target_speed(self.back_drums_target_speed)

        return not self._pygame_window_active, carla.VehicleVelocityControl(self.linear_target_speed, self.angular_target_speed), self._sensor_active


class HumanAgent(AutonomousAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._width = 1280
        self._height = 720
        self._scale = 1

        self._controller = KeyboardControl(self)
        self._hic = HumanInterface(self._width, self._height, self._controller, self._scale)
        self._has_quit = False

        self._clock = pygame.time.Clock()
        self._delta_seconds = 0.05

    def use_fiducials(self):
        return True

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': True, 'light_intensity': 1.0, 'width': self._width, 'height': self._height, 'use_semantic': True
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': self._width, 'height': self._height, 'use_semantic': True
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': False, 'light_intensity': 0, 'width': self._width, 'height': self._height, 'use_semantic': True
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': self._width, 'height': self._height, 'use_semantic': True
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': self._width, 'height': self._height, 'use_semantic': True
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': self._width, 'height': self._height, 'use_semantic': True
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 'light_intensity': 0, 'width': self._width, 'height': self._height, 'use_semantic': True
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 0, 'width': self._width, 'height': self._height, 'use_semantic': True
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""
        self._clock.tick_busy_loop(20)

        quit_, control, active_camera = self._controller.parse_events(self._delta_seconds)
        if quit_:
            self.mission_complete()

        self._hic.run_interface(
            sensor_data=input_data['Grayscale'].get(active_camera, None),
            semantic_data=input_data['Semantic'].get(active_camera, None),
            vehicle_data=self._vehicle_status,
            refresh=active_camera not in input_data['Grayscale'],
            imu_data=self.get_imu_data()
        )
        return control

    def finalize(self):
        """
        Cleanup
        """
        if hasattr(self, '_hic') and not self._has_quit:
            self._hic.set_black_screen()
            self._hic.quit()
            self._has_quit = True
