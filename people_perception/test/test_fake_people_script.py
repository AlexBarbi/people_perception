"""Where fake_people_detector places its scripted people.

The "gazebo" numbers are derived by hand from how Gazebo Harmonic moves a scripted actor (gz-sim SceneManager.cc
ActorTrajectoryAt, gz-common Animation.cc PoseAnimation, gz-math Spline.cc): script time = sim time - delay_start,
wrapped by the length of the trajectory; between two waypoints a cubic Hermite segment
    p(u) = (2u^3 - 3u^2 + 1) P0 + (u^3 - 2u^2 + u) m0 + (-2u^3 + 3u^2) P1 + (u^3 - u^2) m1
with Catmull-Rom tangents m_i = (P_{i+1} - P_{i-1}) / 2, and for a trajectory that ends where it starts
m_0 = m_last = (P_1 - P_{last-1}) / 2.
"""
import importlib.util
import math
import os

import pytest

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'fake_people_detector.py')
_spec = importlib.util.spec_from_file_location('fake_people_detector', _path)
fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake)


def person_1(**extra):
    """person_1 of legged_gazebo/worlds/warehouse.world: 9 m along y and back, one second at each end."""
    entry = {'name': 'person_1', 'size': [0.6, 0.6, 1.8], 'floor_z': 0.0, 'loop': True,
             'waypoints': [[0, 3.5, 4.5], [7.5, 3.5, -4.5], [8.5, 3.5, -4.5], [16, 3.5, 4.5], [17, 3.5, 4.5]]}
    entry.update(extra)
    return fake.Person(entry)


def y_at(person, t):
    (x, y), _ = person.at(t)
    assert x == pytest.approx(3.5)
    return y


def test_linear_is_the_default_straight_lines_at_constant_speed():
    person = person_1()
    assert y_at(person, 0.0) == pytest.approx(4.5)
    assert y_at(person, 1.575) == pytest.approx(4.5 - 9 * 0.21)
    assert y_at(person, 3.75) == pytest.approx(0.0)
    assert y_at(person, 8.0) == pytest.approx(-4.5)       # standing at the far end
    assert y_at(person, 17.0 + 3.75) == pytest.approx(0.0)  # second lap


def test_gazebo_follows_the_spline_of_a_scripted_actor():
    person = person_1(interpolation='gazebo')
    assert y_at(person, 0.0) == pytest.approx(4.5)
    # First leg, u = 0.21: s = -u^3 + 1.5 u^2 + 0.5 u = 0.161889, slower than the straight line at the start
    assert y_at(person, 1.575) == pytest.approx(4.5 - 9 * 0.161889, abs=1e-4)
    assert y_at(person, 3.75) == pytest.approx(0.0)
    assert y_at(person, 7.5) == pytest.approx(-4.5)
    # The "turn" second: the actor does not stand, it swings 9 / 8 m past the waypoint and back
    assert y_at(person, 8.0) == pytest.approx(-5.625)
    assert y_at(person, 8.5) == pytest.approx(-4.5)
    assert y_at(person, 16.5) == pytest.approx(5.625)
    assert y_at(person, 17.0) == pytest.approx(4.5)
    assert y_at(person, 17.0 + 3.75) == pytest.approx(0.0)
    assert y_at(person, 34.0 + 8.0) == pytest.approx(-5.625)


def test_delay_start_holds_the_person_at_the_first_waypoint_once():
    person = person_1(interpolation='gazebo', delay_start=3.0)
    assert y_at(person, 0.0) == pytest.approx(4.5)
    assert y_at(person, 2.9) == pytest.approx(4.5)
    assert y_at(person, 3.0 + 3.75) == pytest.approx(0.0)
    assert y_at(person, 3.0 + 17.0 + 3.75) == pytest.approx(0.0)  # no second wait before the next lap


def test_without_loop_the_person_stays_at_the_last_waypoint():
    person = fake.Person({'name': 'visitor', 'size': [0.5, 0.5, 1.7], 'floor_z': 0.0, 'loop': False,
                          'waypoints': [[0, 0.0, 0.0], [4, 4.0, 0.0]]})
    (x, y), heading = person.at(2.0)
    assert (x, y) == pytest.approx((2.0, 0.0))
    assert heading == pytest.approx(0.0)
    assert person.at(100.0)[0] == pytest.approx((4.0, 0.0))


def test_open_path_uses_one_sided_tangents_at_its_ends():
    # P = 0, 4, 4+4 along x: m0 = (P1 - P0) / 2 = 2, m1 = (P2 - P0) / 2 = 4
    # p(0.5) on the first leg = 0.5 * 4 + 0.125 * 2 - 0.125 * 4 = 1.75
    person = fake.Person({'name': 'visitor', 'size': [0.5, 0.5, 1.7], 'floor_z': 0.0, 'loop': False,
                          'interpolation': 'gazebo', 'waypoints': [[0, 0.0, 0.0], [4, 4.0, 0.0], [8, 8.0, 0.0]]})
    assert person.at(2.0)[0] == pytest.approx((1.75, 0.0))


def test_unknown_interpolation_is_refused():
    with pytest.raises(ValueError):
        person_1(interpolation='bezier')


def test_script_places_the_people_through_its_origin():
    # The waypoints of warehouse.world as they are; the world origin is 4.7165 m ahead and 3 m to the left of where
    # FAST-LIO starts with the robot at its default spawn
    script = fake.Script({'frame': 'camera_init', 'start_time': 0.0, 'origin': [4.7165, 3.0, 0.0],
                          'people': [{'name': 'person_1', 'size': [0.6, 0.6, 1.8], 'floor_z': -0.47, 'loop': True,
                                      'interpolation': 'gazebo',
                                      'waypoints': [[0, 3.5, 4.5], [7.5, 3.5, -4.5], [8.5, 3.5, -4.5], [16, 3.5, 4.5],
                                                    [17, 3.5, 4.5]]}]})
    assert script.frame == 'camera_init'
    (person, x, y, heading), = script.at(3.75)
    assert person.name == 'person_1'
    assert (x, y) == pytest.approx((8.2165, 3.0))
    assert heading == pytest.approx(-math.pi / 2)  # walking towards -y


def test_script_origin_can_be_turned_and_start_later():
    script = fake.Script({'frame': 'camera_init', 'start_time': 10.0, 'origin': [1.0, 2.0, math.pi / 2],
                          'people': [{'name': 'visitor', 'size': [0.5, 0.5, 1.7], 'floor_z': 0.0, 'loop': False,
                                      'waypoints': [[0, 0.0, 0.0], [4, 4.0, 0.0]]}]})
    (_, x, y, heading), = script.at(12.0)  # 2 s into the script: 2 m along its x, which is +y of the frame
    assert (x, y) == pytest.approx((1.0, 4.0))
    assert heading == pytest.approx(math.pi / 2)


def test_script_without_origin_is_in_the_frame_itself():
    script = fake.Script({'frame': 'camera_init', 'start_time': 0.0,
                          'people': [{'name': 'visitor', 'size': [0.5, 0.5, 1.7], 'floor_z': 0.0, 'loop': False,
                                      'waypoints': [[0, 0.0, 0.0], [4, 4.0, 0.0]]}]})
    (_, x, y, _), = script.at(2.0)
    assert (x, y) == pytest.approx((2.0, 0.0))
