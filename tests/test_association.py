from helmet_rider.association.rider_vehicle import RiderVehicleAssociator
from helmet_rider.detection.types import Detection


def det(name, bbox, confidence=0.9):
    return Detection(name, confidence, bbox)


def test_person_sitting_on_motorcycle_is_associated():
    associator = RiderVehicleAssociator()
    people = [det("person", (350, 80, 460, 340))]
    vehicles = [det("motorcycle", (300, 250, 600, 400))]
    matches = associator.associate(people, vehicles)
    assert len(matches) == 1
    assert matches[0].score >= associator.config.threshold


def test_pedestrian_far_from_motorcycle_is_ignored():
    associator = RiderVehicleAssociator()
    people = [det("person", (30, 90, 110, 390))]
    vehicles = [det("motorcycle", (300, 250, 600, 400))]
    assert associator.associate(people, vehicles) == []


def test_person_beside_parked_motorcycle_without_body_contact_is_ignored():
    associator = RiderVehicleAssociator()
    people = [det("person", (615, 120, 700, 420))]
    vehicles = [det("motorcycle", (300, 250, 600, 400))]
    assert associator.associate(people, vehicles) == []


def test_person_close_to_vehicle_edge_without_riding_alignment_is_ignored():
    associator = RiderVehicleAssociator()
    people = [det("person", (570, 150, 670, 430))]
    vehicles = [det("motorcycle", (300, 250, 600, 400))]
    assert associator.associate(people, vehicles) == []


def test_two_riders_can_share_one_vehicle():
    associator = RiderVehicleAssociator()
    people = [
        det("person", (350, 80, 450, 340)),
        det("person", (455, 85, 555, 340)),
    ]
    vehicles = [det("motorcycle", (300, 250, 600, 400))]
    matches = associator.associate(people, vehicles)
    assert {match.person_index for match in matches} == {0, 1}
    assert {match.vehicle_index for match in matches} == {0}


def test_two_motorcycles_receive_their_own_riders():
    associator = RiderVehicleAssociator()
    people = [
        det("person", (80, 60, 180, 300)),
        det("person", (440, 60, 540, 300)),
    ]
    vehicles = [
        det("motorcycle", (40, 210, 250, 350)),
        det("motorcycle", (400, 210, 610, 350)),
    ]
    matches = associator.associate(people, vehicles)
    assert [(item.person_index, item.vehicle_index) for item in matches] == [(0, 0), (1, 1)]


def test_bicycle_rider_is_associated():
    associator = RiderVehicleAssociator()
    people = [det("person", (110, 40, 210, 280))]
    bicycles = [det("bicycle", (60, 180, 300, 330))]
    assert len(associator.associate(people, bicycles)) == 1


def test_partially_visible_person_and_boundary_vehicle_do_not_crash():
    associator = RiderVehicleAssociator()
    people = [det("person", (-18, 45, 80, 290))]
    vehicles = [det("bicycle", (0, 190, 150, 320))]
    matches = associator.associate(people, vehicles)
    assert isinstance(matches, list)
