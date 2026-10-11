"""Geometric rider/vehicle association with one-to-many vehicle support."""

from __future__ import annotations

from dataclasses import dataclass

from helmet_rider.association.geometry import (
    bottom_center,
    expand,
    ioa,
    point_box_distance,
)
from helmet_rider.config import AssociationConfig
from helmet_rider.detection.types import Association, BBox, Detection


@dataclass(frozen=True)
class _Candidate:
    association: Association
    eligible: bool


class RiderVehicleAssociator:
    """Associate people only when their lower-body geometry supports riding.

    The expanded vehicle region is a supporting cue, never a sufficient cue by
    itself. This prevents a person merely standing beside a parked vehicle from
    being sent to the helmet classifier.
    """

    VEHICLES = frozenset({"motorcycle", "bicycle"})

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()

    def score(self, person: BBox, vehicle: BBox, person_index: int = 0,
              vehicle_index: int = 0) -> Association:
        px1, py1, px2, py2 = person
        person_h = max(py2 - py1, 1.0)
        lower = (px1, py2 - person_h * self.config.lower_body_ratio, px2, py2)
        lower_overlap = ioa(lower, vehicle)
        person_overlap = ioa(person, vehicle)
        expanded_vehicle = expand(
            vehicle, self.config.vehicle_expand_x, self.config.vehicle_expand_y
        )
        expanded_overlap = ioa(person, expanded_vehicle)
        person_width = max(px2 - px1, 1.0)
        horizontal_intersection = max(
            0.0, min(px2, vehicle[2]) - max(px1, vehicle[0])
        )
        horizontal_alignment = min(1.0, horizontal_intersection / person_width)
        point = bottom_center(person)
        distance = point_box_distance(point, expanded_vehicle)
        vehicle_diagonal = max(
            ((vehicle[2] - vehicle[0]) ** 2 + (vehicle[3] - vehicle[1]) ** 2) ** 0.5,
            1.0,
        )
        bottom_proximity = max(0.0, 1.0 - distance / vehicle_diagonal)
        # Lower-body/vehicle contact is the strongest signal. Expanded overlap
        # contributes little, so proximity alone cannot pass the threshold.
        score = (
            0.42 * lower_overlap
            + 0.20 * person_overlap
            + 0.08 * expanded_overlap
            + 0.16 * horizontal_alignment
            + 0.14 * bottom_proximity
        )
        return Association(
            person_index=person_index,
            vehicle_index=vehicle_index,
            score=min(1.0, score),
            lower_body_overlap=lower_overlap,
            person_overlap=person_overlap,
            horizontal_alignment=horizontal_alignment,
            bottom_proximity=bottom_proximity,
        )

    def _eligible(self, association: Association) -> bool:
        # A lower-body sliver at the vehicle edge is common for bystanders next
        # to parked bikes. Require meaningful horizontal and whole-person
        # overlap as well as lower-body contact; expanded-box proximity alone
        # can never qualify.
        geometric_contact = (
            association.lower_body_overlap >= self.config.minimum_lower_body_overlap
            and association.horizontal_alignment >= 0.45
            and association.person_overlap >= 0.20
        )
        return geometric_contact and association.score >= self.config.threshold

    def associate(
        self,
        persons: list[Detection],
        vehicles: list[Detection],
    ) -> list[Association]:
        candidates: list[_Candidate] = []
        for person_index, person in enumerate(persons):
            if person.class_name != "person":
                continue
            for vehicle_index, vehicle in enumerate(vehicles):
                if vehicle.class_name not in self.VEHICLES:
                    continue
                association = self.score(
                    person.bbox, vehicle.bbox, person_index, vehicle_index
                )
                candidates.append(_Candidate(association, self.supports(person, vehicle)))

        # Each person gets at most one vehicle, selected by geometric score.
        # There is intentionally no vehicle capacity limit: a bike can have
        # multiple riders.
        selected: dict[int, Association] = {}
        for candidate in sorted(
            candidates, key=lambda item: item.association.score, reverse=True
        ):
            association = candidate.association
            if candidate.eligible and association.person_index not in selected:
                selected[association.person_index] = association
        return sorted(selected.values(), key=lambda item: item.person_index)

    def supports(self, person: Detection, vehicle: Detection) -> bool:
        """Test riding contact without assigning a person to only one vehicle.

        Occluded motorcycles can expose a narrower strip beneath a confidently
        detected rider. Keep substantial body contact and a higher score for
        that case; proximity to a vehicle alone still cannot qualify.
        """
        association = self.score(person.bbox, vehicle.bbox)
        return self._eligible(association) or (
            vehicle.class_name == "motorcycle"
            and person.confidence >= 0.50
            and association.lower_body_overlap >= 0.25
            and association.person_overlap >= 0.20
            and association.horizontal_alignment >= 0.40
            and association.score >= max(self.config.threshold, 0.40)
        )

    def diagnostics(
        self, persons: list[Detection], vehicles: list[Detection]
    ) -> list[Association]:
        """Return every pair score for optional debug overlays."""
        return [
            self.score(person.bbox, vehicle.bbox, person_index, vehicle_index)
            for person_index, person in enumerate(persons)
            for vehicle_index, vehicle in enumerate(vehicles)
            if person.class_name == "person" and vehicle.class_name in self.VEHICLES
        ]
