"""Event metrics require human entity references across source timestamps."""


class TemporalMetrics:
    def __init__(self):
        self.entities = {}
        self.id_switches = 0
        self.false_events = set()
        self.delays = []
        self.observations = 0

    def update(self, sequence, timestamp, predictions, samples, matches):
        for pi, si, _ in matches:
            truth = samples[si]
            entity = truth.get("entity_id")
            if not entity or truth["label"] not in {"helmet", "no_helmet"}:
                continue
            self.observations += 1
            pred = predictions[pi]
            key = (sequence, entity)
            previous = self.entities.get(key)
            if previous and previous["track_id"] != pred["track_id"]:
                self.id_switches += 1
            onset = (
                timestamp
                if previous is None or previous["truth"] != truth["label"]
                else previous["onset"]
            )
            confirmed = (
                previous is not None
                and previous["truth"] == truth["label"]
                and previous["confirmed"]
            )
            if truth["label"] == "helmet" and pred["helmet"]["status"] == "no_helmet":
                self.false_events.add((sequence, pred["track_id"]))
            if (
                truth["label"] == "no_helmet"
                and pred["helmet"]["status"] == "no_helmet"
                and not confirmed
            ):
                self.delays.append(timestamp - onset)
                confirmed = True
            self.entities[key] = {
                "track_id": pred["track_id"],
                "truth": truth["label"],
                "onset": onset,
                "confirmed": confirmed,
            }

    def summary(self):
        import statistics

        return {
            "human_identity_observations": self.observations,
            "id_switches": self.id_switches if self.observations else None,
            "false_violation_events": len(self.false_events) if self.observations else None,
            "mean_confirmation_delay_seconds": statistics.fmean(self.delays)
            if self.delays
            else None,
            "delay_scope": "Time from first reviewed no-helmet observation; unconfirmed events have no delay value.",
            "reason": None
            if self.observations
            else "No matched human temporal identities; model IDs are not ground truth.",
        }
