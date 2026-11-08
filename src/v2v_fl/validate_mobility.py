"""Valida encontros no SUMO com os detectores de produção, sem iniciar Docker/treino."""

import argparse
import itertools
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import traci

from v2v_fl import v2i_docker_orchestrator as v2i
from v2v_fl import v2v_docker_orchestrator as v2v


def contact_summary(events: list[dict], total_clients: int) -> dict:
    counts = Counter(client for event in events for client in event["clients"])
    pairs = Counter(
        pair for event in events for pair in itertools.combinations(sorted(event["clients"]), 2)
    )
    adjacency = {client: set() for client in range(total_clients)}
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(adjacency)
    components = []
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            client = pending.pop()
            if client not in component:
                component.add(client)
                pending.extend(adjacency[client] - component)
        remaining -= component
        components.append(sorted(component))
    return {
        "encounter_count": len(events),
        "participation": {str(c): counts[c] for c in range(total_clients)},
        "unique_pairs": len(pairs),
        "pair_counts": [{"clients": list(pair), "count": n} for pair, n in sorted(pairs.items())],
        "components": components,
        "isolated_clients": [c for c in range(total_clients) if not adjacency[c]],
    }


def simulate(args, mode: str) -> dict:
    rsus = json.loads(args.rsu_config.read_text())["rsus"]
    horizon = float(ET.parse(args.sumo_cfg).find("./time/end").get("value"))
    label = f"mobility-{mode}"
    traci.start(
        ["sumo", "-c", str(args.sumo_cfg), "--seed", str(args.seed),
         "--step-length", str(args.step_length), "--no-step-log", "true",
         "--duration-log.disable", "true", "--no-warnings", "true"],
        label=label,
    )
    connection = traci.getConnection(label)
    mapping = {}
    events = []
    previous = None
    last_time = -float("inf")
    previous_rsu = {}
    last_rsu = {r["id"]: -float("inf") for r in rsus}
    departed = set()
    arrived = set()
    try:
        while connection.simulation.getMinExpectedNumber() > 0:
            if connection.simulation.getTime() >= horizon:
                break
            connection.simulationStep()
            now = connection.simulation.getTime()
            departed.update(connection.simulation.getDepartedIDList())
            arrived.update(connection.simulation.getArrivedIDList())
            vehicles = []
            for vid in connection.vehicle.getIDList():
                if vid not in mapping and len(mapping) < args.total_clients:
                    mapping[vid] = len(mapping)
                if vid in mapping:
                    x, y = connection.vehicle.getPosition(vid)
                    vehicles.append(v2v.VehicleState(
                        vid, x, y, connection.vehicle.getSpeed(vid),
                        connection.vehicle.getRoadID(vid), connection.vehicle.getAngle(vid),
                    ))
            if mode == "v2v":
                if len(vehicles) < 2 or now - last_time < args.cooldown:
                    continue
                clusters = v2v.find_clusters(vehicles, args.radius, args.min_clients)
                if not clusters:
                    previous = None
                    continue
                cluster = max(clusters, key=len)
                key = frozenset(v.veh_id for v in cluster)
                if key == previous:
                    continue
                previous = key
                etc = v2v.cluster_etc(cluster, args.radius)
                if not isinstance(etc, str) and etc < args.min_contact_time:
                    continue
                last_time = now
                events.append({"sim_time": now, "clients": sorted(mapping[v.veh_id] for v in cluster)})
            else:
                groups = v2i.find_vehicles_in_rsu_range(vehicles, rsus)
                for rsu in rsus:
                    rid = rsu["id"]
                    viable = []
                    for vehicle, _ in groups.get(rid, []):
                        etc = v2i.estimate_contact_time_v2i(vehicle, rsu)
                        if isinstance(etc, str) or etc >= args.v2i_min_contact_time:
                            viable.append(mapping[vehicle.veh_id])
                    if len(viable) < args.v2i_min_clients:
                        previous_rsu.pop(rid, None)
                        continue
                    key = frozenset(viable)
                    if now - last_rsu[rid] < args.cooldown or previous_rsu.get(rid) == key:
                        continue
                    previous_rsu[rid] = key
                    last_rsu[rid] = now
                    events.append({"sim_time": now, "rsu_id": rid, "clients": sorted(viable)})
                    if len(events) >= args.encounters:
                        break
            if len(events) >= args.encounters:
                break
        return {
            "mode": mode, **contact_summary(events, args.total_clients), "events": events,
            "sim_time": connection.simulation.getTime(),
            "departed_vehicles": sorted(departed), "arrived_vehicles": sorted(arrived),
            "vehicle_to_client": mapping,
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumo-cfg", type=Path, required=True)
    parser.add_argument("--rsu-config", type=Path, required=True)
    parser.add_argument("--total-clients", type=int, default=10)
    parser.add_argument("--encounters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--radius", type=float, default=300)
    parser.add_argument("--min-clients", type=int, default=3)
    parser.add_argument("--min-contact-time", type=float, default=9)
    parser.add_argument("--v2i-min-clients", type=int, default=3)
    parser.add_argument("--v2i-min-contact-time", type=float, default=9)
    parser.add_argument("--cooldown", type=float, default=15)
    parser.add_argument("--step-length", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [simulate(args, mode) for mode in ("v2v", "v2i")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runs": results,
    }, indent=2) + "\n")
    for result in results:
        print(f"{result['mode'].upper()}: {result['encounter_count']} encontros, "
              f"t={result['sim_time']:.0f}s, participação={result['participation']}, "
              f"componentes={result['components']}")
    if any(r["encounter_count"] < args.encounters or r["isolated_clients"]
           or len(r["departed_vehicles"]) != args.total_clients for r in results):
        parser.exit(1, "Mobilidade insuficiente; consulte o relatório antes de treinar.\n")


if __name__ == "__main__":
    main()
