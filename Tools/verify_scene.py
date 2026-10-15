"""
Check that every ML-Agents agent in a scene presents one observation layout.

    python Tools/verify_scene.py                     # Assets/OutdoorsScene.unity
    python Tools/verify_scene.py Assets/Other.unity

ML-Agents caches the sensor layout of the first agent that reaches a ModelRunner
and asserts that every later agent matches it:

    Sensor shapes must match. [68] != [21]
    Number of Sensors must match. 3 != 2

The assert names neither GameObject, and it only fires once a run is already
under way. This reads the scene and the prefabs it instantiates straight off
disk, so the mismatch has a name before you press play.

  STRUCTURE   the scene and its prefabs parse, every agent resolves to components
  LAYOUT      agents sharing a brain produce the same sensors in the same order
  MODEL       that layout matches the observation inputs of the assigned .onnx
  WIRING      one Agent per GameObject, arena/enemy set, enemy points at that Agent

STRUCTURE, LAYOUT and MODEL set the exit code. WIRING is advisory, because a
deliberately unwired agent is a legitimate thing to have in a scene.

This reads the file, not the editor's in-memory copy, so save the scene first.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

DOC = re.compile(r"^--- !u!(\d+) &(-?\d+)( stripped)?\s*$", re.M)

GAMEOBJECT, TRANSFORM, MONOBEHAVIOUR, PREFAB_INSTANCE = "1", "4", "114", "1001"

# Every SensorComponent subclass shipped with com.unity.ml-agents. Only the ray
# sensors get their observation size computed below; the rest are reported as an
# unknown size, which still catches a differing sensor count or order.
SENSOR_COMPONENTS = {
    "ArticulationBodySensorComponent",
    "BufferSensorComponent",
    "CameraSensorComponent",
    "GridSensorComponent",
    "Match3SensorComponent",
    "RayPerceptionSensorComponent2D",
    "RayPerceptionSensorComponent3D",
    "RenderTextureSensorComponent",
    "RigidBodySensorComponent",
    "VectorSensorComponent",
}

HEURISTIC_ONLY = 1


# --------------------------------------------------------------------------- yaml

def scalar(body, name):
    m = re.search(r"^\s*%s: (.*)$" % re.escape(name), body, re.M)
    return m.group(1).strip() if m else None


def ref(body, name):
    raw = scalar(body, name)
    if raw is None:
        return None
    m = re.search(r"fileID: (-?\d+)", raw)
    return m.group(1) if m else None


def guid_in(raw):
    if raw is None:
        return None
    m = re.search(r"guid: (\w+)", raw)
    return m.group(1) if m else None


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def count_list(body, name):
    """Number of `- item` entries in the block list under `name`."""
    m = re.search(r"^(\s*)%s:\s*$" % re.escape(name), body, re.M)
    if not m:
        return 0
    indent, total = len(m.group(1)), 0
    for line in body[m.end():].splitlines():
        if not line.strip():
            continue
        lead = len(line) - len(line.lstrip())
        if line.lstrip().startswith("- ") and lead >= indent:
            total += 1
        elif lead <= indent:
            break
    return total


def section(body, name):
    """The indented block under `name:`, or "" if absent."""
    m = re.search(r"^(\s*)%s:\s*$" % re.escape(name), body, re.M)
    if not m:
        return ""
    indent, lines = len(m.group(1)), []
    for line in body[m.end():].splitlines():
        if not line.strip():
            lines.append(line)
            continue
        lead = len(line) - len(line.lstrip())
        if lead <= indent and not line.lstrip().startswith("- "):
            break
        lines.append(line)
    return "\n".join(lines)


def component_fids(body):
    block = section(body, "m_Component")
    return re.findall(r"- component: \{fileID: (-?\d+)\}", block)


class Doc:
    __slots__ = ("cls", "fid", "stripped", "body")

    def __init__(self, cls, fid, stripped, body):
        self.cls, self.fid, self.stripped, self.body = cls, fid, stripped, body


class UnityFile:
    def __init__(self, path):
        self.path = Path(path)
        text = self.path.read_text(encoding="utf-8", errors="replace")
        self.docs = {}
        hits = list(DOC.finditer(text))
        for i, m in enumerate(hits):
            end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            self.docs[m.group(2)] = Doc(m.group(1), m.group(2), bool(m.group(3)),
                                        text[m.end():end])

    def of_class(self, cls):
        return [d for d in self.docs.values() if d.cls == cls]

    def get(self, fid):
        return self.docs.get(fid)

    def root_gameobject(self):
        """The prefab root: the GameObject whose Transform has no father."""
        for d in self.of_class(TRANSFORM):
            if ref(d.body, "m_Father") in (None, "0"):
                return ref(d.body, "m_GameObject")
        return None


def guid_index():
    index = {}
    for meta in Path("Assets").rglob("*.meta"):
        try:
            text = meta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r"^guid: (\w+)", text, re.M)
        if m:
            index[m.group(1)] = meta.with_suffix("")
    return index


# ----------------------------------------------------------------- project classes

def project_subclasses(base):
    """Script class names under Assets/ that derive from `base`, transitively."""
    declared = {}
    for path in Path("Assets").rglob("*.cs"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in re.finditer(r"\bclass\s+(\w+)\s*:\s*([\w<>, .]+)", text):
            parents = [p.strip().split("<")[0].split(".")[-1] for p in m.group(2).split(",")]
            declared[m.group(1)] = parents

    found, frontier = set(), {base}
    while frontier:
        nxt = set()
        for name, parents in declared.items():
            if name not in found and frontier.intersection(parents):
                found.add(name)
                nxt.add(name)
        frontier = nxt
    return found


# ----------------------------------------------------------------------- resolving

class Component:
    """A component as the engine sees it: its body plus any instance overrides."""

    def __init__(self, kind, doc, origin, source_fid=None, overrides=None):
        self.kind = kind
        self.doc = doc
        self.origin = origin            # "scene" (own or instance-added) or "prefab"
        self.source_fid = source_fid    # fileID inside the prefab, if prefab-derived
        self.overrides = overrides or {}

    def get(self, name):
        if name in self.overrides:
            return self.overrides[name][0]
        return scalar(self.doc.body, name)

    def get_ref(self, name):
        if name in self.overrides:
            return self.overrides[name][1]
        return ref(self.doc.body, name)

    def get_guid(self, name):
        return guid_in(scalar(self.doc.body, name))

    def get_int(self, name, default=0):
        return as_int(self.get(name), default)

    def array_size(self, name):
        key = name + ".Array.size"
        if key in self.overrides:
            return as_int(self.overrides[key][0])
        return count_list(self.doc.body, name)


class Node:
    """A resolved GameObject."""

    def __init__(self, key, name, self_active):
        self.key = key
        self.name = name
        self.self_active = self_active
        self.parent = None
        self.components = []
        self.children = []
        self.prefab = None      # prefab file this node came from, if any

    def active(self):
        node, seen = self, set()
        while node is not None and node.key not in seen:
            if not node.self_active:
                return False
            seen.add(node.key)
            node = node.parent
        return True

    def path(self):
        parts, node, seen = [], self, set()
        while node is not None and node.key not in seen:
            parts.append(node.name)
            seen.add(node.key)
            node = node.parent
        return "/".join(reversed(parts))

    def subtree(self):
        out, stack, seen = [], [self], set()
        while stack:
            node = stack.pop()
            if node.key in seen:
                continue
            seen.add(node.key)
            out.append(node)
            stack.extend(node.children)
        return out


def kind_of(doc):
    ident = scalar(doc.body, "m_EditorClassIdentifier") or ""
    return ident.split("::")[-1].split(".")[-1] if ident else ""


def modifications(inst_doc):
    """{target fileID: {propertyPath: (value, objectReference or None)}}"""
    out = defaultdict(dict)
    pattern = re.compile(
        r"- target: \{fileID: (-?\d+)[^}]*\}\s*\n"
        r"\s*propertyPath: (\S+)\s*\n"
        r"\s*value: (.*?)\s*\n"
        r"\s*objectReference: \{fileID: (-?\d+)[^}]*\}")
    for m in pattern.finditer(inst_doc.body):
        out[m.group(1)][m.group(2)] = (m.group(3),
                                       m.group(4) if m.group(4) != "0" else None)
    return out


class Scene:
    """The scene with every prefab instance expanded into resolved nodes."""

    def __init__(self, path, guids):
        self.file = UnityFile(path)
        self.guids = guids
        self.prefabs = {}
        self.notes = []
        self.nodes = {}
        self.node_of_component = {}   # scene component fileID -> node
        self._key_of_transform = {}
        self._build()

    def prefab(self, guid):
        if guid not in self.prefabs:
            path = self.guids.get(guid)
            if path is None or not Path(path).is_file():
                self.prefabs[guid] = None
                self.notes.append("prefab guid %s is not under Assets/ - any agents inside "
                                  "it are invisible to this check" % guid)
            else:
                self.prefabs[guid] = UnityFile(path)
        return self.prefabs[guid]

    def _build(self):
        for d in self.file.of_class(GAMEOBJECT):
            if d.stripped:
                continue
            node = Node(("scene", d.fid), scalar(d.body, "m_Name") or "(unnamed)",
                        as_int(scalar(d.body, "m_IsActive"), 1) != 0)
            for fid in component_fids(d.body):
                cd = self.file.get(fid)
                if cd is None:
                    continue
                if cd.cls == MONOBEHAVIOUR:
                    node.components.append(Component(kind_of(cd), cd, "scene"))
                    self.node_of_component[fid] = node
                elif cd.cls == TRANSFORM:
                    self._key_of_transform[fid] = node.key
            self.nodes[node.key] = node

        for inst in self.file.of_class(PREFAB_INSTANCE):
            self._expand(inst)

        self._link()

    def _expand(self, inst):
        guid = guid_in(scalar(inst.body, "m_SourcePrefab"))
        prefab = self.prefab(guid)
        if prefab is None:
            return

        mods = modifications(inst)
        removed = set(re.findall(r"- \{fileID: (-?\d+), guid: \w+, type: 3\}",
                                 section(inst.body, "m_RemovedComponents")))
        added = defaultdict(list)
        for m in re.finditer(r"- targetCorrespondingSourceObject: \{fileID: (-?\d+)[^}]*\}\s*\n"
                             r"\s*insertIndex: -?\d+\s*\n"
                             r"\s*addedObject: \{fileID: (-?\d+)\}", inst.body):
            added[m.group(1)].append(m.group(2))

        for d in prefab.of_class(GAMEOBJECT):
            if d.stripped:
                continue
            own = mods.get(d.fid, {})
            name = own["m_Name"][0] if "m_Name" in own else (scalar(d.body, "m_Name") or "(unnamed)")
            raw_active = own["m_IsActive"][0] if "m_IsActive" in own else scalar(d.body, "m_IsActive")
            node = Node(("inst", inst.fid, d.fid), name, as_int(raw_active, 1) != 0)
            node.prefab = prefab.path

            for fid in component_fids(d.body):
                if fid in removed:
                    continue
                cd = prefab.get(fid)
                if cd is not None and cd.cls == MONOBEHAVIOUR:
                    node.components.append(
                        Component(kind_of(cd), cd, "prefab", fid, mods.get(fid, {})))

            for fid in added.get(d.fid, []):
                cd = self.file.get(fid)
                if cd is not None and cd.cls == MONOBEHAVIOUR:
                    node.components.append(Component(kind_of(cd), cd, "scene"))
                    self.node_of_component[fid] = node

            self.nodes[node.key] = node

        # Stripped Transforms in the scene stand in for the instance's prefab
        # Transforms. Added GameObjects are parented to those, so map them.
        for d in self.file.of_class(TRANSFORM):
            if not d.stripped or ref(d.body, "m_PrefabInstance") != inst.fid:
                continue
            source = ref(d.body, "m_CorrespondingSourceObject")
            tf = prefab.get(source) if source else None
            if tf is None:
                continue
            key = ("inst", inst.fid, ref(tf.body, "m_GameObject"))
            if key in self.nodes:
                self._key_of_transform[d.fid] = key

        # Anything referencing a prefab-derived component does so through a stripped
        # MonoBehaviour doc in the scene, so those fileIDs need resolving too.
        for d in self.file.of_class(MONOBEHAVIOUR):
            if not d.stripped or ref(d.body, "m_PrefabInstance") != inst.fid:
                continue
            source = ref(d.body, "m_CorrespondingSourceObject")
            cd = prefab.get(source) if source else None
            if cd is None:
                continue
            key = ("inst", inst.fid, ref(cd.body, "m_GameObject"))
            if key in self.nodes:
                self.node_of_component[d.fid] = self.nodes[key]

    def _link(self):
        for d in self.file.of_class(TRANSFORM):
            if d.stripped:
                continue
            key = self._key_of_transform.get(d.fid)
            father = ref(d.body, "m_Father")
            if key and father and father != "0":
                self._attach(key, self._key_of_transform.get(father))

        for inst in self.file.of_class(PREFAB_INSTANCE):
            prefab = self.prefabs.get(guid_in(scalar(inst.body, "m_SourcePrefab")))
            if prefab is None:
                continue

            root_key = ("inst", inst.fid, prefab.root_gameobject())
            parent_tf = ref(section(inst.body, "m_Modification"), "m_TransformParent")
            self._attach(root_key, self._key_of_transform.get(parent_tf))

            for d in prefab.of_class(TRANSFORM):
                go, father = ref(d.body, "m_GameObject"), ref(d.body, "m_Father")
                if not go or not father or father == "0":
                    continue
                parent_tf_doc = prefab.get(father)
                if parent_tf_doc is not None:
                    self._attach(("inst", inst.fid, go),
                                 ("inst", inst.fid, ref(parent_tf_doc.body, "m_GameObject")))

    def _attach(self, child_key, parent_key):
        child, parent = self.nodes.get(child_key), self.nodes.get(parent_key)
        if child is None or parent is None or child is parent or child.parent is parent:
            return
        child.parent = parent
        parent.children.append(child)


# ------------------------------------------------------------------------- sensors

def ray_sensor_size(comp):
    rays = comp.get_int("m_RaysPerDirection", 3) * 2 + 1
    tags = comp.array_size("m_DetectableTags")
    return rays * (tags + 2) * max(1, comp.get_int("m_ObservationStacks", 1))


def sensors_of(node, behavior):
    """
    The sensor list the agent would build, in the order ML-Agents sorts it.

    Mirrors Agent.InitializeSensors: sensor components (on children too, when
    UseChildSensors is set), then the CollectObservations VectorSensor, the whole
    list sorted by sensor name. ObservableAttribute sensors are not modelled.
    """
    scope = node.subtree() if behavior.get_int("m_UseChildSensors", 1) != 0 else [node]

    sensors = []
    for host in scope:
        if not host.active():
            continue  # GetComponentsInChildren skips inactive GameObjects
        for comp in host.components:
            if comp.kind not in SENSOR_COMPONENTS:
                continue
            name = comp.get("m_SensorName") or comp.kind
            size = (ray_sensor_size(comp)
                    if comp.kind.startswith("RayPerceptionSensorComponent") else None)
            stacks = max(1, comp.get_int("m_ObservationStacks", 1))
            if stacks > 1:
                name = "StackingSensor_size%d_%s" % (stacks, name)
            sensors.append((name, size, comp, host))

    vector = behavior.get_int("VectorObservationSize", 0)
    if vector > 0:
        stacked = max(1, behavior.get_int("NumStackedVectorObservations", 1))
        name = "VectorSensor_size%d" % vector
        if stacked > 1:
            name = "StackingSensor_size%d_%s" % (stacked, name)
        sensors.append((name, vector * stacked, None, node))

    sensors.sort(key=lambda s: s[0])
    return sensors


def shape_of(size):
    return "[%d]" % size if size is not None else "[?]"


def layout_of(sensors):
    return tuple(shape_of(size) for _, size, _, _ in sensors)


# -------------------------------------------------------------------------- report

def head(title):
    print("\n%s\n%s" % (title, "-" * len(title)))


def collect(scene, agent_classes):
    agents = []
    for node in scene.nodes.values():
        behaviors = [c for c in node.components if c.kind == "BehaviorParameters"]
        brains = [c for c in node.components if c.kind in agent_classes]
        if behaviors or brains:
            agents.append((node, behaviors, brains))
    agents.sort(key=lambda a: a[0].path())
    return agents


def brains_by_runner(agents):
    """
    Agents grouped the way ML-Agents groups them into ModelRunners, plus the ones
    left out. An inactive GameObject or a heuristic-only brain never reaches a
    ModelRunner, so neither can take part in the shape mismatch.
    """
    runners, skipped = defaultdict(list), []
    for node, behaviors, brains in agents:
        if not behaviors:
            continue
        behavior = behaviors[0]
        if not node.active():
            skipped.append((node, "GameObject is inactive"))
            continue
        if behavior.get_int("m_BehaviorType", 0) == HEURISTIC_ONLY:
            skipped.append((node, "brain is HeuristicOnly"))
            continue
        key = (behavior.get("m_BehaviorName") or "(unnamed)",
               behavior.get_guid("m_Model") or "none")
        runners[key].append((node, behavior, brains))
    return runners, skipped


def check_structure(scene, agents, agent_classes):
    head("STRUCTURE  %s" % scene.file.path)
    ok = True

    prefabs = sorted({Path(n.prefab).name for n in scene.nodes.values() if n.prefab})
    print("  Agent subclasses in Assets/ : %s" % (", ".join(sorted(agent_classes)) or "none"))
    print("  prefabs instantiated        : %s" % (", ".join(prefabs) or "none"))
    print("  agent GameObjects           : %d" % len(agents))

    for note in scene.notes:
        print("  ! %s" % note)
        ok = False

    for node, behaviors, brains in agents:
        if not behaviors:
            print("  ! %s has %s but no BehaviorParameters" % (node.path(), brains[0].kind))
            ok = False
        if not brains:
            print("  ! %s has BehaviorParameters but no Agent component" % node.path())
            ok = False

    if ok:
        print("  every agent resolved to a component set")
    return ok


def layouts_of_runner(members):
    layouts = defaultdict(list)
    for node, behavior, brains in members:
        sensors = sensors_of(node, behavior)
        layouts[layout_of(sensors)].append((node, sensors, len(brains)))
    return layouts


def check_layout(agents):
    head("LAYOUT")
    ok = True

    runners, skipped = brains_by_runner(agents)
    by_reason = defaultdict(list)
    for node, why in skipped:
        by_reason[why].append(node)
    for why, nodes in sorted(by_reason.items()):
        print("  %d agents never reach a ModelRunner (%s): %s%s"
              % (len(nodes), why, ", ".join(n.path() for n in nodes[:3]),
                 ", ..." if len(nodes) > 3 else ""))

    for (name, _), members in sorted(runners.items()):
        layouts = layouts_of_runner(members)

        print("  brain %r  %d GameObjects, %d Agent components, %d distinct layout(s)"
              % (name, len(members), sum(c for g in layouts.values() for _, _, c in g),
                 len(layouts)))

        for layout, group in sorted(layouts.items(), key=lambda kv: -len(kv[1])):
            node, sensors, _ = group[0]
            total = sum(s for _, s, _, _ in sensors if s is not None)
            print("    %-24s total obs %-5d x%d GameObjects"
                  % (" ".join(layout) or "(none)", total, len(group)))
            for sensor_name, size, comp, host in sensors:
                where = "CollectObservations"
                if comp is not None:
                    where = comp.origin + ("" if host is node else " on child " + host.name)
                print("      %-46s %-7s %s" % (sensor_name, shape_of(size), where))
            for member, _, _ in group[:4]:
                print("      - %s" % member.path())
            if len(group) > 4:
                print("      - ... and %d more" % (len(group) - 4))

        if len(layouts) > 1:
            ordered = sorted(layouts, key=len, reverse=True)
            print("    ! these layouts share one ModelRunner, so ML-Agents will assert:")
            print("        Number of Sensors must match. %d != %d"
                  % (len(ordered[0]), len(ordered[-1])))
            for i, (a, b) in enumerate(zip(ordered[0], ordered[-1])):
                if a != b:
                    print("        Sensor shapes must match. %s != %s   (at sensor %d)"
                          % (a, b, i))
                    break
            ok = False

    if ok:
        print("  every brain has exactly one sensor layout")
    return ok


def check_model(agents, guids):
    head("MODEL")

    runners, _ = brains_by_runner(agents)
    if not any(guid != "none" for _, guid in runners):
        print("  no model assigned - nothing to compare against")
        return True

    ok = True
    for (name, guid), members in sorted(runners.items()):
        if guid == "none":
            print("  brain %r has no model; it will only run under a trainer" % name)
            continue
        path = guids.get(guid)
        if path is None or not Path(path).is_file():
            print("  ! brain %r points at model guid %s, which is not under Assets/"
                  % (name, guid))
            ok = False
            continue

        blob = Path(path).read_bytes()
        inputs = 0
        while ("obs_%d" % inputs).encode() in blob:
            inputs += 1

        print("  %-24s %d observation inputs" % (Path(path).name, inputs))
        for layout, group in sorted(layouts_of_runner(members).items(),
                                    key=lambda kv: -len(kv[1])):
            matches = len(layout) == inputs
            print("      %-24s %d sensors from %d GameObjects   %s"
                  % (" ".join(layout) or "(none)", len(layout), len(group),
                     "ok" if matches else "MISMATCH"))
            if not matches:
                ok = False
        if not ok:
            print("      the model was exported for %d sensors; a longer list shifts every "
                  "tensor after the first onto the wrong sensor" % inputs)

    if not ok:
        print("  (counted from the obs_N tensor names in the .onnx; sizes are not checked)")
    return ok


def check_wiring(scene, agents, agent_classes):
    head("WIRING")
    clean = True

    duplicates = defaultdict(list)
    for node, behaviors, brains in agents:
        if len(brains) < 2 and len(behaviors) < 2:
            continue
        tally = defaultdict(int)
        for comp in node.components:
            tally[comp.kind] += 1
        signature = (len(brains), len(behaviors),
                     tuple(sorted(k for k, n in tally.items() if n > 1
                                  and k not in agent_classes and k != "BehaviorParameters")))
        duplicates[signature].append(node)

    for (n_brains, n_behaviors, extra), nodes in duplicates.items():
        clean = False
        print("  %d GameObjects carry %d Agent + %d BehaviorParameters components"
              % (len(nodes), n_brains, n_behaviors))
        if extra:
            print("    also duplicated: %s" % ", ".join(extra))
        for node in nodes[:4]:
            print("    - %s" % node.path())
        if len(nodes) > 4:
            print("    - ... and %d more" % (len(nodes) - 4))

    agent_nodes = {node.key for node, _, _ in agents}
    for node, _, brains in agents:
        for comp in brains:
            for field in ("arena", "enemy"):
                if comp.get_ref(field) in (None, "0"):
                    clean = False
                    print("  %s   %s.%s is None   (%s copy)"
                          % (node.path(), comp.kind, field, comp.origin))

            target = comp.get_ref("enemy")
            if target in (None, "0"):
                continue
            owner = scene.node_of_component.get(target)
            if owner is None:
                clean = False
                print("  %s   %s.enemy points at &%s, which resolves to nothing in this "
                      "scene" % (node.path(), comp.kind, target))
            elif owner.key not in agent_nodes:
                clean = False
                print("  %s   %s.enemy points at %s, which is not an agent"
                      % (node.path(), comp.kind, owner.path()))
            elif owner.key == node.key:
                clean = False
                print("  %s   %s.enemy points at itself" % (node.path(), comp.kind))

    if clean:
        print("  one Agent per GameObject, arena and enemy set and resolving on every one")
    return clean


def verify(scene_path, guids, agent_classes):
    scene = Scene(scene_path, guids)
    agents = collect(scene, agent_classes)

    structure_ok = check_structure(scene, agents, agent_classes)
    layout_ok = check_layout(agents)
    model_ok = check_model(agents, guids)
    wiring_ok = check_wiring(scene, agents, agent_classes)

    head("VERDICT")
    for name, ok in (("structure", structure_ok), ("layout", layout_ok),
                     ("model", model_ok), ("wiring", wiring_ok)):
        print("  %-12s %s" % (name, "ok" if ok else "CHECK ABOVE"))

    return structure_ok and layout_ok and model_ok


def main():
    scenes = [Path(a) for a in sys.argv[1:]] or [Path("Assets/OutdoorsScene.unity")]
    for scene_path in scenes:
        if not scene_path.is_file():
            sys.exit("no such scene: %s" % scene_path)

    guids = guid_index()
    agent_classes = project_subclasses("Agent")
    failed = [p for p in scenes if not verify(p, guids, agent_classes)]

    if len(scenes) > 1:
        head("ACROSS %d SCENES" % len(scenes))
        for scene_path in scenes:
            print("  %-12s %s" % ("CHECK ABOVE" if scene_path in failed else "ok",
                                  scene_path))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
