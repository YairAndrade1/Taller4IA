from __future__ import annotations

from planning.pddl import (
    Action,
    Problem,
    apply_action,
    get_all_groundings,
    is_applicable,
)

# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    # Intento inicial
    # from collections import deque
    # queue = deque([[h for h in hlas]])
    # while queue:
    #     plan = queue.popleft()
    #     idx = None
    #     for i, step in enumerate(plan):
    #         if not is_primitive(step):
    #             idx = i
    #             break
    #     if idx is None:
    #         state = problem.getStartState()
    #         ok = True
    #         for a in plan:
    #             if not is_applicable(state, a):
    #                 ok = False
    #                 break
    #             state = apply_action(state, a)
    #         if ok and problem.isGoalState(state):
    #             return plan
    #         continue
    #     hla = plan[idx]
    #     if hla.refinements:
    #         new_plan = plan[:idx] + hla.refinements[0] + plan[idx+1:]
    #         queue.append(new_plan)
    # return []

    # Prompt
    # Mejora la solución inicial para que la búsqueda HTN haga BFS completa sobre todas las refinaciones de la primera HLA no primitiva, simule la ejecución de los planes totalmente primitivos paso a paso y acepte solo aquellos en los que cada acción sea aplicable y se alcance la meta; además, completa build_htn_hierarchy(problem) creando HLAs concretas (Navigate, PrepareSupplies, ExtractPatient y FullRescueMission) a partir de acciones instanciadas con get_all_groundings(problem.domain, problem.objects).

    # Version final
    from collections import deque

    queue = deque()
    queue.append([h for h in hlas])

    while queue:
        plan = queue.popleft()

        idx = None
        for i, step in enumerate(plan):
            if not is_primitive(step):
                idx = i
                break

        if idx is None:
            state = problem.getStartState()
            failed = False
            for step in plan:
                if not is_applicable(state, step):
                    failed = True
                    break
                state = apply_action(state, step)
            if not failed and problem.isGoalState(state):
                return plan
            continue

        hla: HLA = plan[idx]
        for refinement in hla.refinements:
            new_plan = plan[:idx] + refinement + plan[idx + 1 :]
            queue.append(new_plan)

    return []


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    # Intento inicial
    # all_actions = get_all_groundings(problem.domain, problem.objects)
    # hlas = []
    # for p in problem.objects["patients"]:
    #     # pick first supply and medical post naively
    #     s = problem.objects["supplies"][0]
    #     m = problem.objects["medical_posts"][0]
    #     pick = next((a for a in all_actions if "PickUp(" in a.name and ("%s"%s) in a.name), None)
    #     hlas.append(HLA(f"FullRescue({p})", refinements=[]))
    # return hlas

    # Prompt
    # Mejora la solución inicial construyendo una jerarquía HTN concreta en la que Navigate se refine en acciones Move de un solo paso entre celdas adyacentes, PrepareSupplies(s,m) combine Navigate, PickUp(s), otro Navigate y SetupSupplies(s,m), ExtractPatient(p,m) combine Navigate, PickUp(p), otro Navigate y PutDown(p,m), y FullRescueMission(s,p,m) encadene PrepareSupplies, ExtractPatient y la acción primitiva Rescue.

    # Version final
    all_actions = get_all_groundings(problem.domain, problem.objects)

    move_actions: dict[tuple, Action] = {}
    adjacency: dict[object, set] = {}
    for fluent in problem.initial_state:
        if fluent[0] == "Adjacent":
            adjacency.setdefault(fluent[1], set()).add(fluent[2])

    for a in all_actions:
        if a.name.startswith("Move("):
            from_cell = None
            to_cell = None
            for f in a.del_list:
                if f[0] == "At" and f[1] == "robot":
                    from_cell = f[2]
            for f in a.add_list:
                if f[0] == "At" and f[1] == "robot":
                    to_cell = f[2]
            if from_cell is not None and to_cell is not None:
                move_actions[(from_cell, to_cell)] = a

    def shortest_path(start_cell, goal_cell):
        if start_cell == goal_cell:
            return [start_cell]
        from collections import deque

        queue = deque([start_cell])
        parents: dict[object, object | None] = {start_cell: None}
        while queue:
            cell = queue.popleft()
            for nxt in adjacency.get(cell, set()):
                if nxt in parents:
                    continue
                parents[nxt] = cell
                if nxt == goal_cell:
                    path = [goal_cell]
                    cur = goal_cell
                    while parents[cur] is not None:
                        cur = parents[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                queue.append(nxt)
        return None

    def make_navigate(start_cell, goal_cell) -> HLA | None:
        path = shortest_path(start_cell, goal_cell)
        if not path or len(path) < 2:
            return None
        nav = HLA(f"Navigate_{start_cell}_to_{goal_cell}")
        refinement: list[Action] = []
        for frm, to in zip(path, path[1:]):
            action = move_actions.get((frm, to))
            if action is None:
                return None
            refinement.append(action)
        nav.refinements.append(refinement)
        return nav

    def find_pickup(obj_name, loc):
        for a in all_actions:
            if not a.name.startswith("PickUp("):
                continue
            if obj_name not in a.name:
                continue
            if ("At", obj_name, loc) in a.precond_pos:
                return a
        return None

    def find_putdown(obj_name, loc):
        for a in all_actions:
            if not a.name.startswith("PutDown("):
                continue
            if obj_name not in a.name:
                continue
            if ("At", "robot", loc) in a.precond_pos:
                return a
        return None

    def find_setup(supply_name, loc):
        for a in all_actions:
            if not a.name.startswith("SetupSupplies("):
                continue
            if supply_name not in a.name:
                continue
            if ("Holding", "robot", supply_name) in a.precond_pos and (
                "MedicalPost",
                loc,
            ) in a.precond_pos:
                return a
        return None

    def find_rescue(patient_name, loc):
        for a in all_actions:
            if not a.name.startswith("Rescue("):
                continue
            if patient_name not in a.name:
                continue
            if ("At", "robot", loc) in a.precond_pos and (
                "MedicalPost",
                loc,
            ) in a.precond_pos:
                return a
        return None

    hlas_result: list[HLA] = []

    supplies = problem.objects.get("supplies", [])
    medical_posts = problem.objects.get("medical_posts", [])
    patients = problem.objects.get("patients", [])

    for i, p in enumerate(patients):
        s = supplies[i % len(supplies)] if supplies else None
        m = medical_posts[0] if medical_posts else None
        if s is None or m is None:
            continue

        robot_start = problem.getStartState()
        robot_pos = next(
            (f[2] for f in robot_start if f[0] == "At" and f[1] == "robot"), None
        )
        supply_pos = next(
            (f[2] for f in problem.initial_state if f[0] == "At" and f[1] == s), None
        )
        patient_pos = next(
            (f[2] for f in problem.initial_state if f[0] == "At" and f[1] == p), None
        )
        mission_start = robot_pos if i == 0 else m

        prep_name = f"PrepareSupplies_{s}_to_{m}"
        prep = HLA(prep_name)
        pickup_actions = [find_pickup(s, supply_pos)]
        setup_action = find_setup(s, m)
        nav_to_supply = (
            make_navigate(mission_start, supply_pos)
            if mission_start is not None and supply_pos is not None
            else None
        )
        nav_supply_to_med = (
            make_navigate(supply_pos, m) if supply_pos is not None else None
        )

        if nav_to_supply and pickup_actions and nav_supply_to_med and setup_action:
            prep.refinements.append(
                [nav_to_supply, pickup_actions[0], nav_supply_to_med, setup_action]
            )

        extract = HLA(f"ExtractPatient_{p}_to_{m}")
        pickup_patient = [find_pickup(p, patient_pos)]
        putdown_patient = [find_putdown(p, m)]

        nav_to_patient = (
            make_navigate(m, patient_pos) if patient_pos is not None else None
        )
        nav_patient_to_med = (
            make_navigate(patient_pos, m) if patient_pos is not None else None
        )
        if nav_to_patient and pickup_patient and nav_patient_to_med and putdown_patient:
            extract.refinements.append(
                [
                    nav_to_patient,
                    pickup_patient[0],
                    nav_patient_to_med,
                    putdown_patient[0],
                ]
            )

        full = HLA(f"FullRescue_{p}")
        rescue_action = find_rescue(p, m)
        refinement_body: list = []
        if prep.refinements:
            refinement_body.append(prep)
        if extract.refinements:
            refinement_body.append(extract)
        if rescue_action:
            refinement_body.append(rescue_action)
        if refinement_body:
            full.refinements.append(refinement_body)

        hlas_result.append(full)

    return hlas_result
