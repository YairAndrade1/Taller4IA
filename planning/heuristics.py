from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects, get_all_groundings, get_applicable_actions


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    # (1) Versión inicial del código
    #     ### Your code here ###
    #
    #     ### End of your code ###
    
    # (2) Prompts utilizados para refinarla
    # "Por favor ayúdame a implementar esta heurística asumiendo que no hay precondiciones. Usa variables en español como 'fluentes_faltantes' y un ciclo while tradicional para encontrar cuántas acciones cubren el objetivo. Asegúrate de que parezca escrito por un estudiante sin atajos."
    
    # (3) Versión final del código
    fluentes_faltantes = goal - state
    acciones_instanciadas = get_all_groundings(domain, objects)
    acciones_necesarias = 0
    while len(fluentes_faltantes) > 0:
        mejor_accion = None
        max_cobertura = -1
        for accion in acciones_instanciadas:
            fluentes_nuevos = accion.add_list & fluentes_faltantes
            cobertura = len(fluentes_nuevos)
            if cobertura > max_cobertura:
                max_cobertura = cobertura
                mejor_accion = accion
        if mejor_accion is None or max_cobertura == 0:
            # Si ninguna acción puede darnos los fluentes que faltan, es un callejón sin salida
            return float('inf')
        fluentes_faltantes = fluentes_faltantes - mejor_accion.add_list
        acciones_necesarias = acciones_necesarias + 1
    return float(acciones_necesarias)


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    # (1) Versión inicial del código
    #     ### Your code here ###
    #
    #     ### End of your code ###
    
    # (2) Prompts utilizados para refinarla
    # "Por favor ayúdame a implementar esta heurística relajada. Usa una variable 'estado_actual' y un contador 'pasos'. Encuentra iterativamente la acción que aporte más fluentes nuevos al objetivo y añade sus efectos positivos al estado actual sin borrar nada. Hazlo paso a paso."
    
    # (3) Versión final del código
    estado_actual = state
    pasos = 0
    while not goal.issubset(estado_actual):
        acciones_aplicables = get_applicable_actions(estado_actual, domain, objects)
        mejor_accion = None
        max_cobertura = -1
        # Lo que aún nos falta para lograr el objetivo desde el estado actual
        fluentes_faltantes = goal - estado_actual
        for accion in acciones_aplicables:
            fluentes_nuevos = accion.add_list & fluentes_faltantes
            cobertura = len(fluentes_nuevos)
            if cobertura > max_cobertura:
                max_cobertura = cobertura
                mejor_accion = accion
        if mejor_accion is None or max_cobertura == 0:
            return float('inf')
        # Simulamos aplicar la acción pero ignoramos su delete_list (monótono)
        estado_actual = estado_actual | mejor_accion.add_list
        pasos = pasos + 1
    return float(pasos)
