"""
Directed-graph utilities for loop-aware workflow analysis.

What it does:
    Computes strongly connected components (SCCs) and the condensation
    DAG of a directed graph.  An SCC containing a cycle is how the
    system represents an explicit loop drawn in the editor; the
    condensation is always acyclic, so stage ordering never fails.

Entities in it:
    - strongly_connected_components: iterative Tarjan.
    - condense: SCCs plus the inter-component edge set.

How used by other modules:
    backend.orchestration.scheduler condenses the dependency graph to
    build loop-aware execution stages.
"""

from __future__ import annotations


def strongly_connected_components(
    node_ids: list[str],
    edges: list[tuple[str, str]],
) -> list[list[str]]:
    """Compute SCCs with iterative Tarjan, deterministically.

    Args:
        node_ids: All vertices; visit order follows this list.
        edges: Directed (source, target) pairs; endpoints outside
            *node_ids* are ignored.

    Returns:
        Components in reverse-topological order of the condensation,
        each preserving discovery order of its members.
    """
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].append(target)

    counter = 0
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []

    for root in node_ids:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_pos = work[-1]
            if child_pos == 0:
                index[node] = lowlink[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)

            descended = False
            neighbours = adjacency[node]
            while child_pos < len(neighbours):
                neighbour = neighbours[child_pos]
                child_pos += 1
                if neighbour not in index:
                    work[-1] = (node, child_pos)
                    work.append((neighbour, 0))
                    descended = True
                    break
                if neighbour in on_stack:
                    lowlink[node] = min(lowlink[node], index[neighbour])
            if descended:
                continue

            work.pop()
            if lowlink[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                component.reverse()
                components.append(component)
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    return components


def condense(
    node_ids: list[str],
    edges: list[tuple[str, str]],
) -> tuple[list[list[str]], dict[str, int], set[tuple[int, int]]]:
    """Build the condensation DAG of a directed graph.

    Args:
        node_ids: All vertices.
        edges: Directed (source, target) pairs.

    Returns:
        ``(components, component_of, component_edges)`` where
        *component_edges* contains only inter-component pairs, so the
        result is acyclic by construction.
    """
    components = strongly_connected_components(node_ids, edges)
    component_of = {
        member: component_index
        for component_index, members in enumerate(components)
        for member in members
    }
    component_edges = {
        (component_of[source], component_of[target])
        for source, target in edges
        if source in component_of and target in component_of
        and component_of[source] != component_of[target]
    }
    return components, component_of, component_edges
