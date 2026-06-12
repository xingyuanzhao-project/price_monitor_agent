/**
 * Custom ReactFlow edges.
 *
 * DataFlowEdge — single directional line with an arrowhead (marker set
 * by the store's edge converter). When source === target it renders a
 * self-loop arc from the node's output port around the side back into
 * its input port: the visual form of an explicit single-agent loop.
 *
 * ToolCallEdge — one edge, two opposed arcs (call and return), both
 * with arrowheads. One green double edge = the full agent <=> tool
 * round trip. Stroke color comes from the edge's style (owned by the
 * store), not duplicated here.
 */

import { BaseEdge, getSmoothStepPath, type EdgeProps } from "@xyflow/react";

export function DataFlowEdge({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
}: EdgeProps) {
  let path: string;
  if (source === target) {
    // Self-loop: out (bottom) arcs around the right side into in (top).
    const reach = 80;
    path =
      `M ${sourceX} ${sourceY} ` +
      `C ${sourceX + reach} ${sourceY + reach * 0.6}, ` +
      `${targetX + reach} ${targetY - reach * 0.6}, ` +
      `${targetX} ${targetY}`;
  } else {
    [path] = getSmoothStepPath({
      sourceX, sourceY, sourcePosition,
      targetX, targetY, targetPosition,
    });
  }
  return <BaseEdge id={id} path={path} style={style} markerEnd={markerEnd} />;
}

export function ToolCallEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style,
}: EdgeProps) {
  const deltaX = targetX - sourceX;
  const deltaY = targetY - sourceY;
  const length = Math.hypot(deltaX, deltaY) || 1;
  // Unit normal: offsets the two arcs to opposite sides of the chord.
  const normalX = -deltaY / length;
  const normalY = deltaX / length;
  const offset = Math.min(18, 6 + length / 20);
  const midX = (sourceX + targetX) / 2;
  const midY = (sourceY + targetY) / 2;

  const callPath =
    `M ${sourceX} ${sourceY} ` +
    `Q ${midX + normalX * offset} ${midY + normalY * offset} ${targetX} ${targetY}`;
  const returnPath =
    `M ${targetX} ${targetY} ` +
    `Q ${midX - normalX * offset} ${midY - normalY * offset} ${sourceX} ${sourceY}`;

  const stroke = (style?.stroke as string) ?? "currentColor";
  const markerId = `tool-call-arrow-${id}`;

  return (
    <>
      <defs>
        <marker
          id={markerId}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill={stroke} />
        </marker>
      </defs>
      <BaseEdge id={id} path={callPath} style={style} markerEnd={`url(#${markerId})`} />
      <path
        d={returnPath}
        className="react-flow__edge-path"
        fill="none"
        stroke={stroke}
        strokeWidth={1}
        markerEnd={`url(#${markerId})`}
      />
    </>
  );
}
