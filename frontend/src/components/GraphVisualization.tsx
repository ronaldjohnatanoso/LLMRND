"use client";

import { useEffect, useState, useRef, useMemo } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";
import { SimulationResponse, SimulationStep, GraphNode, GraphEdge } from "@/types";

interface GraphVisualizationProps {
  simulation: SimulationResponse;
  currentStep: number;
}

const roleColors: Record<string, string> = {
  QUERY: "#FF6B6B",
  FACT: "#4ECDC4",
  OBSERVATION: "#95E1D3",
  GOAL: "#FF6B6B",
  CONSTRAINT: "#C44D58",
  DECISION: "#FFB74D",
  PROPAGATED: "#88CC88",
};

export default function GraphVisualization({ simulation, currentStep }: GraphVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [hoveredNode, setHoveredNode] = useState<{ node: any; x: number; y: number } | null>(null);
  const [showAllEdges, setShowAllEdges] = useState(true);
  const [edgeOpacity, setEdgeOpacity] = useState(0.3);

  // Store stable positions that persist across step changes
  const stablePositionsRef = useRef(new Map<string, { x: number; y: number }>());
  const usedPositionsRef = useRef(new Map<string, Set<string>>());

  // Build nodes and edges UP TO CURRENT STEP (not full simulation)
  // This rebuilds as currentStep changes to show progressive graph growth
  const fullGraphState = useMemo(() => {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const nodeMap = new Map<string, GraphNode>();

    // Track nodes by layer for better positioning
    const layerNodes = new Map<number, string[]>();
    layerNodes.set(0, ["query"]);

    // Track positions used to avoid overlap (use ref to persist across renders)
    if (!usedPositionsRef.current) {
      usedPositionsRef.current = new Map();
    }

    // Helper to find next available position in a layer
    const findAvailablePosition = (layer: number, preferredX: number, y: number, nodeId: string) => {
      // Check if we already have a stable position for this node
      if (stablePositionsRef.current.has(nodeId)) {
        return stablePositionsRef.current.get(nodeId)!;
      }

      const layerPositions = usedPositionsRef.current.get(String(layer)) || new Set();
      const spacing = 35; // Reduced spacing to fit more nodes
      const maxX = 400; // Increased range

      const gridX = Math.round(preferredX / spacing) * spacing;
      const gridY = Math.round(y / spacing) * spacing;
      const key = `${gridX},${gridY}`;

      if (!layerPositions.has(key)) {
        layerPositions.add(key);
        usedPositionsRef.current.set(String(layer), layerPositions);
        const pos = { x: gridX, y };
        stablePositionsRef.current.set(nodeId, pos);
        return pos;
      }

      // Spiral outward with finer angle resolution for better distribution
      for (let radius = 1; radius < 30; radius++) {
        for (let angle = 0; angle < 360; angle += 30) { // 30-degree increments instead of 45
          const rad = (angle * Math.PI) / 180;
          const testX = Math.round((preferredX + radius * spacing * Math.cos(rad)) / spacing) * spacing;
          const testY = Math.round((y + radius * spacing * Math.sin(rad)) / spacing) * spacing;
          const testKey = `${testX},${testY}`;

          if (!layerPositions.has(testKey) && Math.abs(testX) < maxX) {
            layerPositions.add(testKey);
            usedPositionsRef.current.set(String(layer), layerPositions);
            const pos = { x: testX, y: testY };
            stablePositionsRef.current.set(nodeId, pos);
            return pos;
          }
        }
      }

      return { x: preferredX, y };
    };

    // Process timeline UP TO CURRENT STEP only
    for (let stepIdx = 0; stepIdx <= currentStep; stepIdx++) {
      const stepData = simulation.timeline[stepIdx];
      const stepType = stepData.type;

      if (stepType === "search") {
        if (!nodeMap.has("query")) {
          nodeMap.set("query", {
            data: {
              id: "query",
              label: simulation.query.substring(0, 30),
              role: "QUERY",
              activation: 1.0,
              layer: 0,
              stepAdded: stepIdx,
            },
            classes: "inactive-hop",
          });
        }
      } else if (stepType === "layer_1") {
        const nodesData = stepData.nodes_data || [];
        layerNodes.set(1, []);

        nodesData.forEach((nodeData, i) => {
          if (!nodeMap.has(nodeData.id)) {
            const spread = nodesData.length > 1 ? 100 / nodesData.length : 0;
            const preferredX = nodesData.length > 1 ? -50 + i * spread : 0;
            const y = 120;
            const pos = findAvailablePosition(1, preferredX, y, nodeData.id);

            nodeMap.set(nodeData.id, {
              data: {
                id: nodeData.id,
                label: nodeData.text.substring(0, 12) + (nodeData.text.length > 12 ? "..." : ""),
                role: nodeData.role,
                activation: nodeData.activation,
                layer: 1,
                stepAdded: stepIdx,
              },
              position: { x: pos.x * 10, y },
              classes: "inactive-hop",
            });

            edges.push({
              data: {
                id: `query-${nodeData.id}`,
                source: "query",
                target: nodeData.id,
                strength: nodeData.activation,
                stepAdded: stepIdx,
              },
            });
          }
        });
      } else if (stepType === "propagation") {
        const parentId = stepData.parent_id || "";
        const childId = stepData.child_id || "";
        const hop = stepData.hop || 1;

        const parent = nodeMap.get(parentId);
        if (parent && parentId !== "query" && !nodeMap.has(childId)) {
          let labelText = `L${hop + 1}`;
          let nodeRole = "PROPAGATED";
          const allFinalStates = simulation.final_states || [];
          const finalNode = allFinalStates.find(n => n.id === childId);
          if (finalNode?.text) {
            labelText = finalNode.text.substring(0, 20) + (finalNode.text.length > 20 ? "..." : "");
          }
          if (finalNode?.role) {
            nodeRole = finalNode.role;
          }

          const parentX = parent.position?.x || 0;
          const parentY = parent.position?.y || 0;
          const layer = hop + 1;
          const y = parentY + 100;

          const existingInLayer = layerNodes.get(layer) || [];
          const spread = existingInLayer.length > 0 ? 100 / (existingInLayer.length + 1) : 0;
          const preferredX = parentX + (existingInLayer.length % 2 === 0 ? (existingInLayer.length + 1) * spread : -(existingInLayer.length + 1) * spread);

          const pos = findAvailablePosition(layer, preferredX, y, childId);

          nodeMap.set(childId, {
            data: {
              id: childId,
              label: labelText.substring(0, 12) + (labelText.length > 12 ? "..." : ""),
              role: nodeRole,
              activation: stepData.delta || 0,
              layer: hop + 1,
              stepAdded: stepIdx,
            },
            position: { x: pos.x, y: pos.y },
            classes: "inactive-hop",
          });

          layerNodes.set(layer, [...existingInLayer, childId]);
        }

        edges.push({
          data: {
            id: `${parentId}-${childId}-${stepIdx}`,
            source: parentId,
            target: childId,
            strength: stepData.delta || 0,
            stepAdded: stepIdx,
          },
        });
      }
    }

    nodes.push(...Array.from(nodeMap.values()));
    return { nodes, edges };
  }, [simulation, currentStep]);

  // Calculate which nodes are active in CURRENT step (vs just present)
  const currentStepState = useMemo(() => {
    const activeNodeIds = new Set<string>();
    const propagatedIds = new Set<string>();
    const failedGate2Ids = new Set<string>();

    const currentStepData = simulation.timeline[currentStep];

    // Determine which nodes are active in the CURRENT step
    if (currentStepData) {
      if (currentStepData.type === "search") {
        activeNodeIds.add("query");
      } else if (currentStepData.type === "layer_1") {
        activeNodeIds.add("query");
        const nodesData = currentStepData.nodes_data || [];
        nodesData.forEach(n => activeNodeIds.add(n.id));
      } else if (currentStepData.type === "hop_start") {
        activeNodeIds.add("query");
        const layer1Step = simulation.timeline.find(s => s.type === "layer_1");
        if (layer1Step?.nodes_data) {
          layer1Step.nodes_data.forEach(n => activeNodeIds.add(n.id));
        }
      } else if (currentStepData.type === "propagation") {
        activeNodeIds.add(currentStepData.parent_id || "");
        activeNodeIds.add(currentStepData.child_id || "");
      }
    }

    // Track propagated and failed gate 2 nodes across all steps up to current
    for (let stepIdx = 0; stepIdx <= currentStep; stepIdx++) {
      const step = simulation.timeline[stepIdx];
      if (step.type === "propagation" && step.newly_activated === true) {
        propagatedIds.add(step.parent_id || "");
      } else if (step.type === "gate_2_fail") {
        failedGate2Ids.add(step.node_id || "");
      }
    }

    return { activeNodeIds, propagatedIds, failedGate2Ids };
  }, [simulation, currentStep]);

  // Calculate dynamic scaling based on node count
  const nodeCount = fullGraphState.nodes.length;
  const baseNodeSize = Math.max(15, 40 - nodeCount * 0.5); // Shrink as nodes increase
  const baseFontSize = Math.max(6, 12 - nodeCount * 0.1);

  // Initialize/update Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;
    if (fullGraphState.nodes.length === 0) return;

    const cy = cyRef.current;

    if (!cy) {
      // First time initialization - build the FULL graph once
      const newCy = cytoscape({
        container: containerRef.current,
        elements: [...fullGraphState.nodes, ...fullGraphState.edges] as ElementDefinition[],
        style: [
          {
            selector: "node",
            style: {
              "background-color": (ele: any) => roleColors[ele.data("role")] || "#888",
              label: "data(label)",
              color: "#fff",
              "text-valign": "center",
              "text-halign": "center",
              width: (ele: any) => baseNodeSize * (0.8 + ele.data("activation") * 0.2),
              height: (ele: any) => baseNodeSize * (0.8 + ele.data("activation") * 0.2),
              "font-size": (ele: any) => baseFontSize * 0.8,
              "font-weight": "bold",
              "border-width": 2,
              "border-color": "#fff",
              "text-outline-width": 1.5,
              "text-outline-color": "#000",
              "text-max-width": "50px",
              "text-wrap": "wrap",
              "text-overflow-wrap": "anywhere",
              opacity: 0.6,
              "transition-property": "border-width, border-color, opacity, width, height, background-color",
              "transition-duration": 200,
              "transition-timing-function": "ease-in-out",
            },
          },
          {
            selector: "node.active-node",
            style: {
              opacity: 1,
              "border-width": 6,
              "border-color": "#00FF88",
              "border-opacity": 1,
              "overlay-color": "#00FF88",
              "overlay-padding": 8,
              "overlay-opacity": 0.3,
              "z-index": 100,
            },
          },
          {
            selector: "node.propagated-node",
            style: {
              opacity: 0.7,
              "border-width": 3,
              "border-color": "#4A9EFF",
              "border-opacity": 1,
              "overlay-color": "#4A9EFF",
              "overlay-padding": 4,
              "overlay-opacity": 0.2,
              "z-index": 50,
            },
          },
          {
            selector: "node.failed-gate2",
            style: {
              opacity: 0.5,
              "border-width": 3,
              "border-color": "#FF6B6B",
              "border-opacity": 1,
              "border-style": "dashed",
              "border-dash-pattern": [4, 2],
              "overlay-color": "#FF6B6B",
              "overlay-padding": 3,
              "overlay-opacity": 0.15,
              "z-index": 45,
            },
          },
          {
            selector: "node.inactive-hop",
            style: {
              opacity: 0.4,
              "border-width": 2,
              "border-color": "#666",
              "overlay-color": "#666",
              "overlay-padding": 2,
              "overlay-opacity": 0.2,
            },
          },
          {
            selector: "edge",
            style: {
              width: (ele: any) => 1 + ele.data("strength") * 3,
              "line-color": "#4ECDC4",
              "target-arrow-color": "#4ECDC4",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              opacity: edgeOpacity,
              "arrow-scale": 0.8,
              "transition-property": "width, line-color, target-arrow-color, opacity",
              "transition-duration": 200,
              "transition-timing-function": "ease-in-out",
            },
          },
          {
            selector: "edge.new-edge",
            style: {
              "line-color": "#FFD700",
              "target-arrow-color": "#FFD700",
              "line-style": "dashed",
              "line-dash-pattern": [6, 3],
            },
          },
        ],
        layout: {
          name: "preset",
        },
        minZoom: 0.3,
        maxZoom: 3,
      });

      cyRef.current = newCy;

      // Fit on initial creation
      setTimeout(() => {
        if (cyRef.current && !cyRef.current.destroyed()) {
          cyRef.current.fit(undefined, 50);
        }
      }, 100);

      // Add hover event handlers
      newCy.on("mouseover", "node", (evt) => {
        const node = evt.target;
        const pos = node.renderedPosition();
        const data = node.data();

        // Get full text from timeline data or final states
        let fullText = data.label;

        // First check timeline nodes_data (Layer 1 nodes)
        const allTimelineNodes = simulation.timeline.flatMap(step => step.nodes_data || []);
        let nodeData = allTimelineNodes.find(n => n.id === data.id);
        if (nodeData?.text) {
          fullText = nodeData.text;
        }

        // If not found in timeline, check final_states (hop/propagated nodes)
        if (!nodeData) {
          const allFinalStates = simulation.final_states || [];
          const finalNode = allFinalStates.find(n => n.id === data.id);
          if (finalNode?.text) {
            fullText = finalNode.text;
          }
        }

        // Get neighbor count
        const neighborhood = node.neighborhood().nodes();
        const neighborCount = neighborhood.length - 1; // Exclude the node itself

        // Collect all signal reception events for this node UP TO CURRENT STEP
        const propagationsToThisNode: Array<{
          fromId: string;
          fromText: string;
          delta: number;
          oldActivation: number;
          newActivation: number;
          hop: number;
          edgeWeight: number;
          roleBoost: number;
          decayFactor: number;
        }> = [];
        let totalDeltaReceived = 0;
        let timesReceived = 0;

        for (let stepIdx = 0; stepIdx <= currentStep; stepIdx++) {
          const step = simulation.timeline[stepIdx];

          // Track propagation events
          if (step.type === "propagation" && step.child_id === data.id) {
            totalDeltaReceived += step.delta || 0;
            timesReceived++;

            // Get parent node info
            const parentNode = simulation.final_states?.find(n => n.id === step.parent_id);
            propagationsToThisNode.push({
              fromId: step.parent_id || "",
              fromText: parentNode?.text || "Unknown",
              delta: step.delta || 0,
              oldActivation: step.old_activation || 0,
              newActivation: step.new_activation || 0,
              hop: step.hop || 0,
              edgeWeight: step.edge_weight || 0,
              roleBoost: step.role_boost || 1.0,
              decayFactor: step.decay_factor || 1.0,
            });
          }

          // Track layer_1 initial activations (layer 1 nodes receiving from query)
          if (step.type === "layer_1") {
            const nodesData = step.nodes_data || [];
            const nodeData = nodesData.find(n => n.id === data.id);
            if (nodeData) {
              totalDeltaReceived += nodeData.activation || 0;
              timesReceived++;

              // Add query as the source
              propagationsToThisNode.push({
                fromId: "query",
                fromText: simulation.query,
                delta: nodeData.activation || 0,
                oldActivation: 0,
                newActivation: nodeData.activation || 0,
                hop: 0,
                edgeWeight: 1.0, // Default weight for layer 0 -> layer 1
                roleBoost: 1.0,
                decayFactor: 1.0,
              });
            }
          }
        }

        setHoveredNode({
          node: {
            ...data,
            fullText,
            neighborCount,
            activated: data.wasActivated || false,
            propagated: (data as any).propagated || false,
            totalDeltaReceived,
            timesReceived,
            propagations: propagationsToThisNode,
          },
          x: pos.x,
          y: pos.y,
        });
      });

      newCy.on("mouseout", "node", () => {
        // Hide immediately, but let tooltip handle its own hover state
        setHoveredNode(null);
      });

      return () => {
        if (cyRef.current) {
          cyRef.current.destroy();
          cyRef.current = null;
        }
      };
    }
  }, [fullGraphState, baseNodeSize, baseFontSize, edgeOpacity, simulation]);

  // Update classes reactively - only updates CSS, never touches structure or positions
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || fullGraphState.nodes.length === 0) return;

    cy.batch(() => {
      // Update node classes based on current step state
      fullGraphState.nodes.forEach(node => {
        const cyNode = cy.getElementById(node.data.id);
        if (cyNode) {
          const isActive = currentStepState.activeNodeIds.has(node.data.id);
          const propagated = currentStepState.propagatedIds.has(node.data.id);
          const failedGate2 = currentStepState.failedGate2Ids.has(node.data.id);

          const newClass = isActive
            ? "active-node"
            : propagated
              ? "propagated-node"
              : failedGate2
                ? "failed-gate2"
                : "inactive-hop";

          // Only update if class actually changed
          if (!cyNode.hasClass(newClass)) {
            cyNode.classes(newClass);
          }
        }
      });

      // Update edge visibility based on showAllEdges setting
      // When showAllEdges is false, only show edges added in current step
      fullGraphState.edges.forEach(edge => {
        const cyEdge = cy.getElementById(edge.data.id);
        if (cyEdge) {
          const shouldBeVisible = showAllEdges || edge.data.stepAdded === currentStep;

          const currentDisplay = cyEdge.style('display');
          const newDisplay = shouldBeVisible ? 'element' : 'none';

          if (currentDisplay !== newDisplay) {
            cyEdge.style('display', newDisplay);
          }
        }
      });
    });
  }, [currentStep, showAllEdges, fullGraphState, currentStepState]);

  return (
    <div className="relative w-full h-[600px] bg-slate-900 rounded-lg overflow-hidden border border-slate-700">
      {/* Legend */}
      <div className="absolute top-3 left-3 z-40 bg-slate-800/95 backdrop-blur-sm rounded-lg p-3 border border-slate-600 text-xs space-y-3">
        <div className="font-semibold text-white mb-2">Legend</div>

        {/* Controls */}
        <div className="space-y-2">
          {/* Edge visibility toggle */}
          <button
            onClick={() => setShowAllEdges(!showAllEdges)}
            className={`w-full px-3 py-2 rounded text-left transition-all ${
              showAllEdges
                ? "bg-cyan-900/50 border border-cyan-600 text-cyan-300"
                : "bg-slate-700/50 border border-slate-600 text-slate-400 hover:bg-slate-700"
            }`}
          >
            <div className="flex items-center justify-between">
              <span>{showAllEdges ? "All Edges" : "Current Edges"}</span>
              <span className="text-[10px] opacity-70">
                {showAllEdges ? "(🕸️)" : "(📍)"}
              </span>
            </div>
          </button>

          {/* Edge opacity toggle */}
          <button
            onClick={() => setEdgeOpacity(edgeOpacity === 0.3 ? 0.1 : edgeOpacity === 0.1 ? 0.05 : 0.3)}
            className={`w-full px-3 py-2 rounded text-left transition-all ${
              edgeOpacity === 0.3
                ? "bg-purple-900/50 border border-purple-600 text-purple-300"
                : edgeOpacity === 0.1
                  ? "bg-purple-900/30 border border-purple-700 text-purple-400"
                  : "bg-slate-700/50 border border-slate-600 text-slate-400 hover:bg-slate-700"
            }`}
          >
            <div className="flex items-center justify-between">
              <span>Edge Opacity</span>
              <span className="text-[10px] font-mono">
                {edgeOpacity === 0.3 ? "30%" : edgeOpacity === 0.1 ? "10%" : "5%"}
              </span>
            </div>
          </button>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded border-[3px] border-green-400 bg-green-400/30"></div>
            <span className="text-slate-300">Currently Active</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded opacity-70 bg-blue-400/20" style={{ border: "2px solid #4A9EFF" }}></div>
            <span className="text-slate-300">Propagated</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded opacity-40 bg-gray-500 border-2 border-gray-500 border-dashed"></div>
            <span className="text-slate-300">Inactive</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded opacity-50 bg-red-400/20" style={{ border: "2px dashed #FF6B6B" }}></div>
            <span className="text-slate-300">Gate 2 Failed</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 border-t-2 border-dashed border-yellow-400"></div>
            <span className="text-slate-300">New Edge</span>
          </div>
        </div>
      </div>

      <div ref={containerRef} className="w-full h-full" />

      {/* Tooltip */}
      {hoveredNode && (
        <div
          data-tooltip="true"
          className="absolute z-50 bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-4 max-w-sm pointer-events-auto"
          style={{
            left: hoveredNode.x + 20,
            top: hoveredNode.y - 20,
            transform: "translate(0, -50%)",
          }}
          onMouseEnter={() => {
            // Keep tooltip visible when hovering over it
          }}
          onMouseLeave={() => {
            setHoveredNode(null);
          }}
        >
          <div className="space-y-3">
            {/* Role Badge */}
            <div className="flex items-center gap-2">
              <span
                className="px-2 py-1 rounded text-xs font-semibold text-white"
                style={{
                  backgroundColor: roleColors[hoveredNode.node.role] || "#888",
                }}
              >
                {hoveredNode.node.role}
              </span>
              <span className="text-slate-400 text-xs">ID: {hoveredNode.node.id?.slice(0, 8)}...</span>
            </div>

            {/* Full Text */}
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">Text</div>
              <div className="text-white text-sm leading-relaxed">{hoveredNode.node.fullText}</div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-2 text-xs">
              <div className="bg-slate-900/50 rounded p-2">
                <div className="text-slate-400">Activation</div>
                <div className="text-blue-400 font-mono font-bold">
                  {hoveredNode.node.activation?.toFixed(3) || "N/A"}
                </div>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <div className="text-slate-400">Layer</div>
                <div className="text-purple-400 font-mono font-bold">
                  {hoveredNode.node.layer ?? "N/A"}
                </div>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <div className="text-slate-400">Step</div>
                <div className="text-green-400 font-mono font-bold">
                  #{hoveredNode.node.stepAdded ?? "?"}
                </div>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <div className="text-slate-400">Connections</div>
                <div className="text-orange-400 font-mono font-bold">
                  {hoveredNode.node.neighborCount ?? 0}
                </div>
              </div>
            </div>

            {/* Status badges */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className={`bg-slate-900/50 rounded p-2 flex items-center justify-between gap-2 ${
                (hoveredNode.node as any).activated ? "border border-green-500/50" : "opacity-50"
              }`}>
                <span className="text-slate-400">Activated</span>
                <span className={`font-mono font-bold ${
                  (hoveredNode.node as any).activated ? "text-green-400" : "text-gray-500"
                }`}>
                  {(hoveredNode.node as any).activated ? "✓" : "✗"}
                </span>
              </div>
              <div className={`bg-slate-900/50 rounded p-2 flex items-center justify-between gap-2 ${
                (hoveredNode.node as any).propagated ? "border border-purple-500/50" : "opacity-50"
              }`}>
                <span className="text-slate-400">Propagated</span>
                <span className={`font-mono font-bold ${
                  (hoveredNode.node as any).propagated ? "text-purple-400" : "text-gray-500"
                }`}>
                  {(hoveredNode.node as any).propagated ? "✓" : "✗"}
                </span>
              </div>
              <div className={`bg-slate-900/50 rounded p-2 flex items-center justify-between gap-2 ${
                (hoveredNode.node as any).failedGate2 ? "border border-red-500/50" : "opacity-50"
              }`}>
                <span className="text-slate-400">Gate 2 Fail</span>
                <span className={`font-mono font-bold ${
                  (hoveredNode.node as any).failedGate2 ? "text-red-400" : "text-gray-500"
                }`}>
                  {(hoveredNode.node as any).failedGate2 ? "✓" : "✗"}
                </span>
              </div>
            </div>

            {/* Accumulated signal info - always show */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-slate-900/50 rounded p-2">
                <div className="text-slate-400">Times Received</div>
                <div className="text-cyan-400 font-mono font-bold">
                  {(hoveredNode.node as any).timesReceived || 0}
                </div>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <div className="text-slate-400">Total Δ Received</div>
                <div className="text-cyan-400 font-mono font-bold">
                  {((hoveredNode.node as any).totalDeltaReceived || 0).toFixed(3)}
                </div>
              </div>
            </div>

            {/* Propagation calculation breakdown */}
            {(hoveredNode.node as any).propagations && (hoveredNode.node as any).propagations.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs text-slate-400 uppercase tracking-wide font-semibold">
                  Signal Received (Δ = parent_sim × weight × boost × decay)
                </div>
                <div className="space-y-2 max-h-[200px] overflow-y-auto pr-1">
                  {(hoveredNode.node as any).propagations.map((prop: any, i: number) => (
                    <div key={i} className="bg-slate-900/70 rounded p-2 text-xs border-l-2 border-cyan-500">
                      <div className="flex justify-between items-start mb-1">
                        <span className="text-slate-300 font-medium truncate mr-2" title={prop.fromText}>
                          From: {prop.fromText.substring(0, 20)}...
                        </span>
                        <span className="text-cyan-400 font-mono font-bold">
                          +{prop.delta.toFixed(3)}
                        </span>
                      </div>
                      <div className="text-slate-500 text-[10px] space-y-0.5">
                        <div>Activation: {prop.oldActivation.toFixed(3)} → {prop.newActivation.toFixed(3)}</div>
                        <div>
                          calc: {((prop.oldActivation > 0 ? prop.oldActivation : (prop.delta / (prop.edgeWeight * prop.roleBoost * prop.decayFactor))).toFixed(3))} × {prop.edgeWeight.toFixed(2)} × {prop.roleBoost.toFixed(2)} × {prop.decayFactor.toFixed(2)}
                        </div>
                        <div className="text-slate-600">
                          hop={prop.hop} | weight={prop.edgeWeight.toFixed(2)} | boost={prop.roleBoost.toFixed(2)}x | decay={prop.decayFactor.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Similarity if available */}
            {hoveredNode.node.similarity_to_query !== undefined && (
              <div className="bg-slate-900/50 rounded p-2">
                <div className="text-slate-400 text-xs">Similarity to Query</div>
                <div className="text-yellow-400 font-mono font-bold">
                  {hoveredNode.node.similarity_to_query.toFixed(3)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
