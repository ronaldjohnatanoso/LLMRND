"use client";

import { useEffect, useState, useRef } from "react";
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
  const [elements, setElements] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] }>({
    nodes: [],
    edges: [],
  });

  // Build graph state from timeline up to current step
  useEffect(() => {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const nodeMap = new Map<string, GraphNode>();

    // Track layer structure and active nodes for current step
    const layerStructure = new Map<number, string[]>();
    layerStructure.set(0, ["query"]);

    // Determine active node IDs for current step
    const activeNodeIds = new Set<string>();
    const currentStepData = simulation.timeline[currentStep];

    if (currentStepData) {
      // Add nodes based on step type
      if (currentStepData.type === "search") {
        activeNodeIds.add("query");
      } else if (currentStepData.type === "layer_1") {
        activeNodeIds.add("query");
        const nodesData = currentStepData.nodes_data || [];
        nodesData.forEach(n => activeNodeIds.add(n.id));
      } else if (currentStepData.type === "hop_start") {
        const hop = currentStepData.hop || 1;
        activeNodeIds.add("query");
        // Add layer 1 nodes
        const layer1Step = simulation.timeline.find(s => s.type === "layer_1");
        if (layer1Step?.nodes_data) {
          layer1Step.nodes_data.forEach(n => activeNodeIds.add(n.id));
        }
      } else if (currentStepData.type === "propagation") {
        activeNodeIds.add(currentStepData.parent_id || "");
        activeNodeIds.add(currentStepData.child_id || "");
      }
    }

    // Process timeline up to current step
    for (let stepIdx = 0; stepIdx <= currentStep; stepIdx++) {
      const stepData = simulation.timeline[stepIdx];
      const stepType = stepData.type;

      if (stepType === "search") {
        // Add query node
        const isActive = activeNodeIds.has("query");
        nodeMap.set("query", {
          data: {
            id: "query",
            label: simulation.query.substring(0, 30),
            role: "QUERY",
            activation: 1.0,
            layer: 0,
            stepAdded: stepIdx,
            isActive,
          },
          classes: isActive ? "active-node" : "",
        });
      } else if (stepType === "layer_1") {
        const nodesData = stepData.nodes_data || [];
        layerStructure.set(1, []);

        nodesData.forEach((nodeData, i) => {
          if (!nodeMap.has(nodeData.id)) {
            const spread = nodesData.length > 1 ? 80 / (nodesData.length - 1) : 0;
            const x = nodesData.length > 1 ? -40 + i * spread : 0;
            const isActive = activeNodeIds.has(nodeData.id);

            nodeMap.set(nodeData.id, {
              data: {
                id: nodeData.id,
                label: nodeData.text.substring(0, 20) + (nodeData.text.length > 20 ? "..." : ""),
                role: nodeData.role,
                activation: nodeData.activation,
                layer: 1,
                stepAdded: stepIdx,
                isActive,
              },
              position: { x: x * 10, y: 150 },
              classes: isActive ? "active-node" : "",
            });
            layerStructure.set(1, [...(layerStructure.get(1) || []), nodeData.id]);

            // Add edge from query
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
        const newlyActivated = stepData.newly_activated || false;
        const hop = stepData.hop || 1;
        const delta = stepData.delta || 0;

        const parent = nodeMap.get(parentId);
        if (parent && parentId !== "query") {
          // Calculate child position
          const existingChildren = edges.filter((e) => e.data.source === parentId);
          const childNum = existingChildren.length;
          const offset = (childNum + 1) * 30;
          const x = (parent.position?.x || 0) + (childNum % 2 === 0 ? offset : -offset);

          if (newlyActivated && !nodeMap.has(childId)) {
            const intensity = Math.min(1.0, delta * 3);
            const r = Math.floor(100 + 155 * intensity);
            const g = Math.floor(200 - 100 * intensity);
            const b = 150;
            const isActive = activeNodeIds.has(childId);

            nodeMap.set(childId, {
              data: {
                id: childId,
                label: `Hop${hop}`,
                role: `L${hop + 1}`,
                activation: delta,
                layer: hop + 1,
                stepAdded: stepIdx,
                isActive,
              },
              position: { x: Math.max(-200, Math.min(200, x)), y: 150 + hop * 120 },
              classes: `${newlyActivated ? "newly-activated" : ""} ${isActive ? "active-node" : ""}`.trim(),
            });
          }

          // Update active state for existing parent node
          if (nodeMap.has(parentId) && activeNodeIds.has(parentId)) {
            const parentNode = nodeMap.get(parentId);
            if (parentNode) {
              parentNode.data.isActive = true;
              parentNode.classes = `${parentNode.classes || ""} active-node`.trim();
            }
          }

          edges.push({
            data: {
              id: `${parentId}-${childId}-${stepIdx}`,
              source: parentId,
              target: childId,
              strength: delta,
              stepAdded: stepIdx,
            },
            classes: stepIdx === currentStep ? "new-edge" : "",
          });
        }
      }
    }

    nodes.push(...Array.from(nodeMap.values()));
    setElements({ nodes, edges });
  }, [simulation, currentStep]);

  // Initialize/update Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements: [...elements.nodes, ...elements.edges] as ElementDefinition[],
      style: [
        {
          selector: "node",
          style: {
            "background-color": (ele: any) => roleColors[ele.data("role")] || "#888",
            label: "data(label)",
            color: "#fff",
            "text-valign": "center",
            "text-halign": "center",
            width: (ele: any) => 30 + ele.data("activation") * 30,
            height: (ele: any) => 30 + ele.data("activation") * 30,
            "font-size": (ele: any) => Math.max(8, ele.data("activation") * 10),
            "font-weight": "bold",
            "border-width": 2,
            "border-color": "#fff",
            "text-outline-width": 2,
            "text-outline-color": "#000",
            "text-max-width": "80px",
            "text-wrap": "wrap",
            opacity: 0.6,
            "transition-property": "border-width, border-color, opacity, width, height",
            "transition-duration": "0.3s",
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
          selector: "node.newly-activated",
          style: {
            "border-width": 4,
            "border-color": "#FFD700",
            "border-opacity": 0.8,
            "overlay-color": "#FFD700",
            "overlay-padding": 6,
            "overlay-opacity": 0.4,
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
            opacity: (ele: any) => 0.3 + Math.min(0.7, ele.data("strength") * 2),
            "arrow-scale": 0.8,
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
      minZoom: 0.5,
      maxZoom: 2,
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [elements]);

  return (
    <div className="w-full h-[600px] bg-slate-900 rounded-lg overflow-hidden border border-slate-700">
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
}
