"use client";

import { useEffect, useRef } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";
import { Node as NodeType } from "@/types";

interface NodeDetailsPanelProps {
  node: NodeType;
  neighbors: NodeType[];
  onClose: () => void;
}

const roleColors: Record<string, string> = {
  QUERY: "#FF6B6B",
  FACT: "#4ECDC4",
  OBSERVATION: "#95E1D3",
  GOAL: "#FF6B6B",
  CONSTRAINT: "#C44D58",
  DECISION: "#FFB74D",
};

export default function NodeDetailsPanel({ node, neighbors, onClose }: NodeDetailsPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const elements: ElementDefinition[] = [
      // Center node
      {
        data: {
          id: node.id,
          label: node.text.substring(0, 30) + (node.text.length > 30 ? "..." : ""),
          role: node.role,
          activation: node.activation,
          confidence: node.confidence,
        },
        classes: "center",
      },
      // Neighbors
      ...neighbors.map((n) => ({
        data: {
          id: n.id,
          label: n.text.substring(0, 30) + (n.text.length > 30 ? "..." : ""),
          role: n.role,
          activation: n.activation,
          confidence: n.confidence,
        },
      })),
      // Edges
      ...neighbors.map((n) => ({
        data: {
          id: `${node.id}-${n.id}`,
          source: node.id,
          target: n.id,
        },
      })),
    ];

    const cy = cytoscape({
      container: containerRef.current,
      elements,
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
          },
        },
        {
          selector: "node.center",
          style: {
            "border-width": 4,
            "border-color": "#FFD700",
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": "#4ECDC4",
            "target-arrow-color": "#4ECDC4",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            opacity: 0.6,
            "arrow-scale": 0.8,
          },
        },
      ],
      layout: {
        name: "concentric",
        concentric: (node: any) => (node.id === node.id ? 1 : 0),
        minNodeSpacing: 50,
      },
      minZoom: 0.5,
      maxZoom: 2,
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [node, neighbors]);

  return (
    <div className="fixed right-0 top-0 h-full w-[500px] bg-slate-900 border-l border-slate-700 overflow-y-auto z-50">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Node Details</h2>
          <button
            onClick={onClose}
            className="px-3 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg transition-all"
          >
            ✕
          </button>
        </div>

        {/* Node Info */}
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <div className="text-sm text-slate-500 font-mono mb-2">{node.id}</div>
          <div className="text-lg text-white mb-4">{node.text}</div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-900 rounded p-3">
              <div className="text-slate-400 text-xs">Role</div>
              <div
                className="font-semibold"
                style={{ color: roleColors[node.role] || "#888" }}
              >
                {node.role}
              </div>
            </div>
            <div className="bg-slate-900 rounded p-3">
              <div className="text-slate-400 text-xs">Activation</div>
              <div className="font-semibold text-blue-400">
                {node.activation.toFixed(3)}
              </div>
            </div>
            <div className="bg-slate-900 rounded p-3">
              <div className="text-slate-400 text-xs">Confidence</div>
              <div className="font-semibold text-purple-400">
                {node.confidence.toFixed(3)}
              </div>
            </div>
            <div className="bg-slate-900 rounded p-3">
              <div className="text-slate-400 text-xs">Neighbors</div>
              <div className="font-semibold text-green-400">{node.neighbors.length}</div>
            </div>
          </div>
        </div>

        {/* Neighbor Graph */}
        <div className="space-y-3">
          <h3 className="font-semibold">Neighborhood Graph</h3>
          <div className="w-full h-[400px] bg-slate-950 rounded-lg overflow-hidden border border-slate-700">
            <div ref={containerRef} className="w-full h-full" />
          </div>
        </div>

        {/* Neighbors List */}
        <div className="space-y-3">
          <h3 className="font-semibold">
            Connected Nodes ({neighbors.length})
          </h3>
          <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
            {neighbors.length === 0 ? (
              <div className="text-slate-500 text-sm py-4 text-center">
                No connections
              </div>
            ) : (
              neighbors.map((n) => (
                <div
                  key={n.id}
                  className="bg-slate-800 rounded-lg p-3 border border-slate-700"
                >
                  <div className="text-sm text-slate-500 font-mono mb-1">{n.id}</div>
                  <div className="text-white text-sm mb-2">{n.text}</div>
                  <div className="flex gap-2 text-xs">
                    <span
                      className="px-2 py-1 rounded"
                      style={{
                        backgroundColor: roleColors[n.role] || "#888",
                        color: "#fff",
                      }}
                    >
                      {n.role}
                    </span>
                    <span className="px-2 py-1 bg-slate-700 rounded text-slate-300">
                      Act: {n.activation.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
