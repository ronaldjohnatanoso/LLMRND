"use client";

import { useEffect, useState } from "react";
import { getNodes, clearMemory } from "@/lib/api";
import { Node } from "@/types";

interface AllNodesTabProps {
  onSelectNode?: (node: Node) => void;
}

const roleColors: Record<string, string> = {
  QUERY: "#FF6B6B",
  FACT: "#4ECDC4",
  OBSERVATION: "#95E1D3",
  GOAL: "#FFE66D",
  CONSTRAINT: "#C44D58",
  DECISION: "#FFB74D",
};

export default function AllNodesTab({ onSelectNode }: AllNodesTabProps) {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [filteredNodes, setFilteredNodes] = useState<Node[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState<string>("ALL");
  const [sortBy, setSortBy] = useState<"activation" | "confidence" | "text">("activation");
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    loadNodes();
  }, []);

  useEffect(() => {
    filterAndSortNodes();
  }, [nodes, search, roleFilter, sortBy]);

  const loadNodes = async () => {
    setLoading(true);
    try {
      const data = await getNodes();
      setNodes(data);
    } catch (error) {
      console.error("Error loading nodes:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleResetDatabase = async () => {
    if (!showResetConfirm) {
      setShowResetConfirm(true);
      return;
    }

    setResetting(true);
    try {
      await clearMemory();
      setNodes([]);
      setShowResetConfirm(false);
      alert("✅ Database cleared successfully!");
    } catch (error) {
      console.error("Error clearing database:", error);
      alert("❌ Failed to clear database. Check if backend is running.");
    } finally {
      setResetting(false);
    }
  };

  const filterAndSortNodes = () => {
    let result = [...nodes];

    // Filter by role
    if (roleFilter !== "ALL") {
      result = result.filter((n) => n.role === roleFilter);
    }

    // Filter by search
    if (search) {
      const searchLower = search.toLowerCase();
      result = result.filter(
        (n) =>
          n.text.toLowerCase().includes(searchLower) ||
          n.id.toLowerCase().includes(searchLower)
      );
    }

    // Sort
    result.sort((a, b) => {
      if (sortBy === "activation") return b.activation - a.activation;
      if (sortBy === "confidence") return b.confidence - a.confidence;
      return a.text.localeCompare(b.text);
    });

    setFilteredNodes(result);
  };

  const uniqueRoles = Array.from(new Set(nodes.map((n) => n.role)));

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search nodes..."
          className="flex-1 min-w-64 px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
        />

        <select
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
          className="px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
        >
          <option value="ALL">All Roles</option>
          {uniqueRoles.map((role) => (
            <option key={role} value={role}>
              {role}
            </option>
          ))}
        </select>

        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
          className="px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
        >
          <option value="activation">Sort by Activation</option>
          <option value="confidence">Sort by Confidence</option>
          <option value="text">Sort by Text</option>
        </select>

        <button
          onClick={loadNodes}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-all"
        >
          Refresh
        </button>

        <button
          onClick={handleResetDatabase}
          disabled={resetting}
          className={`px-4 py-2 rounded-lg transition-all ${
            showResetConfirm
              ? "bg-red-600 hover:bg-red-700"
              : "bg-slate-700 hover:bg-slate-600"
          } disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          {resetting ? "Clearing..." : showResetConfirm ? "⚠️ Confirm Reset" : "🗑️ Reset Database"}
        </button>
      </div>

      {showResetConfirm && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
          <p className="text-red-300 mb-2">⚠️ <strong>Warning:</strong> This will permanently delete all nodes from the database!</p>
          <p className="text-slate-400 text-sm mb-3">This action cannot be undone. All stored commitments will be lost.</p>
          <div className="flex gap-2">
            <button
              onClick={() => setShowResetConfirm(false)}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-all text-sm"
            >
              Cancel
            </button>
            <button
              onClick={() => handleResetDatabase()}
              disabled={resetting}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-all text-sm"
            >
              {resetting ? "Deleting..." : "Yes, Delete All"}
            </button>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <div className="text-slate-400 text-sm">Total Nodes</div>
          <div className="text-2xl font-bold">{nodes.length}</div>
        </div>
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <div className="text-slate-400 text-sm">Filtered</div>
          <div className="text-2xl font-bold">{filteredNodes.length}</div>
        </div>
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <div className="text-slate-400 text-sm">Avg Activation</div>
          <div className="text-2xl font-bold">
            {nodes.length > 0
              ? (nodes.reduce((sum, n) => sum + n.activation, 0) / nodes.length).toFixed(3)
              : "0.00"}
          </div>
        </div>
        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
          <div className="text-slate-400 text-sm">Avg Confidence</div>
          <div className="text-2xl font-bold">
            {nodes.length > 0
              ? (nodes.reduce((sum, n) => sum + n.confidence, 0) / nodes.length).toFixed(3)
              : "0.00"}
          </div>
        </div>
      </div>

      {/* Nodes List */}
      {loading ? (
        <div className="text-center py-12 text-slate-400">Loading nodes...</div>
      ) : filteredNodes.length === 0 ? (
        <div className="text-center py-12 text-slate-400">
          {nodes.length === 0 ? "No nodes in memory" : "No nodes match your filters"}
        </div>
      ) : (
        <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
          {filteredNodes.map((node) => (
            <div
              key={node.id}
              onClick={() => onSelectNode?.(node)}
              className="bg-slate-800 hover:bg-slate-750 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-all cursor-pointer"
            >
              <div className="flex items-start gap-4">
                <div
                  className="w-3 h-3 rounded-full mt-1.5 flex-shrink-0"
                  style={{ backgroundColor: roleColors[node.role] || "#888" }}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-slate-500 font-mono mb-1">{node.id}</div>
                  <div className="text-white mb-2">{node.text}</div>
                  <div className="flex flex-wrap gap-3 text-sm">
                    <span className="px-2 py-1 bg-slate-700 rounded text-slate-300">
                      {node.role}
                    </span>
                    <span className="px-2 py-1 bg-blue-900/50 rounded text-blue-400">
                      Act: {node.activation.toFixed(3)}
                    </span>
                    <span className="px-2 py-1 bg-purple-900/50 rounded text-purple-400">
                      Conf: {node.confidence.toFixed(3)}
                    </span>
                    <span className="px-2 py-1 bg-slate-700 rounded text-slate-300">
                      {node.neighbors.length} neighbors
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
