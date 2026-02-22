"use client";

import { useState } from "react";
import { addNodes, getNodes } from "@/lib/api";
import { Node } from "@/types";

interface AddNodesTabProps {
  onNodesAdded?: () => void;
}

const roleDescriptions: Record<string, string> = {
  FACT: "Factual information that can be verified",
  OBSERVATION: "Direct observations or experiences",
  GOAL: "Goals or objectives to achieve",
  CONSTRAINT: "Limitations or restrictions",
  DECISION: "Decisions made or conclusions reached",
};

export default function AddNodesTab({ onNodesAdded }: AddNodesTabProps) {
  const [texts, setTexts] = useState("");
  const [role, setRole] = useState<string>("FACT");
  const [loading, setLoading] = useState(false);
  const [addedNodes, setAddedNodes] = useState<Node[]>([]);

  const handleAdd = async () => {
    const lines = texts
      .split("\n")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    if (lines.length === 0) {
      alert("Please enter at least one node");
      return;
    }

    setLoading(true);
    try {
      const result = await addNodes(lines, role);
      setAddedNodes(result);
      setTexts("");
      onNodesAdded?.();
    } catch (error) {
      console.error("Error adding nodes:", error);
      alert("Failed to add nodes. Check if backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const loadSampleData = async () => {
    // Sample knowledge about computer science history
    setTexts(
      `Alan Turing is considered the father of computer science
Turing invented the Turing machine which is a mathematical model of computation
The Turing test is a test of a machine's ability to exhibit intelligent behavior
Turing worked at Bletchley Park during World War II
He helped crack the Enigma code used by the Germans
Charles Babbage designed the first mechanical computer called the Analytical Engine
Ada Lovelace wrote the first algorithm intended to be processed by a machine
She is often called the first computer programmer
John von Neumann contributed to the design of the stored-program computer architecture
The von Neumann architecture is the basis of most modern computers`
    );
  };

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Add New Nodes</h2>
          <button
            onClick={loadSampleData}
            type="button"
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm transition-all"
          >
            Load Sample Data
          </button>
        </div>

        <textarea
          value={texts}
          onChange={(e) => setTexts(e.target.value)}
          placeholder="Enter facts (one per line)&#10;Each line will become a separate node in memory"
          className="w-full h-64 px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white placeholder-slate-500 font-mono text-sm resize-none"
        />

        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm text-slate-400 mb-2">Role</label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              className="w-full px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
            >
              {Object.entries(roleDescriptions).map(([r, desc]) => (
                <option key={r} value={r}>
                  {r} - {desc}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleAdd}
            disabled={loading}
            className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all self-end"
          >
            {loading ? "Adding..." : `Add ${texts.split("\n").filter((t) => t.trim()).length} Nodes`}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {addedNodes.length > 0 && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-xl font-semibold mb-4">
            Successfully Added {addedNodes.length} Nodes
          </h3>
          <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
            {addedNodes.map((node) => (
              <div
                key={node.id}
                className="bg-slate-900 rounded-lg p-4 border border-slate-700"
              >
                <div className="text-sm text-slate-500 font-mono mb-1">{node.id}</div>
                <div className="text-white mb-2">{node.text}</div>
                <div className="flex gap-3 text-sm">
                  <span className="px-2 py-1 bg-slate-700 rounded text-slate-300">
                    {node.role}
                  </span>
                  <span className="px-2 py-1 bg-blue-900/50 rounded text-blue-400">
                    Act: {node.activation.toFixed(3)}
                  </span>
                  <span className="px-2 py-1 bg-purple-900/50 rounded text-purple-400">
                    Conf: {node.confidence.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <button
            onClick={() => setAddedNodes([])}
            className="mt-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm transition-all"
          >
            Clear Results
          </button>
        </div>
      )}

      {/* Tips */}
      <div className="bg-blue-900/20 rounded-xl p-6 border border-blue-700">
        <h3 className="text-lg font-semibold mb-3 text-blue-400">Tips for Adding Nodes</h3>
        <ul className="space-y-2 text-slate-300 text-sm">
          <li>• Enter one statement per line - each line becomes a separate node</li>
          <li>• Use specific, factual statements for better retrieval</li>
          <li>• Nodes with similar concepts will be automatically connected</li>
          <li>• Click "Load Sample Data" to see an example of computer science history</li>
        </ul>
      </div>
    </div>
  );
}
