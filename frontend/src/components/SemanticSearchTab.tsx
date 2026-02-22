"use client";

import { useState } from "react";
import { query } from "@/lib/api";
import { Node } from "@/types";

export default function SemanticSearchTab() {
  const [searchQuery, setSearchQuery] = useState("");
  const [results, setResults] = useState<Node[]>([]);
  const [loading, setLoading] = useState(false);
  const [k, setK] = useState(5);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const result = await query({
        query_text: searchQuery,
        top_k: k,
        propagation_depth: 0,
        min_similarity_threshold: 0.0,
        candidate_multiplier: 1,
        decay_per_hop: 0.7,
      });
      setResults(result);
    } catch (error) {
      console.error("Search error:", error);
      alert("Failed to search. Check if backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Input */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 space-y-4">
        <h2 className="text-xl font-semibold">Semantic Search</h2>
        <p className="text-slate-400 text-sm">
          Search for semantically similar nodes using vector embeddings
        </p>

        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="e.g., 'What are the goals?' or 'computer science pioneers'"
          className="w-full px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white placeholder-slate-500"
        />

        <div className="flex items-center gap-4">
          <label className="text-slate-400 text-sm">Results:</label>
          <input
            type="range"
            min="1"
            max="20"
            value={k}
            onChange={(e) => setK(parseInt(e.target.value))}
            className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <span className="text-white font-mono w-8">{k}</span>
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleSearch}
            disabled={loading || !searchQuery.trim()}
            className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-xl font-semibold mb-4">
            Found {results.length} Results
          </h3>
          <div className="space-y-3">
            {results.map((node, i) => (
              <details
                key={node.id}
                open={i === 0}
                className="bg-slate-900 rounded-lg border border-slate-700 overflow-hidden group"
              >
                <summary className="px-4 py-3 cursor-pointer hover:bg-slate-800 transition-colors flex items-center gap-3">
                  <span className="px-2 py-1 bg-blue-600 rounded text-xs font-semibold">
                    #{i + 1}
                  </span>
                  <span
                    className="px-2 py-1 rounded text-xs font-semibold"
                    style={{
                      backgroundColor:
                        node.role === "FACT"
                          ? "#4ECDC4"
                          : node.role === "GOAL"
                            ? "#FFE66D"
                            : node.role === "CONSTRAINT"
                              ? "#C44D58"
                              : node.role === "OBSERVATION"
                                ? "#95E1D3"
                                : node.role === "DECISION"
                                  ? "#FFB74D"
                                  : "#888",
                      color: "#fff",
                    }}
                  >
                    {node.role}
                  </span>
                  <span className="flex-1 text-white truncate">
                    {node.text.substring(0, 80)}
                    {node.text.length > 80 ? "..." : ""}
                  </span>
                </summary>
                <div className="px-4 pb-4 space-y-3">
                  <div>
                    <div className="text-slate-400 text-xs uppercase tracking-wide mb-1">
                      Text
                    </div>
                    <div className="text-white">{node.text}</div>
                  </div>

                  <div className="grid grid-cols-4 gap-3 text-sm">
                    <div>
                      <div className="text-slate-400 text-xs uppercase tracking-wide mb-1">
                        Role
                      </div>
                      <div className="text-white capitalize">
                        {node.role.toLowerCase().replace("_", " ")}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-400 text-xs uppercase tracking-wide mb-1">
                        Confidence
                      </div>
                      <div className="text-white">
                        {"★".repeat(Math.floor(node.confidence * 5))}
                        {"☆".repeat(5 - Math.floor(node.confidence * 5))}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-400 text-xs uppercase tracking-wide mb-1">
                        Activation
                      </div>
                      <div className="text-blue-400 font-mono">
                        {node.activation.toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-400 text-xs uppercase tracking-wide mb-1">
                        Neighbors
                      </div>
                      <div className="text-green-400">{node.neighbors.length}</div>
                    </div>
                  </div>

                  <div>
                    <div className="text-slate-400 text-xs uppercase tracking-wide mb-1">
                      ID
                    </div>
                    <div className="text-slate-500 font-mono text-xs">{node.id}</div>
                  </div>
                </div>
              </details>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && results.length === 0 && searchQuery && (
        <div className="bg-slate-800 rounded-xl p-12 border border-slate-700 text-center">
          <div className="text-4xl mb-4">🔍</div>
          <h3 className="text-lg font-semibold mb-2">No Results Found</h3>
          <p className="text-slate-400">Try a different search query</p>
        </div>
      )}

      {!loading && results.length === 0 && !searchQuery && (
        <div className="bg-slate-800 rounded-xl p-12 border border-slate-700 text-center">
          <div className="text-4xl mb-4">🔍</div>
          <h3 className="text-lg font-semibold mb-2">Ready to Search</h3>
          <p className="text-slate-400">
            Enter a query above to search for semantically similar commitments
          </p>
        </div>
      )}
    </div>
  );
}
