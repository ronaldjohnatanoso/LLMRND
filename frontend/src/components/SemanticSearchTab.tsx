"use client";

import { useState } from "react";
import { query } from "@/lib/api";
import { Node } from "@/types";

export default function SemanticSearchTab() {
  const [searchQuery, setSearchQuery] = useState("");
  const [results, setResults] = useState<Node[]>([]);
  const [loading, setLoading] = useState(false);
  const [k, setK] = useState(5);
  const [minSimilarity, setMinSimilarity] = useState(0.55);
  const [redYellowThreshold, setRedYellowThreshold] = useState(0.60);
  const [yellowGreenThreshold, setYellowGreenThreshold] = useState(0.65);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const getQualityIndicator = (similarity: number) => {
    if (similarity >= yellowGreenThreshold) return { emoji: "🟢", color: "text-green-400", label: "High" };
    if (similarity >= redYellowThreshold) return { emoji: "🟡", color: "text-yellow-400", label: "Medium" };
    return { emoji: "🔴", color: "text-red-400", label: "Low" };
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const result = await query({
        query_text: searchQuery,
        top_k: k,
        propagation_depth: 0,
        min_similarity_threshold: minSimilarity,
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

        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center gap-3">
            <label className="text-slate-400 text-sm whitespace-nowrap">Results:</label>
            <input
              type="range"
              min="1"
              max="20"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <span className="text-white font-mono w-8 text-center">{k}</span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-slate-400 text-sm whitespace-nowrap">Min Similarity:</label>
            <input
              type="range"
              min="0.50"
              max="0.75"
              step="0.01"
              value={minSimilarity}
              onChange={(e) => setMinSimilarity(parseFloat(e.target.value))}
              className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <span className="text-white font-mono w-14 text-center">{minSimilarity.toFixed(2)}</span>
          </div>
        </div>

        {/* Advanced Settings Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-slate-400 text-sm hover:text-white transition-colors flex items-center gap-2"
        >
          <span>{showAdvanced ? "▼" : "▶"}</span>
          <span>Quality Thresholds (Advanced)</span>
        </button>

        {showAdvanced && (
          <div className="bg-slate-900/50 rounded-lg p-4 space-y-4 border border-slate-700">
            <div className="flex items-center gap-3">
              <label className="text-slate-400 text-sm whitespace-nowrap">🔴 Red / 🟡 Yellow:</label>
              <input
                type="range"
                min="0.55"
                max="0.65"
                step="0.01"
                value={redYellowThreshold}
                onChange={(e) => setRedYellowThreshold(parseFloat(e.target.value))}
                className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-red-500"
              />
              <span className="text-white font-mono w-14 text-center">{redYellowThreshold.toFixed(2)}</span>
            </div>

            <div className="flex items-center gap-3">
              <label className="text-slate-400 text-sm whitespace-nowrap">🟡 Yellow / 🟢 Green:</label>
              <input
                type="range"
                min="0.60"
                max="0.75"
                step="0.01"
                value={yellowGreenThreshold}
                onChange={(e) => setYellowGreenThreshold(parseFloat(e.target.value))}
                className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-500"
              />
              <span className="text-white font-mono w-14 text-center">{yellowGreenThreshold.toFixed(2)}</span>
            </div>

            {redYellowThreshold >= yellowGreenThreshold && (
              <div className="text-red-400 text-sm">⚠️ Yellow/Green threshold must be higher than Red/Yellow!</div>
            )}

            <div className="text-slate-500 text-xs">
              <div>🔴 Red: similarity &lt; {redYellowThreshold.toFixed(2)} (low quality)</div>
              <div>🟡 Yellow: {redYellowThreshold.toFixed(2)} ≤ similarity &lt; {yellowGreenThreshold.toFixed(2)} (medium quality)</div>
              <div>🟢 Green: similarity ≥ {yellowGreenThreshold.toFixed(2)} (high quality)</div>
            </div>
          </div>
        )}

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
            {results.map((node, i) => {
              const similarity = node.similarity;
              const quality = getQualityIndicator(similarity);

              return (
                <details
                  key={node.id}
                  open={i === 0}
                  className={`bg-slate-900 rounded-lg border overflow-hidden group ${
                    similarity < redYellowThreshold
                      ? "border-red-900"
                      : similarity < yellowGreenThreshold
                        ? "border-yellow-900"
                        : "border-green-900"
                  }`}
                >
                  <summary className="px-4 py-3 cursor-pointer hover:bg-slate-800 transition-colors flex items-center gap-3">
                    <span className="px-2 py-1 bg-blue-600 rounded text-xs font-semibold">
                      #{i + 1}
                    </span>
                    <span className="text-lg">{quality.emoji}</span>
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
                    <span className={`font-mono text-sm ${quality.color}`}>
                      Sim={similarity.toFixed(3)}
                    </span>
                    <span className="flex-1 text-white truncate">
                      {node.text.substring(0, 60)}
                      {node.text.length > 60 ? "..." : ""}
                    </span>
                  </summary>
                  <div className="px-4 pb-4 space-y-3">
                    <div>
                      <div className="text-slate-400 text-xs uppercase tracking-wide mb-1">
                        Text
                      </div>
                      <div className="text-white">{node.text}</div>
                    </div>

                    {/* Quality Message */}
                    {similarity < redYellowThreshold ? (
                      <div className="bg-red-900/20 border border-red-800 rounded px-3 py-2 text-sm">
                        ⚠️ Low similarity (&lt; {redYellowThreshold.toFixed(2)}) - might not be relevant
                      </div>
                    ) : similarity < yellowGreenThreshold ? (
                      <div className="bg-yellow-900/20 border border-yellow-800 rounded px-3 py-2 text-sm">
                        ⚡ Medium similarity ({redYellowThreshold.toFixed(2)} - {yellowGreenThreshold.toFixed(2)}) - review for relevance
                      </div>
                    ) : (
                      <div className="bg-green-900/20 border border-green-800 rounded px-3 py-2 text-sm">
                        ✅ High similarity (≥ {yellowGreenThreshold.toFixed(2)}) - likely relevant
                      </div>
                    )}

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
              );
            })}
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
