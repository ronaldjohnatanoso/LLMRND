"use client";

import { useState, useEffect } from "react";
import GraphVisualization from "@/components/GraphVisualization";
import AllNodesTab from "@/components/AllNodesTab";
import AddNodesTab from "@/components/AddNodesTab";
import SemanticSearchTab from "@/components/SemanticSearchTab";
import NodeDetailsPanel from "@/components/NodeDetailsPanel";
import { querySimulation, getNodes } from "@/lib/api";
import { SimulationResponse, Node as NodeType } from "@/types";

type TabType = "query" | "nodes" | "search" | "add";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("nodes");
  const [query, setQuery] = useState("father of computer");
  const [simulation, setSimulation] = useState<SimulationResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [totalNodes, setTotalNodes] = useState(0);
  const [selectedNode, setSelectedNode] = useState<NodeType | null>(null);
  const [selectedNodeNeighbors, setSelectedNodeNeighbors] = useState<NodeType[]>([]);

  // Query parameters
  const [topK, setTopK] = useState(5);
  const [propagationDepth, setPropagationDepth] = useState(2);
  const [minSimilarityThreshold, setMinSimilarityThreshold] = useState(0.55);
  const [candidateMultiplier, setCandidateMultiplier] = useState(2);
  const [decayPerHop, setDecayPerHop] = useState(0.7);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  // Load total nodes on mount
  useEffect(() => {
    refreshNodeCount();
  }, []);

  const refreshNodeCount = async () => {
    try {
      const nodes = await getNodes();
      setTotalNodes(nodes.length);
    } catch (error) {
      console.error("Error loading nodes:", error);
    }
  };

  const handleRunQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const result = await querySimulation({
        query_text: query,
        top_k: topK,
        propagation_depth: propagationDepth,
        min_similarity_threshold: minSimilarityThreshold,
        candidate_multiplier: candidateMultiplier,
        decay_per_hop: decayPerHop,
      });
      setSimulation(result);
      setCurrentStep(0);
    } catch (error) {
      console.error("Query error:", error);
      alert("Failed to run query. Check if backend is running on http://localhost:8000");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectNode = async (node: NodeType) => {
    setSelectedNode(node);
    // Load neighbors
    try {
      const allNodes = await getNodes();
      const neighborMap = new Map(allNodes.map((n) => [n.id, n]));
      const neighbors = node.neighbors
        .map((id) => neighborMap.get(id))
        .filter((n): n is NodeType => n !== undefined);
      setSelectedNodeNeighbors(neighbors);
    } catch (error) {
      console.error("Error loading neighbors:", error);
      setSelectedNodeNeighbors([]);
    }
  };

  const currentStepData = simulation?.timeline[currentStep];
  const progress = simulation ? ((currentStep + 1) / simulation.total_steps) * 100 : 0;

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                CogMemory
              </h1>
              <p className="text-slate-400">Cognitive Graph Memory for LLMs</p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-blue-400">{totalNodes}</div>
              <div className="text-sm text-slate-500">nodes in memory</div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mt-6">
            <button
              onClick={() => setActiveTab("query")}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === "query"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              🎬 Query Simulation
            </button>
            <button
              onClick={() => setActiveTab("nodes")}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === "nodes"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              📋 All Nodes
            </button>
            <button
              onClick={() => setActiveTab("search")}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === "search"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              🔍 Semantic Search
            </button>
            <button
              onClick={() => setActiveTab("add")}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                activeTab === "add"
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              ➕ Ingest Text
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-8 py-8">
        {activeTab === "query" && (
          <div className="space-y-8">
            {/* Query Input */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700 space-y-4">
              <div className="flex gap-4">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleRunQuery()}
                  placeholder="Enter your query..."
                  className="flex-1 px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white placeholder-slate-500"
                />
                <button
                  onClick={handleRunQuery}
                  disabled={loading}
                  className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? "Thinking..." : "Query"}
                </button>
              </div>

              {/* Basic Parameters */}
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center gap-3">
                  <label className="text-slate-400 text-sm whitespace-nowrap">Top K:</label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <span className="text-white font-mono w-8 text-center">{topK}</span>
                </div>

                <div className="flex items-center gap-3">
                  <label className="text-slate-400 text-sm whitespace-nowrap">Depth:</label>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={propagationDepth}
                    onChange={(e) => setPropagationDepth(parseInt(e.target.value))}
                    className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <span className="text-white font-mono w-8 text-center">{propagationDepth}</span>
                </div>
              </div>

              {/* Advanced Settings Toggle */}
              <button
                onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                className="text-slate-400 text-sm hover:text-white transition-colors flex items-center gap-2"
              >
                <span>{showAdvancedSettings ? "▼" : "▶"}</span>
                <span>Advanced Settings</span>
              </button>

              {showAdvancedSettings && (
                <div className="bg-slate-900/50 rounded-lg p-4 space-y-4 border border-slate-700">
                  <div className="flex items-center gap-3">
                    <label className="text-slate-400 text-sm whitespace-nowrap w-48">Min Similarity:</label>
                    <input
                      type="range"
                      min="0.30"
                      max="0.75"
                      step="0.05"
                      value={minSimilarityThreshold}
                      onChange={(e) => setMinSimilarityThreshold(parseFloat(e.target.value))}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <span className="text-white font-mono w-14 text-center">{minSimilarityThreshold.toFixed(2)}</span>
                  </div>

                  <div className="flex items-center gap-3">
                    <label className="text-slate-400 text-sm whitespace-nowrap w-48">Candidate Multiplier:</label>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      step="1"
                      value={candidateMultiplier}
                      onChange={(e) => setCandidateMultiplier(parseInt(e.target.value))}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <span className="text-white font-mono w-14 text-center">{candidateMultiplier}x</span>
                  </div>

                  <div className="flex items-center gap-3">
                    <label className="text-slate-400 text-sm whitespace-nowrap w-48">Decay Per Hop:</label>
                    <input
                      type="range"
                      min="0.3"
                      max="1.0"
                      step="0.1"
                      value={decayPerHop}
                      onChange={(e) => setDecayPerHop(parseFloat(e.target.value))}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <span className="text-white font-mono w-14 text-center">{decayPerHop.toFixed(1)}</span>
                  </div>

                  <div className="text-slate-500 text-xs grid grid-cols-2 gap-2">
                    <div>• <strong>Min Similarity:</strong> Filter out weak matches</div>
                    <div>• <strong>Candidate Multiplier:</strong> How many extra candidates to consider per hop</div>
                    <div>• <strong>Decay Per Hop:</strong> How much activation decreases per hop</div>
                    <div>• <strong>Depth:</strong> How many hops to propagate activation</div>
                  </div>
                </div>
              )}
            </div>

            {/* Simulation Results */}
            {simulation && (
              <div className="space-y-6">
                {/* Step Navigation */}
                <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700 space-y-4">
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setCurrentStep(0)}
                      disabled={currentStep === 0}
                      className="px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      ⏮️ First
                    </button>
                    <button
                      onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
                      disabled={currentStep === 0}
                      className="px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      ◀️ Prev
                    </button>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {currentStep + 1} / {simulation.total_steps}
                      </div>
                      <div className="text-sm text-slate-400">Step</div>
                    </div>
                    <button
                      onClick={() =>
                        setCurrentStep((s) => Math.min(simulation.total_steps - 1, s + 1))
                      }
                      disabled={currentStep === simulation.total_steps - 1}
                      className="px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      Next ▶️
                    </button>
                    <button
                      onClick={() => setCurrentStep(simulation.total_steps - 1)}
                      disabled={currentStep === simulation.total_steps - 1}
                      className="px-4 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      Last ⏭️
                    </button>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>

                  {/* Step Slider */}
                  <input
                    type="range"
                    min="0"
                    max={simulation.total_steps - 1}
                    value={currentStep}
                    onChange={(e) => setCurrentStep(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>

                {/* Current Step Info */}
                {currentStepData && (
                  <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
                    <div className="flex items-center gap-3 mb-4">
                      <span className="text-2xl">
                        {currentStepData.type === "search" && "🔍"}
                        {currentStepData.type === "search_results" && "📊"}
                        {currentStepData.type === "layer_1" && "✅"}
                        {currentStepData.type === "hop_start" && "🌊"}
                        {currentStepData.type === "propagation" && "➡️"}
                        {currentStepData.type === "gate_1_fail" && "🚫"}
                        {currentStepData.type === "gate_2_fail" && "🚫"}
                        {currentStepData.type === "complete" && "✨"}
                      </span>
                      <h3 className="text-xl font-semibold">{currentStepData.message}</h3>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div className="bg-slate-900/50 rounded-lg p-3">
                        <div className="text-slate-400">Step Type</div>
                        <div className="font-mono text-blue-400">{currentStepData.type}</div>
                      </div>
                      {currentStepData.hop !== undefined && (
                        <div className="bg-slate-900/50 rounded-lg p-3">
                          <div className="text-slate-400">Hop</div>
                          <div className="font-mono text-purple-400">{currentStepData.hop}</div>
                        </div>
                      )}
                      {currentStepData.delta !== undefined && (
                        <div className="bg-slate-900/50 rounded-lg p-3">
                          <div className="text-slate-400">Signal</div>
                          <div className="font-mono text-green-400">
                            Δ{currentStepData.delta.toFixed(4)}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Graph Visualization */}
                <GraphVisualization simulation={simulation} currentStep={currentStep} />
              </div>
            )}

            {/* Empty State */}
            {!simulation && !loading && (
              <div className="bg-slate-800/50 backdrop-blur rounded-xl p-12 border border-slate-700 text-center">
                <div className="text-6xl mb-4">🎬</div>
                <h3 className="text-xl font-semibold mb-2">Query Simulation</h3>
                <p className="text-slate-400">
                  Enter a query above to see step-by-step activation propagation through the
                  cognitive graph
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === "nodes" && (
          <AllNodesTab
            onSelectNode={totalNodes > 0 ? handleSelectNode : undefined}
          />
        )}

        {activeTab === "search" && <SemanticSearchTab />}

        {activeTab === "add" && <AddNodesTab onNodesAdded={refreshNodeCount} />}
      </div>

      {/* Node Details Panel */}
      {selectedNode && (
        <NodeDetailsPanel
          node={selectedNode}
          neighbors={selectedNodeNeighbors}
          onClose={() => {
            setSelectedNode(null);
            setSelectedNodeNeighbors([]);
          }}
        />
      )}
    </main>
  );
}
