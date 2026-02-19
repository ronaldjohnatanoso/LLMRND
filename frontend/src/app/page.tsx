"use client";

import { useState, useEffect } from "react";
import GraphVisualization from "@/components/GraphVisualization";
import { querySimulation, getNodes } from "@/lib/api";
import { SimulationResponse } from "@/types";

export default function Home() {
  const [query, setQuery] = useState("father of computer");
  const [simulation, setSimulation] = useState<SimulationResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [totalNodes, setTotalNodes] = useState(0);

  // Load total nodes on mount
  useEffect(() => {
    getNodes().then((nodes) => setTotalNodes(nodes.length)).catch(console.error);
  }, []);

  const handleRunQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const result = await querySimulation({
        query_text: query,
        top_k: 5,
        propagation_depth: 2,
        min_similarity_threshold: 0.55,
        candidate_multiplier: 2,
        decay_per_hop: 0.7,
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

  const currentStepData = simulation?.timeline[currentStep];
  const progress = simulation ? ((currentStep + 1) / simulation.total_steps) * 100 : 0;

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            CogMemory
          </h1>
          <p className="text-slate-400 text-lg">Cognitive Graph Memory for LLMs</p>
          <p className="text-sm text-slate-500">{totalNodes} nodes in memory</p>
        </div>

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
                  <div className="text-2xl font-bold">{currentStep + 1} / {simulation.total_steps}</div>
                  <div className="text-sm text-slate-400">Step</div>
                </div>
                <button
                  onClick={() => setCurrentStep((s) => Math.min(simulation.total_steps - 1, s + 1))}
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
                      <div className="font-mono text-green-400">Δ{currentStepData.delta.toFixed(4)}</div>
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
            <div className="text-6xl mb-4">🧠</div>
            <h3 className="text-xl font-semibold mb-2">Ready to Query</h3>
            <p className="text-slate-400">Enter a query above to see the cognitive graph in action</p>
          </div>
        )}
      </div>
    </main>
  );
}
