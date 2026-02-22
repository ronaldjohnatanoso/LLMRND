export interface Node {
  id: string;
  text: string;
  role: string;
  activation: number;
  confidence: number;
  neighbors: string[];
  similarity: number;
}

export interface QueryRequest {
  query_text: string;
  top_k?: number;
  propagation_depth?: number;
  min_similarity_threshold?: number;
  candidate_multiplier?: number;
  decay_per_hop?: number;
}

export interface SimulationStep {
  step: number;
  type: string;
  message: string;
  nodes_activated?: string[];
  candidates?: Array<{ id: string; text: string; similarity: number }>;
  nodes_data?: Array<{
    id: string;
    text: string;
    role: string;
    activation: number;
    similarity_to_query: number;
  }>;
  hop?: number;
  parent_id?: string;
  child_id?: string;
  delta?: number;
  newly_activated?: boolean;
  threshold?: number;
  activation?: number;
  old_activation?: number;
  new_activation?: number;
  edge_weight?: number;
  role_boost?: number;
  decay_factor?: number;
  node_id?: string;
  final_states?: Array<{
    id: string;
    text: string;
    role: string;
    activation: number;
  }>;
}

export interface SimulationResponse {
  query: string;
  timeline: SimulationStep[];
  total_steps: number;
  final_states: Array<{
    id: string;
    text: string;
    role: string;
    activation: number;
  }>;
  settings: {
    top_k: number;
    propagation_depth: number;
    min_similarity_threshold: number;
    candidate_multiplier: number;
    decay_per_hop: number;
  };
}

export interface GraphNode {
  data: {
    id: string;
    label: string;
    role: string;
    activation: number;
    layer: number;
    stepAdded?: number;
    isActive?: boolean;
    wasActivated?: boolean;
    propagated?: boolean;
    failedGate2?: boolean;
  };
  position?: { x: number; y: number };
  classes?: string;
}

export interface GraphEdge {
  data: {
    id: string;
    source: string;
    target: string;
    strength: number;
    stepAdded?: number;
  };
  classes?: string;
}
