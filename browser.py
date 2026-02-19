"""Streamlit UI for browsing CogMemory database.

Run with: streamlit run browser.py

Features:
- View all nodes in searchable table
- Semantic search
- Filter by role
- Ingest new text
- Statistics dashboard
- 3D Graph visualization
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from cog_memory.lance_store import LanceStore
from cog_memory.query_interface import CognitiveMemory

st.set_page_config(
    page_title="CogMemory Browser",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 CogMemory Browser")


@st.cache_resource
def get_memory():
    """Get cached memory instance."""
    return CognitiveMemory()


@st.cache_data(ttl=10)
def get_all_nodes():
    """Get all nodes from database."""
    store = LanceStore()
    df = store.table.search().limit(None).to_pandas()
    # Parse JSON fields safely
    df["neighbors"] = df["neighbors"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df["metadata"] = df["metadata"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def _get_role_boost(source_role, target_role):
    """Get role-based boost multiplier for propagation."""
    from cog_memory.node import Role

    # Simplified role boosts (matching cognitive_graph.py)
    boosts = {
        (Role.GOAL, Role.DECISION): 1.5,
        (Role.CONSTRAINT, Role.DECISION): -0.8,
        (Role.OBSERVATION, Role.DECISION): 1.2,
        (Role.FACT, Role.FACT): 1.3,
        (Role.CONDITIONAL_DEPENDENCY, Role.DECISION): 1.1,
        (Role.CONDITIONAL_DEPENDENCY, Role.CONSTRAINT): 1.1,
        (Role.GOAL, Role.CONDITIONAL_DEPENDENCY): 1.2,
    }
    return boosts.get((source_role, target_role), 1.0)


def main():
    """Main UI."""
    memory = get_memory()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        # Stats
        st.subheader("📊 Statistics")
        df = get_all_nodes()
        total = len(df)
        if total > 0:
            role_counts = df["role"].value_counts().to_dict()
            st.metric("Total Nodes", total)
            for role, count in role_counts.items():
                st.metric(role.capitalize(), count)
            avg_conf = df["confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
            avg_act = df["activation"].mean()
            st.metric("Avg Activation", f"{avg_act:.3f}")

        st.divider()

        # Ingest new text
        st.subheader("📥 Ingest Text")
        new_text = st.text_area(
            "Enter paragraph to extract commitments:",
            height=150,
            help="Paste any text and the LLM will extract commitments with meta-roles",
        )
        if st.button("Extract & Store", type="primary"):
            if new_text.strip():
                with st.spinner("Extracting commitments with LLM..."):
                    nodes = memory.ingest_paragraph(new_text)
                    st.success(f"✅ Extracted and stored {len(nodes)} commitments!")
                    st.cache_data.clear()  # Clear cache to refresh data
                    st.rerun()
            else:
                st.warning("⚠️ Please enter some text")

        st.divider()

        # Reset database
        st.subheader("🗑️ Reset Database")
        st.caption("⚠️ This will delete all stored commitments")

        # Confirmation state
        if "confirm_reset" not in st.session_state:
            st.session_state.confirm_reset = False

        if not st.session_state.confirm_reset:
            if st.button("Reset Database", type="secondary"):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⚠️ Yes, Delete All", type="primary"):
                    import shutil

                    with st.spinner("Deleting database..."):
                        db_path = memory.store.db_path
                        if db_path.exists():
                            shutil.rmtree(db_path)
                        # Clear all caches including memory instance
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        # Force garbage collection to close file handles
                        import gc
                        gc.collect()
                        st.session_state.confirm_reset = False
                        st.success("✅ Database reset complete!")
                        st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()

        st.divider()

        # Clear cache button
        st.subheader("🔄 Clear Cache")
        st.caption("If you see errors after code changes, clear the cache")
        if st.button("Clear All Caches", type="secondary"):
            with st.spinner("Clearing caches..."):
                st.cache_data.clear()
                st.cache_resource.clear()
                import gc
                gc.collect()
                st.success("✅ Caches cleared!")
                st.rerun()

        st.divider()
        st.caption(f"Database: {memory.store.db_path}")

        # Tech stack info
        with st.expander("🔧 Tech Stack"):
            st.markdown("""
**Graph Visualization:**

**3D Graph (Plotly):**
- `plotly.graph_objects` - True 3D rendering
- `go.Scatter3d` - Nodes in 3D space
- Z-axis = activation level
- Drag/zoom/pan controls

**2D Graph (NetworkX + Plotly):**
- `networkx` - Graph algorithms
- `nx.spring_layout()` - Physics positioning
- `nx.kamada_kawai_layout()` - Energy minimization
- `nx.circular_layout()` - Circle arrangement

**Color coding:**
- Blue = Facts
- Green = Observations
- Red = Goals
- Purple = Constraints
- Orange = Decisions
- Teal = Conditional Dependencies

**Size = Activation level**
**Edges = Semantic connections**
            """)

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 All Nodes",
        "🔍 Semantic Search",
        "📖 Details",
        "🧠 Propagation Query",
        "🕸️ 3D Graph",
        "🎨 Canvas (vis.js)"
    ])

    with tab1:
        st.subheader("All Commitments")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            role_filter = st.selectbox(
                "Filter by Role",
                ["All"] + sorted(df["role"].unique().tolist()),
            )

        with col2:
            min_conf = st.slider(
                "Min Confidence",
                0.0, 1.0, 0.0, 0.1,
                help="Show only nodes above this confidence"
            )

        # Apply filters
        filtered_df = df.copy()
        if role_filter != "All":
            filtered_df = filtered_df[filtered_df["role"] == role_filter]
        if min_conf > 0:
            filtered_df = filtered_df[filtered_df["confidence"] >= min_conf]

        st.caption(f"Showing {len(filtered_df)} of {total} nodes")

        # Display table
        if not filtered_df.empty:
            display_df = filtered_df[[
                "text",
                "role",
                "confidence",
                "activation",
                "neighbors",
            ]].copy()

            # Format columns
            display_df["role"] = display_df["role"].str.upper().str.replace("_", " ")
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{'★' * int(x * 5)}")
            display_df["activation"] = display_df["activation"].apply(lambda x: f"{x:.3f}")
            display_df["connections"] = display_df["neighbors"].apply(lambda x: len(x) if isinstance(x, dict) else 0)
            display_df = display_df.drop("neighbors", axis=1)

            display_df.columns = ["Text", "Role", "Confidence", "Activation", "Connections"]
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                column_config={
                    "Connections": st.column_config.NumberColumn(
                        "Connections",
                        help="Number of edges to other nodes",
                        format="%d"
                    )
                }
            )
        else:
            st.info("No nodes match the filters")

    with tab2:
        st.subheader("Semantic Search")

        query = st.text_input(
            "Search query:",
            placeholder="e.g., 'What are the goals?' or 'budget constraints'",
        )

        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("Results", 1, 20, 5)
        with col2:
            min_similarity = st.slider(
                "Min Similarity",
                0.50, 0.75, 0.55, 0.01,
                help="Filter out weak matches"
            )

        # Quality thresholds for color coding
        with st.expander("🎨 Quality Thresholds (optional)"):
            col1, col2 = st.columns(2)
            with col1:
                red_yellow_threshold = st.slider(
                    "🔴 Red / 🟡 Yellow Threshold",
                    0.55, 0.65, 0.60, 0.01,
                    help="Below this = Red (low quality), Above = Yellow/Green"
                )
            with col2:
                yellow_green_threshold = st.slider(
                    "🟡 Yellow / 🟢 Green Threshold",
                    0.60, 0.75, 0.65, 0.01,
                    help="Below this = Yellow, Above = Green (high quality)"
                )

            if red_yellow_threshold >= yellow_green_threshold:
                st.error("⚠️ Yellow/Green threshold must be higher than Red/Yellow threshold!")

        if query:
            with st.spinner("Searching..."):
                all_results = memory.store.query_similar(
                    memory.embedding_manager.generate_embedding(query),
                    k=k * 2,  # Get more to filter
                )

                # Filter by similarity
                results = [r for r in all_results if r.get('similarity', 0) >= min_similarity]

                # Count filtered
                filtered = len(all_results) - len(results)
                if filtered > 0:
                    st.caption(f"⚠️ Filtered out {filtered} weak matches (similarity < {min_similarity})")

            if results:
                for i, r in enumerate(results, 1):
                    similarity = r.get('similarity', 0)
                    quality_emoji = "🟢" if similarity >= yellow_green_threshold else "🟡" if similarity >= red_yellow_threshold else "🔴"

                    with st.expander(
                        f"{i}. [{r['role'].upper()}] {quality_emoji} Sim={similarity:.3f} | {r['text'][:80]}...",
                        expanded=i == 1,
                    ):
                        st.write(f"**Text:** {r['text']}")
                        st.write(f"**Role:** {r['role'].replace('_', ' ').title()}")
                        st.write(f"**Confidence:** {r['confidence']:.2f}")
                        st.write(f"**Similarity:** {similarity:.3f} {quality_emoji}")

                        if similarity < red_yellow_threshold:
                            st.warning(f"⚠️ Low similarity (< {red_yellow_threshold:.2f}) - might not be relevant")
                        elif similarity < yellow_green_threshold:
                            st.caption(f"⚡ Medium similarity ({red_yellow_threshold:.2f} - {yellow_green_threshold:.2f}) - review for relevance")
                        else:
                            st.success(f"✅ High similarity (≥ {yellow_green_threshold:.2f}) - likely relevant")

                        st.write(f"**Activation:** {r['activation']:.3f}")
            else:
                st.info(f"No results found above similarity threshold {min_similarity:.2f}")
        else:
            st.info("Enter a search query above")

    with tab3:
        st.subheader("Node Details")

        # Get nodes from in-memory graph (more up-to-date)
        if len(memory.graph) > 0:
            # Sort nodes by text for selection
            sorted_nodes = sorted(memory.graph.nodes.values(), key=lambda n: n.text[:50])
            node_options = [f"{n.text[:50]}... [{n.role.value}]" for n in sorted_nodes]
            selected = st.selectbox("Select node", node_options)

            if selected:
                idx = node_options.index(selected)
                node = sorted_nodes[idx]

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Role:**", node.role.value.replace("_", " ").title())
                    st.write("**Confidence:**", f"{node.confidence:.2f}")
                    st.write("**Activation:**", f"{node.activation:.3f}")

                with col2:
                    st.write("**ID:**", node.id[:8] + "...")
                    st.json(node.metadata)

                st.divider()
                st.subheader("Text")
                st.write(node.text)

                if node.neighbors and len(node.neighbors) > 0:
                    st.subheader(f"Connected Nodes ({len(node.neighbors)} connections)")

                    # Show each neighbor with their text
                    for neighbor_id, weight in node.neighbors.items():
                        neighbor = memory.graph.get_node(neighbor_id)
                        if neighbor:
                            with st.expander(
                                f"💪 {weight:.3f} | [{neighbor.role.value.upper()}] {neighbor.text[:60]}..."
                            ):
                                st.write(f"**Text:** {neighbor.text}")
                                st.write(f"**Role:** {neighbor.role.value.replace('_', ' ').title()}")
                                st.write(f"**Confidence:** {neighbor.confidence:.2f}")
                                st.write(f"**Connection Strength:** {weight:.3f}")
                        else:
                            st.write(f"⚠️ Node {neighbor_id[:8]}... (strength: {weight:.3f}) - not in graph")
                else:
                    st.info("🔗 No connections (this node is isolated)")

                st.divider()
                st.subheader("Vector (first 10 dims)")
                vector = node.embedding
                if vector is not None and len(vector) > 0:
                    st.write(str(vector[:10]) + "...")

                # Debug section: compare graph vs database
                with st.expander("🔍 Debug Info (Graph vs Database)"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**In-Memory Graph:**")
                        st.write(f"Neighbors count: {len(node.neighbors)}")
                        if node.neighbors:
                            st.json(node.neighbors)
                        else:
                            st.write("No neighbors in graph")

                    with col2:
                        st.write("**Database Record:**")
                        db_record = memory.store.get_node(node.id)
                        if db_record:
                            db_neighbors = db_record.get("neighbors", {})
                            st.write(f"Neighbors count: {len(db_neighbors)}")
                            if db_neighbors:
                                st.json(db_neighbors)
                            else:
                                st.write("No neighbors in database")
                        else:
                            st.write("Node not found in database!")

    with tab4:
        st.subheader("🧠 Propagation Query (Direct + Network Results)")

        # Query controls
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            query = st.text_input(
                "Enter your query:",
                placeholder="e.g., 'database limits' or 'cache performance'",
                key="prop_query"
            )
        with col2:
            direct_k = st.slider(
                "Direct",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                help="Direct matches from vector search",
                key="prop_direct_k"
            )
        with col3:
            total_k = st.slider(
                "Total",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                help="Total results (direct + propagated)",
                key="prop_total_k"
            )
        with col4:
            depth = st.selectbox(
                "Depth",
                [1, 2, 3],
                index=1,
                help="How many hops to propagate",
                key="prop_depth"
            )

        # Rerun button
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            # Initialize session state for last query
            if "last_prop_query" not in st.session_state:
                st.session_state.last_prop_query = ""

            # Update last query when user types
            if query and query != st.session_state.last_prop_query:
                st.session_state.last_prop_query = query

        with col2:
            rerun_clicked = st.button(
                "🔄 Rerun",
                type="primary",
                disabled=not st.session_state.last_prop_query,
                help="Re-run propagation with current threshold settings"
            )
        with col3:
            if st.button("Clear"):
                st.session_state.last_prop_query = ""
                st.rerun()

        # Use last query if rerun clicked
        if rerun_clicked and st.session_state.last_prop_query:
            query = st.session_state.last_prop_query

        # Second row: filters
        col1, col2 = st.columns([2, 1])
        with col1:
            min_similarity = st.slider(
                "Min Similarity",
                min_value=0.50,
                max_value=0.75,
                value=0.55,
                step=0.01,
                help="Filter out weak matches (increase to reduce noise)",
                key="prop_min_sim"
            )
        with col2:
            candidate_multiplier = st.slider(
                "Candidates",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Fetch multiplier: Total × Candidates = how many to fetch before filtering (default: 2)",
                key="prop_candidate_multiplier"
            )

        # Quality thresholds
        with st.expander("🎨 Quality Thresholds"):
            col1, col2 = st.columns(2)
            with col1:
                prop_red_yellow = st.slider(
                    "🔴/🟡 Threshold",
                    0.55, 0.65, 0.60, 0.01,
                    help="Below = Red, Above = Yellow/Green",
                    key="prop_red_yellow"
                )
            with col2:
                prop_yellow_green = st.slider(
                    "🟡/🟢 Threshold",
                    0.60, 0.75, 0.65, 0.01,
                    help="Below = Yellow, Above = Green",
                    key="prop_yellow_green"
                )

            if prop_red_yellow >= prop_yellow_green:
                st.error("⚠️ Green threshold must be higher than Red threshold!")

        # Propagation thresholds
        with st.expander("🌊 Propagation Thresholds"):
            st.caption("Control how activation spreads through the network")
            col1, col2 = st.columns(2)

            with col1:
                prop_activation_threshold = st.slider(
                    "Min Node Activation",
                    0.1, 0.9, 0.5, 0.05,
                    help="🚪 **Gate 2**: Can this activated node CONTINUE propagating?\n\nNodes below this activation won't propagate further to their children (default: 0.5).\n\nThink: \"Is this node strong enough to promote?\"",
                    key="prop_activation_threshold"
                )
            with col2:
                prop_min_delta = st.slider(
                    "Min Signal Strength",
                    0.1, 0.5, 0.3, 0.05,
                    help="🚪 **Gate 1**: Can this node BE activated at all?\n\nMinimum signal required to activate a child node (default: 0.3).\n\nThink: \"Is this signal strong enough to interview?\"",
                    key="prop_min_delta"
                )

            st.caption("💡 Higher values = less propagation, more focused results")

            # Add explanation section
            with st.expander("📖 How propagation gates work"):
                st.markdown("""
                ### Two-Gate System

                Think of it like a hiring process with two checkpoints:

                **Gate 1: Min Signal Strength** (default: 0.3)
                - Question: "Is this signal strong enough to interview?"
                - Applied: Before activating a child node
                - Filters: Weak parent→child signals

                **Gate 2: Min Node Activation** (default: 0.5)
                - Question: "Is this node strong enough to promote?"
                - Applied: After activation, before continuing
                - Filters: Nodes that exist but are too weak to propagate further

                ### Example Flow

                ```
                Layer 1: banana = 0.873 ✅
                         ↓ Check Gate 1: 0.873 × 0.42 × 1.0 = 0.367 ≥ 0.3 ✅
                         Update child: guava = 0.370
                         Check Gate 2: 0.370 < 0.5 ❌

                Layer 2: guava = 0.370 ✅ (activated, but stops here)
                         ↓ Won't propagate further (fails Gate 2)

                Layer 3: (never reached)
                ```

                ### Summary

                - **Gate 1**: Controls who ENTERS the system
                - **Gate 2**: Controls who KEEPS GOING
                """)

            st.info("ℹ️ Changes require clearing cache to take effect")

        st.caption("📊 Results include both direct vector matches AND nodes activated through network propagation")

        if query:
            try:
                with st.spinner("Querying with propagation..."):
                    # Update propagation thresholds from UI
                    memory.graph.activation_threshold = prop_activation_threshold
                    memory.graph.min_delta = prop_min_delta  # Use the slider value!

                    # Get direct matches first (without propagation)
                    query_emb = memory.embedding_manager.generate_embedding(query)
                    all_direct_results = memory.store.query_similar(query_emb, k=direct_k * candidate_multiplier)

                    # Filter by minimum similarity
                    direct_results = [r for r in all_direct_results if r.get('similarity', 0) >= min_similarity]

                    # Count what was filtered
                    filtered_count = len(all_direct_results) - len(direct_results)
                    if filtered_count > 0:
                        st.caption(f"⚠️ Filtered out {filtered_count} weak matches (similarity < {min_similarity:.2f})")

                    if not direct_results:
                        st.warning(f"No matches found with similarity ≥ {min_similarity:.2f}. Try lowering the threshold.")
                        st.stop()

                    # Now run full query with propagation
                    final_results = memory.query(
                        query,
                        top_k=total_k,
                        propagation_depth=depth,
                        min_similarity_threshold=min_similarity,
                        candidate_multiplier=candidate_multiplier
                    )

                    # Show rerun confirmation
                    if rerun_clicked:
                        st.success(f"✅ Re-ran query with updated threshold settings")
            except Exception as e:
                st.error(f"❌ Error during query: {str(e)}")
                st.info("💡 Tip: If you're seeing a network error, check your internet connection or try using local embeddings.")
                st.stop()

            # Count direct vs propagated
            direct_count = 0
            propagated_count = 0
            for node in final_results:
                is_direct = any(dr["id"] == node.id for dr in direct_results)
                if is_direct:
                    direct_count += 1
                else:
                    propagated_count += 1

            # Display results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Results", len(final_results))
            with col2:
                st.metric("Direct Matches", direct_count, delta="🎯 Vector search")
            with col3:
                st.metric("Propagated", propagated_count, delta="🌊 Network spread")

            # View mode selector
            col1, col2 = st.columns([3, 1])
            with col1:
                view_mode = st.radio(
                    "View Mode",
                    ["📊 Compact Graph", "📄 Detailed Tree", "🎬 Animated"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
            with col2:
                with st.popover("❓ Role Boosts"):
                    st.markdown("### 📈 Role Boost Matrix")
                    st.markdown("*Activation multiplier based on source → target role*")
                    st.caption("| From | To | Boost | Why |")
                    st.caption("|------|-----|-------|-----|")
                    st.caption("| GOAL | DECISION | **1.5×** | Goals drive decisions |")
                    st.caption("| CONSTRAINT | DECISION | **-0.8×** | Constraints inhibit decisions |")
                    st.caption("| OBSERVATION | DECISION | **1.2×** | Observations inform decisions |")
                    st.caption("| FACT | FACT | **1.3×** | Facts reinforce each other |")
                    st.caption("| CONDITIONAL | DECISION | **1.1×** | Conditionals guide decisions |")
                    st.caption("| CONDITIONAL | CONSTRAINT | **1.1×** | Conditionals create constraints |")
                    st.caption("| GOAL | CONDITIONAL | **1.2×** | Goals create dependencies |")
                    st.markdown("---")
                    st.caption("*All other combinations: 1.0× (neutral)*")

            st.markdown("---")

            # Build propagation tree recursively
            from collections import deque

            # Track (parent, child) edges to prevent cycles while allowing multi-parent nodes
            visited_edges = set()

            # Tree structure: node_id -> {"children": [(child_id, weight, boost), ...]}
            tree = {}
            root_children = []

            # Add Layer 1 (direct matches)
            for direct_rec in direct_results:
                node_id = direct_rec["id"]
                similarity = direct_rec.get('similarity', 0.5)
                tree[node_id] = {"children": []}
                root_children.append((node_id, similarity, 1.0, direct_rec))

            # Build tree via BFS from each direct match
            visited = set(dr["id"] for dr in direct_results)
            queue = deque([(node_id, 1) for node_id, _, _, _ in root_children])

            while queue:
                current_id, current_depth = queue.popleft()

                if current_depth >= depth:
                    continue

                current_node = memory.graph.get_node(current_id)
                if not current_node or not current_node.neighbors:
                    continue

                # Add neighbors as children
                for neighbor_id, weight in sorted(current_node.neighbors.items(), key=lambda x: -x[1]):
                    # Skip if this specific edge was already traversed (prevents cycles)
                    if (current_id, neighbor_id) in visited_edges:
                        continue

                    neighbor = memory.graph.get_node(neighbor_id)
                    if not neighbor:
                        continue


                    visited.add(neighbor_id)
                    visited_edges.add((current_id, neighbor_id))  # Track edge to prevent cycles
                    role_boost = _get_role_boost(current_node.role, neighbor.role)

                    # Initialize node if not exists
                    if neighbor_id not in tree:
                        tree[neighbor_id] = {"children": []}

                    # Add as child of current (allows multi-parent nodes)
                    tree[current_id]["children"].append((neighbor_id, weight, role_boost))
                    queue.append((neighbor_id, current_depth + 1))

            if view_mode == "📊 Compact Graph":
                # COMPACT GRAPH MODE - Using HTML/CSS for proper nesting
                st.subheader("📊 Propagation Graph (Compact)")

                # Build HTML tree
                def build_html_tree(node_id, weight, boost, level, parent_rec=None, parent_similarity=None, visited_edges=None):
                    """Build HTML tree recursively with collapsible nodes.

                    Args:
                        node_id: Current node ID
                        weight: Edge weight from parent
                        boost: Role boost from parent
                        level: Current depth level
                        parent_rec: Parent record with 'id' key
                        parent_similarity: Parent's ORIGINAL similarity to query (not propagated activation)
                        visited_edges: Set of (parent, child) pairs already rendered to prevent cycles
                    """
                    if visited_edges is None:
                        visited_edges = set()

                    node = memory.graph.get_node(node_id)
                    if not node:
                        return ""

                    # Use final activation (node.activation) which is the MAX of all incoming signals
                    # This shows the true activation value, not the propagated value from a specific parent
                    activation = node.activation

                    # Track parent's similarity for children propagation calculations
                    if parent_rec is None:
                        # Layer 1: Direct match - use similarity_to_query
                        parent_similarity = node.similarity_to_query if node.similarity_to_query > 0 else activation
                    # For Layer 2+, we keep using the parent's parent_similarity (passed down)

                    # Check if in final results
                    in_final = node_id in {n.id for n in final_results}
                    bg_color = "#d4edda" if in_final else "#f8f9fa"

                    # Role colors
                    role_colors = {
                        "FACT": "#007bff",
                        "OBSERVATION": "#28a745",
                        "GOAL": "#dc3545",
                        "CONSTRAINT": "#6f42c1",
                        "DECISION": "#fd7e14",
                        "CONDITIONAL_DEPENDENCY": "#ffc107"
                    }
                    role_color = role_colors.get(node.role.value, "#6c757d")

                    # Node HTML
                    indent_px = (level - 1) * 24
                    full_text = node.text.replace('"', '&quot;').replace("'", '&#39;').replace('<', '&lt;').replace('>', '&gt;')

                    # Check if has children
                    has_children = node_id in tree and tree[node_id]["children"]
                    child_count = len(tree[node_id]["children"]) if has_children else 0

                    html = f"<div style='margin-left:{indent_px}px;margin-top:4px;margin-bottom:4px;'>"

                    if has_children:
                        # Collapsible node
                        also_direct = node.similarity_to_query > 0 and parent_rec is not None
                        html += f"<details style='margin:0;' {'open' if level <= 1 else ''}><summary style='list-style:none;cursor:pointer;padding:0;display:block;width:100%;'><div style='background:{bg_color};border-left:3px solid {role_color};border-radius:4px;padding:8px 12px;transition:all 0.2s;' onmouseover=\"this.style.transform='translateX(4px)'\" onmouseout=\"this.style.transform='translateX(0)'\" title=\"{full_text}\"><div style=\"display:flex;align-items:center;gap:8px;flex-wrap:wrap;\"><span style=\"color:#6c757d;font-size:14px;\">{'▾' if level <= 1 else '▸'}</span><span style=\"color:{role_color};font-weight:bold;font-size:12px;\">{node.role.value.upper()}</span><span style=\"color:#666;font-size:11px;\">⚡{activation:.3f}</span><span style=\"color:#666;font-size:11px;\">💪{weight:.2f}×{boost:.1f}</span>{f'<span style=\"color:#28a745;font-size:10px;\">(also direct: {node.similarity_to_query:.3f})</span>' if also_direct else ''}<span style=\"color:#adb5bd;font-size:10px;\">({child_count} children)</span></div><div style=\"margin-top:4px;font-size:13px;color:#333;\">{node.text[:55]}...</div></div></summary><div style='margin-left:12px;border-left:1px dashed #ccc;padding-left:8px;margin-top:4px;'>"
                        for child_id, child_weight, child_boost in tree[node_id]["children"]:
                            # Check for cycles - skip if we've already rendered this edge
                            if (node_id, child_id) in visited_edges:
                                continue
                            visited_edges.add((node_id, child_id))
                            html += build_html_tree(child_id, child_weight, child_boost, level + 1, {"id": node_id}, parent_similarity, visited_edges)
                        html += "</div></details>"
                    else:
                        # Leaf node
                        also_direct = node.similarity_to_query > 0 and parent_rec is not None
                        html += f"<div style='background:{bg_color};border-left:3px solid {role_color};border-radius:4px;padding:8px 12px;cursor:help;transition:all 0.2s;' onmouseover=\"this.style.transform='translateX(4px)'\" onmouseout=\"this.style.transform='translateX(0)'\" title=\"{full_text}\"><div style=\"display:flex;align-items:center;gap:8px;flex-wrap:wrap;\"><span style=\"color:#adb5bd;font-size:14px;\">•</span><span style=\"color:{role_color};font-weight:bold;font-size:12px;\">{node.role.value.upper()}</span><span style=\"color:#666;font-size:11px;\">⚡{activation:.3f}</span><span style=\"color:#666;font-size:11px;\">💪{weight:.2f}×{boost:.1f}</span>{f'<span style=\"color:#28a745;font-size:10px;\">(also direct: {node.similarity_to_query:.3f})</span>' if also_direct else ''}</div><div style=\"margin-top:4px;font-size:13px;color:#333;\">{node.text[:55]}...</div></div>"

                    html += "</div>"
                    return html

                # Render tree
                st.markdown(f"**🔍 Query:** `{query}`")
                st.markdown("---")

                # Scrollable container for tree (both vertical and horizontal)
                st.markdown(
                    '<div style="max-height: 600px; overflow-x: auto; overflow-y: auto; padding: 10px; border: 1px solid #dee2e6; border-radius: 4px;">',
                    unsafe_allow_html=True
                )

                # Share visited_edges across all roots to prevent cycles from bidirectional edges
                global_visited_edges = set()
                for node_id, weight, boost, direct_rec in root_children:
                    tree_html = build_html_tree(node_id, weight, boost, level=1, parent_rec=None, visited_edges=global_visited_edges)
                    st.markdown(tree_html, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Legend
                st.markdown("---")
                st.caption("💡 Hover over nodes to see full text")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Status:**")
                    st.markdown("🟢 In Results | ⚪ Excluded")
                with col2:
                    st.markdown("**Roles:**")
                    st.markdown("🔵 FACT | 🟢 OBSERVATION | 🔴 GOAL")
                with col3:
                    st.markdown("**More:**")
                    st.markdown("🟣 CONSTRAINT | 🟠 DECISION | 🟡 CONDITIONAL")

            elif view_mode == "📄 Detailed Tree":
                # DETAILED MODE - Using HTML/CSS with expandable sections
                st.subheader("🌳 Propagation Tree (Detailed)")

                def build_detailed_html_tree(node_id, weight, boost, level, parent_rec=None, parent_similarity=None, visited_edges=None):
                    """Build detailed HTML tree with expandable sections.

                    Args:
                        parent_similarity: Parent's ORIGINAL similarity to query (not propagated activation)
                        visited_edges: Set of (parent, child) pairs already rendered to prevent cycles
                    """
                    if visited_edges is None:
                        visited_edges = set()

                    node = memory.graph.get_node(node_id)
                    if not node:
                        return ""

                    # Use final activation (node.activation) which is the MAX of all incoming signals
                    activation = node.activation

                    # Track parent's similarity for children propagation calculations
                    if parent_rec is None:
                        # Layer 1: Direct match - use similarity_to_query
                        parent_similarity = node.similarity_to_query if node.similarity_to_query > 0 else activation
                    # For Layer 2+, we keep using the parent's parent_similarity (passed down)

                    # Check if in final results
                    in_final = node_id in {n.id for n in final_results}
                    bg_color = "#d4edda" if in_final else "#ffffff"

                    # Role colors
                    role_colors = {
                        "FACT": "#007bff", "OBSERVATION": "#28a745", "GOAL": "#dc3545",
                        "CONSTRAINT": "#6f42c1", "DECISION": "#fd7e14", "CONDITIONAL_DEPENDENCY": "#ffc107"
                    }
                    role_color = role_colors.get(node.role.value, "#6c757d")
                    indent_px = (level - 1) * 24
                    has_children = node_id in tree and tree[node_id]["children"]

                    # Build HTML
                    html = f"<div style='margin-left:{indent_px}px;margin-bottom:8px;'>"
                    html += f"<details style='background:{bg_color};border-left:4px solid {role_color};border-radius:6px;padding:12px;cursor:pointer;'>"
                    html += "<summary style='list-style:none;cursor:pointer;display:block;width:100%;'>"
                    html += "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;'>"
                    html += "<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
                    html += f"<span style='background:{role_color};color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;'>{node.role.value.upper()}</span>"
                    html += f"<span style='font-size:14px;color:#333;font-weight:500;'>{node.text[:60]}...</span>"
                    html += "</div>"
                    html += "<div style='display:flex;align-items:center;gap:12px;font-size:12px;color:#666;'>"
                    html += f"<span title='Layer {level}'>📍L{level}</span>"
                    html += f"<span title='Activation'>⚡{activation:.3f}</span>"
                    html += f"<span title='Edge Weight × Role Boost'>💪{weight:.2f}×{boost:.1f}</span>"
                    also_direct = node.similarity_to_query > 0 and parent_rec is not None
                    if also_direct:
                        html += f"<span style='color:#28a745;font-size:10px;' title='Also direct match'>(direct: {node.similarity_to_query:.3f})</span>"
                    html += f"<span style='color:{'#28a745' if in_final else '#6c757d'}'>{('✅' if in_final else '💤')}</span>"
                    html += "</div></div></summary>"

                    # Content inside details
                    html += "<div style='margin-top:12px;padding-top:12px;border-top:1px solid #dee2e6;'>"
                    html += "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px;'>"
                    html += f"<div><div style='font-size:11px;color:#6c757d;margin-bottom:4px;'>ACTIVATION</div><div style='font-size:14px;font-weight:500;color:#333;'>{activation:.3f}</div></div>"
                    html += f"<div><div style='font-size:11px;color:#6c757d;margin-bottom:4px;'>EDGE WEIGHT</div><div style='font-size:14px;font-weight:500;color:#333;'>{weight:.3f}</div></div>"
                    html += f"<div><div style='font-size:11px;color:#6c757d;margin-bottom:4px;'>ROLE BOOST</div><div style='font-size:14px;font-weight:500;color:{role_color};'>{boost:.1f}×</div></div>"
                    html += "</div>"

                    # Formula
                    if parent_rec and parent_similarity is not None:
                        parent_node = memory.graph.get_node(parent_rec["id"])
                        # Calculate what this node received from THIS parent (for display)
                        propagated_from_this_parent = parent_similarity * weight * boost
                        html += f"<div style='background:#f8f9fa;padding:10px;border-radius:4px;margin-bottom:12px;'>"
                        html += "<div style='font-size:11px;color:#6c757d;margin-bottom:6px;'>🧮 PROPAGATION FROM THIS PARENT</div>"
                        html += f"<code style='font-size:12px;color:#495057;'>[{parent_similarity:.3f}] parent_sim × [{weight:.3f}] edge × [{boost:.1f}] boost = <strong>{propagated_from_this_parent:.3f}</strong></code>"
                        html += "<div style='font-size:11px;color:#6c757d;margin-top:6px;'>"
                        html += f"📍 Parent's similarity to query × 💪 Connection strength × 📈 Role multiplier"
                        html += f"<br/>Parent: <strong>{parent_node.text[:40]}...</strong> ({parent_node.role.value.upper()})"
                        html += f"<br/>Received: <strong>{propagated_from_this_parent:.3f}</strong> from this parent"

                        # Show if final activation is higher (due to multiple parents or direct match)
                        if activation > propagated_from_this_parent:
                            diff = activation - propagated_from_this_parent
                            html += f"<br/>✨ Final activation: <strong>{activation:.3f}</strong> (+{diff:.3f} from other sources)"
                        if node.similarity_to_query > 0:
                            html += f"<br/><span style='color:#28a745;'>⚠️ Also direct match: {node.similarity_to_query:.3f}</span>"
                        html += "</div>"
                        html += "</div>"
                    else:
                        html += f"<div style='background:#f8f9fa;padding:10px;border-radius:4px;margin-bottom:12px;'>"
                        html += "<div style='font-size:11px;color:#6c757d;margin-bottom:6px;'>📊 VECTOR SIMILARITY</div>"
                        html += f"<div style='font-size:14px;font-weight:500;color:#333;'>{weight:.3f}</div>"
                        html += "<div style='font-size:11px;color:#6c757d;margin-top:4px;'>Direct match from query (cosine similarity)</div>"
                        html += "</div>"

                    # Children
                    if has_children:
                        html += f"<div style='font-size:11px;color:#6c757d;margin-bottom:8px;'>🌊 {len(tree[node_id]['children'])} child node(s)</div>"
                        html += "</div></details>"
                        html += f"<div style='margin-left:12px;border-left:1px dashed #ccc;'>"
                        for child_id, child_weight, child_boost in tree[node_id]["children"]:
                            # Check for cycles - skip if we've already rendered this edge
                            if (node_id, child_id) in visited_edges:
                                continue
                            visited_edges.add((node_id, child_id))
                            html += build_detailed_html_tree(child_id, child_weight, child_boost, level + 1, {"id": node_id}, parent_similarity, visited_edges)
                        html += "</div>"
                    else:
                        html += "<div style='font-size:11px;color:#adb5bd;'>🍃 No children (leaf node)</div>"
                        html += "</div></details>"

                    html += "</div>"
                    return html

                # Render tree
                st.markdown(f"**🔍 Query:** `{query}`")
                st.markdown(f"**📍 Layer 1:** {len(root_children)} direct match(es)")
                st.markdown("---")

                # Scrollable container for tree (both vertical and horizontal)
                st.markdown(
                    '<div style="max-height: 600px; overflow-x: auto; overflow-y: auto; padding: 10px; border: 1px solid #dee2e6; border-radius: 4px;">',
                    unsafe_allow_html=True
                )

                # Share visited_edges across all roots to prevent cycles from bidirectional edges
                global_visited_edges = set()
                for node_id, weight, boost, direct_rec in root_children:
                    tree_html = build_detailed_html_tree(node_id, weight, boost, level=1, parent_rec=None, visited_edges=global_visited_edges)
                    st.markdown(tree_html, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                st.caption("💡 Click ▶ to expand nodes and see details. Layer = hops from query.")

            elif view_mode == "🎬 Animated":
                # ANIMATED MODE - Step-by-step propagation
                st.subheader("🎬 Animated Propagation")

                # Run simulation
                if "sim_data" not in st.session_state or st.session_state.sim_data.get("query") != query:
                    with st.spinner("Running simulation..."):
                        sim_data = memory.query_simulation(
                            query_text=query,
                            top_k=total_k,
                            propagation_depth=depth,
                            min_similarity_threshold=min_similarity,
                            candidate_multiplier=candidate_multiplier,
                            decay_per_hop=0.7  # Default decay rate
                        )
                        st.session_state.sim_data = sim_data
                        st.session_state.anim_step = 0

                sim_data = st.session_state.sim_data
                total_steps = sim_data["total_steps"]

                # Navigation controls - using number_input for instant navigation
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                with col1:
                    if st.button("⏮️", key="anim_first"):
                        st.session_state.anim_step = 0
                with col2:
                    if st.button("◀️", key="anim_prev"):
                        if st.session_state.anim_step > 0:
                            st.session_state.anim_step -= 1
                with col3:
                    # Use number_input for instant slider-like navigation
                    step = st.number_input(
                        "Step",
                        min_value=0,
                        max_value=total_steps - 1,
                        value=st.session_state.anim_step,
                        step=1,
                        label_visibility="collapsed",
                        key="anim_step_input",
                        format="%d"
                    )
                    # Sync session state
                    st.session_state.anim_step = int(step)
                    st.markdown(f"<div style='text-align:center;'>Step {step + 1}/{total_steps}</div>", unsafe_allow_html=True)
                with col4:
                    if st.button("▶️", key="anim_next"):
                        if st.session_state.anim_step < total_steps - 1:
                            st.session_state.anim_step += 1
                with col5:
                    if st.button("⏭️", key="anim_last"):
                        st.session_state.anim_step = total_steps - 1

                # Manual control info
                st.caption("💡 Use buttons or drag the slider to navigate instantly")

                # Current step display
                step_data = sim_data["timeline"][step]
                step_type = step_data["type"]

                # Status indicator
                status_colors = {
                    "search": "#6c757d", "search_results": "#17a2b8", "layer_1": "#28a745",
                    "hop_start": "#fd7e14", "propagation": "#007bff", "gate_1_fail": "#dc3545",
                    "gate_2_fail": "#dc3545", "filtered": "#6c757d", "complete": "#28a745",
                }
                status_color = status_colors.get(step_type, "#6c757d")

                st.markdown(
                    f"<div style='background:{status_color};color:white;padding:12px;border-radius:6px;margin-bottom:16px;'>"
                    f"<strong>Step {step + 1}:</strong> {step_data['message']}</div>",
                    unsafe_allow_html=True
                )

                # Display step-specific content
                if step_type == "search":
                    st.info("🔍 Generating embeddings and searching...")

                elif step_type == "search_results":
                    candidates = step_data.get("candidates", [])
                    st.write(f"**Found {len(candidates)} candidates**")
                    for c in candidates[:5]:
                        st.write(f"- {c['text'][:50]}... (sim: {c['similarity']:.3f})")

                elif step_type == "layer_1":
                    st.success("✅ Layer 1: Direct matches activated")
                    for node_data in step_data.get("nodes_data", []):
                        role_emoji = {"FACT": "🔵", "OBSERVATION": "🟢", "GOAL": "🔴", "CONSTRAINT": "🟣", "DECISION": "🟠"}
                        emoji = role_emoji.get(node_data["role"], "⚪")
                        st.markdown(f"{emoji} **{node_data['role']}** - Activation: {node_data['activation']:.3f}")

                elif step_type == "propagation":
                    st.write(f"🌊 **Hop {step_data.get('hop', 0)}** - Propagating activation")
                    parent_id = step_data.get("parent_id", "")[:8]
                    child_id = step_data.get("child_id", "")[:8]
                    delta = step_data.get("delta", 0)
                    st.write(f"From: {parent_id}... → To: {child_id}...")
                    st.write(f"Signal strength: Δ{delta:.4f}")
                    if step_data.get("newly_activated"):
                        st.success("🆕 Node activated!")
                    else:
                        st.caption("Node already activated")

                elif step_type == "gate_1_fail":
                    st.warning(f"🚫 Gate 1: Signal too weak (Δ {step_data.get('delta', 0):.3f} < {step_data.get('threshold', 0)})")

                elif step_type == "gate_2_fail":
                    st.warning(f"🚫 Gate 2: Node too weak to propagate further (activation: {step_data.get('activation', 0):.3f})")

                elif step_type == "complete":
                    st.success("✨ Propagation complete!")
                    final = step_data.get("final_states", [])
                    st.write(f"**Total activated:** {len(final)}")
                    for i, state in enumerate(final[:10], 1):
                        st.write(f"{i}. **{state['role']}** - Act: {state['activation']:.3f} - {state['text'][:40]}...")

                # Progress bar
                progress = (step + 1) / total_steps
                st.markdown(
                    f"<div style='width:100%;background:#e9ecef;height:8px;border-radius:4px;margin-top:16px;'>"
                    f"<div style='width:{progress*100}%;background:{status_color};height:100%;border-radius:4px;'></div></div>",
                    unsafe_allow_html=True
                )

        else:
            st.info("Enter a query above to see propagation results")

    with tab5:
        st.subheader("🕸️ 3D Cognitive Graph (True 3D)")

        col1, col2, col3 = st.columns(3)
        with col1:
            min_activation = st.slider(
                "Min Activation",
                0.0, 1.0, 0.0, 0.1,
                help="Show only nodes above this activation"
            )
        with col2:
            show_labels = st.checkbox("Show Labels", value=True)
        with col3:
            layout = st.selectbox("Layout", ["Force Directed", "Circular", "Random"])

        if not df.empty:
            with st.spinner("Generating 3D graph..."):
                # Filter by activation
                filtered_df = df[df["activation"] >= min_activation].copy()

                if len(filtered_df) == 0:
                    st.warning("No nodes match the activation filter")
                else:
                    # Create NetworkX graph
                    G = nx.Graph()

                    # Color map by role
                    role_colors = {
                        "fact": "rgb(52, 152, 219)",           # blue
                        "observation": "rgb(46, 204, 113)",     # green
                        "goal": "rgb(231, 76, 60)",           # red
                        "constraint": "rgb(155, 89, 182)",     # purple
                        "decision": "rgb(243, 156, 18)",       # orange
                        "conditional_dependency": "rgb(26, 188, 156)",  # teal
                    }

                    # Add nodes
                    for _, row in filtered_df.iterrows():
                        G.add_node(
                            row["id"],
                            text=row["text"],
                            role=row["role"],
                            confidence=row["confidence"],
                            activation=row["activation"],
                            color=role_colors.get(row["role"], "rgb(149, 165, 166)"),
                        )

                    # Add edges from neighbors
                    for _, row in filtered_df.iterrows():
                        neighbors = row.get("neighbors", {})
                        if neighbors:
                            for neighbor_id, weight in neighbors.items():
                                if neighbor_id in G.nodes:
                                    G.add_edge(row["id"], neighbor_id, weight=weight)

                    # 3D layout
                    if layout == "Force Directed":
                        # Use spring layout in 3D
                        pos_2d = nx.spring_layout(G, dim=2, k=2, iterations=50)
                        # Add Z dimension based on activation
                        pos_3d = {}
                        for node, (x, y) in pos_2d.items():
                            z = G.nodes[node].get("activation", 0) * 10
                            pos_3d[node] = (x, y, z)
                    elif layout == "Circular":
                        pos_2d = nx.circular_layout(G, dim=2)
                        pos_3d = {}
                        for node, (x, y) in pos_2d.items():
                            z = G.nodes[node].get("activation", 0) * 10
                            pos_3d[node] = (x, y, z)
                    else:  # Random
                        pos_3d = nx.random_layout(G, dim=3)

                    # Extract node coordinates and attributes
                    node_x = []
                    node_y = []
                    node_z = []
                    node_text = []
                    node_colors = []
                    node_sizes = []

                    for node in G.nodes():
                        x, y, z = pos_3d[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_z.append(z)

                        node_data = G.nodes[node]
                        activation = node_data.get("activation", 0)
                        size = 5 + (activation * 30)
                        node_sizes.append(size)

                        node_colors.append(node_data.get("color", "rgb(149, 165, 166)"))

                        if show_labels:
                            node_text.append(
                                f"{node_data.get('role', 'unknown').upper()}<br>"
                                f"{node_data.get('text', '')[:60]}...<br>"
                                f"Conf: {node_data.get('confidence', 0):.2f}<br>"
                                f"Act: {activation:.3f}"
                            )
                        else:
                            node_text.append("")

                    # Create 3D scatter for nodes
                    fig = go.Figure()

                    fig.add_trace(go.Scatter3d(
                        x=node_x,
                        y=node_y,
                        z=node_z,
                        mode='markers+text' if show_labels else 'markers',
                        marker=dict(
                            size=node_sizes,
                            color=node_colors,
                            opacity=0.8,
                            line=dict(width=1, color='white')
                        ),
                        text=node_text,
                        hovertemplate='<b>%{text}</b><extra></extra>',
                        name='Nodes'
                    ))

                    # Add edges
                    edge_x = []
                    edge_y = []
                    edge_z = []

                    for edge in G.edges(data=True):
                        x0, y0, z0 = pos_3d[edge[0]]
                        x1, y1, z1 = pos_3d[edge[1]]

                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_z.extend([z0, z1, None])

                    fig.add_trace(go.Scatter3d(
                        x=edge_x,
                        y=edge_y,
                        z=edge_z,
                        mode='lines',
                        line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
                        hoverinfo='none',
                        name='Edges'
                    ))

                    # Update layout for dark theme
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(visible=False, showgrid=False, showticklabels=False),
                            yaxis=dict(visible=False, showgrid=False, showticklabels=False),
                            zaxis=dict(visible=False, showgrid=False, showticklabels=False),
                            bgcolor='rgba(30, 30, 30, 1)',
                        ),
                        paper_bgcolor='rgba(30, 30, 30, 1)',
                        plot_bgcolor='rgba(30, 30, 30, 1)',
                        margin=dict(l=0, r=0, b=0, t=0),
                        showlegend=False,
                        height=700,
                    )

                    # Add camera controls info
                    st.caption("🖱️ Drag to rotate • Scroll to zoom • Right-click to pan")

                    # Render 3D plot
                    st.plotly_chart(fig, use_container_width=True)

                    # Legend
                    st.subheader("Legend")
                    cols = st.columns(len(role_colors))
                    for col, (role, color) in zip(cols, role_colors.items()):
                        with col:
                            st.markdown(
                                f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; color: white;">{role.replace("_", " ").title()}</div>',
                                unsafe_allow_html=True,
                            )
        else:
            st.info("No nodes in database yet. Ingest some text first!")

    with tab6:
        st.subheader("🎨 Animated Canvas (vis.js)")
        st.info("Canvas animation coming soon - for now use the Propagation Query tab with Compact/Detailed views")


if __name__ == "__main__":
    main()
