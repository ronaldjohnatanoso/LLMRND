"""Streamlit UI for browsing CogMemory database.

Run with: streamlit run browser.py

Features:
- View all nodes in searchable table
- Semantic search
- Filter by role
- Ingest new text
- Statistics dashboard
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
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
        st.caption(f"Database: {memory.store.db_path}")

    # Main content
    tab1, tab2, tab3 = st.tabs(["📋 All Nodes", "🔍 Semantic Search", "📖 Details"])

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
            ]].copy()

            # Format columns
            display_df["role"] = display_df["role"].str.upper().str.replace("_", " ")
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{'★' * int(x * 5)}")
            display_df["activation"] = display_df["activation"].apply(lambda x: f"{x:.3f}")

            display_df.columns = ["Text", "Role", "Confidence", "Activation"]
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
            )
        else:
            st.info("No nodes match the filters")

    with tab2:
        st.subheader("Semantic Search")

        query = st.text_input(
            "Search query:",
            placeholder="e.g., 'What are the goals?' or 'budget constraints'",
        )

        k = st.slider("Results", 1, 20, 5)

        if query:
            with st.spinner("Searching..."):
                results = memory.store.query_similar(
                    memory.embedding_manager.generate_embedding(query),
                    k=k,
                )

            if results:
                for i, r in enumerate(results, 1):
                    with st.expander(
                        f"{i}. [{r['role'].upper()}] {r['text'][:80]}...",
                        expanded=i == 1,
                    ):
                        st.write(f"**Text:** {r['text']}")
                        st.write(f"**Role:** {r['role'].replace('_', ' ').title()}")
                        st.write(f"**Confidence:** {r['confidence']:.2f}")
                        st.write(f"**Similarity:** {r['similarity']:.3f}")
                        st.write(f"**Activation:** {r['activation']:.3f}")
            else:
                st.info("No results found")
        else:
            st.info("Enter a search query above")

    with tab3:
        st.subheader("Node Details")

        # Node selector
        if not df.empty:
            node_options = df["text"].str[:50] + "..."
            selected = st.selectbox("Select node", node_options)

            if selected:
                idx = node_options[node_options == selected].index[0]
                node = df.iloc[idx]

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Role:**", node["role"].replace("_", " ").title())
                    st.write("**Confidence:**", f"{node['confidence']:.2f}")
                    st.write("**Activation:**", f"{node['activation']:.3f}")

                with col2:
                    st.write("**ID:**", node["id"])
                    st.json(node.get("metadata", {}))

                st.divider()
                st.subheader("Text")
                st.write(node["text"])

                if node.get("neighbors"):
                    st.subheader("Neighbors")
                    st.json(node["neighbors"])

                st.subheader("Vector (first 10 dims)")
                vector = node.get("vector", [])
                if vector:
                    st.write(str(vector[:10]) + "...")


if __name__ == "__main__":
    main()
