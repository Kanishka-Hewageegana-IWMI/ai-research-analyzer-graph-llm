import streamlit as st
import google.generativeai as genai
import pandas as pd
import networkx as nx
from pyvis.network import Network
import json
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in .env file. Please set it and try again.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# Load the dataset
data_path = "data/Updated_Cleaned_Dataset.csv"
data = pd.read_csv(data_path)

def llm_parse_query(user_query):
    prompt = f"""
    You are an expert in parsing research queries.
    Dataset columns:
    ['Main Author','Corresponding Authors','title','geography_focus',
     'CGIAR Main Center','keywords','sdgs','impact_area','url']
    Extract filters from: "{user_query}".
    Return JSON with keys:
    - author_name
    - search_both_author_fields
    - cgiar_center
    - geography
    - title_keywords
    - keywords_filter
    - include_keywords
    - include_collaborators
    - include_related_authors
    - show_cgiar_centers
    - selected_legends (list of node types to display: Author, Co-Author, Publication, Geography, CGIAR Center, Keyword)
    ONLY return JSON, nothing else.
    """
    response = model.generate_content(prompt)
    cleaned = response.text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()
    return json.loads(cleaned)

def normalize_filters(filters):
    for k, v in filters.items():
        if isinstance(v, list) and k != "selected_legends":
            if len(v) == 1:
                filters[k] = v[0]
            elif len(v) > 1:
                filters[k] = "|".join(v)
            else:
                filters[k] = None
    return filters

def apply_filters(df, filters):
    if filters.get("author_name"):
        author_name = filters["author_name"]
        if filters.get("search_both_author_fields", False):
            main_match = df["Main Author"].str.contains(author_name, na=False, case=False)
            corr_match = df["Corresponding Authors"].str.contains(author_name, na=False, case=False)
            df = df[main_match | corr_match]
        else:
            df = df[df["Main Author"].str.contains(author_name, na=False, case=False)]
    if filters.get("cgiar_center"):
        df = df[df["CGIAR Main Center"].str.contains(filters["cgiar_center"], na=False, case=False)]
    if filters.get("geography"):
        df = df[df["geography_focus"].str.contains(filters["geography"], na=False, case=False)]
    if filters.get("title_keywords"):
        df = df[df["title"].str.contains(filters["title_keywords"], na=False, case=False)]
    if filters.get("keywords_filter"):
        df = df[df["keywords"].str.contains(filters["keywords_filter"], na=False, case=False)]
    return df

def build_graph_html(df, selected_legends):
    if df.empty:
        return "No matching results."

    G = nx.Graph()

    # Define node type properties
    node_types = {
        "Author": {"color": "lightblue", "title_prefix": "Author: "},
        "Co-Author": {"color": "yellow", "title_prefix": "Co-Author: "},
        "Publication": {"color": "lightgreen", "title_prefix": "Publication: "},
        "Geography": {"color": "orange", "title_prefix": "Geography: "},
        "CGIAR Center": {"color": "red", "title_prefix": "CGIAR Center: "},
        "Keyword": {"color": "purple", "title_prefix": "Keyword: "}
    }

    publication_centers = {}

    for _, row in df.iterrows():
        pub_title = row["title"]

        if "Publication" in selected_legends:
            G.add_node(pub_title, node_type="Publication", color="lightgreen", title=f"Publication: {pub_title}")

        # Main author
        if "Author" in selected_legends and pd.notna(row["Main Author"]):
            main_author = row["Main Author"]
            G.add_node(main_author, node_type="Author", color="lightblue", title=f"Author: {main_author}")
            if "Publication" in selected_legends:
                G.add_edge(main_author, pub_title)

        # Co-authors
        if "Co-Author" in selected_legends and pd.notna(row["Corresponding Authors"]) and row["Corresponding Authors"] != "No Data":
            for coauthor in [c.strip() for c in str(row["Corresponding Authors"]).split(";") if c.strip()]:
                G.add_node(coauthor, node_type="Co-Author", color="yellow", title=f"Co-Author: {coauthor}")
                if "Publication" in selected_legends:
                    G.add_edge(coauthor, pub_title)

        # Geography
        if "Geography" in selected_legends and pd.notna(row["geography_focus"]) and row["geography_focus"] != "No Data":
            geography = row["geography_focus"]
            G.add_node(geography, node_type="Geography", color="orange", title=f"Geography: {geography}")
            if "Publication" in selected_legends:
                G.add_edge(pub_title, geography)

        # Keywords
        if "Keyword" in selected_legends and pd.notna(row["keywords"]) and row["keywords"] != "No Data":
            for keyword in [k.strip() for k in str(row["keywords"]).split(";") if k.strip()]:
                G.add_node(keyword, node_type="Keyword", color="purple", title=f"Keyword: {keyword}")
                if "Publication" in selected_legends:
                    G.add_edge(pub_title, keyword)

        # Centers collection
        if "CGIAR Center" in selected_legends:
            if pub_title not in publication_centers:
                publication_centers[pub_title] = set()
            if pd.notna(row["CGIAR Main Center"]) and row["CGIAR Main Center"] != "No Data":
                publication_centers[pub_title].add(row["CGIAR Main Center"])

    # Connect centers
    if "CGIAR Center" in selected_legends:
        for pub_title, centers in publication_centers.items():
            for center in centers:
                color = "darkred" if len(centers) > 1 else "red"
                G.add_node(center, node_type="CGIAR Center", color=color, title=f"CGIAR Center: {center}")
                if "Publication" in selected_legends:
                    G.add_edge(pub_title, center)

    # PyVis Graph
    net = Network(height="650px", width="100%", bgcolor="#222222", font_color="white", cdn_resources="in_line")
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data["node_type"] in selected_legends:
            net.add_node(node, label=str(node), color=node_data.get('color', 'lightblue'),
                         title=node_data.get('title', str(node)))
    for edge in G.edges():
        if G.nodes[edge[0]]["node_type"] in selected_legends and G.nodes[edge[1]]["node_type"] in selected_legends:
            net.add_edge(edge[0], edge[1])
    net.repulsion(node_distance=180, spring_length=180, damping=0.95)

    # Save to temp file
    html_file = "graph_llm.html"
    net.save_graph(html_file)

    # Dynamic legend based on selected node types
    legend_items = [
        f'<span style="color:{props["color"]};">■</span> {node_type}<br>'
        for node_type, props in node_types.items() if node_type in selected_legends
    ]
    legend_html = f"""
    <div style="
        position: fixed;
        top: 20px; right: 20px;
        background: rgba(0,0,0,0.7);
        padding: 10px;
        border-radius: 8px;
        color: white; font-size: 14px;">
      <b>Legend</b><br>
      {"".join(legend_items)}
    </div>
    """
    with open(html_file, "r") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", legend_html + "</body>")
    with open(html_file, "w") as f:
        f.write(html_content)

    return html_content

# Streamlit UI
st.title("AI Research Dashboard - Graph LLM Analyzer")

user_query = st.text_input("Enter the prompt:")

process = st.button("Process Query")

legend_options = ["Author", "Co-Author", "Publication", "Geography", "CGIAR Center", "Keyword"]

if 'selected_legends' not in st.session_state:
    st.session_state.selected_legends = legend_options.copy()

st.subheader("Select Node Types to Display")
selected_legends = st.multiselect(
    "Choose legends:",
    options=legend_options,
    default=st.session_state.selected_legends
)
st.session_state.selected_legends = selected_legends

if process:
    with st.spinner("Processing query..."):
        filters = llm_parse_query(user_query)
        filters = normalize_filters(filters)
        df = apply_filters(data.copy(), filters)
        if df.empty:
            st.session_state.msg = "No matching results."
            st.session_state.df_display = None
        else:
            df_display = df[['Main Author', 'title', 'geography_focus', 'CGIAR Main Center', 'keywords', 'url']].copy()
            df_display['title'] = df_display['title'].str[:60] + '...'
            df_display['keywords'] = df_display['keywords'].astype(str).str[:50] + '...'
            df_display['url'] = df_display['url'].astype(str).str[:50] + '...'
            st.session_state.df_display = df_display
            st.session_state.df = df
            st.session_state.msg = None
        if filters.get("selected_legends"):
            st.session_state.selected_legends = filters["selected_legends"]

# Display results
if 'msg' in st.session_state:
    if st.session_state.msg:
        st.text_area("Output:", value=st.session_state.msg, height=100)
    elif 'df_display' in st.session_state and st.session_state.df_display is not None:
        st.subheader("Output DataFrame")
        st.dataframe(st.session_state.df_display)

        with st.spinner("Building graph..."):
            html_content = build_graph_html(st.session_state.df, st.session_state.selected_legends)

        st.subheader("Network Graph")
        st.components.v1.html(html_content, height=700, scrolling=True)