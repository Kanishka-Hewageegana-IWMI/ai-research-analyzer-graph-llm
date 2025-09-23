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
data_path = "data/Cleaned Dataset - Gold_Data.csv"
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
    ONLY return JSON, nothing else.
    """
    response = model.generate_content(prompt)
    cleaned = response.text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()
    return json.loads(cleaned)


def normalize_filters(filters):
    for k, v in filters.items():
        if isinstance(v, list):
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


def build_graph_and_html(filters):
    G = nx.Graph()
    df = apply_filters(data.copy(), filters)
    if df.empty:
        return None, None, "No matching results."

    df_display = df[['Main Author', 'title', 'geography_focus', 'CGIAR Main Center', 'keywords', 'url']].copy()
    df_display['title'] = df_display['title'].str[:60] + '...'
    df_display['keywords'] = df_display['keywords'].astype(str).str[:50] + '...'
    df_display['url'] = df_display['url'].astype(str).str[:50] + '...'

    # Build author + publication + co-author links
    publication_centers = {}
    for _, row in df.iterrows():
        pub_title = row["title"]
        # Add publication node
        if pub_title not in publication_centers:
            publication_centers[pub_title] = set()
            G.add_node(pub_title, node_type="Publication", color="lightgreen", title=f"Publication: {pub_title}")
        # Main author
        main_author = row["Main Author"]
        if pd.notna(main_author):
            G.add_node(main_author, node_type="Author", color="lightblue", title=f"Author: {main_author}")
            G.add_edge(main_author, pub_title)
        # Co-authors
        if pd.notna(row["Corresponding Authors"]) and row["Corresponding Authors"] != "No Data":
            for coauthor in [c.strip() for c in str(row["Corresponding Authors"]).split(";") if c.strip()]:
                G.add_node(coauthor, node_type="Co-Author", color="yellow", title=f"Co-Author: {coauthor}")
                G.add_edge(coauthor, pub_title)
        # Centers
        if pd.notna(row["CGIAR Main Center"]) and row["CGIAR Main Center"] != "No Data":
            publication_centers[pub_title].add(row["CGIAR Main Center"])
        # Geography
        if pd.notna(row["geography_focus"]) and row["geography_focus"] != "No Data":
            geography = row["geography_focus"]
            G.add_node(geography, node_type="Geography", color="orange", title=f"Geography: {geography}")
            G.add_edge(pub_title, geography)

    # Connect centers + multiple center co-links
    for pub_title, centers in publication_centers.items():
        for center in centers:
            color = "darkred" if len(centers) > 1 else "red"
            G.add_node(center, node_type="CGIAR Center", color=color, title=f"CGIAR Center: {center}")
            G.add_edge(pub_title, center)

    # PyVis Graph
    net = Network(height="650px", width="100%", bgcolor="#222222", font_color="white", cdn_resources="in_line")
    for node in G.nodes():
        node_data = G.nodes[node]
        net.add_node(node, label=str(node), color=node_data.get('color', 'lightblue'),
                     title=node_data.get('title', str(node)))
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    net.repulsion(node_distance=180, spring_length=180, damping=0.95)

    # Save to temp file
    html_file = "graph_llm.html"
    net.save_graph(html_file)

    # Add legend
    legend_html = """
    <div style="
        position: fixed;
        top: 20px; right: 20px;
        background: rgba(0,0,0,0.7);
        padding: 10px;
        border-radius: 8px;
        color: white; font-size: 14px;">
      <b>Legend</b><br>
      <span style="color:lightblue;">■</span> Author<br>
      <span style="color:yellow;">■</span> Co-Author<br>
      <span style="color:lightgreen;">■</span> Publication<br>
      <span style="color:orange;">■</span> Geography<br>
      <span style="color:red;">■</span> CGIAR Center<br>
      <span style="color:purple;">■</span> Keyword
    </div>
    """
    with open(html_file, "r") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", legend_html + "</body>")
    with open(html_file, "w") as f:
        f.write(html_content)

    return G, df_display, html_content


# Streamlit UI
st.title("AI Research Dashboard - Graph LLM Analyzer")

user_query = st.text_input("Enter the prompt:")

if user_query:
    with st.spinner("Processing query..."):
        filters = llm_parse_query(user_query)
        filters = normalize_filters(filters)
        G, df_display, html_content_or_msg = build_graph_and_html(filters)

    if html_content_or_msg == "No matching results.":
        st.text_area("Output:", value=html_content_or_msg, height=100)
    else:
        st.subheader("Output DataFrame")
        st.dataframe(df_display)

        st.subheader("Network Graph")
        st.components.v1.html(html_content_or_msg, height=700, scrolling=True)
