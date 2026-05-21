import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Player Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("data/player_clusters.csv")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("⚽ Player Performance Dashboard")

# ---------------------------------------------------
# QUICK STATS
# ---------------------------------------------------

st.subheader("Quick Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Players", len(df))

with col2:
    st.metric("Clusters", df["cluster_name"].nunique())

with col3:
    elite_count = (df["cluster_name"] == "Elite Stars").sum()
    st.metric("Elite Stars", elite_count)

st.divider()

# ---------------------------------------------------
# PLAYER SEARCH
# ---------------------------------------------------

st.subheader("🔍 Compare Players")

col_search1, col_search2 = st.columns(2)

with col_search1:
    player_name_1 = st.text_input(
        "Player 1",
        placeholder="e.g. Messi"
    )

with col_search2:
    player_name_2 = st.text_input(
        "Player 2",
        placeholder="e.g. Ronaldo"
    )

# ---------------------------------------------------
# CLUSTER FILTER
# ---------------------------------------------------

st.subheader("Select Clusters")

clusters = st.multiselect(
    "Choose Player Archetypes",
    options=df["cluster_name"].unique(),
    default=df["cluster_name"].unique()
)

filtered_df = df[
    df["cluster_name"].astype(str).isin(
        [str(c) for c in clusters]
    )
]

st.divider()

# ---------------------------------------------------
# PLAYER VARIABLES
# ---------------------------------------------------

player_1 = None
player_2 = None

# ---------------------------------------------------
# PLAYER 1 SEARCH
# ---------------------------------------------------

if player_name_1:

    results_1 = filtered_df[
        filtered_df["name"].str.contains(
            player_name_1,
            case=False,
            na=False
        )
    ]

    if not results_1.empty:
        player_1 = results_1.iloc[0]

# ---------------------------------------------------
# PLAYER 2 SEARCH
# ---------------------------------------------------

if player_name_2:

    results_2 = filtered_df[
        filtered_df["name"].str.contains(
            player_name_2,
            case=False,
            na=False
        )
    ]

    if not results_2.empty:
        player_2 = results_2.iloc[0]

# ---------------------------------------------------
# PLAYER CARDS
# ---------------------------------------------------

if player_1 is not None:

    colA, colB = st.columns(2)

    # ===============================================
    # PLAYER 1 CARD
    # ===============================================

    with colA:

        st.subheader(player_1["name"])

        try:
            st.image(player_1["image_url"], width=220)
        except:
            st.warning("Image unavailable")

        st.markdown(
            f"""
            ### {player_1['cluster_name']}

            **Position:** {player_1['position']}  
            **Sub Position:** {player_1['sub_position']}  
            **Nationality:** {player_1['country_of_citizenship']}  

            **Market Value:** €{player_1['market_value_in_eur']:,.0f}  
            **Highest Market Value:** €{player_1['highest_market_value_in_eur']:,.0f}  

            **Goals / 90:** {player_1['goals_per_90']:.2f}  
            **Assists / 90:** {player_1['assists_per_90']:.2f}  
            **Cards / 90:** {player_1['cards_per_90']:.2f}  

            **Matches Played:** {player_1['matches_played']}  
            **International Caps:** {player_1['international_caps']}  
            """
        )

    # ===============================================
    # PLAYER 2 CARD
    # ===============================================

    if player_2 is not None:

        with colB:

            st.subheader(player_2["name"])

            try:
                st.image(player_2["image_url"], width=220)
            except:
                st.warning("Image unavailable")

            st.markdown(
                f"""
                ### {player_2['cluster_name']}

                **Position:** {player_2['position']}  
                **Sub Position:** {player_2['sub_position']}  
                **Nationality:** {player_2['country_of_citizenship']}  

                **Market Value:** €{player_2['market_value_in_eur']:,.0f}  
                **Highest Market Value:** €{player_2['highest_market_value_in_eur']:,.0f}  

                **Goals / 90:** {player_2['goals_per_90']:.2f}  
                **Assists / 90:** {player_2['assists_per_90']:.2f}  
                **Cards / 90:** {player_2['cards_per_90']:.2f}  

                **Matches Played:** {player_2['matches_played']}  
                **International Caps:** {player_2['international_caps']}  
                """
            )

# ---------------------------------------------------
# RADAR CHART
# ---------------------------------------------------

if player_1 is not None:

    st.divider()

    st.subheader("⚔️ Player Comparison Radar")

    radar_metrics_1 = {
        "Goals/90": player_1["goals_per_90"],
        "Assists/90": player_1["assists_per_90"],
        "Cards/90": player_1["cards_per_90"],
        "Minutes/Match": player_1["minutes_per_match"] / 100,
        "International Caps": player_1["international_caps"] / 200
    }

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=list(radar_metrics_1.values()),
        theta=list(radar_metrics_1.keys()),
        fill='toself',
        name=player_1["name"]
    ))

    # ===============================================
    # PLAYER 2 RADAR
    # ===============================================

    if player_2 is not None:

        radar_metrics_2 = {
            "Goals/90": player_2["goals_per_90"],
            "Assists/90": player_2["assists_per_90"],
            "Cards/90": player_2["cards_per_90"],
            "Minutes/Match": player_2["minutes_per_match"] / 100,
            "International Caps": player_2["international_caps"] / 200
        }

        fig_radar.add_trace(go.Scatterpolar(
            r=list(radar_metrics_2.values()),
            theta=list(radar_metrics_2.keys()),
            fill='toself',
            name=player_2["name"]
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        template="plotly_dark",
        height=650,
        margin=dict(l=100, r=100, t=50, b=50)
    )

    st.plotly_chart(
        fig_radar,
        use_container_width=True
    )

# ---------------------------------------------------
# KPI METRICS
# ---------------------------------------------------

if player_1 is not None:

    st.divider()

    st.subheader("Player Metrics")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Goals / 90",
            f"{player_1['goals_per_90']:.2f}"
        )

    with metric2:
        st.metric(
            "Assists / 90",
            f"{player_1['assists_per_90']:.2f}"
        )

    with metric3:
        st.metric(
            "Minutes / Match",
            f"{player_1['minutes_per_match']:.1f}"
        )

    with metric4:
        st.metric(
            "International Caps",
            int(player_1['international_caps'])
        )

# ---------------------------------------------------
# PCA PLOT
# ---------------------------------------------------

st.divider()

st.subheader("Football Player Archetypes")

fig = px.scatter(
    filtered_df,
    x="pca_1",
    y="pca_2",
    color="cluster_name",
    hover_data=[
        "name",
        "position",
        "country_of_citizenship",
        "market_value_in_eur",
        "goals_per_90",
        "assists_per_90"
    ],
    title="Player Clusters",
    height=700
)

fig.update_layout(
    template="plotly_dark",
    legend_title="Player Archetype"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# DATASET TABLE
# ---------------------------------------------------

with st.expander("View Dataset"):

    st.dataframe(
        filtered_df[[
            "name",
            "position",
            "country_of_citizenship",
            "cluster_name",
            "market_value_in_eur",
            "goals_per_90",
            "assists_per_90"
        ]],
        use_container_width=True
    )