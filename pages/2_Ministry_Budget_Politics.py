from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "Neo4j GDS" / "stock_return_baseline"

MINISTRY_LABEL_EN = {
    "กระทรวงการคลัง": "Finance",
    "กระทรวงพลังงาน": "Energy",
    "กระทรวงคมนาคม": "Transport",
    "กระทรวงดิจิทัลเพื่อเศรษฐกิจและสังคม": "Digital Economy",
    "กระทรวงสาธารณสุข": "Public Health",
    "กระทรวงพาณิชย์": "Commerce",
    "กระทรวงเกษตรและสหกรณ์": "Agriculture",
    "กระทรวงมหาดไทย": "Interior",
    "กระทรวงอุตสาหกรรม": "Industry",
    "กระทรวงการท่องเที่ยวและกีฬา": "Tourism & Sports",
}

PARTY_ROLE_LABEL = {
    "core": "Core ruling party",
    "coalition": "Coalition partner",
    "independent": "Independent / technocrat",
    "none": "2014-2019 NCPO (no party)",
}


@st.cache_data
def load_data():
    cabinet = pd.read_csv(DATA_DIR / "cabinet_history.csv")
    budget = pd.read_csv(DATA_DIR / "budget_by_ministry.csv")
    correlation = pd.read_csv(DATA_DIR / "correlation_comparison.csv")
    party_summary = pd.read_csv(DATA_DIR / "party_effect_summary.csv")
    model_comparison = pd.read_csv(DATA_DIR / "model_comparison.csv")
    minister_net = pd.read_csv(DATA_DIR / "minister_network_centrality.csv")

    budget["ministry_en"] = budget["min_name"].map(MINISTRY_LABEL_EN)
    cabinet["ministry_en"] = cabinet["ministry"].map(MINISTRY_LABEL_EN)
    return cabinet, budget, correlation, party_summary, model_comparison, minister_net


def main() -> None:
    st.set_page_config(page_title="Ministry Budget & Politics", layout="wide")
    st.title("Ministry Budget, Cabinet Politics & SET50 Returns")
    st.caption(
        "Does government budget allocation, or which party runs a ministry, show up in the "
        "stocks tied to that ministry? Built from real Thai cabinet history (2014-present, "
        "sourced verbatim from Wikipedia) and real budget data (FY2559-2569, re-fetched from "
        "the Thailand Government Spending open-data API after the original endpoint was retired)."
    )

    if not (DATA_DIR / "cabinet_history.csv").exists():
        st.error(f"Data not found under {DATA_DIR} - run the pipeline scripts in that folder first.")
        return

    cabinet, budget, correlation, party_summary, model_comparison, minister_net = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cabinets covered", cabinet["cabinet_no"].nunique())
    col2.metric("Ministers tracked", cabinet["minister"].nunique())
    col3.metric("Fiscal years of budget data", budget["year"].nunique())
    col4.metric("Ministries", 10)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Cabinet & Party History",
        "Ministry Budget Over Time",
        "Ruling Party vs Returns",
        "Minister Career Network",
        "Prediction Model Results",
    ])

    with tab1:
        st.subheader("Who ran which ministry, and which party they belonged to")
        ministries = sorted(cabinet["ministry_en"].dropna().unique())
        selected = st.multiselect("Filter by ministry", ministries, default=ministries)
        filtered = cabinet[cabinet["ministry_en"].isin(selected)].copy()
        filtered["party_role_label"] = filtered["party_role"].map(PARTY_ROLE_LABEL)
        st.dataframe(
            filtered[["cabinet_no", "pm", "ministry_en", "minister", "party", "party_role_label", "start_date", "end_date"]]
            .rename(columns={"ministry_en": "ministry", "party_role_label": "party_role"})
            .sort_values(["cabinet_no", "ministry"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "`party_role = none` (Cabinets 61, 2014-2019) means the NCPO military government - "
            "no elected party was involved at all, not \"zero ruling-party effect.\""
        )

    with tab2:
        st.subheader("Ministry budget allocation, FY2559-2569")
        budget_ministries = sorted(budget["ministry_en"].unique())
        selected_b = st.multiselect("Ministries to plot", budget_ministries, default=budget_ministries[:5])
        plot_df = budget[budget["ministry_en"].isin(selected_b)]
        fig = px.line(
            plot_df, x="year", y="total_budget_million_baht", color="ministry_en",
            labels={"year": "Fiscal year (B.E.)", "total_budget_million_baht": "Budget (million baht)", "ministry_en": "Ministry"},
            markers=True,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            budget[["year", "ministry_en", "total_budget_million_baht", "total_disbursed_million_baht"]]
            .rename(columns={"ministry_en": "ministry"})
            .sort_values(["ministry", "year"]),
            use_container_width=True, hide_index=True,
        )

    with tab3:
        st.subheader("Mean stock return by which party ran the ministry")
        fig2 = go.Figure(go.Bar(
            x=[PARTY_ROLE_LABEL.get(r, r) for r in party_summary["party_role"]],
            y=party_summary["mean"],
            error_y=dict(type="data", array=party_summary["std"] / party_summary["count"] ** 0.5),
            text=[f"n={n}" for n in party_summary["count"]],
        ))
        fig2.update_layout(yaxis_title="Mean annual return (%)", height=450)
        st.plotly_chart(fig2, use_container_width=True)

        anova_p = party_summary["anova_p"].iloc[0]
        st.metric("One-way ANOVA p-value (across all 4 groups)", f"{anova_p:.4f}",
                   help="Below 0.05 = conventionally significant. This one is borderline.")
        st.dataframe(party_summary.drop(columns=["anova_f", "anova_p"]), use_container_width=True, hide_index=True)

        st.subheader("Ministry budget YoY% correlation with stock return: old summary vs recomputed (real 11-year data)")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Old (aggregate summary)", x=correlation["label_en"], y=correlation["old_simple_r"]))
        fig3.add_trace(go.Bar(name="Recomputed (real data)", x=correlation["label_en"], y=correlation["new_simple_r_full_data"]))
        fig3.update_layout(barmode="group", yaxis_title="Correlation (r)", height=450)
        st.plotly_chart(fig3, use_container_width=True)
        st.warning(
            "Finance (0.29→0.09) and Commerce (-0.60→-0.19) differ materially between the old "
            "hardcoded summary and the recomputed value from the real budget series. The budget "
            "YoY% figures themselves were spot-checked as an exact match to pre-deletion data, so "
            "this isn't a bug here - the discrepancy likely comes from the stock-return data "
            "vintage or exact methodology behind the original number, which can't be reconstructed."
        )

    with tab4:
        st.subheader("Minister career-movement network (2014-2026)")
        png_path = DATA_DIR / "minister_network.png"
        if png_path.exists():
            st.image(str(png_path), use_container_width=True)
        st.caption(
            "Tripartite Minister-Ministry-Stock graph. Node size = betweenness centrality - "
            "ministers who held multiple ministries act as structural bridges, regardless of how "
            "long they served."
        )
        top_ministers = (
            minister_net[minister_net["node_type"] == "minister"]
            .sort_values("betweenness", ascending=False)
            .head(15)
        )
        st.dataframe(
            top_ministers[["node", "betweenness", "weighted_degree", "pagerank"]]
            .rename(columns={"node": "minister"}),
            use_container_width=True, hide_index=True,
        )

    with tab5:
        st.subheader("Does any of this help predict next year's stock return?")
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)
        fig4 = px.bar(model_comparison, x="model", y="R2", labels={"R2": "R² (test years 2024-2026)"})
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        fig4.update_layout(height=450, xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)
        st.info(
            "Naive baselines score *negative* R² (worse than predicting everyone's average "
            "return). Centrality features push R² positive. Adding party_role or real budget "
            "YoY% doesn't improve held-out R² further, but both rank in the top few features by "
            "importance - real signal exists, but 443 rows isn't enough to convert it into a "
            "reliably better forecast."
        )

    st.divider()
    st.caption(
        "Full pipeline and methodology notes: "
        "`Neo4j GDS/stock_return_baseline/README.md`."
    )


if __name__ == "__main__":
    main()
