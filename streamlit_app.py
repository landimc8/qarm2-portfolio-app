import streamlit as st

st.set_page_config(page_title="Feedstock Hedging Optimizer", layout="wide")

st.title("Biodiesel Feedstock Hedging Portfolio")
st.caption("QARM II — HEC Lausanne")

st.info("This app is under construction. Full version coming soon.")

st.header("About the project")
st.write(
    "This app will help a biodiesel producer manage price risk on its feedstocks "
    "(UCO, tallow, soybean oil, vegetable oils) using portfolio optimization "
    "and risk budgeting."
)

tab1, tab2, tab3 = st.tabs(["Data", "Optimization", "Results"])
with tab1:
    st.write("Feedstock price data will appear here.")
with tab2:
    st.write("Hedging strategy and constraints will be set here.")
with tab3:
    st.write("Optimal hedge weights and risk metrics will be shown here.")

st.divider()
st.caption("Team: [add names here]")
