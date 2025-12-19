import os
import yaml
import streamlit as st
import requests

CONFIG_PATH_DEFAULT = "configs/config.yaml"

def load_config():
    path = os.getenv("OUTCOME_WRITER_CONFIG", CONFIG_PATH_DEFAULT)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    st.set_page_config(page_title="Outcome Writer", layout="wide")
    st.title("Outcome Writer — Outcome-Conditioned Sales Email Drafts")

    with st.sidebar:
        st.header("Scenario")
        industry = st.selectbox("Industry", ["SaaS","Healthcare","FinTech","Manufacturing","Retail","Defense","Education","Real Estate"])
        persona = st.selectbox("Persona", ["VP of Sales","Head of Ops","CFO","CTO","Program Manager","Procurement Lead","Marketing Director"])
        product = st.text_input("Product", "SecureRAG")
        value_prop = st.text_input("Value prop", "private document Q&A with citations")
        tone = st.selectbox("Tone", ["consultative","executive","direct","friendly","warm"])
        goal = st.text_input("Goal", "book a 15-min intro call")
        company = st.text_input("Prospect company (optional)", "")
        constraints = st.text_area("Constraints (optional)", "Keep it under 140 words. Include one CTA.")
        include_examples = st.checkbox("Show retrieved examples", False)

        st.divider()
        api_base = st.text_input("API base URL", f"http://{cfg['server']['host']}:{cfg['server']['port']}")

    if st.button("Generate"):
        payload = {
            "industry": industry,
            "persona": persona,
            "product": product,
            "value_prop": value_prop,
            "tone": tone,
            "goal": goal,
            "company_name": company or None,
            "constraints": constraints or None,
            "include_examples": include_examples,
        }
        try:
            r = requests.post(f"{api_base}/v1/generate_email", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()

            st.subheader("Subject")
            st.code(data["subject"])

            st.subheader("Body")
            st.text_area("Draft", data["body"], height=220)

            cols = st.columns(3)
            cols[0].metric("Retrieved examples", data.get("retrieved_count", 0))
            score = data.get("success_likelihood")
            cols[1].metric("Success likelihood (toy)", f"{score:.3f}" if score is not None else "n/a")
            cols[2].info("Notes:\n" + "\n".join(data.get("notes", [])))

            if include_examples and data.get("retrieved_examples"):
                st.divider()
                st.subheader("Retrieved successful examples")
                for ex in data["retrieved_examples"]:
                    st.markdown(f"**Similarity:** {ex.get('similarity',0):.3f} • **Subject:** {ex.get('subject','')}")
                    st.caption(f"{ex.get('industry','')} • {ex.get('persona','')} • {ex.get('product','')} • {ex.get('tone','')}")
                    st.text(ex.get("body","")[:1200])
                    st.divider()
        except Exception as e:
            st.error(f"Request failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
