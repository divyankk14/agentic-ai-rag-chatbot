import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖")

st.title("🤖 Agentic AI eBook Chatbot")
st.caption("Ask questions strictly grounded in the Agentic AI eBook (Konverge AI)")

# Text input for the question
question = st.text_input("Ask a question:", placeholder="e.g. What is the role of memory in an agentic AI system?")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"question": question})
            response.raise_for_status()  # raises an error if the API returned a non200 status
            data = response.json()

            #  Show the answer 
            st.subheader("Answer")
            st.write(data["answer"])

            #  Show confidence score 
            st.metric("Confidence Score", f"{data['confidence']:.2f}")

            # show retrieved chunks in an expandable section
            with st.expander("📄 View Retrieved Source Chunks"):
                for i, source in enumerate(data["sources"]):
                    st.markdown(f"**{i+1}. Page {source['page']} — {source['heading']}** (score: {source['score']:.2f})")
                    st.text(source["text"][:300] + "...")
                    st.divider()

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
