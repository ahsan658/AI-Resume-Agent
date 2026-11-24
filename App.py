import os
import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task


def get_groq_api_key() -> str:
    # Check Streamlit Secrets first if deployed on Streamlit, then environment variables
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def build_llm(api_key: str) -> LLM:
    os.environ["GROQ_API_KEY"] = api_key
    return LLM(
        model="groq/openai/gpt-oss-120b",
        temperature=0.3,
    )


def run_analysis(topic: str, api_key: str) -> str:
    llm = build_llm(api_key)

    analyst = Agent(
        role="Data & Systems Analyst",
        goal="Provide clear, accurate, and actionable analysis on the requested topic.",
        backstory="You are a concise technical analyst who structures findings clearly.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=(
            f"Analyze the following topic thoroughly and concisely: {topic}\n"
            "Include key takeaways, practical implications, and any caveats."
        ),
        expected_output="A well-structured analysis with bullet points and clear headings.",
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result)


def main() -> None:
    st.set_page_config(page_title="CrewAI Groq Analysis", layout="centered")
    st.title("Analysis Runner")

    api_key = get_groq_api_key()
    if not api_key:
        st.error("Missing `GROQ_API_KEY`. Add it to your platform secrets or environment variables.")
        st.stop()

    topic = st.text_area("Enter topic or data to analyze", height=120)

    if st.button("Run Analysis", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic before running analysis.")
            st.stop()

        with st.spinner("Running analysis..."):
            try:
                output = run_analysis(topic.strip(), api_key)
                st.markdown(output)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()
