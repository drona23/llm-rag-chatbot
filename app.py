"""
Gradio web UI for the Student Loan RAG Chatbot.
Layout: chat on left, live sources + confidence on right.
"""
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Global agent -- initialized once at startup.
# gr.State can't deepcopy objects that hold live SSL sockets (Pinecone/Voyage),
# so we keep the agent at module level and close over it in callbacks.
_agent = None


def build_agent():
    from src.llm import LLMChat
    from src.vector_db import VectorStore
    from src.rag_agent import RAGAgent

    print("Initializing RAG agent...")
    llm = LLMChat()
    store = VectorStore()
    return RAGAgent(vector_store=store, llm=llm)


def format_sources(sources: list[str], scores: list[float]) -> str:
    if not sources:
        return "*No sources retrieved.*"
    parts = []
    for i, (text, score) in enumerate(zip(sources, scores), 1):
        preview = text[:400] + "..." if len(text) > 400 else text
        parts.append(f"### Source {i} &nbsp; `score: {score:.3f}`\n\n{preview}")
    return "\n\n---\n\n".join(parts)


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return f"Confidence: HIGH ({score:.2f})"
    elif score >= 0.50:
        return f"Confidence: MEDIUM ({score:.2f})"
    else:
        return f"Confidence: LOW ({score:.2f})"


def chat(message, history):
    if not message.strip():
        return history, "*Ask a question to see retrieved sources here.*"

    result = _agent.answer(message)

    sources_md = format_sources(result["sources"], result["retrieval_scores"])
    confidence_md = f"**{confidence_label(result['confidence'])}**"
    sidebar_md = f"{confidence_md}\n\n---\n\n{sources_md}"

    # Gradio 6 uses messages format: list of {"role": ..., "content": ...}
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["response"]},
    ]
    return history, sidebar_md


def clear():
    _agent.reset_conversation()
    return [], "*Ask a question to see retrieved sources here.*"


def build_ui():
    with gr.Blocks(title="Student Loan RAG Chatbot") as demo:
        gr.Markdown("# Student Loan RAG Chatbot\nPowered by Claude + Pinecone + Voyage AI")

        with gr.Row():
            # Left: chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=520)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask about student loans...",
                        show_label=False,
                        scale=8,
                        container=False,
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")
                clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")

            # Right: live sources panel
            with gr.Column(scale=2):
                sources_panel = gr.Markdown(
                    value="*Ask a question to see retrieved sources here.*",
                    label="Retrieved Sources",
                )

        gr.Examples(
            examples=[
                "What types of student loans are available?",
                "How does Public Service Loan Forgiveness work?",
                "What are income-driven repayment plans?",
                "Can I refinance my federal student loans?",
                "What happens if I default on my student loans?",
            ],
            inputs=msg_box,
            label="Example questions",
        )

        send_btn.click(
            fn=chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, sources_panel],
        ).then(lambda: "", outputs=msg_box)

        msg_box.submit(
            fn=chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, sources_panel],
        ).then(lambda: "", outputs=msg_box)

        clear_btn.click(fn=clear, outputs=[chatbot, sources_panel])

    return demo


if __name__ == "__main__":
    _agent = build_agent()
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,   # set True to get a public gradio.live URL
        theme=gr.themes.Soft(),
    )
