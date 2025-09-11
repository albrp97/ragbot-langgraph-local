from app.ui.gradio_app import build_app
from app.llm.chat import preload_model
import os

def main():
    preload_model()
    demo = build_app()
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

if __name__ == "__main__":
    main()
