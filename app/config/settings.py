from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# load settings from environment or .env file
class Settings(BaseSettings):
    model_id: str = Field(default="Qwen/Qwen3-0.6B", alias="MODEL_ID")
    embedding_id: str = Field(default="Qwen/Qwen3-Embedding-0.6B", alias="EMBEDDING_ID")

    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")

    token_limit_history: int = Field(default=3200, alias="TOKEN_LIMIT_HISTORY")
    token_limit_summary: int = Field(default=1000, alias="TOKEN_LIMIT_SUMMARY")

    device: str = Field(default="auto", alias="DEVICE")
    gradio_port: int = Field(default=7860, alias="GRADIO_PORT")

    # ---- LLM generation defaults ----
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_top_p: float = Field(default=0.9, alias="LLM_TOP_P")
    llm_top_k: int = Field(default=0, alias="LLM_TOP_K")
    llm_do_sample: bool = Field(default=True, alias="LLM_DO_SAMPLE")
    llm_repetition_penalty: float = Field(default=1.05, alias="LLM_REPETITION_PENALTY")
    llm_max_new_tokens: int = Field(default=256, alias="LLM_MAX_NEW_TOKENS")
    llm_seed: int = Field(default=0, alias="LLM_SEED")

    llm_reasoning: bool = Field(default=False, alias="LLM_REASONING")
    
    # ---- Embeddings / Ingestion knobs ----
    embedding_batch_size: int = Field(default=8, alias="EMBEDDING_BATCH_SIZE")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    # ---- Retrieval knobs ----
    retriever_k: int = Field(default=5, alias="RETRIEVER_K")
    
    # ---- Image captions (VLM) ----
    image_retrieval_id: str = Field(default="Salesforce/blip-image-captioning-large", alias="IMAGE_RETRIEVAL_ID")
    images_dir: str = Field(default="./data/processed/images", alias="IMAGES_DIR")
    
    detector_model: str = Field(default="yolov11n.pt", alias="DETECTOR_MODEL")
    detector_port: int = Field(default=8000, alias="DETECTOR_PORT")


    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

settings = Settings()