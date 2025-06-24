import httpx
import io
from PIL import Image
from sentence_transformers import SentenceTransformer


class ImageEmbeddingGenerator:
    """
    Singleton class to manage the CLIP model from sentence-transformers.
    Ensures the model is loaded only once.
    """
    _instance = None
    model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageEmbeddingGenerator, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, model_id="clip-ViT-B-32"):
        if self.model is None:
            print(f"[*] Loading CLIP model from Hugging Face: {model_id}")
            print("[*] This process will download the model if not already available. Please wait...")
            try:
                self.model = SentenceTransformer(model_id)
                print("[+] CLIP model successfully loaded.")
            except Exception as e:
                print(f"[!] FAILED to load CLIP model: {e}")
                self.model = None

    async def generate_embedding(self, image_url: str):
        """
        Generate an embedding vector from an image URL.
        """
        if not self.model or not image_url:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url, timeout=20)
                response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))

            embedding = self.model.encode(image, convert_to_tensor=True)
            return embedding
        except Exception as e:
            print(f"Failed to generate embedding for {image_url}: {e}")
            return None

# Initialize the embedding generator
embedding_generator_instance = ImageEmbeddingGenerator()

async def get_image_embedding(image_url: str):
    """
    Async wrapper function to call the generator instance.
    """
    return await embedding_generator_instance.generate_embedding(image_url)

