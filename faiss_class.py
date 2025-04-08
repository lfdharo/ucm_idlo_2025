import numpy as np
import faiss
import logging
import os
from typing import List, Dict, Optional
from utils import find_files, ensures_dir
from vector_embedding import exctract_vector_embedding
from sklearn.preprocessing import normalize
import pickle

class FaissClass:
    """A class to handle speaker verification using FAISS for similarity search."""
    
    def __init__(self, model_name: str,  model: object,
                           feature_extractor: Optional[object] = None, threshold: float = 0.5):
        """Initialize the FaissClass.
        
        Args:
            model_name (str): Name of the model being used
            threshold (float): Similarity threshold for verification
        """
        self.model_name = model_name
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.index = None
        self.speaker_ids = []
        self.audio_files = []
        self.logger = logging.getLogger(__name__)
        
    def build_index(self, enrollment_dir: str) -> None:
        """Build FAISS index from enrollment audio files.
        
        Args:
            enrollment_dir (str): Directory containing enrollment audio files
        """
        self.logger.info(f"Building FAISS index from {enrollment_dir}")
        
        # Find all audio files in enrollment directory
        self.audio_files = find_files(enrollment_dir)
        if not self.audio_files:
            raise ValueError(f"No audio files found in {enrollment_dir}")
            
        # Extract speaker IDs from filenames
        self.speaker_ids = [f.split('/')[-1] for f in self.audio_files]
        
        # Initialize FAISS index
        # self.index = faiss.IndexFlatL2(len(self.audio_files))


        # Calculate the embeddings
        embeddings = None
        for audio_file in self.audio_files:
            # vector = self._extract_vector(audio_file)
            vector = exctract_vector_embedding(audio_file, self.model_name, self.model, self.feature_extractor)
            if embeddings is None:
                embeddings = vector
            else:
                embeddings = np.append(embeddings, vector, axis=0)  # Append all the generated embeddings for faster storing in the Faiss index later
       
       
        # self.index.add(np.array([vector], dtype=np.float32))            
        idx_int = np.arange(len(self.speaker_ids))  # Ids in the Faiss index must be integer. Therefore we need to create a dictionary to map the string IDs and the numeric ones
        self.int2idx = {str(i): self.speaker_ids[i] for i in range(0, len(self.speaker_ids))}

        # Step 1: Instantiate the index for the subselected embeddings
        self.index = faiss.IndexFlatIP(embeddings.shape[1])

        # Step 2: Pass the index to IndexIDMap
        self.index = faiss.IndexIDMap(self.index)

        # Step 3: Add vectors and their IDs
        # start = time.time()
        faiss.normalize_L2(embeddings)
        self.index.add_with_ids(embeddings, idx_int)          
        self.logger.info(f"Built index with {len(self.audio_files)} vectors")
    
    def save_index(self, index_file: str) -> None:
        """Save the built index to a file.
        
        Args:
            index_file (str): Path to save the index file
        """
        ensures_dir(os.path.dirname(index_file))
        faiss.write_index(self.index, index_file)
        with open(index_file + '.pkl', 'wb') as h:
            pickle.dump(self.int2idx, h)
            pickle.dump(self.speaker_ids, h)
        self.logger.info(f"Index saved to {index_file}")

    def load_index(self, index_file: str) -> None:
        """Load a saved index from a file.
        
        Args:
            index_file (str): Path to the saved index file
        """       
        self.index = faiss.read_index(index_file)
        with open(index_file + '.pkl', 'rb') as h:
            self.int2idx = pickle.load(h)
            self.speaker_ids = pickle.load(h)

        self.logger.info(f"Index loaded from {index_file}")


    def verify_speaker(self, test_file: str) -> Dict:
        """Verify if a test audio file matches any enrolled speaker.
        
        Args:
            test_file (str): Path to test audio file
            
        Returns:
            Dict: Verification results including matched speaker and similarity score
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        # Extract vector from test file
        # test_vector = self._extract_vector(test_file)
        spk1 = os.path.basename(test_file).split('_')[0]
        test_vector = exctract_vector_embedding(test_file, self.model_name, self.model, self.feature_extractor)
        
        # Search for nearest neighbor
        D, I = self.vector_search(test_vector) # Distance and index id
        for r, d in zip(I, D):
            self.logger.info(f'Closest element found: {self.int2idx[str(r[0])]} with cosine distance {d[0]}')


        # Get results
        matched_speaker_file = self.speaker_ids[I[0][0]]
        similarity_score = D[0][0]
        
        result = {
            'matched_speaker': matched_speaker_file,
            'similarity_score': similarity_score,
            'is_match': similarity_score >= self.threshold and matched_speaker_file.split('_')[0] == spk1
        }
        
        self.logger.info(f"Verification result: {result}")
        return result

    def vector_search(self, query_vector, num_results=10, tl=0.0, th=1.0):
        """Search closer vectors in the FAISS index.
        Args:
            query_vector (numpy array): Vector embedding to search for closer stored items
            num_results (int): Number of closer vectors to return.
        Returns:
            D (:obj:`numpy.array` of `float`): Distance between retrieved results and query.
            I (:obj:`numpy.array` of `int`): ID of the retrieved items.

        """
        query_vector = normalize(query_vector)
        D, I = self.index.search(np.array(query_vector).astype("float32"), k=num_results)
        I = [[I[0][i]] for i, d in enumerate(D[0]) if (d>=tl and d<=th)]
        D = [[d] for d in D[0] if (d>=tl and d<=th)]
        return D, I  # Return distances and items

    # def _extract_vector(self, audio_file: str) -> np.ndarray:
    #     """Extract speaker embedding vector from audio file.
        
    #     Args:
    #         audio_file (str): Path to audio file
            
    #     Returns:
    #         np.ndarray: Speaker embedding vector
    #     """
    #     # This method should be implemented based on the specific model being used
    #     # For now, we'll raise NotImplementedError
    #     raise NotImplementedError("Vector extraction method not implemented") 