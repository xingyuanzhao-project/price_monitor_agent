"""
Text analysis tools for chunking, searching, extracting, classifying, scoring, and summarizing.

What it does:
    Provides seven concrete tool implementations for natural language processing:
    text chunking, semantic search via TF-IDF cosine similarity, entity/relationship
    extraction, text classification, multi-dimensional scoring, summarization
    (single, multi-synthesis, delta), and cross-modal alignment.

Entities in it:
    - ChunkTextTool: Splits text into overlapping chunks for processing.
    - SemanticSearchTool: Searches text corpus using TF-IDF cosine similarity.
    - ExtractEntitiesTool: Extracts entities, relationships, and claims from text.
    - ClassifyTextTool: Classifies text into provided categories.
    - ScoreTextTool: Scores text on sentiment, relevance, and credibility dimensions.
    - SummarizeTextTool: Produces summaries in single, multi_synthesis, or delta modes.
    - CrossModalAlignmentTool: Aligns information across text from different modalities.

How used by other modules:
    - Registered in the ToolRegistry at application startup.
    - Called by agents during workflow execution for text processing tasks.
    - Operates on text data from upstream data acquisition or user input.
"""

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from backend.tools.base import BaseTool, ToolExecutionError


class ChunkTextTool(BaseTool):
    """
    Splits text into overlapping chunks with deduplication and format normalization.

    Description:
        Divides input text into chunks of a specified size with configurable
        overlap to preserve context at chunk boundaries.  Optionally deduplicates
        chunks with identical normalized content and normalizes whitespace/encoding
        across all chunks.

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'chunk_text'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for chunking parameters.
        execute: Splits input text into chunks.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'chunk_text'
        """
        return "chunk_text"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the text chunking functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Splits text into overlapping chunks of specified size with "
            "optional deduplication and format normalization."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines text, chunk_size, overlap, deduplicate, and normalize parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to chunk"},
                "chunk_size": {
                    "type": "integer",
                    "description": "Maximum characters per chunk",
                    "default": 1000,
                },
                "overlap": {
                    "type": "integer",
                    "description": "Number of overlapping characters between chunks",
                    "default": 200,
                },
                "deduplicate": {
                    "type": "boolean",
                    "description": "Remove chunks with identical normalized content",
                    "default": False,
                },
                "normalize_format": {
                    "type": "boolean",
                    "description": "Normalize whitespace and unicode in each chunk",
                    "default": False,
                },
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Split text into overlapping chunks with optional deduplication and normalization.

        Description:
            Divides the input text into chunks of chunk_size characters with
            overlap characters repeated between consecutive chunks.  When
            deduplicate is True, chunks whose normalized fingerprint has been
            seen before are dropped.  When normalize_format is True, each chunk's
            whitespace is collapsed and leading/trailing space removed.

        Params:
            **kwargs (Any): Must include 'text'. Optional: 'chunk_size', 'overlap',
                'deduplicate', 'normalize_format'.

        Returns:
            dict: Dictionary with 'chunks' list and 'total_chunks' count.

        Raises:
            ToolExecutionError: If text is empty or parameters are invalid.
        """
        text = kwargs.get("text", "")
        if not text:
            raise ToolExecutionError("Cannot chunk empty text")

        chunk_size = kwargs.get("chunk_size", 1000)
        overlap = kwargs.get("overlap", 200)
        deduplicate = kwargs.get("deduplicate", False)
        normalize_format = kwargs.get("normalize_format", False)

        if chunk_size <= 0:
            raise ToolExecutionError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ToolExecutionError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ToolExecutionError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        chunks = []
        start_position = 0
        step_size = chunk_size - overlap
        seen_fingerprints: set[str] = set()

        while start_position < len(text):
            end_position = start_position + chunk_size
            chunk_content = text[start_position:end_position]

            if normalize_format:
                chunk_content = re.sub(r"\s+", " ", chunk_content).strip()

            if deduplicate:
                fingerprint = re.sub(r"\s+", "", chunk_content.lower())
                if fingerprint in seen_fingerprints:
                    start_position += step_size
                    continue
                seen_fingerprints.add(fingerprint)

            chunks.append({
                "index": len(chunks),
                "start": start_position,
                "end": min(end_position, len(text)),
                "content": chunk_content,
            })
            start_position += step_size

        return {"chunks": chunks, "total_chunks": len(chunks)}


class SemanticSearchTool(BaseTool):
    """
    Searches a text corpus using TF-IDF cosine similarity.

    Description:
        Computes TF-IDF vectors for a query and corpus documents, then ranks
        documents by cosine similarity to the query vector.

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'semantic_search'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for search parameters.
        execute: Searches the corpus and returns ranked results.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'semantic_search'
        """
        return "semantic_search"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the TF-IDF cosine similarity search functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return "Searches a text corpus using TF-IDF cosine similarity to find relevant documents."

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines query, corpus, and top_k parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "corpus": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of document strings to search",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return",
                    "default": 5,
                },
            },
            "required": ["query", "corpus"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Search the corpus using TF-IDF cosine similarity.

        Description:
            Tokenizes query and documents, computes TF-IDF vectors, calculates
            cosine similarity, and returns the top_k most similar documents.

        Params:
            **kwargs (Any): Must include 'query' and 'corpus'. Optional: 'top_k'.

        Returns:
            dict: Dictionary with 'results' list of {index, score, content} dicts.

        Raises:
            ToolExecutionError: If query is empty or corpus is empty.
        """
        query = kwargs.get("query", "")
        corpus = kwargs.get("corpus", [])
        top_k = kwargs.get("top_k", 5)

        if not query:
            raise ToolExecutionError("Search query cannot be empty")
        if not corpus:
            raise ToolExecutionError("Search corpus cannot be empty")

        all_documents = [query] + list(corpus)
        tokenized_documents = [self._tokenize(document) for document in all_documents]

        document_frequency = Counter()
        for tokens in tokenized_documents:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                document_frequency[token] += 1

        total_documents = len(all_documents)
        tfidf_vectors = []
        vocabulary = sorted(document_frequency.keys())
        vocabulary_index = {term: index for index, term in enumerate(vocabulary)}

        for tokens in tokenized_documents:
            term_frequency = Counter(tokens)
            vector = np.zeros(len(vocabulary))
            for term, count in term_frequency.items():
                if term in vocabulary_index:
                    tf_value = count / len(tokens) if tokens else 0
                    idf_value = math.log(total_documents / document_frequency[term])
                    vector[vocabulary_index[term]] = tf_value * idf_value
            tfidf_vectors.append(vector)

        query_vector = tfidf_vectors[0]
        similarities = []

        for document_index in range(1, len(tfidf_vectors)):
            document_vector = tfidf_vectors[document_index]
            similarity_score = self._cosine_similarity(query_vector, document_vector)
            similarities.append((document_index - 1, similarity_score))

        similarities.sort(key=lambda item: item[1], reverse=True)
        top_results = similarities[:top_k]

        results = [
            {
                "index": corpus_index,
                "score": float(score),
                "content": corpus[corpus_index],
            }
            for corpus_index, score in top_results
            if score > 0
        ]

        return {"results": results, "total_matches": len(results)}

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into lowercase alphanumeric tokens.

        Description:
            Splits on non-alphanumeric characters and lowercases all tokens.

        Params:
            text (str): Input text to tokenize.

        Returns:
            list[str]: List of lowercase tokens.
        """
        return [token.lower() for token in re.split(r"\W+", text) if token]

    def _cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Description:
            Returns dot product divided by product of magnitudes. Returns 0 if
            either vector has zero magnitude.

        Params:
            vector_a (np.ndarray): First vector.
            vector_b (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity value between 0 and 1.
        """
        dot_product = np.dot(vector_a, vector_b)
        magnitude_a = np.linalg.norm(vector_a)
        magnitude_b = np.linalg.norm(vector_b)
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        return float(dot_product / (magnitude_a * magnitude_b))


class ExtractEntitiesTool(BaseTool):
    """
    Extracts entities, relationships, and claims from text.

    Description:
        Uses pattern-based extraction to identify named entities, relationships
        between entities, and factual claims within the input text.

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'extract_entities'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for extraction parameters.
        execute: Extracts entities, relationships, and claims from text.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'extract_entities'
        """
        return "extract_entities"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the entity/relationship/claim extraction functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return "Extracts entities, relationships, and claims from text content."

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines text and entity_types parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to extract entities from"},
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of entities to extract (e.g., 'organization', 'person', 'asset')",
                },
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Extract entities, relationships, and claims from text.

        Description:
            Identifies named entities with type distinctions (TICKER, COMPANY,
            PERSON, DATE, NUMBER), relationship pairs with attribution context,
            and factual claims containing numeric values.

        Params:
            **kwargs (Any): Must include 'text'. Optional: 'entity_types'.

        Returns:
            dict: Dictionary with 'entities', 'relationships', and 'claims' lists.

        Raises:
            ToolExecutionError: If text is empty.
        """
        text = kwargs.get("text", "")
        if not text:
            raise ToolExecutionError("Cannot extract entities from empty text")

        requested_types: list[str] = kwargs.get("entity_types", [])

        entities = self._extract_named_entities(text, requested_types)
        relationships = self._extract_relationships(text, entities)
        claims = self._extract_claims(text)

        return {
            "entities": entities,
            "relationships": relationships,
            "claims": claims,
        }

    def _classify_entity_type(self, token: str) -> str:
        """
        Classify a named entity token into a specific entity type.

        Description:
            Applies pattern-based heuristics to determine whether the token
            is a TICKER, NUMBER, DATE, COMPANY, or PERSON.

        Params:
            token (str): The extracted entity string.

        Returns:
            str: One of TICKER, NUMBER, DATE, COMPANY, PERSON.
        """
        # Ticker: all-caps 1-5 letter sequence, optionally preceded by $
        if re.match(r"^\$?[A-Z]{1,5}$", token):
            return "TICKER"
        # Number with currency/percentage
        if re.match(r"^\$?[\d,]+(?:\.\d+)?%?$", token):
            return "NUMBER"
        # Date patterns
        if re.match(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?",
            token,
        ) or re.match(r"\b\d{4}\b", token):
            return "DATE"
        # Company suffix indicators
        if re.search(
            r"\b(?:Inc|Corp|Ltd|LLC|Group|Holdings|Technologies|Capital|Partners|Fund)\b",
            token,
        ):
            return "COMPANY"
        # Single-word capitalized names are tentatively PERSON; multi-word may be either
        words = token.split()
        if len(words) <= 2:
            return "PERSON"
        return "COMPANY"

    def _extract_named_entities(
        self, text: str, requested_types: list[str]
    ) -> list[dict]:
        """
        Extract named entities with type classification.

        Description:
            Finds ticker symbols, numbers, dates, and capitalized multi-word
            phrases, classifying each by type (TICKER, NUMBER, DATE, COMPANY,
            PERSON).  If requested_types is non-empty, only those types are
            returned.

        Params:
            text (str): Input text to process.
            requested_types (list[str]): Optional filter of types to return.

        Returns:
            list[dict]: List of entity dicts with 'name' and 'type' fields.
        """
        seen_entities: set[str] = set()
        entities: list[dict] = []

        # Extract ticker symbols (e.g. $AAPL or AAPL in finance context)
        ticker_pattern = re.compile(r"\$[A-Z]{1,5}\b|\b[A-Z]{2,5}\b")
        for match in ticker_pattern.finditer(text):
            candidate = match.group()
            if candidate not in seen_entities and len(candidate) >= 2:
                entity_type = self._classify_entity_type(candidate)
                seen_entities.add(candidate)
                entities.append({"name": candidate, "type": entity_type})

        # Extract multi-word capitalized phrases (companies, people, places)
        phrase_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        for match in phrase_pattern.finditer(text):
            candidate = match.group()
            if candidate not in seen_entities:
                seen_entities.add(candidate)
                entity_type = self._classify_entity_type(candidate)
                entities.append({"name": candidate, "type": entity_type})

        # Extract date patterns
        date_pattern = re.compile(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
            re.IGNORECASE,
        )
        for match in date_pattern.finditer(text):
            candidate = match.group()
            if candidate not in seen_entities:
                seen_entities.add(candidate)
                entities.append({"name": candidate, "type": "DATE"})

        if requested_types:
            normalised_requested = {t.upper() for t in requested_types}
            entities = [
                entity for entity in entities
                if entity["type"].upper() in normalised_requested
            ]

        return entities

    def _extract_relationships(self, text: str, entities: list[dict]) -> list[dict]:
        """
        Extract relationships between entities with attribution context.

        Description:
            Finds entity pairs appearing in the same sentence and extracts
            the verb phrase between them as the predicate.  When the sentence
            contains attribution verbs (said, reported, announced, according),
            the relationship is tagged as an attribution and the attributing
            entity is identified.

        Params:
            text (str): Input text to process.
            entities (list[dict]): Previously extracted entities.

        Returns:
            list[dict]: List of relationship dicts with 'subject', 'predicate',
                'object', 'attribution', and optionally 'attributed_by'.
        """
        attribution_verbs = {"said", "reported", "announced", "stated",
                             "according", "disclosed", "claimed", "noted"}
        sentences = re.split(r"[.!?]+", text)
        entity_names = {entity["name"] for entity in entities}
        relationships: list[dict] = []

        for sentence in sentences:
            sentence = sentence.strip()
            entities_in_sentence = [name for name in entity_names if name in sentence]
            if len(entities_in_sentence) < 2:
                continue

            sentence_tokens = set(re.split(r"\W+", sentence.lower()))
            is_attribution = bool(attribution_verbs & sentence_tokens)

            for subject_index in range(len(entities_in_sentence)):
                for object_index in range(subject_index + 1, len(entities_in_sentence)):
                    subject_entity = entities_in_sentence[subject_index]
                    object_entity = entities_in_sentence[object_index]
                    subject_position = sentence.find(subject_entity)
                    object_position = sentence.find(object_entity)
                    between_text = sentence[
                        subject_position + len(subject_entity): object_position
                    ].strip()
                    if not between_text:
                        continue

                    relationship: dict[str, Any] = {
                        "subject": subject_entity,
                        "predicate": between_text,
                        "object": object_entity,
                        "is_attribution": is_attribution,
                    }
                    if is_attribution:
                        # The entity appearing before an attribution verb is the source.
                        attribution_match = re.search(
                            r"(\w[\w\s]*)\s+(?:said|reported|announced|stated|"
                            r"according to|disclosed|claimed|noted)",
                            sentence,
                            re.IGNORECASE,
                        )
                        if attribution_match:
                            relationship["attributed_by"] = attribution_match.group(1).strip()

                    relationships.append(relationship)

        return relationships

    def _extract_claims(self, text: str) -> list[dict]:
        """
        Extract factual claims containing numeric values.

        Description:
            Identifies sentences with numbers, percentages, or currency amounts
            as potential factual claims.

        Params:
            text (str): Input text to process.

        Returns:
            list[dict]: List of claim dicts with 'text' and 'has_numeric_value'.
        """
        sentences = re.split(r"[.!?]+", text)
        numeric_pattern = re.compile(r"\d+[\d,.]*%?|\$[\d,.]+")
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and numeric_pattern.search(sentence):
                claims.append({
                    "text": sentence,
                    "has_numeric_value": True,
                })

        return claims


class ClassifyTextTool(BaseTool):
    """
    Classifies text into user-provided categories using keyword matching.

    Description:
        Assigns text to one or more categories based on keyword overlap
        between the text content and category-defining terms.

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'classify_text'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for classification parameters.
        execute: Classifies the input text.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'classify_text'
        """
        return "classify_text"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the text classification functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return "Classifies text into provided categories based on content analysis."

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines text and categories parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to classify"},
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "description": "Categories with associated keywords",
                },
            },
            "required": ["text", "categories"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Classify text into categories by keyword overlap.

        Description:
            Tokenizes the text, counts keyword matches for each category,
            and returns categories ranked by match score.

        Params:
            **kwargs (Any): Must include 'text' and 'categories'.

        Returns:
            dict: Dictionary with 'classifications' list of {category, score} dicts.

        Raises:
            ToolExecutionError: If text is empty or categories list is empty.
        """
        text = kwargs.get("text", "")
        categories = kwargs.get("categories", [])

        if not text:
            raise ToolExecutionError("Cannot classify empty text")
        if not categories:
            raise ToolExecutionError("Categories list cannot be empty")

        text_tokens = set(re.split(r"\W+", text.lower()))
        classifications = []

        for category in categories:
            category_name = category.get("name", "")
            keywords = category.get("keywords", [])
            if not keywords:
                continue

            matching_keywords = [
                keyword for keyword in keywords
                if keyword.lower() in text_tokens or keyword.lower() in text.lower()
            ]
            score = len(matching_keywords) / len(keywords)
            classifications.append({
                "category": category_name,
                "score": score,
                "matching_keywords": matching_keywords,
            })

        classifications.sort(key=lambda item: item["score"], reverse=True)
        return {"classifications": classifications}


class ScoreTextTool(BaseTool):
    """
    Scores text on sentiment, relevance, and credibility dimensions.

    Description:
        Computes multi-dimensional scores for text using lexicon-based
        sentiment analysis, keyword-based relevance, and credibility
        indicators (source attribution, numeric evidence, hedging language).

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'score_text'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for scoring parameters.
        execute: Scores the text on requested dimensions.
    """

    POSITIVE_LEXICON = {
        "good", "great", "excellent", "positive", "bullish", "growth", "profit",
        "gain", "surge", "strong", "improve", "rise", "high", "up", "best",
        "success", "increase", "advance", "rally", "outperform",
    }

    NEGATIVE_LEXICON = {
        "bad", "poor", "negative", "bearish", "loss", "decline", "drop",
        "weak", "crash", "fall", "down", "worst", "fail", "decrease",
        "risk", "debt", "sell", "plunge", "underperform", "recession",
    }

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'score_text'
        """
        return "score_text"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the multi-dimensional text scoring functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return "Scores text on sentiment, relevance, and credibility dimensions."

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines text, dimensions, context, and entities parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to score"},
                "dimensions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["sentiment", "relevance", "credibility"],
                    },
                    "description": "Scoring dimensions to compute",
                },
                "relevance_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords for relevance scoring",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Named entities to score sentiment for individually",
                },
            },
            "required": ["text", "dimensions"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Score text on specified dimensions with optional per-entity sentiment.

        Description:
            Computes scores for each requested dimension and returns them
            with the component analysis.  When 'entities' is provided and
            'sentiment' is in dimensions, also computes a per-entity sentiment
            score by scoring the sentences containing each entity.

        Params:
            **kwargs (Any): Must include 'text' and 'dimensions'.
                Optional: 'relevance_keywords', 'entities'.

        Returns:
            dict: Dictionary with 'scores' for each dimension and optionally
                'entity_sentiments' mapping entity name to sentiment score.

        Raises:
            ToolExecutionError: If text is empty or dimensions is empty.
        """
        text = kwargs.get("text", "")
        dimensions = kwargs.get("dimensions", [])

        if not text:
            raise ToolExecutionError("Cannot score empty text")
        if not dimensions:
            raise ToolExecutionError("Dimensions list cannot be empty")

        scores = {}
        text_lower = text.lower()
        tokens = set(re.split(r"\W+", text_lower))

        for dimension in dimensions:
            if dimension == "sentiment":
                scores["sentiment"] = self._score_sentiment(tokens)
            elif dimension == "relevance":
                relevance_keywords = kwargs.get("relevance_keywords", [])
                scores["relevance"] = self._score_relevance(text_lower, tokens, relevance_keywords)
            elif dimension == "credibility":
                scores["credibility"] = self._score_credibility(text)
            else:
                raise ToolExecutionError(
                    f"Unsupported scoring dimension: '{dimension}'. "
                    f"Must be one of: sentiment, relevance, credibility"
                )

        result: dict[str, Any] = {"scores": scores}

        # Per-entity sentiment: score the sentences that contain each entity.
        entity_names: list[str] = kwargs.get("entities", [])
        if entity_names and "sentiment" in dimensions:
            result["entity_sentiments"] = self._score_per_entity_sentiment(
                text, entity_names
            )

        return result

    def _score_per_entity_sentiment(
        self, text: str, entity_names: list[str]
    ) -> dict[str, dict]:
        """
        Compute sentiment scores for the context sentences of named entities.

        Description:
            For each entity, locates all sentences in the text that mention
            it and computes the sentiment score over those sentences' tokens.
            Entities with no matching sentences receive a neutral score.

        Params:
            text (str): Full input text.
            entity_names (list[str]): Named entities to score individually.

        Returns:
            dict[str, dict]: Mapping from entity name to its sentiment score dict.
        """
        sentences = [sentence.strip() for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
        entity_sentiments: dict[str, dict] = {}

        for entity_name in entity_names:
            entity_sentences = [
                sentence for sentence in sentences
                if entity_name.lower() in sentence.lower()
            ]
            if not entity_sentences:
                entity_sentiments[entity_name] = {
                    "value": 0.0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "sentence_count": 0,
                }
                continue

            combined_entity_text = " ".join(entity_sentences)
            entity_tokens = set(re.split(r"\W+", combined_entity_text.lower()))
            sentiment_score = self._score_sentiment(entity_tokens)
            sentiment_score["sentence_count"] = len(entity_sentences)
            entity_sentiments[entity_name] = sentiment_score

        return entity_sentiments

    def _score_sentiment(self, tokens: set[str]) -> dict:
        """
        Compute sentiment score from token overlap with lexicons.

        Description:
            Counts positive and negative keyword matches and computes a
            normalized score from -1.0 (most negative) to 1.0 (most positive).

        Params:
            tokens (set[str]): Lowercase tokens from the text.

        Returns:
            dict: Sentiment score with positive_count, negative_count, and value.
        """
        positive_matches = tokens & self.POSITIVE_LEXICON
        negative_matches = tokens & self.NEGATIVE_LEXICON
        positive_count = len(positive_matches)
        negative_count = len(negative_matches)
        total = positive_count + negative_count

        if total == 0:
            score_value = 0.0
        else:
            score_value = (positive_count - negative_count) / total

        return {
            "value": score_value,
            "positive_count": positive_count,
            "negative_count": negative_count,
        }

    def _score_relevance(
        self, text_lower: str, tokens: set[str], relevance_keywords: list[str]
    ) -> dict:
        """
        Compute relevance score based on keyword presence.

        Description:
            Counts how many relevance keywords appear in the text.

        Params:
            text_lower (str): Lowercased full text.
            tokens (set[str]): Lowercase tokens from the text.
            relevance_keywords (list[str]): Keywords indicating relevance.

        Returns:
            dict: Relevance score with matched_keywords and value.
        """
        if not relevance_keywords:
            return {"value": 0.0, "matched_keywords": []}

        matched_keywords = [
            keyword for keyword in relevance_keywords
            if keyword.lower() in tokens or keyword.lower() in text_lower
        ]
        score_value = len(matched_keywords) / len(relevance_keywords)
        return {"value": score_value, "matched_keywords": matched_keywords}

    def _score_credibility(self, text: str) -> dict:
        """
        Compute credibility score from structural indicators.

        Description:
            Checks for source attribution, numeric evidence, and absence
            of excessive hedging language.

        Params:
            text (str): Original text.

        Returns:
            dict: Credibility score with component indicators and value.
        """
        has_source = bool(re.search(r"(?:according to|source:|cited|reported by)", text, re.IGNORECASE))
        has_numbers = bool(re.search(r"\d+[\d,.]*%?|\$[\d,.]+", text))
        hedging_words = {"might", "maybe", "possibly", "perhaps", "could", "allegedly", "rumored"}
        text_tokens = set(re.split(r"\W+", text.lower()))
        hedging_count = len(text_tokens & hedging_words)

        source_score = 0.4 if has_source else 0.0
        numeric_score = 0.3 if has_numbers else 0.0
        hedging_penalty = min(hedging_count * 0.1, 0.3)

        credibility_value = min(1.0, max(0.0, source_score + numeric_score + 0.3 - hedging_penalty))

        return {
            "value": credibility_value,
            "has_source_attribution": has_source,
            "has_numeric_evidence": has_numbers,
            "hedging_count": hedging_count,
        }


class SummarizeTextTool(BaseTool):
    """
    Produces text summaries in single, multi_synthesis, or delta modes.

    Description:
        Supports three summarization modes: 'single' for summarizing one text,
        'multi_synthesis' for combining multiple texts into a unified summary,
        and 'delta' for highlighting differences between two text versions.

    Attributes:
        SUPPORTED_MODES: Class-level set of supported summarization modes.

    Methods:
        name: Returns 'summarize_text'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for summarization parameters.
        execute: Produces the requested type of summary.
    """

    SUPPORTED_MODES = {"single", "multi_synthesis", "delta"}

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'summarize_text'
        """
        return "summarize_text"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the summarization modes available.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Produces text summaries. Modes: 'single' (one text), "
            "'multi_synthesis' (combine multiple), 'delta' (differences between two versions)."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines mode, texts, max_length, and comparison parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_MODES),
                    "description": "Summarization mode",
                },
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Input texts (1 for single, multiple for multi_synthesis, 2 for delta)",
                },
                "max_sentences": {
                    "type": "integer",
                    "description": "Maximum sentences in the summary",
                    "default": 5,
                },
            },
            "required": ["mode", "texts"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Produce a summary based on the specified mode.

        Description:
            Dispatches to the appropriate summarization strategy based on mode.

        Params:
            **kwargs (Any): Must include 'mode' and 'texts'.

        Returns:
            dict: Dictionary with 'summary' and mode-specific metadata.

        Raises:
            ToolExecutionError: If mode is unsupported or texts are insufficient.
        """
        mode = kwargs.get("mode")
        texts = kwargs.get("texts", [])
        max_sentences = kwargs.get("max_sentences", 5)

        if mode not in self.SUPPORTED_MODES:
            raise ToolExecutionError(
                f"Unsupported summarization mode: '{mode}'. "
                f"Must be one of: {sorted(self.SUPPORTED_MODES)}"
            )

        if not texts:
            raise ToolExecutionError("Cannot summarize empty texts list")

        if mode == "single":
            if len(texts) < 1:
                raise ToolExecutionError("Single mode requires at least 1 text")
            return self._summarize_single(texts[0], max_sentences)
        elif mode == "multi_synthesis":
            if len(texts) < 2:
                raise ToolExecutionError("Multi-synthesis mode requires at least 2 texts")
            return self._summarize_multi_synthesis(texts, max_sentences)
        elif mode == "delta":
            if len(texts) < 2:
                raise ToolExecutionError("Delta mode requires exactly 2 texts")
            return self._summarize_delta(texts[0], texts[1], max_sentences)

        raise ToolExecutionError(f"Unhandled summarization mode: '{mode}'")

    def _summarize_single(self, text: str, max_sentences: int) -> dict:
        """
        Summarize a single text by extracting important sentences.

        Description:
            Scores sentences by word frequency and selects top sentences.

        Params:
            text (str): Text to summarize.
            max_sentences (int): Maximum sentences in summary.

        Returns:
            dict: Dictionary with 'summary', 'mode', and 'sentence_count'.
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return {"summary": "", "mode": "single", "sentence_count": 0}

        word_frequencies = Counter(re.split(r"\W+", text.lower()))
        sentence_scores = []
        for sentence in sentences:
            words = re.split(r"\W+", sentence.lower())
            score = sum(word_frequencies.get(word, 0) for word in words if word)
            sentence_scores.append(score)

        ranked_indices = sorted(
            range(len(sentences)),
            key=lambda index: sentence_scores[index],
            reverse=True,
        )
        selected_indices = sorted(ranked_indices[:max_sentences])
        summary = ". ".join(sentences[index] for index in selected_indices) + "."

        return {"summary": summary, "mode": "single", "sentence_count": len(selected_indices)}

    def _summarize_multi_synthesis(self, texts: list[str], max_sentences: int) -> dict:
        """
        Synthesize multiple texts into a unified summary with conflict reconciliation.

        Description:
            Computes per-document summaries, then identifies conflicting facts
            (sentences with opposing numeric or sentiment signals across sources)
            and notes them explicitly in the output.  Non-conflicting key sentences
            are merged into the final summary.

        Params:
            texts (list[str]): Multiple texts to synthesize.
            max_sentences (int): Maximum sentences in summary.

        Returns:
            dict: Dictionary with 'summary', 'mode', 'source_count', and 'conflicts'.
        """
        # Extract key sentences per source.
        per_source_sentences: list[list[str]] = []
        for source_text in texts:
            source_sentences = [
                sentence.strip()
                for sentence in re.split(r"[.!?]+", source_text)
                if sentence.strip()
            ]
            per_source_sentences.append(source_sentences)

        # Detect conflicts: sentences from different sources that share a numeric
        # claim but with opposing direction words (up/down, positive/negative).
        positive_direction_words = {"up", "rise", "increase", "surged", "gained", "above"}
        negative_direction_words = {"down", "fall", "decline", "dropped", "lost", "below"}
        numeric_pattern = re.compile(r"\d+[\d,.]*%?|\$[\d,.]+")
        conflicts: list[dict] = []

        if len(per_source_sentences) >= 2:
            for source_index_a in range(len(per_source_sentences)):
                for source_index_b in range(source_index_a + 1, len(per_source_sentences)):
                    for sentence_a in per_source_sentences[source_index_a]:
                        for sentence_b in per_source_sentences[source_index_b]:
                            if not numeric_pattern.search(sentence_a):
                                continue
                            if not numeric_pattern.search(sentence_b):
                                continue
                            tokens_a = set(re.split(r"\W+", sentence_a.lower()))
                            tokens_b = set(re.split(r"\W+", sentence_b.lower()))
                            direction_a_positive = bool(tokens_a & positive_direction_words)
                            direction_a_negative = bool(tokens_a & negative_direction_words)
                            direction_b_positive = bool(tokens_b & positive_direction_words)
                            direction_b_negative = bool(tokens_b & negative_direction_words)
                            if (direction_a_positive and direction_b_negative) or (
                                direction_a_negative and direction_b_positive
                            ):
                                conflicts.append({
                                    "source_a": source_index_a,
                                    "source_b": source_index_b,
                                    "sentence_a": sentence_a,
                                    "sentence_b": sentence_b,
                                })

        # Build combined summary from non-conflicting sentences.
        conflicting_texts: set[str] = set()
        for conflict in conflicts:
            conflicting_texts.add(conflict["sentence_a"])
            conflicting_texts.add(conflict["sentence_b"])

        all_sentences: list[str] = []
        for source_sentences in per_source_sentences:
            for sentence in source_sentences:
                if sentence not in conflicting_texts:
                    all_sentences.append(sentence)

        combined_text = ". ".join(all_sentences)
        single_result = self._summarize_single(combined_text, max_sentences) if combined_text else {
            "summary": "", "mode": "single", "sentence_count": 0
        }

        conflict_notes = [
            f"Source {conflict['source_a']} vs Source {conflict['source_b']}: "
            f"'{conflict['sentence_a'][:80]}' contradicts '{conflict['sentence_b'][:80]}'"
            for conflict in conflicts[:5]
        ]

        return {
            "summary": single_result["summary"],
            "mode": "multi_synthesis",
            "source_count": len(texts),
            "conflicts": conflict_notes,
            "conflict_count": len(conflicts),
        }

    def _summarize_delta(self, text_before: str, text_after: str, max_sentences: int) -> dict:
        """
        Summarize the differences between two text versions.

        Description:
            Identifies sentences unique to each version and presents them
            as additions and removals.

        Params:
            text_before (str): Original text version.
            text_after (str): Updated text version.
            max_sentences (int): Maximum sentences to report.

        Returns:
            dict: Dictionary with 'additions', 'removals', and 'summary'.
        """
        sentences_before = set(s.strip() for s in re.split(r"[.!?]+", text_before) if s.strip())
        sentences_after = set(s.strip() for s in re.split(r"[.!?]+", text_after) if s.strip())

        additions = list(sentences_after - sentences_before)[:max_sentences]
        removals = list(sentences_before - sentences_after)[:max_sentences]

        summary_parts = []
        if additions:
            summary_parts.append(f"Added {len(additions)} new point(s)")
        if removals:
            summary_parts.append(f"Removed {len(removals)} point(s)")
        if not summary_parts:
            summary_parts.append("No significant changes detected")

        return {
            "mode": "delta",
            "summary": ". ".join(summary_parts) + ".",
            "additions": additions,
            "removals": removals,
        }


class CrossModalAlignmentTool(BaseTool):
    """
    Aligns information across text from different modalities or sources.

    Description:
        Takes text segments from different modalities (e.g., news, technical,
        social) and identifies common themes, contradictions, and confidence
        levels across them.

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'cross_modal_alignment'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for alignment parameters.
        execute: Aligns information across modal text segments.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'cross_modal_alignment'
        """
        return "cross_modal_alignment"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the cross-modal alignment functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Aligns information across text from different modalities or sources "
            "to identify common themes, contradictions, and confidence levels."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines segments (modal texts) and structured_data (fetched data
            points from data acquisition tools) parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "modality": {"type": "string", "description": "Source modality name"},
                            "text": {"type": "string", "description": "Text content from this modality"},
                        },
                        "required": ["modality", "text"],
                    },
                    "description": "Text segments from different modalities",
                },
                "structured_data": {
                    "type": "object",
                    "description": (
                        "Structured data fetched by data acquisition tools to anchor "
                        "text claims against (e.g. {'price': 150.5, 'volume': 1000000, "
                        "'change_pct': -2.3})."
                    ),
                },
            },
            "required": ["segments"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Align information across modal text segments and anchor claims to structured data.

        Description:
            Tokenises each segment and identifies overlapping vocabulary as common
            themes.  Measures agreement/disagreement using sentiment polarity.
            When structured_data is provided, extracts numeric claims from text and
            checks whether they correspond to the structured data values.

        Params:
            **kwargs (Any): Must include 'segments' list of {modality, text} objects.
                Optional: 'structured_data' dict of field→value pairs.

        Returns:
            dict: Dictionary with 'common_themes', 'contradictions', 'confidence',
                  'modality_summary', and 'anchored_claims'.

        Raises:
            ToolExecutionError: If segments is empty or has fewer than 2 entries.
        """
        segments = kwargs.get("segments", [])
        if len(segments) < 2:
            raise ToolExecutionError(
                "Cross-modal alignment requires at least 2 segments from different modalities"
            )

        structured_data: dict[str, Any] = kwargs.get("structured_data") or {}

        modality_tokens: dict[str, set[str]] = {}
        modality_sentiments: dict[str, float] = {}

        positive_words = {
            "good", "great", "positive", "bullish", "growth", "up", "strong", "rise", "gain",
        }
        negative_words = {
            "bad", "negative", "bearish", "decline", "down", "weak", "fall", "loss", "drop",
        }

        for segment in segments:
            modality = segment["modality"]
            text = segment["text"]
            tokens = set(re.split(r"\W+", text.lower())) - {""}
            modality_tokens[modality] = tokens

            positive_count = len(tokens & positive_words)
            negative_count = len(tokens & negative_words)
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words > 0:
                modality_sentiments[modality] = (positive_count - negative_count) / total_sentiment_words
            else:
                modality_sentiments[modality] = 0.0

        all_token_sets = list(modality_tokens.values())
        common_tokens = all_token_sets[0]
        for token_set in all_token_sets[1:]:
            common_tokens = common_tokens & token_set

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "for", "of", "and", "or",
        }
        common_themes = sorted(common_tokens - stopwords)

        contradictions = []
        modality_names = list(modality_sentiments.keys())
        for index_a in range(len(modality_names)):
            for index_b in range(index_a + 1, len(modality_names)):
                name_a = modality_names[index_a]
                name_b = modality_names[index_b]
                sentiment_difference = abs(modality_sentiments[name_a] - modality_sentiments[name_b])
                if sentiment_difference > 0.5:
                    contradictions.append({
                        "modality_a": name_a,
                        "modality_b": name_b,
                        "sentiment_a": modality_sentiments[name_a],
                        "sentiment_b": modality_sentiments[name_b],
                        "disagreement_level": sentiment_difference,
                    })

        total_tokens_union: set[str] = set()
        for token_set in all_token_sets:
            total_tokens_union |= token_set
        significant_common = common_tokens - stopwords
        overlap_ratio = len(significant_common) / max(len(total_tokens_union - stopwords), 1)
        confidence = min(1.0, overlap_ratio * 5)

        modality_summary = {
            modality: {"sentiment": modality_sentiments[modality], "token_count": len(tokens)}
            for modality, tokens in modality_tokens.items()
        }

        # Anchor numeric claims in text to structured data values.
        anchored_claims = self._anchor_claims_to_structured_data(segments, structured_data)

        return {
            "common_themes": common_themes[:20],
            "contradictions": contradictions,
            "confidence": confidence,
            "modality_summary": modality_summary,
            "anchored_claims": anchored_claims,
        }

    def _anchor_claims_to_structured_data(
        self,
        segments: list[dict],
        structured_data: dict[str, Any],
    ) -> list[dict]:
        """
        Match numeric claims in text segments to structured data field values.

        Description:
            Extracts numeric values from each segment's text and checks whether
            any structured_data field is numerically close to a claim value.
            Returns a list of anchored claim dicts indicating which data field
            each text claim corresponds to, whether they agree, and the delta.

        Params:
            segments (list[dict]): Text segments with modality and text fields.
            structured_data (dict[str, Any]): Numeric field→value data points.

        Returns:
            list[dict]: Anchored claim dicts with 'claim', 'modality',
                'matched_field', 'data_value', 'text_value', and 'agrees'.
        """
        if not structured_data:
            return []

        numeric_token_pattern = re.compile(r"\$?[\d,]+(?:\.\d+)?%?")
        anchored_claims: list[dict] = []

        for segment in segments:
            modality = segment["modality"]
            text = segment["text"]
            numeric_matches = numeric_token_pattern.findall(text)

            for raw_match in numeric_matches:
                cleaned = raw_match.replace("$", "").replace(",", "").replace("%", "")
                try:
                    claim_value = float(cleaned)
                except ValueError:
                    continue

                for field_name, field_value in structured_data.items():
                    try:
                        data_value = float(field_value)
                    except (TypeError, ValueError):
                        continue

                    # Consider a match if within 5% relative tolerance.
                    if data_value == 0:
                        continue
                    relative_delta = abs(claim_value - data_value) / abs(data_value)
                    if relative_delta <= 0.05:
                        anchored_claims.append({
                            "claim": raw_match,
                            "modality": modality,
                            "matched_field": field_name,
                            "data_value": data_value,
                            "text_value": claim_value,
                            "agrees": True,
                            "relative_delta": relative_delta,
                        })
                    elif relative_delta <= 0.20:
                        anchored_claims.append({
                            "claim": raw_match,
                            "modality": modality,
                            "matched_field": field_name,
                            "data_value": data_value,
                            "text_value": claim_value,
                            "agrees": False,
                            "relative_delta": relative_delta,
                        })

        return anchored_claims
