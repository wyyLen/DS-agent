import json
import os
from typing import Any, Optional, Union

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.ingestion.pipeline import run_transformations
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    get_response_synthesizer,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeWithScore,
    QueryBundle,
    QueryType,
    TransformComponent,
)
from llama_index.retrievers.bm25 import BM25Retriever

from dsagent_core.const import EXAMPLE_DATA_PATH
from metagpt.rag.factories import (
    get_rag_embedding,
    get_rag_llm,
    get_rankers,
    get_retriever,
)
from metagpt.rag.factories.retriever import RetrieverFactory
from metagpt.rag.interface import NoEmbedding
from metagpt.rag.retrievers.base import ModifiableRAGRetriever, PersistableRAGRetriever
from dsagent_core.rag.retrievers.customMixtureRetriever import CustomMixtureRetriever
from dsagent_core.rag.retrievers.hybrid_retriever import SimpleHybridRetriever
from metagpt.rag.schema import (
    BaseRankerConfig,
    BaseRetrieverConfig,
)
from metagpt.utils.common import import_class


class CustomMixtureEngine(RetrieverQueryEngine):
    def __init__(
        self,
        retriever: CustomMixtureRetriever,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[list[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        index: Optional[VectorStoreIndex] = None,
    ) -> None:
        super().__init__(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager,
        )
        self.index = index

    @classmethod
    def from_docs(
        cls,
        input_dir: str = None,
        input_files: list[str] = None,
        transformations: Optional[list[TransformComponent]] = None,
        embed_model: BaseEmbedding = None,
        llm: LLM = None,
        retriever_configs: list[BaseRetrieverConfig] = None,
        ranker_configs: list[BaseRankerConfig] = None,
    ) -> "CustomMixtureEngine":
        if not input_dir and not input_files:
            raise ValueError("Can not find input dir or input files.")
        documents = cls._load_exp_data(input_files)
        cls._fix_document_metadata(documents)
        index = VectorStoreIndex.from_documents(
            documents=documents,
            transformations=transformations or [SentenceSplitter()],
            embed_model=cls._resolve_embed_model(embed_model, retriever_configs),
            show_progress=True,
        )
        return cls._from_index(index, llm=llm, retriever_configs=retriever_configs, ranker_configs=ranker_configs)

    @classmethod
    def _load_exp_data(cls, input_files: list[str]):
        documents = []
        for file in input_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for exp in data:
                exp_content = {
                    'question': exp['task'],
                    'solution': exp['solution'],
                    'metadata': exp['metadata']
                }
                document = Document(text=json.dumps(exp_content))
                document.metadata["type"] = exp["metadata"]
                for task_type in exp["metadata"]:
                    document.excluded_embed_metadata_keys.append(task_type)
                documents.append(document)
        return documents

    async def aretrieve(self, query: QueryType) -> list[NodeWithScore]:
        """Allow query to be str."""
        query_bundle = QueryBundle(query) if isinstance(query, str) else query
        nodes = await super().aretrieve(query_bundle)
        self._try_reconstruct_obj(nodes)
        return nodes

    def add_exp(self, exp: str):
        """
            this function allow to add new docs to the index, without reloading the whole rag engine.
            requires: retriever must has add_nodes function.
        """
        self._ensure_retriever_modifiable()
        doc_chunk = [Document(text=exp)]
        nodes = run_transformations(doc_chunk, transformations=self.index._transformations)
        self._save_nodes(nodes)

    def add_docs(self, input_files: list[str]):
        """
            Add docs to retriever. retriever must has add_nodes function.
        """
        self._ensure_retriever_modifiable()
        documents = SimpleDirectoryReader(input_files=input_files).load_data()
        self._fix_document_metadata(documents)
        nodes = run_transformations(documents, transformations=self.index._transformations)
        self._save_nodes(nodes)

    def persist(self, persist_dir: Union[str, os.PathLike], **kwargs):
        """Persist."""
        self._ensure_retriever_persistable()
        self._persist(str(persist_dir), **kwargs)

    @classmethod
    def _from_index(
            cls,
            index: VectorStoreIndex,
            llm: LLM = None,
            retriever_configs: list[BaseRetrieverConfig] = None,
            ranker_configs: list[BaseRankerConfig] = None,
    ) -> "CustomMixtureEngine":
        llm = llm or get_rag_llm()
        retriever = get_retriever(configs=retriever_configs, index=index)  # Default index.as_retriever
        # retriever = RetrieverFactory().create_mixture_retriever(config=retriever_configs[0])
        rankers = get_rankers(configs=ranker_configs, llm=llm)  # Default []

        return cls(
            retriever=retriever,
            node_postprocessors=rankers,
            response_synthesizer=get_response_synthesizer(llm=llm),
            index=index,
        )


    @classmethod
    def _from_mixture_index(cls, index: VectorStoreIndex, llm: LLM = None, retriever_configs: list[BaseRetrieverConfig] = None,
                            ranker_configs: list[BaseRankerConfig] = None, bm25_index: BM25Retriever = None) -> "MixtureTypeEngine":
        llm = llm or get_rag_llm()
        retriever = get_retriever(configs=retriever_configs, index=index)  # Default index.as_retriever
        rankers = get_rankers(configs=ranker_configs, llm=llm)
        pass

    # def update_rag_engine(self, index: VectorStoreIndex) -> "CustomEngine":
    #     return self._from_index(index, retriever_configs=[FAISSRetrieverConfig()])

    def _ensure_retriever_modifiable(self):
        self._ensure_retriever_of_type(ModifiableRAGRetriever)

    def _ensure_retriever_persistable(self):
        self._ensure_retriever_of_type(PersistableRAGRetriever)

    def _ensure_retriever_of_type(self, required_type: BaseRetriever):
        """Ensure that self.retriever is required_type, or at least one of its components, if it's a SimpleHybridRetriever.
        Args:
            required_type: The class that the retriever is expected to be an instance of.
        """
        if isinstance(self.retriever, SimpleHybridRetriever):
            if not any(isinstance(r, required_type) for r in self.retriever.retrievers):
                raise TypeError(
                    f"Must have at least one retriever of type {required_type.__name__} in SimpleHybridRetriever"
                )

        if not isinstance(self.retriever, required_type):
            raise TypeError(f"The retriever is not of type {required_type.__name__}: {type(self.retriever)}")

    def _save_nodes(self, nodes: list[BaseNode]):
        self.retriever.add_nodes(nodes)

    def _persist(self, persist_dir: str, **kwargs):
        self.retriever.persist(persist_dir, **kwargs)

    @staticmethod
    def _try_reconstruct_obj(nodes: list[NodeWithScore]):
        """If node is object, then dynamically reconstruct object, and save object to node.metadata["obj"]."""
        for node in nodes:
            if node.metadata.get("is_obj", False):
                obj_cls = import_class(node.metadata["obj_cls_name"], node.metadata["obj_mod_name"])
                obj_dict = json.loads(node.metadata["obj_json"])
                node.metadata["obj"] = obj_cls(**obj_dict)

    @staticmethod
    def _fix_document_metadata(documents: list[Document]):
        pass
        # """LlamaIndex keep metadata['file_path'], which is unnecessary, maybe deleted in the near future."""
        # for doc in documents:
        #     doc.excluded_embed_metadata_keys.append("file_path")

    @staticmethod
    def _resolve_embed_model(embed_model: BaseEmbedding = None, configs: list[Any] = None) -> BaseEmbedding:
        if configs and all(isinstance(c, NoEmbedding) for c in configs):
            return MockEmbedding(embed_dim=1)

        return embed_model or get_rag_embedding()

    def get_index(self):
        return self.index

    def get_retriever(self):
        return self.retriever

    async def _aquery(self, query_bundle: QueryBundle) -> tuple[list[NodeWithScore], QueryBundle]:
        """Answer a query."""
        nodes = await self.aretrieve(query_bundle)
        return nodes, query_bundle

    async def get_synthesizer_response(self, nodes: list[NodeWithScore], query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
                CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

